import sys
import time
import traceback
import datetime
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import httpx
from sqlalchemy.orm import Session
from loguru import logger

from enem_pipeline.config import settings
from enem_pipeline.db.models import Source, CollectionRun, Document, Indicator
from enem_pipeline.storage.raw_store import RawStore

# Injeta truststore para suporte a certificados Windows nativos
if sys.platform == "win32":
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


class BaseCollector(ABC):
    """Classe base para coletores de dados resilientes e auditáveis."""

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.raw_store = RawStore()
        self.seen_hashes: set = set()
        self.client = httpx.Client(
            timeout=settings.DEFAULT_HTTP_TIMEOUT,
            headers={"User-Agent": settings.USER_AGENT, "Accept": "application/json"},
            follow_redirects=True,
        )

    def get_or_create_source(self, db: Session) -> Source:
        """Obtém ou registra a fonte no banco de dados."""
        source = db.query(Source).filter_by(name=self.source_name).first()
        if not source:
            source = Source(
                name=self.source_name,
                category="general",
                base_url="",
                access_type="api_rest",
                reliability_level="high",
                active=True,
            )
            db.add(source)
            db.commit()
            db.refresh(source)
        return source

    def start_run(self, db: Session) -> CollectionRun:
        """Inicializa e registra uma nova execução de coleta."""
        self.seen_hashes.clear()
        source = self.get_or_create_source(db)
        now = datetime.datetime.now(datetime.timezone.utc)
        run = CollectionRun(
            source_id=source.id,
            started_at=now,
            status="running",
            records_found=0,
            records_inserted=0,
            records_updated=0,
            records_rejected=0,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        logger.info(f"[{self.source_name.upper()}] Execução #{run.id} iniciada.")
        return run

    def finish_run(
        self,
        db: Session,
        run: CollectionRun,
        status: str = "success",
        error_details: Optional[str] = None,
    ):
        """Finaliza a execução atualizando os contadores e status no banco."""
        run.finished_at = datetime.datetime.now(datetime.timezone.utc)
        run.status = status
        run.error_details = error_details
        db.commit()
        duration = (run.finished_at - run.started_at).total_seconds()
        logger.info(
            f"[{self.source_name.upper()}] Execução #{run.id} finalizada em {duration:.1f}s com status '{status}'. "
            f"Encontrados: {run.records_found}, Inseridos: {run.records_inserted}, "
            f"Atualizados: {run.records_updated}, Rejeitados: {run.records_rejected}."
        )

    def fetch_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Executa requisições HTTP GET controladas com retries e exponential backoff."""
        retries = settings.DEFAULT_MAX_RETRIES
        delay = 1.0

        for attempt in range(1, retries + 1):
            try:
                response = self.client.get(url, params=params)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    logger.warning(f"[{self.source_name}] Rate limit (429) em {url}. Aguardando {delay * 2}s...")
                    time.sleep(delay * 2)
                else:
                    logger.warning(
                        f"[{self.source_name}] Resposta HTTP {response.status_code} na tentativa {attempt}/{retries}: {url}"
                    )
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                logger.warning(
                    f"[{self.source_name}] Falha na requisição ({exc}) tentativa {attempt}/{retries} para {url}"
                )

            if attempt < retries:
                time.sleep(delay)
                delay *= 2

        raise RuntimeError(f"[{self.source_name}] Falha após {retries} tentativas para {url}")

    def save_document(
        self,
        db: Session,
        run: CollectionRun,
        source: Source,
        external_id: Optional[str],
        title: Optional[str],
        original_content: str,
        url: Optional[str] = None,
        language: str = "pt",
        published_at: Optional[datetime.datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        processed_content: Optional[str] = None,
    ) -> Optional[Document]:
        """Valida, deduplica e salva um documento textual no banco e na camada RAW."""
        run.records_found += 1

        if not original_content or not original_content.strip():
            run.records_rejected += 1
            logger.debug(f"[{self.source_name}] Documento rejeitado: conteúdo vazio.")
            return None

        # Deduplicação via SHA-256 do conteúdo original
        content_hash = RawStore.calculate_hash(original_content)
        if content_hash in self.seen_hashes:
            run.records_rejected += 1
            logger.debug(f"[{self.source_name}] Documento duplicado ignorado no lote em memória ({content_hash[:8]}).")
            return None

        existing = db.query(Document).filter_by(content_hash=content_hash).first()
        if existing:
            self.seen_hashes.add(content_hash)
            run.records_rejected += 1
            logger.debug(f"[{self.source_name}] Documento duplicado ignorado do banco ({content_hash[:8]}).")
            return existing

        self.seen_hashes.add(content_hash)

        # Salva na camada RAW imutável em disco
        self.raw_store.save_raw(
            source_name=self.source_name,
            external_id=external_id or content_hash[:12],
            raw_data=original_content,
            metadata=metadata or {},
        )

        doc = Document(
            source_id=source.id,
            external_id=str(external_id) if external_id else None,
            title=title[:1000] if title else None,
            original_content=original_content,
            processed_content=processed_content,
            url=url,
            language=language,
            published_at=published_at,
            collected_at=datetime.datetime.now(datetime.timezone.utc),
            content_hash=content_hash,
            metadata_json=metadata,
        )
        db.add(doc)
        db.flush()
        run.records_inserted += 1
        return doc

    def save_indicator(
        self,
        db: Session,
        run: CollectionRun,
        source: Source,
        name: str,
        value: float,
        unit: str,
        description: Optional[str] = None,
        geographic_location: str = "Brasil",
        reference_period: Optional[str] = None,
        methodology: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Indicator]:
        """Salva um indicador estatístico com deduplicação por série, período e localidade."""
        run.records_found += 1

        existing = (
            db.query(Indicator)
            .filter_by(
                source_id=source.id,
                name=name,
                reference_period=reference_period,
                geographic_location=geographic_location,
            )
            .first()
        )

        if existing:
            # Atualiza valor se for mais recente
            existing.value = float(value)
            existing.metadata_json = metadata
            run.records_updated += 1
            return existing

        indicator = Indicator(
            source_id=source.id,
            name=name,
            description=description,
            value=float(value),
            unit=unit,
            geographic_location=geographic_location,
            reference_period=reference_period,
            methodology=methodology,
            collected_at=datetime.datetime.now(datetime.timezone.utc),
            metadata_json=metadata,
        )
        db.add(indicator)
        run.records_inserted += 1
        return indicator

    @abstractmethod
    def collect(self, db: Session, max_items: int = 50) -> CollectionRun:
        """Método principal que orquestra a coleta da fonte específica."""
        pass
