import datetime
from typing import Optional
from sqlalchemy.orm import Session
from loguru import logger
from enem_pipeline.collectors.base import BaseCollector


class CamaraCollector(BaseCollector):
    """Coletor para projetos de lei, emendas e proposições legislativas da Câmara dos Deputados."""

    def __init__(self):
        super().__init__(source_name="camara")
        self.proposicoes_url = "https://dadosabertos.camara.leg.br/api/v2/proposicoes"

    def _parse_camara_date(self, date_str: str) -> Optional[datetime.datetime]:
        """Converte ISO ou data da câmara para objeto datetime."""
        if not date_str:
            return None
        try:
            return datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            try:
                return datetime.datetime.strptime(date_str[:10], "%Y-%m-%d")
            except Exception:
                return None

    def collect(self, db: Session, max_items: int = 50) -> any:
        run = self.start_run(db)
        source = self.get_or_create_source(db)

        try:
            # Filtra proposições de alto impacto social: Projetos de Lei (PL) e Propostas de Emenda à Constituição (PEC)
            params = {
                "siglaTipo": "PL,PEC,PLP",
                "ordem": "DESC",
                "ordenarPor": "id",
                "itens": max_items,
            }
            logger.info(f"[CÂMARA] Consultando API de Proposições: {self.proposicoes_url}")
            data = self.fetch_json(self.proposicoes_url, params=params)

            items = data.get("dados", []) if isinstance(data, dict) else []
            logger.info(f"[CÂMARA] {len(items)} proposições legislativas recebidas.")

            for item in items:
                ext_id = str(item.get("id"))
                sigla = item.get("siglaTipo", "PL")
                numero = item.get("numero", "")
                ano = item.get("ano", "")
                ementa = item.get("ementa", "").strip()
                uri = item.get("uri", "")
                pub_date = self._parse_camara_date(item.get("dataApresentacao"))

                if not ementa:
                    run.records_rejected += 1
                    continue

                title = f"{sigla} {numero}/{ano}: {ementa[:150]}"
                content = (
                    f"PROPOSIÇÃO LEGISLATIVA: {sigla} {numero}/{ano}\n\n"
                    f"Ementa Oficial:\n{ementa}\n\n"
                    f"Tipo: {sigla} | Ano: {ano}\n"
                    f"Link Oficial: {uri}"
                )

                metadata = {
                    "siglaTipo": sigla,
                    "numero": numero,
                    "ano": ano,
                    "uri": uri,
                }

                self.save_document(
                    db=db,
                    run=run,
                    source=source,
                    external_id=f"CAMARA_{ext_id}",
                    title=f"[Câmara] {title}",
                    original_content=content,
                    url=uri,
                    language="pt",
                    published_at=pub_date,
                    metadata=metadata,
                )

            db.commit()
            self.finish_run(db, run, status="success")
        except Exception as exc:
            db.rollback()
            self.finish_run(db, run, status="failed", error_details=str(exc))
            raise

        return run
