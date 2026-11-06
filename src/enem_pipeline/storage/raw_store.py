import json
import hashlib
import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger
from enem_pipeline.config import settings


class RawStore:
    """Gerencia o armazenamento imutável de dados brutos na Camada RAW (Data Lake local)."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or settings.RAW_DATA_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def calculate_hash(content: str) -> str:
        """Calcula o hash SHA-256 do conteúdo para deduplicação estrita."""
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

    def save_raw(
        self,
        source_name: str,
        external_id: str,
        raw_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        file_ext: str = "json",
    ) -> Path:
        """Salva o payload bruto original em disco mantendo estrutura de diretórios por data."""
        now = datetime.datetime.now(datetime.timezone.utc)
        year_str = now.strftime("%Y")
        month_str = now.strftime("%m")
        day_str = now.strftime("%d")

        target_dir = self.base_dir / source_name / year_str / month_str / day_str
        target_dir.mkdir(parents=True, exist_ok=True)

        clean_ext_id = "".join(c for c in str(external_id) if c.isalnum() or c in ("-", "_"))
        timestamp_str = now.strftime("%H%M%S_%f")
        filename = f"{clean_ext_id}_{timestamp_str}.{file_ext}"
        filepath = target_dir / filename

        envelope = {
            "source": source_name,
            "external_id": external_id,
            "collected_at": now.isoformat() + "Z",
            "metadata": metadata or {},
            "raw_payload": raw_data,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(envelope, f, ensure_ascii=False, indent=2)
            logger.debug(f"Payload RAW salvo em: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Erro ao salvar arquivo RAW {filepath}: {e}")
            raise
