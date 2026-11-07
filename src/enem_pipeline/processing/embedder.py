import os
import time
from typing import List, Optional
import httpx
from sqlalchemy.orm import Session
from loguru import logger

from enem_pipeline.config import settings
from enem_pipeline.db.models import Document


class GeminiEmbedder:
    """Gerador de Embeddings Semânticos utilizando a Google Gemini API (text-embedding-004)."""

    MODEL_NAME = "text-embedding-004"
    DIMENSION = 768

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        self.client = httpx.Client(timeout=30.0)

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Gera o vetor de 768 dimensões para o texto informado."""
        if not self.api_key:
            logger.error("[EMBEDDER] GEMINI_API_KEY não configurada. Impossível gerar embeddings.")
            return None

        # Trunca texto se for excessivamente longo para evitar estouro de tokens
        truncated_text = text[:8000]

        # Tenta usar a biblioteca oficial google-genai se instalada
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            result = client.models.embed_content(
                model=self.MODEL_NAME,
                contents=truncated_text,
            )
            if hasattr(result, "embeddings") and result.embeddings:
                return result.embeddings[0].values
        except Exception:
            # Fallback direto via REST API caso o pacote não esteja no path local
            pass

        # Chamada REST direta para a API do Google Gemini
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL_NAME}:embedContent?key={self.api_key}"
        payload = {
            "model": f"models/{self.MODEL_NAME}",
            "content": {
                "parts": [{"text": truncated_text}]
            }
        }

        try:
            response = self.client.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                return data.get("embedding", {}).get("values")
            elif response.status_code == 429:
                logger.warning("[EMBEDDER] Rate limit da Gemini API. Aguardando 2s...")
                time.sleep(2.0)
            else:
                logger.warning(f"[EMBEDDER] Erro HTTP {response.status_code}: {response.text}")
        except Exception as exc:
            logger.error(f"[EMBEDDER] Falha na requisição de embedding: {exc}")

        return None

    def embed_pending_documents(self, db: Session, limit: int = 50) -> int:
        """Busca documentos com embedding nulo no banco e preenche seus vetores."""
        if not self.api_key:
            logger.warning("[EMBEDDER] GEMINI_API_KEY não encontrada. Pulando etapa de embeddings.")
            return 0

        # Seleciona documentos com embedding nulo
        docs = (
            db.query(Document)
            .filter(Document.embedding.is_(None))
            .order_by(Document.id.desc())
            .limit(limit)
            .all()
        )

        if not docs:
            logger.info("[EMBEDDER] Nenhum documento pendente de vetorização.")
            return 0

        logger.info(f"[EMBEDDER] Iniciando vetorização de {len(docs)} documentos...")
        processed = 0

        for doc in docs:
            # Combina título e conteúdo processado para máxima precisão semântica
            content_to_embed = f"{doc.title or ''}\n\n{doc.processed_content or doc.original_content}"
            vector = self.generate_embedding(content_to_embed)

            if vector and len(vector) == self.DIMENSION:
                doc.embedding = vector
                processed += 1
                if processed % 10 == 0:
                    db.commit()
                    logger.info(f"[EMBEDDER] {processed}/{len(docs)} documentos vetorizados.")
                time.sleep(0.1)  # Respeito a rate limits

        db.commit()
        logger.info(f"[EMBEDDER] Vetorização finalizada. Total processado: {processed}/{len(docs)}.")
        return processed
