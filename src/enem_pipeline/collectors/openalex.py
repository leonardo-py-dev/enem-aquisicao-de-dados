import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from loguru import logger
from enem_pipeline.collectors.base import BaseCollector


class OpenAlexCollector(BaseCollector):
    """Coletor para produção científica e artigos acadêmicos sobre a realidade brasileira."""

    def __init__(self):
        super().__init__(source_name="openalex")
        self.works_url = "https://api.openalex.org/works"

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[Dict[str, Any]]) -> str:
        """Reconstrói o abstract completo a partir do inverted index do OpenAlex."""
        if not inverted_index or not isinstance(inverted_index, dict):
            return ""
        try:
            words = sorted([(pos, word) for word, positions in inverted_index.items() for pos in positions])
            return " ".join(w for _, w in words)
        except Exception:
            return ""

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime.datetime]:
        if not date_str:
            return None
        try:
            return datetime.datetime.strptime(date_str[:10], "%Y-%m-%d")
        except Exception:
            return None

    def collect(self, db: Session, max_items: int = 50) -> any:
        run = self.start_run(db)
        source = self.get_or_create_source(db)

        # Temas sociais de alto impacto para o ENEM
        search_terms = [
            "desigualdade brasil educacao",
            "direitos humanos meio ambiente brasil",
            "saude publica tecnologia sociedade brasil",
        ]

        per_search = max(5, max_items // len(search_terms))

        try:
            for term in search_terms:
                params = {
                    "filter": f"default.search:{term}",
                    "per_page": per_search,
                    "sort": "publication_date:desc",
                }
                logger.info(f"[OPENALEX] Consultando busca: '{term}'...")
                data = self.fetch_json(self.works_url, params=params)

                results = data.get("results", []) if isinstance(data, dict) else []

                for item in results:
                    openalex_id = item.get("id", "")
                    clean_id = openalex_id.split("/")[-1] if "/" in openalex_id else openalex_id
                    title = item.get("title", "").strip() if item.get("title") else ""
                    if not title:
                        run.records_rejected += 1
                        continue

                    abstract = self._reconstruct_abstract(item.get("abstract_inverted_index"))
                    doi = item.get("doi")
                    pub_date = self._parse_date(item.get("publication_date"))
                    pub_year = item.get("publication_year")

                    # Extrai conceitos/tópicos identificados pelo OpenAlex
                    concepts = [c.get("display_name") for c in item.get("concepts", []) if c.get("display_name")]
                    concepts_str = ", ".join(concepts[:5])

                    content = (
                        f"ARTIGO CIENTÍFICO: {title}\n\n"
                        f"Resumo / Abstract:\n{abstract if abstract else 'Resumo não disponibilizado.'}\n\n"
                        f"Ano: {pub_year} | DOI: {doi}\n"
                        f"Conceitos Acadêmicos: {concepts_str}\n"
                        f"Identificador OpenAlex: {openalex_id}"
                    )

                    metadata = {
                        "openalex_id": openalex_id,
                        "doi": doi,
                        "publication_year": pub_year,
                        "concepts": concepts[:10],
                        "cited_by_count": item.get("cited_by_count", 0),
                    }

                    self.save_document(
                        db=db,
                        run=run,
                        source=source,
                        external_id=f"OPENALEX_{clean_id}",
                        title=f"[Artigo] {title}",
                        original_content=content,
                        url=doi or openalex_id,
                        language=item.get("language") or "pt",
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
