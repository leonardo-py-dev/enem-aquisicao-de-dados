import datetime
from sqlalchemy.orm import Session
from loguru import logger
from enem_pipeline.collectors.base import BaseCollector
from enem_pipeline.db.models import HistoricalTheme


class InepCollector(BaseCollector):
    """Coletor oficial do INEP para histórico de provas, temas de redação e metadados."""

    def __init__(self):
        super().__init__(source_name="inep")

    def collect(self, db: Session, max_items: int = 50) -> any:
        run = self.start_run(db)
        source = self.get_or_create_source(db)

        try:
            # 1. Carrega temas consolidados da tabela historical_themes e sincroniza como Documentos textuais
            themes = db.query(HistoricalTheme).order_by(HistoricalTheme.year.desc()).all()
            if not themes:
                logger.info("[INEP] Semeando histórico oficial do ENEM...")
                from enem_pipeline.db.seed import seed_database
                seed_database(db)
                themes = db.query(HistoricalTheme).order_by(HistoricalTheme.year.desc()).all()

            for theme in themes[:max_items]:
                content = (
                    f"TEMA DA REDAÇÃO DO ENEM {theme.year}: {theme.official_theme}\n\n"
                    f"Eixo Temático Oficial: {theme.thematic_axis}\n"
                    f"Problema Social Central: {theme.central_problem}\n"
                    f"Aplicação: Exame Nacional do Ensino Médio - Prova Oficial de Redação.\n"
                    f"Fonte Oficial: INEP / Ministério da Educação."
                )

                metadata = {
                    "year": theme.year,
                    "thematic_axis": theme.thematic_axis,
                    "central_problem": theme.central_problem,
                    "source_type": "official_exam_theme",
                }

                # Publicação oficial estimada em novembro de cada ano
                pub_date = datetime.datetime(theme.year, 11, 5)

                self.save_document(
                    db=db,
                    run=run,
                    source=source,
                    external_id=f"ENEM_{theme.year}",
                    title=f"Tema da Redação ENEM {theme.year}: {theme.official_theme}",
                    original_content=content,
                    url=theme.source_url or "https://www.gov.br/inep/pt-br/areas-de-atuacao/avaliacao-e-exames-educacionais/enem",
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
