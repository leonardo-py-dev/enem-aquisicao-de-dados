import time
from apscheduler.schedulers.background import BlockingScheduler
from loguru import logger
from enem_pipeline.db.session import get_db
from enem_pipeline.collectors import InepCollector, IbgeCollector, CamaraCollector, OpenAlexCollector
from enem_pipeline.processing.classifier import ThematicClassifier


def job_collect_news_and_propositions():
    """Job de alta frequência: notícias do IBGE e matérias da Câmara dos Deputados."""
    logger.info("[CRON] Executando coleta periódica de notícias e proposições...")
    with get_db() as db:
        IbgeCollector().collect(db, max_items=20)
        CamaraCollector().collect(db, max_items=25)
        ThematicClassifier().process_and_classify_all(db, limit=100)


def job_collect_academic():
    """Job semanal: pesquisas acadêmicas do OpenAlex."""
    logger.info("[CRON] Executando coleta de pesquisas científicas (OpenAlex)...")
    with get_db() as db:
        OpenAlexCollector().collect(db, max_items=30)
        ThematicClassifier().process_and_classify_all(db, limit=100)


def start_scheduler():
    """Inicia o agendador em background com intervalos configurados."""
    scheduler = BlockingScheduler()

    # Job a cada 6 horas para notícias e câmara
    scheduler.add_job(job_collect_news_and_propositions, "interval", hours=6, id="news_propositions")

    # Job diário para artigos acadêmicos
    scheduler.add_job(job_collect_academic, "interval", days=1, id="academic_papers")

    logger.info("Agendador APScheduler iniciado com sucesso. Pressione Ctrl+C para encerrar.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Agendador encerrado com segurança.")


if __name__ == "__main__":
    start_scheduler()
