from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from enem_pipeline.db.models import Base, Source, Document, DocumentTopic
from enem_pipeline.db.seed import seed_database
from enem_pipeline.collectors.inep import InepCollector
from enem_pipeline.processing.classifier import ThematicClassifier
from enem_pipeline.processing.cleaner import TextCleaner


def test_end_to_end_pipeline_flow(tmp_path):
    """Testa o fluxo integrado de ponta a ponta: Coleta -> RAW -> Validação -> Processamento -> Classificação."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        # 1. Seed
        seed_database(db)
        sources = db.query(Source).count()
        assert sources == 7

        # 2. Coleta INEP (5 temas)
        collector = InepCollector()
        collector.raw_store.base_dir = tmp_path / "raw"
        run = collector.collect(db, max_items=5)
        assert run.status == "success"
        assert run.records_inserted == 5

        # 3. Validação de documentos
        docs = db.query(Document).all()
        assert len(docs) == 5
        for d in docs:
            assert d.content_hash is not None
            assert len(d.content_hash) == 64

        # 4. Processamento & Normalização
        for d in docs:
            d.processed_content = TextCleaner.normalize_text(d.original_content)
        db.commit()

        # 5. Classificação Temática
        classifier = ThematicClassifier()
        classified_count = classifier.process_and_classify_all(db, limit=10)
        assert classified_count >= 5

        associations = db.query(DocumentTopic).all()
        assert len(associations) >= 5
        for assoc in associations:
            assert 0.0 <= assoc.relevance_score <= 1.0
            assert assoc.classification_method == "lexical_rules_v1"

    finally:
        db.close()
