import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from enem_pipeline.db.models import Base, Source, Document, Indicator, Topic, DocumentTopic, HistoricalTheme
from enem_pipeline.storage.raw_store import RawStore


@pytest.fixture
def test_db():
    """Cria um banco SQLite em memória isolado para os testes."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_source_creation(test_db):
    """Testa criação e integridade da tabela sources."""
    source = Source(
        name="test_source",
        category="education",
        base_url="https://example.com",
        access_type="api_rest",
        reliability_level="high",
    )
    test_db.add(source)
    test_db.commit()

    saved = test_db.query(Source).filter_by(name="test_source").first()
    assert saved is not None
    assert saved.category == "education"
    assert saved.active is True


def test_document_deduplication_by_content_hash(test_db):
    """Testa a restrição de unicidade do content_hash para documentos."""
    source = Source(name="s1", category="cat", base_url="http://x", access_type="api")
    test_db.add(source)
    test_db.commit()

    content = "Texto importante para teste de unicidade."
    h = RawStore.calculate_hash(content)

    doc1 = Document(
        source_id=source.id,
        title="Doc 1",
        original_content=content,
        content_hash=h,
    )
    test_db.add(doc1)
    test_db.commit()

    # Tentativa de inserir segundo documento com mesmo hash
    with pytest.raises(Exception):
        doc2 = Document(
            source_id=source.id,
            title="Doc 2",
            original_content=content,
            content_hash=h,
        )
        test_db.add(doc2)
        test_db.commit()
    test_db.rollback()


def test_indicator_creation_and_update(test_db):
    """Testa integridade e persistência de indicadores estatísticos."""
    source = Source(name="s2", category="stats", base_url="http://y", access_type="api")
    test_db.add(source)
    test_db.commit()

    ind = Indicator(
        source_id=source.id,
        name="Taxa de Desemprego",
        value=7.8,
        unit="%",
        reference_period="2024",
        geographic_location="Brasil",
    )
    test_db.add(ind)
    test_db.commit()

    saved = test_db.query(Indicator).filter_by(name="Taxa de Desemprego").first()
    assert saved.value == 7.8
    assert saved.unit == "%"
