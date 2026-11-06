import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from enem_pipeline.db.models import Base, Source, Document, CollectionRun
from enem_pipeline.collectors.base import BaseCollector
from enem_pipeline.collectors.inep import InepCollector
from enem_pipeline.collectors.ibge import IbgeCollector
from enem_pipeline.collectors.camara import CamaraCollector
from enem_pipeline.collectors.openalex import OpenAlexCollector


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


class DummyCollector(BaseCollector):
    def __init__(self):
        super().__init__(source_name="dummy_source")

    def collect(self, db, max_items=10):
        run = self.start_run(db)
        source = self.get_or_create_source(db)
        self.save_document(
            db=db,
            run=run,
            source=source,
            external_id="DUMMY_1",
            title="Título de Teste",
            original_content="Conteúdo de teste para validação de pipeline.",
        )
        self.finish_run(db, run, status="success")
        return run


def test_base_collector_lifecycle_and_dedup(memory_db):
    """Testa ciclo de vida do coletor e deduplicação estrita."""
    collector = DummyCollector()

    # Execução 1: Deve inserir
    run1 = collector.collect(memory_db)
    assert run1.records_found == 1
    assert run1.records_inserted == 1
    assert run1.records_rejected == 0
    assert run1.status == "success"

    # Execução 2: Mesmo conteúdo deve ser rejeitado como duplicata
    run2 = collector.collect(memory_db)
    assert run2.records_found == 1
    assert run2.records_inserted == 0
    assert run2.records_rejected == 1
    assert run2.status == "success"


def test_inep_collector_seeding_and_ingestion(memory_db):
    """Testa coletor do INEP ingerindo histórico."""
    collector = InepCollector()
    run = collector.collect(memory_db, max_items=5)
    assert run.records_found == 5
    assert run.records_inserted == 5

    docs = memory_db.query(Document).filter_by(source_id=run.source_id).all()
    assert len(docs) == 5
    assert "ENEM" in docs[0].title


@patch("enem_pipeline.collectors.ibge.IbgeCollector.fetch_json")
def test_ibge_collector_mock(mock_fetch, memory_db):
    """Testa coletor do IBGE com payload simulado."""
    mock_fetch.return_value = {
        "items": [
            {
                "id": "1001",
                "tipo": "Notícia",
                "titulo": "Pesquisa do IBGE aponta avanço educacional",
                "introducao": "Estudo detalha taxas de matrícula e escolaridade.",
                "data_publicacao": "01/10/2026 10:00:00",
                "editorias": "educacao",
                "link": "https://agenciadenoticias.ibge.gov.br/exemplo",
            }
        ]
    }

    collector = IbgeCollector()
    run = collector.collect(memory_db, max_items=1)
    assert run.records_inserted >= 1
    doc = memory_db.query(Document).filter(Document.title.contains("IBGE")).first()
    assert doc is not None
    assert "avanço educacional" in doc.title


@patch("enem_pipeline.collectors.camara.CamaraCollector.fetch_json")
def test_camara_collector_mock(mock_fetch, memory_db):
    """Testa coletor da Câmara dos Deputados com payload simulado."""
    mock_fetch.return_value = {
        "dados": [
            {
                "id": 99999,
                "siglaTipo": "PL",
                "numero": 1234,
                "ano": 2026,
                "ementa": "Dispõe sobre diretrizes para inteligência artificial na educação pública.",
                "uri": "https://camara.leg.br/proposicoes/99999",
                "dataApresentacao": "2026-05-10T14:30:00",
            }
        ]
    }

    collector = CamaraCollector()
    run = collector.collect(memory_db, max_items=1)
    assert run.records_inserted == 1
    doc = memory_db.query(Document).filter(Document.title.contains("PL 1234/2026")).first()
    assert doc is not None
    assert "educação pública" in doc.original_content


@patch("enem_pipeline.collectors.openalex.OpenAlexCollector.fetch_json")
def test_openalex_collector_mock(mock_fetch, memory_db):
    """Testa coletor do OpenAlex com payload acadêmico simulado."""
    mock_fetch.return_value = {
        "results": [
            {
                "id": "https://openalex.org/W999999",
                "title": "Desigualdades Sociais e Educação no Brasil Contemporâneo",
                "publication_year": 2026,
                "publication_date": "2026-03-15",
                "doi": "https://doi.org/10.1000/182",
                "abstract_inverted_index": {
                    "O": [0],
                    "estudo": [1],
                    "analisa": [2],
                    "a": [3],
                    "educacao": [4],
                },
                "concepts": [{"display_name": "Education"}, {"display_name": "Sociology"}],
            }
        ]
    }

    collector = OpenAlexCollector()
    run = collector.collect(memory_db, max_items=1)
    assert run.records_inserted >= 1
    doc = memory_db.query(Document).filter(Document.title.contains("Desigualdades Sociais")).first()
    assert doc is not None
    assert "O estudo analisa a educacao" in doc.original_content
