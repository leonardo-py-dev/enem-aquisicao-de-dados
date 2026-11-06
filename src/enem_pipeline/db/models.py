import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.types import JSON, TypeDecorator
from sqlalchemy.orm import relationship
from enem_pipeline.db.session import Base


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


class CompatibleVector(TypeDecorator):
    """Vector type that gracefully resolves to pgvector on PostgreSQL and JSON/TEXT on SQLite."""
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector
                return dialect.type_descriptor(Vector(1536))
            except Exception:
                return dialect.type_descriptor(JSON())
        return dialect.type_descriptor(JSON())


class Source(Base):
    """Cadastro das fontes de dados coletadas."""
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(String(50), nullable=False)  # education, statistics, legislative, academic, news
    base_url = Column(String(500), nullable=False)
    access_type = Column(String(50), nullable=False)  # api_rest, structured_json, odata, scraping
    reliability_level = Column(String(20), default="high")  # high, medium, low
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)

    runs = relationship("CollectionRun", back_populates="source", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="source", cascade="all, delete-orphan")
    indicators = relationship("Indicator", back_populates="source", cascade="all, delete-orphan")


class CollectionRun(Base):
    """Histórico e auditoria de cada execução de coleta."""
    __tablename__ = "collection_runs"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)
    started_at = Column(DateTime, default=utc_now, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String(30), default="running")  # running, success, failed, partial
    records_found = Column(Integer, default=0)
    records_inserted = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_rejected = Column(Integer, default=0)
    error_details = Column(Text, nullable=True)

    source = relationship("Source", back_populates="runs")


class Document(Base):
    """Documentos textuais coletados (notícias, proposições legislativas, artigos, redações)."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)
    external_id = Column(String(255), nullable=True, index=True)
    title = Column(String(1000), nullable=True)
    original_content = Column(Text, nullable=False)
    processed_content = Column(Text, nullable=True)
    url = Column(String(1000), nullable=True)
    language = Column(String(10), default="pt")
    published_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)
    collected_at = Column(DateTime, default=utc_now, nullable=False)
    content_hash = Column(String(64), unique=True, nullable=False, index=True)
    metadata_json = Column(JSON, nullable=True)
    embedding = Column(CompatibleVector, nullable=True)

    source = relationship("Source", back_populates="documents")
    topics = relationship("DocumentTopic", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_documents_source_published", "source_id", "published_at"),
    )


class Indicator(Base):
    """Indicadores estatísticos e séries temporais estruturadas (IBGE, IPEA)."""
    __tablename__ = "indicators"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=False)
    geographic_location = Column(String(100), default="Brasil")
    reference_period = Column(String(50), nullable=True)
    methodology = Column(Text, nullable=True)
    collected_at = Column(DateTime, default=utc_now, nullable=False)
    metadata_json = Column(JSON, nullable=True)

    source = relationship("Source", back_populates="indicators")

    __table_args__ = (
        Index("ix_indicators_name_period", "name", "reference_period"),
    )


class Topic(Base):
    """Catálogo dos 10 eixos temáticos do ENEM e subtemas/problemas sociais específicos."""
    __tablename__ = "topics"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    thematic_axis = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    parent_topic_id = Column(Integer, ForeignKey("topics.id"), nullable=True)

    parent = relationship("Topic", remote_side=[id], backref="subtopics")
    document_associations = relationship("DocumentTopic", back_populates="topic", cascade="all, delete-orphan")


class DocumentTopic(Base):
    """Relacionamento n:n entre documentos e assuntos/eixos com grau de confiança."""
    __tablename__ = "document_topics"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=False, index=True)
    relevance_score = Column(Float, default=1.0)
    classification_method = Column(String(50), default="lexical_rules_v1")
    classifier_version = Column(String(20), default="1.0.0")
    created_at = Column(DateTime, default=utc_now)

    document = relationship("Document", back_populates="topics")
    topic = relationship("Topic", back_populates="document_associations")

    __table_args__ = (
        UniqueConstraint("document_id", "topic_id", "classification_method", name="uq_doc_topic_method"),
    )


class HistoricalTheme(Base):
    """Histórico oficial dos temas de redação do ENEM."""
    __tablename__ = "historical_themes"

    id = Column(Integer, primary_key=True, index=True)
    year = Column(Integer, unique=True, nullable=False, index=True)
    official_theme = Column(String(500), nullable=False)
    thematic_axis = Column(String(100), nullable=False)
    central_problem = Column(Text, nullable=False)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=True)
    source_url = Column(String(500), nullable=True)
    metadata_json = Column(JSON, nullable=True)


class DatasetVersion(Base):
    """Controle de versão para conjuntos de dados de treinamento de Machine Learning."""
    __tablename__ = "dataset_versions"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=utc_now)
    description = Column(Text, nullable=False)
    collection_period = Column(String(100), nullable=False)
    processing_version = Column(String(50), nullable=False)
    record_count = Column(Integer, default=0)
