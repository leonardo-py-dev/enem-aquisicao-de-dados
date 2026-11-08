import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from loguru import logger

from enem_pipeline.db.models import HistoricalTheme, Topic, Document, DocumentTopic, Source
from enem_pipeline.db.seed import HISTORICAL_ENEM_THEMES, INITIAL_TOPICS


def build_training_matrix(
    db: Optional[Session] = None,
    min_year: int = 2005,
    max_year: int = 2025,
    historical_themes: Optional[List[Dict]] = None,
    topics: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """Constrói a matriz tabular de atributos históricos e debates sociais para os 10 eixos do ENEM.

    Gera uma linha por (ano, eixo), permitindo treinar modelos de Ranking / Classificação como XGBoost.
    Suporta execução desacoplada sem sessão de banco (modo in-memory / fallback).
    """
    # 1. Carrega todos os temas históricos e mapeia ano -> eixo
    if historical_themes is not None:
        history_by_year: Dict[int, str] = {
            t["year"]: t["thematic_axis"] for t in historical_themes
        }
    elif db is not None:
        try:
            themes = db.query(HistoricalTheme).order_by(HistoricalTheme.year.asc()).all()
            history_by_year = {t.year: t.thematic_axis for t in themes}
            if not history_by_year:
                history_by_year = {t["year"]: t["thematic_axis"] for t in HISTORICAL_ENEM_THEMES}
        except Exception as e:
            logger.warning(f"[ML] Falha ao consultar HistoricalTheme no DB: {e}. Usando seed estático.")
            history_by_year = {t["year"]: t["thematic_axis"] for t in HISTORICAL_ENEM_THEMES}
    else:
        history_by_year = {t["year"]: t["thematic_axis"] for t in HISTORICAL_ENEM_THEMES}

    # 2. Carrega todos os 10 eixos oficiais
    if topics is not None:
        axes = [top.get("thematic_axis") or top.get("name") for top in topics]
    elif db is not None:
        try:
            db_topics = db.query(Topic).all()
            axes = [top.thematic_axis for top in db_topics]
        except Exception as e:
            logger.warning(f"[ML] Falha ao consultar Topic no DB: {e}. Usando eixos canônicos.")
            axes = []
    else:
        axes = []

    if not axes:
        axes = [top["thematic_axis"] for top in INITIAL_TOPICS]

    # 3. Carrega contagem de documentos por ano, eixo e fonte
    doc_records = []
    if db is not None:
        try:
            doc_query = (
                db.query(
                    Document.published_at,
                    DocumentTopic.relevance_score,
                    Topic.thematic_axis,
                    Source.name.label("source_name"),
                )
                .join(DocumentTopic, Document.id == DocumentTopic.document_id)
                .join(Topic, DocumentTopic.topic_id == Topic.id)
                .join(Source, Document.source_id == Source.id)
                .all()
            )
            for row in doc_query:
                if row.published_at:
                    doc_records.append({
                        "year": row.published_at.year,
                        "axis": row.thematic_axis,
                        "score": row.relevance_score or 1.0,
                        "source": row.source_name,
                    })
        except Exception as e:
            logger.warning(f"[ML] Falha ao consultar documentos no DB: {e}. Usando contagens padrão neutras.")
            doc_records = []

    doc_df = pd.DataFrame(doc_records) if doc_records else pd.DataFrame(columns=["year", "axis", "score", "source"])

    rows = []

    # 4. Itera ano a ano para construir a série temporal sem data leakage
    for target_year in range(min_year, max_year + 1):
        actual_theme_axis = history_by_year.get(target_year, None)

        for axis in axes:
            # --- Features Históricas do INEP ---
            past_occurrences = [y for y, ax in history_by_year.items() if y < target_year and ax == axis]
            
            # Anos desde a última vez que o eixo caiu
            if past_occurrences:
                years_since_last = target_year - max(past_occurrences)
            else:
                years_since_last = target_year - 1998  # Limite inferior do ENEM

            # Veto de repetição consecutiva (caiu no ano anterior?)
            prev_year_axis = history_by_year.get(target_year - 1, None)
            occurred_previous_year = 1 if prev_year_axis == axis else 0

            # Ocorrências nos últimos 3 anos
            recent_3_years = [target_year - 1, target_year - 2, target_year - 3]
            occurred_last_3_years = sum(1 for y in recent_3_years if history_by_year.get(y) == axis)

            # Frequência histórica total acumulada até target_year - 1
            total_past_exams = sum(1 for y in history_by_year.keys() if y < target_year)
            cumulative_frequency = (len(past_occurrences) / total_past_exams) if total_past_exams > 0 else 0.0

            # --- Features de Debate Social no Ano da Prova ---
            if not doc_df.empty:
                year_mask = (doc_df["year"] == target_year) & (doc_df["axis"] == axis)
                prev_mask = (doc_df["year"] == target_year - 1) & (doc_df["axis"] == axis)

                vol_ibge = len(doc_df[year_mask & (doc_df["source"] == "ibge")])
                vol_camara = len(doc_df[year_mask & (doc_df["source"] == "camara")])
                vol_openalex = len(doc_df[year_mask & (doc_df["source"] == "openalex")])
                score_total = doc_df[year_mask]["score"].sum()

                prev_score_total = doc_df[prev_mask]["score"].sum()
                # Aceleração percentual do debate em relação ao ano anterior
                if prev_score_total > 0:
                    debate_acceleration = (score_total - prev_score_total) / prev_score_total
                else:
                    debate_acceleration = 0.0
            else:
                vol_ibge = 0
                vol_camara = 0
                vol_openalex = 0
                score_total = 0.0
                debate_acceleration = 0.0

            # Target: O eixo foi o escolhido no ENEM daquele ano?
            is_target = 1 if actual_theme_axis and actual_theme_axis == axis else 0

            rows.append({
                "year": target_year,
                "axis": axis,
                "years_since_last": years_since_last,
                "occurred_previous_year": occurred_previous_year,
                "occurred_last_3_years": occurred_last_3_years,
                "cumulative_frequency": cumulative_frequency,
                "vol_ibge": vol_ibge,
                "vol_camara": vol_camara,
                "vol_openalex": vol_openalex,
                "score_total": float(score_total),
                "debate_acceleration": float(debate_acceleration),
                "target": is_target,
            })

    matrix_df = pd.DataFrame(rows)
    logger.info(f"[ML] Matriz de treino gerada com {len(matrix_df)} registros ({min_year}-{max_year}).")
    return matrix_df
