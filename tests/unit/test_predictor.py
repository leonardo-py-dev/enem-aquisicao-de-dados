"""
Suíte de testes unitários para o módulo preditivo ENEM 2026 e Competência 5.
Valida:
1. Aplicação determinística da regra de veto consecutivo (P=0.0% para Demografia e Envelhecimento).
2. Normalização estrita da distribuição de probabilidades (soma = 1.0).
3. Monotonicidade decrescente e estabilidade do ranking.
4. Presença dos 10 eixos oficiais canônicos.
5. Posicionamento do eixo vetado no 10º lugar.
6. Correspondência exata entre os 3 eixos líderes e as propostas da Competência 5.
7. Presença obrigatória e substantiva de todos os 5 elementos (Agente, Ação, Meio/Modo, Efeito e Detalhamento).
8. Resiliência e fallback offline determinístico na ausência ou falha da API Gemini.
9. Invariantes de integridade validadas em __post_init__.
10. Isolamento temporal estrito contra data leakage.
"""

import json
import math
from unittest.mock import patch
import pytest
import pandas as pd
import numpy as np

from enem_pipeline.ml.backtest import OFFICIAL_AXES
from enem_pipeline.ml.predictor import (
    VETOED_AXIS_2025,
    InterventionProposal,
    Prediction2026Result,
    ENEM2026Predictor,
    CANONICAL_PROPOSALS,
)


def _make_dummy_valid_proposals(axes: list[str]) -> list[InterventionProposal]:
    """Helper para criar propostas válidas para testes de validação de dataclass."""
    return [
        InterventionProposal(
            axis=ax,
            candidate_theme=f"Tema de teste para {ax}",
            agente=f"Ministério competente de {ax}",
            acao="Implementar plano nacional estratégico",
            meio_modo="Mediante dotação orçamentária vinculada",
            detalhamento="Com priorização de áreas vulneráveis e indicadores do SAEB",
            efeito="A fim de assegurar garantias fundamentais e cidadania",
        )
        for ax in axes
    ]


def _make_dummy_valid_ranking() -> list[tuple[str, float]]:
    """Helper para criar um ranking decrescente válido com veto a 2025."""
    non_vetoed = [ax for ax in OFFICIAL_AXES if ax != VETOED_AXIS_2025]
    weights = [float(10 - i) for i in range(len(non_vetoed))]
    total_w = sum(weights)
    ranking = [(ax, w / total_w) for ax, w in zip(non_vetoed, weights)]
    ranking.append((VETOED_AXIS_2025, 0.0))
    return ranking


def test_veto_rule_2025_strictly_zero():
    """Valida que o eixo do ENEM 2025 ('Demografia e Envelhecimento') recebe probabilidade P=0.0%."""
    predictor = ENEM2026Predictor(offline=True)
    result = predictor.predict()

    assert result.vetoed_axis == VETOED_AXIS_2025
    ranking_dict = dict(result.ranking)
    assert VETOED_AXIS_2025 in ranking_dict
    assert ranking_dict[VETOED_AXIS_2025] == 0.0


def test_probabilities_sum_to_one():
    """Valida que a soma de todas as probabilidades da distribuição 2026 totaliza exatamente 1.0 (100%)."""
    predictor = ENEM2026Predictor(offline=True)
    result = predictor.predict()

    total_prob = sum(prob for _, prob in result.ranking)
    assert abs(total_prob - 1.0) < 1e-6, f"Soma observada: {total_prob}"


def test_ranking_ordering_descending():
    """Valida que o ranking é estritamente decrescente em termos de probabilidade."""
    predictor = ENEM2026Predictor(offline=True)
    result = predictor.predict()

    probs = [prob for _, prob in result.ranking]
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1] - 1e-9, f"Violação de ordenação: {probs[i]} < {probs[i+1]}"


def test_exactly_ten_unique_official_axes():
    """Valida que o ranking contém os 10 eixos oficiais do ENEM sem duplicatas ou ausências."""
    predictor = ENEM2026Predictor(offline=True)
    result = predictor.predict()

    ranked_axes = [axis for axis, _ in result.ranking]
    assert len(ranked_axes) == 10
    assert len(set(ranked_axes)) == 10
    assert set(ranked_axes) == set(OFFICIAL_AXES)


def test_vetoed_axis_placed_at_last_position():
    """Valida que o eixo vetado (probabilidade 0.0) é posicionado no final do ranking (10º lugar)."""
    predictor = ENEM2026Predictor(offline=True)
    result = predictor.predict()

    last_axis, last_prob = result.ranking[-1]
    assert last_axis == VETOED_AXIS_2025
    assert last_prob == 0.0


def test_top3_proposals_correspond_to_top3_axes():
    """Valida que há exatamente 3 propostas e que seus eixos correspondem aos 3 líderes do ranking."""
    predictor = ENEM2026Predictor(offline=True)
    result = predictor.predict()

    assert len(result.top3_proposals) == 3
    top3_ranked_axes = [axis for axis, _ in result.ranking[:3]]
    proposal_axes = [p.axis for p in result.top3_proposals]
    assert proposal_axes == top3_ranked_axes


def test_competence_5_all_five_elements_present():
    """Valida que cada proposta de intervenção possui os 5 elementos devidamente preenchidos e substantivos."""
    predictor = ENEM2026Predictor(offline=True)
    result = predictor.predict()

    for idx, p in enumerate(result.top3_proposals):
        assert isinstance(p, InterventionProposal)
        assert len(p.axis.strip()) > 0
        assert len(p.candidate_theme.strip()) >= 15, f"Tema da proposta #{idx+1} muito curto."
        assert len(p.agente.strip()) >= 5, f"Agente da proposta #{idx+1} muito curto."
        assert len(p.acao.strip()) >= 15, f"Ação da proposta #{idx+1} muito curta."
        assert len(p.meio_modo.strip()) >= 15, f"Meio/modo da proposta #{idx+1} muito curto."
        assert len(p.efeito.strip()) >= 15, f"Efeito da proposta #{idx+1} muito curto."
        assert len(p.detalhamento.strip()) >= 20, f"Detalhamento da proposta #{idx+1} insuficiente."


def test_offline_fallback_deterministic_catalog(monkeypatch):
    """Valida que na ausência da variável GEMINI_API_KEY o preditor executa perfeitamente via catálogo offline."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    predictor = ENEM2026Predictor(offline=False)
    result = predictor.predict()

    assert len(result.top3_proposals) == 3
    for p in result.top3_proposals:
        assert p.detalhamento and len(p.detalhamento) > 10


def test_gemini_api_exception_graceful_fallback():
    """Valida que erros ou exceções na biblioteca Gemini acionam o fallback sem interromper o pipeline."""
    with patch(
        "enem_pipeline.ml.predictor._call_gemini_for_proposals",
        side_effect=RuntimeError("API Gateway Timeout"),
    ):
        predictor = ENEM2026Predictor(offline=False)
        result = predictor.predict()
        assert len(result.top3_proposals) == 3
        for p in result.top3_proposals:
            assert p.axis in OFFICIAL_AXES
            assert len(p.detalhamento) > 10


def test_prediction_2026_result_post_init_validations():
    """Valida que dados corrompidos passados ao dataclass disparam ValueError imediatamente."""
    valid_ranking = _make_dummy_valid_ranking()
    top3_axes = [ax for ax, _ in valid_ranking[:3]]
    valid_proposals = _make_dummy_valid_proposals(top3_axes)

    # Caso válido inicial
    res = Prediction2026Result(
        ranking=valid_ranking,
        vetoed_axis=VETOED_AXIS_2025,
        top3_proposals=valid_proposals,
    )
    assert res is not None

    # Erro 1: Veto violado (probabilidade do eixo vetado > 0.0)
    bad_veto_ranking = [(ax, (0.05 if ax == VETOED_AXIS_2025 else p * 0.95)) for ax, p in valid_ranking]
    with pytest.raises(ValueError, match="probabilidade do eixo vetado deve ser estritamente 0.0"):
        Prediction2026Result(
            ranking=bad_veto_ranking,
            vetoed_axis=VETOED_AXIS_2025,
            top3_proposals=valid_proposals,
        )

    # Erro 2: Soma diferente de 1.0
    bad_prob_ranking = [(ax, 0.05) for ax in OFFICIAL_AXES[:-1]] + [(VETOED_AXIS_2025, 0.0)]
    with pytest.raises(ValueError, match="soma das probabilidades"):
        Prediction2026Result(
            ranking=bad_prob_ranking,
            vetoed_axis=VETOED_AXIS_2025,
            top3_proposals=valid_proposals,
        )

    # Erro 3: Quantidade incorreta de eixos no ranking
    with pytest.raises(ValueError, match="ranking deve conter exatamente 10 eixos"):
        Prediction2026Result(
            ranking=valid_ranking[:9],
            vetoed_axis=VETOED_AXIS_2025,
            top3_proposals=valid_proposals,
        )

    # Erro 4: Quantidade incorreta de propostas
    with pytest.raises(ValueError, match="top3_proposals deve conter exatamente 3"):
        Prediction2026Result(
            ranking=valid_ranking,
            vetoed_axis=VETOED_AXIS_2025,
            top3_proposals=valid_proposals[:2],
        )

    # Erro 5: Eixos das propostas divergindo do Top-3 do ranking
    inverted_proposals = _make_dummy_valid_proposals([top3_axes[1], top3_axes[0], top3_axes[2]])
    with pytest.raises(ValueError, match="não coincidem com o Top-3"):
        Prediction2026Result(
            ranking=valid_ranking,
            vetoed_axis=VETOED_AXIS_2025,
            top3_proposals=inverted_proposals,
        )


def test_intervention_proposal_post_init_validations():
    """Valida as invariantes em InterventionProposal.__post_init__."""
    # Válido
    prop = InterventionProposal(
        axis="Educação",
        candidate_theme="Tema válido para educação",
        agente="MEC",
        acao="Ação concreta pedagógica",
        meio_modo="Por meio de dotação orçamentária do FNDE",
        detalhamento="Com laboratórios interdisciplinares",
        efeito="Para superar o analfabetismo funcional",
    )
    assert prop.axis == "Educação"

    # Erro de campo vazio
    with pytest.raises(ValueError, match="Invariante da Competência 5 violada: o campo 'detalhamento' não pode ser vazio"):
        InterventionProposal(
            axis="Educação",
            candidate_theme="Tema válido",
            agente="MEC",
            acao="Ação concreta",
            meio_modo="Meio",
            detalhamento="   ",
            efeito="Efeito",
        )

    # Erro de eixo desconhecido fora da taxonomia oficial
    with pytest.raises(ValueError, match="Eixo 'Eixo Inexistente' desconhecido"):
        InterventionProposal(
            axis="Eixo Inexistente",
            candidate_theme="Tema válido",
            agente="MEC",
            acao="Ação concreta",
            meio_modo="Meio",
            detalhamento="Detalhamento substantivo",
            efeito="Efeito",
        )


def test_temporal_training_cutoff_no_future_leakage():
    """Valida que o treinamento é estritamente limitado a y <= 2025 e que o sensor de leakage barra anos futuros."""
    predictor = ENEM2026Predictor(offline=True)
    df = predictor._build_dataset(target_year=2026)

    # 1. Verifica que dados de treino são estritamente anteriores a 2026
    train_mask = df["year"] < 2026
    assert df.loc[train_mask, "year"].max() == 2025

    # 2. Testa o disparo do sensor de auditoria ativa caso train_mask inclua anos futuros
    with patch.object(pd.Series, "__lt__", lambda self, other: self <= other):
        with pytest.raises(RuntimeError, match=r"\[DATA LEAKAGE DETECTED\]"):
            predictor._train_and_predict_raw(df, target_year=2026)


def test_json_serialization_roundtrip():
    """Valida a serialização e desserialização JSON de Prediction2026Result."""
    predictor = ENEM2026Predictor(offline=True)
    result = predictor.predict()

    json_str = result.to_json()
    data = json.loads(json_str)

    assert "vetoed_axis" in data
    assert data["vetoed_axis"] == VETOED_AXIS_2025
    assert "ranking" in data
    assert len(data["ranking"]) == 10
    assert "top3_proposals" in data
    assert len(data["top3_proposals"]) == 3

    assert result.get_top_axis() == result.ranking[0][0]
    assert result.get_top3_axes() == [r[0] for r in result.ranking[:3]]
    assert result.get_axis_probability(VETOED_AXIS_2025) == 0.0
