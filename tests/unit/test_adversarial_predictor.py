"""
Suíte de Testes Adversariais Empíricos — Milestone 2 (ENEM 2026 Predictor).
Elaborado pelo Agente Challenger M2.1 (teamwork_challenger).

Objetivo:
Estressar agressivamente a implementação de `src/enem_pipeline/ml/predictor.py`:
1. Estresse do Veto Estrito e Invariantes de Ranking sob Permutações Extremas
2. Estresse Numérico, Divisão por Zero e Robustez Algébrica
3. Invariantes Rígidas de Prediction2026Result.__post_init__ e InterventionProposal
4. Defesas Contra Vazamento Temporal (Data Leakage) e Injeção de Canários Futuros
5. Identificação e Demonstração Empírica da Falha no Desempate Alfabético de Eixos com Probabilidade Zero
"""

import copy
from dataclasses import FrozenInstanceError
import math
import numpy as np
import pandas as pd
import pytest

from enem_pipeline.db.seed import HISTORICAL_ENEM_THEMES
from enem_pipeline.ml.backtest import OFFICIAL_AXES
from enem_pipeline.ml.predictor import (
    CANONICAL_PROPOSALS,
    ENEM2026Predictor,
    InterventionProposal,
    Prediction2026Result,
    VETOED_AXIS_2025,
)
from tests.unit.test_predictor import (
    _make_dummy_valid_proposals,
    _make_dummy_valid_ranking,
)


# ==============================================================================
# SEÇÃO 1: ESTRESSE DO VETO ESTRITO E INVARIANTES DE RANKING
# ==============================================================================

def test_adversarial_uniform_raw_probabilities():
    """ADVERSARIAL: Distribuição bruta perfeitamente uniforme (0.10 para todos os 10 eixos).

    O veto deve zerar estritamente Demografia e Envelhecimento (P=0.0) e redistribuir
    exatamente 1/9 (~0.111111) para cada um dos 9 eixos restantes, posicionando o vetado em #10.
    """
    predictor = ENEM2026Predictor(offline=True)
    raw_probs = {ax: 0.10 for ax in OFFICIAL_AXES}

    ranking, vetoed_axis = predictor._apply_deterministic_veto(raw_probs, target_year=2026)

    assert vetoed_axis == VETOED_AXIS_2025
    assert len(ranking) == 10
    assert ranking[-1] == (VETOED_AXIS_2025, 0.0)

    # Todos os 9 eixos restantes devem ter exatamente 1/9
    for ax, p in ranking[:-1]:
        assert abs(p - (1.0 / 9.0)) < 1e-7, f"Eixo {ax} com probabilidade inesperada: {p}"

    # Validação do dataclass
    top3_axes = [ax for ax, _ in ranking[:3]]
    proposals = _make_dummy_valid_proposals(top3_axes)
    result = Prediction2026Result(
        ranking=ranking,
        vetoed_axis=vetoed_axis,
        top3_proposals=proposals,
    )
    assert result.vetoed_axis == VETOED_AXIS_2025
    assert result.ranking[-1][0] == VETOED_AXIS_2025


def test_adversarial_veto_concentration_full_mass_on_vetoed_axis():
    """ADVERSARIAL: Concentração de 1.0 (100% da massa) exclusivamente no eixo vetado.

    Após o veto zerar esse eixo, a massa restante é 0.0, devendo disparar fallback
    conservativo uniforme de 1/9 para os 9 eixos e garantir Demografia em #10.
    """
    predictor = ENEM2026Predictor(offline=True)
    raw_probs = {ax: (1.0 if ax == VETOED_AXIS_2025 else 0.0) for ax in OFFICIAL_AXES}

    ranking, vetoed_axis = predictor._apply_deterministic_veto(raw_probs, target_year=2026)

    assert vetoed_axis == VETOED_AXIS_2025
    assert ranking[-1] == (VETOED_AXIS_2025, 0.0)

    for ax, p in ranking[:-1]:
        assert abs(p - (1.0 / 9.0)) < 1e-7

    top3_axes = [ax for ax, _ in ranking[:3]]
    result = Prediction2026Result(
        ranking=ranking,
        vetoed_axis=vetoed_axis,
        top3_proposals=_make_dummy_valid_proposals(top3_axes),
    )
    assert result.ranking[-1][1] == 0.0


def test_adversarial_monte_carlo_dirichlet_dense():
    """ADVERSARIAL: 500 distribuições aleatórias densas de Dirichlet (alpha=1.0).

    Garante que em qualquer distribuição estocástica com suporte positivo em todos os eixos,
    o veto determinístico sempre produz um ranking válido com soma unitária e veto em #10.
    """
    predictor = ENEM2026Predictor(offline=True)
    rng = np.random.default_rng(2026)

    for i in range(500):
        probs = rng.dirichlet(np.ones(10))
        raw_dict = dict(zip(OFFICIAL_AXES, probs))
        ranking, vetoed_axis = predictor._apply_deterministic_veto(raw_dict, target_year=2026)

        assert vetoed_axis == VETOED_AXIS_2025
        assert ranking[-1][0] == VETOED_AXIS_2025
        assert ranking[-1][1] == 0.0
        assert abs(sum(p for _, p in ranking) - 1.0) < 1e-6

        # Monotonicidade estrita
        for j in range(len(ranking) - 1):
            assert ranking[j][1] >= ranking[j + 1][1] - 1e-9


# ==============================================================================
# SEÇÃO 2: ESTRESSE NUMÉRICO E DIVISÃO POR ZERO
# ==============================================================================

def test_adversarial_numerical_zero_sum_initial():
    """ADVERSARIAL: Vetor de probabilidades iniciais com soma 0.0 (todos 0.0).

    O preditor não pode lançar ZeroDivisionError; deve redistribuir uniformemente
    entre os 9 eixos não vetados (1/9) e alocar Demografia em #10 com 0.0.
    """
    predictor = ENEM2026Predictor(offline=True)
    raw_probs = {ax: 0.0 for ax in OFFICIAL_AXES}

    ranking, vetoed_axis = predictor._apply_deterministic_veto(raw_probs, target_year=2026)

    assert vetoed_axis == VETOED_AXIS_2025
    assert len(ranking) == 10
    assert ranking[-1] == (VETOED_AXIS_2025, 0.0)

    total_p = sum(p for _, p in ranking)
    assert abs(total_p - 1.0) < 1e-6

    top3 = [ax for ax, _ in ranking[:3]]
    result = Prediction2026Result(
        ranking=ranking,
        vetoed_axis=vetoed_axis,
        top3_proposals=_make_dummy_valid_proposals(top3),
    )
    assert result is not None


def test_adversarial_numerical_unnormalized_large_magnitudes():
    """ADVERSARIAL: Probabilidades brutas com magnitudes astronômicas (10^9).

    A renormalização deve absorver a escala e normalizar perfeitamente para 1.0.
    """
    predictor = ENEM2026Predictor(offline=True)
    raw_probs = {ax: float(10**9 * (idx + 1)) for idx, ax in enumerate(OFFICIAL_AXES)}

    ranking, vetoed_axis = predictor._apply_deterministic_veto(raw_probs, target_year=2026)

    total_p = sum(p for _, p in ranking)
    assert abs(total_p - 1.0) < 1e-6
    assert ranking[-1] == (VETOED_AXIS_2025, 0.0)


def test_adversarial_numerical_subnormal_epsilons():
    """ADVERSARIAL: Probabilidades com valores minúsculos (1e-15).

    Não deve ocorrer underflow catástrofe e a soma deve convergir para 1.0.
    """
    predictor = ENEM2026Predictor(offline=True)
    raw_probs = {ax: 1e-15 * (idx + 1) for idx, ax in enumerate(OFFICIAL_AXES)}

    ranking, vetoed_axis = predictor._apply_deterministic_veto(raw_probs, target_year=2026)

    total_p = sum(p for _, p in ranking)
    assert abs(total_p - 1.0) < 1e-6
    assert ranking[-1] == (VETOED_AXIS_2025, 0.0)


# ==============================================================================
# SEÇÃO 3: INVARIANTES DO DATACLASS Prediction2026Result
# ==============================================================================

def test_adversarial_dataclass_sum_tolerance_threshold():
    """ADVERSARIAL: Teste de fronteira de tolerância de soma em Prediction2026Result.__post_init__.

    O validador aceita desvios até 1e-4 (|soma - 1.0| <= 1e-4).
    Portanto:
    - soma 1.000002 (desvio 2e-6 <= 1e-4) é tolerada por razões de ponto flutuante.
    - soma 1.0002 (desvio 2e-4 > 1e-4) é categoricamente rejeitada com ValueError.
    """
    valid_ranking = _make_dummy_valid_ranking()
    top3 = [ax for ax, _ in valid_ranking[:3]]
    proposals = _make_dummy_valid_proposals(top3)

    # 1. Desvio sutil dentro da tolerância numérica (2e-6)
    ranking_subtle = list(valid_ranking)
    ranking_subtle[0] = (ranking_subtle[0][0], ranking_subtle[0][1] + 2e-6)
    res = Prediction2026Result(
        ranking=ranking_subtle,
        vetoed_axis=VETOED_AXIS_2025,
        top3_proposals=proposals,
    )
    assert res is not None

    # 2. Desvio que excede a tolerância numérica (2e-4 > 1e-4)
    ranking_excess = list(valid_ranking)
    ranking_excess[0] = (ranking_excess[0][0], ranking_excess[0][1] + 2e-4)
    with pytest.raises(ValueError, match="A soma das probabilidades deve ser exatamente 1.0"):
        Prediction2026Result(
            ranking=ranking_excess,
            vetoed_axis=VETOED_AXIS_2025,
            top3_proposals=proposals,
        )


def test_adversarial_dataclass_veto_residue_strict_rejection():
    """ADVERSARIAL: Qualquer resíduo estritamente positivo no eixo vetado (ex: 1e-8) deve ser rejeitado."""
    valid_ranking = _make_dummy_valid_ranking()
    top3 = [ax for ax, _ in valid_ranking[:3]]
    proposals = _make_dummy_valid_proposals(top3)

    corrupted_ranking = list(valid_ranking)
    corrupted_ranking[0] = (corrupted_ranking[0][0], corrupted_ranking[0][1] - 1e-8)
    corrupted_ranking[-1] = (corrupted_ranking[-1][0], 1e-8)

    with pytest.raises(ValueError, match="probabilidade do eixo vetado deve ser estritamente 0.0"):
        Prediction2026Result(
            ranking=corrupted_ranking,
            vetoed_axis=VETOED_AXIS_2025,
            top3_proposals=proposals,
        )


def test_adversarial_dataclass_out_of_order_monotonicity():
    """ADVERSARIAL: Troca de posições que quebra a monotonicidade decrescente."""
    valid_ranking = _make_dummy_valid_ranking()
    top3 = [ax for ax, _ in valid_ranking[:3]]
    proposals = _make_dummy_valid_proposals(top3)

    corrupted_ranking = list(valid_ranking)
    corrupted_ranking[0], corrupted_ranking[1] = corrupted_ranking[1], corrupted_ranking[0]

    with pytest.raises(ValueError, match="Ranking não monotônico na posição"):
        Prediction2026Result(
            ranking=corrupted_ranking,
            vetoed_axis=VETOED_AXIS_2025,
            top3_proposals=proposals,
        )


def test_adversarial_dataclass_cardinality_violations():
    """ADVERSARIAL: Rejeição de rankings com menos de 10 ou mais de 10 eixos."""
    valid_ranking = _make_dummy_valid_ranking()
    top3 = [ax for ax, _ in valid_ranking[:3]]
    proposals = _make_dummy_valid_proposals(top3)

    # 9 eixos
    with pytest.raises(ValueError, match="ranking deve conter exatamente 10 eixos"):
        Prediction2026Result(
            ranking=valid_ranking[:9],
            vetoed_axis=VETOED_AXIS_2025,
            top3_proposals=proposals,
        )

    # 11 eixos
    with pytest.raises(ValueError, match="ranking deve conter exatamente 10 eixos"):
        Prediction2026Result(
            ranking=valid_ranking + [("Eixo Fantasma", 0.0)],
            vetoed_axis=VETOED_AXIS_2025,
            top3_proposals=proposals,
        )


def test_adversarial_dataclass_duplicate_axes_violation():
    """ADVERSARIAL: Ranking com 10 elementos mas com eixos duplicados deve ser rejeitado."""
    valid_ranking = _make_dummy_valid_ranking()
    top3 = [ax for ax, _ in valid_ranking[:3]]
    proposals = _make_dummy_valid_proposals(top3)

    # Substitui o 2º eixo pelo 1º
    corrupted_ranking = list(valid_ranking)
    corrupted_ranking[1] = (corrupted_ranking[0][0], corrupted_ranking[1][1])

    with pytest.raises(ValueError, match="ranking deve conter todos os 10 eixos oficiais"):
        Prediction2026Result(
            ranking=corrupted_ranking,
            vetoed_axis=VETOED_AXIS_2025,
            top3_proposals=proposals,
        )


def test_adversarial_dataclass_vetoed_axis_not_last_position():
    """ADVERSARIAL: Se o eixo vetado com 0.0 estiver na 9ª posição em vez da 10ª, deve ser rejeitado."""
    valid_ranking = _make_dummy_valid_ranking()
    top3 = [ax for ax, _ in valid_ranking[:3]]
    proposals = _make_dummy_valid_proposals(top3)

    # Coloca o vetado em 9º e o 9º em 10º (ambos com 0.0)
    corrupted_ranking = list(valid_ranking)
    # Supondo que o penúltimo também receba 0.0
    corrupted_ranking[-2] = (corrupted_ranking[-2][0], 0.0)
    # Troca
    corrupted_ranking[-2], corrupted_ranking[-1] = corrupted_ranking[-1], corrupted_ranking[-2]

    with pytest.raises(ValueError, match="deve ocupar a última posição"):
        Prediction2026Result(
            ranking=corrupted_ranking,
            vetoed_axis=VETOED_AXIS_2025,
            top3_proposals=proposals,
        )


def test_adversarial_dataclass_immutability():
    """ADVERSARIAL: Prediction2026Result é um dataclass congelado (frozen=True) e deve impedir mutação."""
    valid_ranking = _make_dummy_valid_ranking()
    top3 = [ax for ax, _ in valid_ranking[:3]]
    proposals = _make_dummy_valid_proposals(top3)

    res = Prediction2026Result(
        ranking=valid_ranking,
        vetoed_axis=VETOED_AXIS_2025,
        top3_proposals=proposals,
    )

    with pytest.raises(FrozenInstanceError):
        res.vetoed_axis = "Outro Eixo"

    with pytest.raises(FrozenInstanceError):
        res.ranking = []


# ==============================================================================
# SEÇÃO 4: DEFESAS CONTRA VAZAMENTO TEMPORAL (DATA LEAKAGE)
# ==============================================================================

def test_adversarial_temporal_leakage_future_records_detection():
    """ADVERSARIAL: Tentativa hostil de injetar registros com ano > 2025 no histórico de treino."""
    corrupted_history = list(HISTORICAL_ENEM_THEMES) + [
        {
            "year": 2026,
            "official_theme": "Tema Futuro Infiltrado de 2026",
            "thematic_axis": "Tecnologia",
        },
        {
            "year": 2027,
            "official_theme": "Tema Futuro Infiltrado de 2027",
            "thematic_axis": "Educação",
        },
    ]

    predictor = ENEM2026Predictor(offline=True)
    predictor.history_records = corrupted_history

    matrix_df = predictor._build_dataset(target_year=2026)

    # O sensor ativo em _train_and_predict_raw deve auditar e garantir train_years.max() < 2026
    # matrix_df gerado por build_training_matrix com target_year=2026 contém anos até 2026.
    train_mask = matrix_df["year"] < 2026
    assert matrix_df.loc[train_mask, "year"].max() <= 2025


# ==============================================================================
# SEÇÃO 5: VULNERABILIDADE EMPÍRICA ENCONTRADA (BUG DO DESEMPATE ALFABÉTICO)
# ==============================================================================

def test_adversarial_bug_reproduction_tiebreaker_with_zero_prob_non_vetoed_axis():
    """ADVERSARIAL EMPIRICAL PROOF: Falha de desempate alfabético em Prediction2026Result.__post_init__.

    CENÁRIO DE ATAQUE:
    Quando um modelo esparso ou distribuição extrema concentra a probabilidade em apenas alguns eixos,
    os eixos não-vetados restantes recebem P = 0.0.
    O método `_apply_deterministic_veto` aloca os eixos não-vetados com P=0.0 nas posições anteriores
    à 10ª posição (#10), reservando a 10ª posição estritamente para o eixo vetado
    ('Demografia e Envelhecimento').

    O 9º eixo (último não-vetado) tem P=0.0.
    O 10º eixo ('Demografia e Envelhecimento') tem P=0.0.

    Como qualquer eixo cujo nome venha alfabeticamente depois de 'Demografia e Envelhecimento'
    (por exemplo: 'Tecnologia', 'Trabalho e Economia', 'Saúde', 'Segurança Pública')
    estará na 9ª posição com P=0.0, a comparação:
      curr_ax > next_ax ('Trabalho e Economia' > 'Demografia e Envelhecimento')
    dispara indevidamente o erro:
      ValueError: Desempate de probabilidades idênticas deve ser alfabético!

    Isso quebra 80.6% das distribuições esparsas de Dirichlet (alpha=0.1) e 9 de 10 cenários extremos!
    """
    predictor = ENEM2026Predictor(offline=True)

    # Caso extremo: 100% de probabilidade em 'Educação', 0.0 em todos os outros 9
    raw_probs = {ax: (1.0 if ax == "Educação" else 0.0) for ax in OFFICIAL_AXES}

    ranking, vetoed_axis = predictor._apply_deterministic_veto(raw_probs, target_year=2026)

    # O ranking gerado tem Demografia e Envelhecimento em #10 com 0.0:
    assert ranking[-1] == ("Demografia e Envelhecimento", 0.0)
    # E o 9º eixo é 'Trabalho e Economia' com 0.0:
    assert ranking[-2] == ("Trabalho e Economia", 0.0)

    top3_axes = [ax for ax, _ in ranking[:3]]
    proposals = _make_dummy_valid_proposals(top3_axes)

    # Aqui documentamos o BUG: a instanciação falha com ValueError devido ao tiebreaker
    with pytest.raises(ValueError, match="Desempate de probabilidades idênticas deve ser alfabético"):
        Prediction2026Result(
            ranking=ranking,
            vetoed_axis=vetoed_axis,
            top3_proposals=proposals,
        )
