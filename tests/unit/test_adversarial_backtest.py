"""
Suíte de Testes Adversariais Empíricos de Vazamento Temporal e Fronteiras.
Elaborado pelo Agente Challenger M1.1 (teamwork_challenger).

Objetivo:
Estressar agressivamente a implementação de `src/enem_pipeline/ml/backtest.py`:
1. Injeção de Canários Temporais Futuros (verificação de contaminação cruzada zero)
2. Injeção de Canários no Ano de Teste (verificação de independência preditiva de treino)
3. Disparo do Sensor Ativo de Data Leakage em Runtime
4. Fronteiras de Horizontes Extremos e Subconjuntos Arbitrários (N=1, N=11, janelas móveis)
5. Veto Determinístico Rígido (enforce_hard_veto=True) e Conservação de Probabilidades
6. Resiliência Numérica, Imutabilidade e Invariantes Algébricas dos Dataclasses
"""

import copy
import math
import numpy as np
import pandas as pd
import pytest
from dataclasses import FrozenInstanceError

from enem_pipeline.db.seed import HISTORICAL_ENEM_THEMES, INITIAL_TOPICS
from enem_pipeline.ml.backtest import (
    OFFICIAL_AXES,
    BacktestYearResult,
    BacktestSummary,
    WalkForwardSimulator,
    compute_brier_score,
    compute_log_loss,
    compute_exact_binomial_pvalue,
    compute_relative_lift,
)
from tests.unit.test_backtest import make_valid_ranking


# ==============================================================================
# SEÇÃO 1: TESTES ADVERSARIAIS DE VAZAMENTO TEMPORAL E CANÁRIOS
# ==============================================================================

def test_adversarial_future_canary_leakage_invariance():
    """ADVERSARIAL: Injeção de canários bizarros no futuro distante (2024, 2025)

    não pode alterar absolutamente NADA na distribuição de probabilidades de 2018 (bit-level).
    """
    # 1. Simulação base limpa até 2018
    sim_clean = WalkForwardSimulator(min_train_year=2003, random_state=42)
    summary_clean = sim_clean.run(start_year=2018, end_year=2018)
    clean_result = summary_clean.yearly_results[0]
    clean_ranking = clean_result.predicted_ranking

    # 2. Injeção de canário no futuro: distorcer temas de 2024 e 2025 para um eixo específico
    corrupted_history = [
        copy.deepcopy(t) for t in HISTORICAL_ENEM_THEMES
        if t["year"] not in [2024, 2025]
    ]
    corrupted_history.append({
        "year": 2024,
        "official_theme": "CANÁRIO CORRUPTOR DO FUTURO 2024",
        "thematic_axis": "Tecnologia",
        "central_problem": "Injeção hostil de dados futuros",
    })
    corrupted_history.append({
        "year": 2025,
        "official_theme": "CANÁRIO CORRUPTOR DO FUTURO 2025",
        "thematic_axis": "Tecnologia",
        "central_problem": "Injeção hostil de dados futuros",
    })

    sim_corrupted = WalkForwardSimulator(min_train_year=2003, random_state=42)
    sim_corrupted.history_records = corrupted_history
    sim_corrupted.history_map = {
        item["year"]: {
            "axis": item["thematic_axis"],
            "theme": item["official_theme"],
        }
        for item in corrupted_history
    }
    summary_corrupted = sim_corrupted.run(start_year=2018, end_year=2018)
    corrupted_result = summary_corrupted.yearly_results[0]
    corrupted_ranking = corrupted_result.predicted_ranking

    # Verificação estrita de invariância: cada eixo deve ter EXATAMENTE a mesma probabilidade
    assert len(clean_ranking) == len(corrupted_ranking) == 10
    for (ax1, p1), (ax2, p2) in zip(clean_ranking, corrupted_ranking):
        assert ax1 == ax2, f"Ordem dos eixos divergiu: {ax1} != {ax2}"
        assert abs(p1 - p2) < 1e-12, (
            f"Vazamento temporal detectado! O canário de 2024/2025 alterou a probabilidade do eixo {ax1} em 2018: "
            f"limpo={p1:.10f}, corrompido={p2:.10f}"
        )


def test_adversarial_target_year_canary_training_blindness():
    """ADVERSARIAL: Alterar o tema do próprio ano avaliado (ex: 2020) para um canário

    não pode influenciar a matriz de treino ou o modelo, pois dados de treino devem ser < 2020.
    O ranking de probabilidades deve permanecer identicamente inalterado.
    """
    sim_clean = WalkForwardSimulator(min_train_year=2003, random_state=42)
    summary_clean = sim_clean.run(start_year=2020, end_year=2020)
    clean_ranking = summary_clean.yearly_results[0].predicted_ranking

    # Substitui o tema de 2020 por um canário hostil com eixo diferente
    corrupted_history = [
        copy.deepcopy(t) for t in HISTORICAL_ENEM_THEMES if t["year"] != 2020
    ]
    corrupted_history.append({
        "year": 2020,
        "official_theme": "TEMA CANÁRIO INJETADO EM 2020",
        "thematic_axis": "Segurança Pública",
        "central_problem": "Canário hostil no ano avaliado",
    })

    sim_canary = WalkForwardSimulator(min_train_year=2003, random_state=42)
    sim_canary.history_records = corrupted_history
    sim_canary.history_map = {
        item["year"]: {
            "axis": item["thematic_axis"],
            "theme": item["official_theme"],
        }
        for item in corrupted_history
    }
    summary_canary = sim_canary.run(start_year=2020, end_year=2020)
    canary_ranking = summary_canary.yearly_results[0].predicted_ranking

    for (ax1, p1), (ax2, p2) in zip(clean_ranking, canary_ranking):
        assert ax1 == ax2
        assert abs(p1 - p2) < 1e-12, (
            f"Data leakage no ano alvo: o modelo de treino para 2020 acessou o tema de 2020! "
            f"Eixo {ax1}: clean={p1}, canary={p2}"
        )


def test_adversarial_active_data_leakage_sensor_trips():
    """ADVERSARIAL: Se uma regressão de código (ex: '<=' em vez de '<') ou corrupção de índice

    fizer com que dados de ano >= target_year entrem em train_years, o sensor em
    `_train_and_predict_year` DEVE barrar o treinamento com RuntimeError('[DATA LEAKAGE DETECTED]').
    """
    sim = WalkForwardSimulator(min_train_year=2003, random_state=42)
    matrix_df = sim._build_dataset(max_year=2020)

    # Simula uma regressão hostil onde train_mask inclui o target_year (ex: operador <= em vez de <)
    from unittest.mock import patch

    with patch.object(pd.Series, "__lt__", lambda self, other: self <= other):
        with pytest.raises(RuntimeError, match=r"\[DATA LEAKAGE DETECTED\]"):
            sim._train_and_predict_year(target_year=2020, matrix_df=matrix_df)



# ==============================================================================
# SEÇÃO 2: TESTES DE FRONTEIRAS EXTREMAS, HORIZONTES E SUBCONJUNTOS
# ==============================================================================

@pytest.mark.parametrize("single_year", [2015, 2018, 2020, 2024, 2025])
def test_adversarial_single_year_horizon(single_year):
    """ADVERSARIAL: Execução em horizonte unitário (N=1)

    Garante que média, soma, lift e teste binomial calculam sem divisão por zero
    e atendem a todas as invariantes para um único ano.
    """
    sim = WalkForwardSimulator(min_train_year=2003, random_state=42)
    summary = sim.run(start_year=single_year, end_year=single_year)

    assert isinstance(summary, BacktestSummary)
    assert summary.years_evaluated == [single_year]
    assert len(summary.yearly_results) == 1

    res = summary.yearly_results[0]
    assert res.year == single_year
    assert res.top1_hit in [True, False]
    assert res.top3_hit in [True, False]

    # Invariantes numéricas do resumo unitário
    assert summary.top1_accuracy == (1.0 if res.top1_hit else 0.0)
    assert summary.top3_hit_rate == (1.0 if res.top3_hit else 0.0)
    assert summary.mrr == res.reciprocal_rank
    assert summary.mean_brier_score == res.brier_score
    assert summary.mean_log_loss == res.log_loss

    # Teste binomial para n=1
    k = 1 if res.top3_hit else 0
    expected_p = 0.30 if k == 1 else 1.0
    assert pytest.approx(summary.binomial_test_p_value, rel=1e-3) == expected_p


def test_adversarial_arbitrary_temporal_subsets():
    """ADVERSARIAL: Execução em subconjuntos arbitrários e janelas móveis

    Garante consistência em sub-intervalos:
    - (2016, 2019)
    - (2021, 2023)
    - (2019, 2025)
    """
    sim = WalkForwardSimulator(min_train_year=2003, random_state=42)

    intervals = [(2016, 2019), (2021, 2023), (2019, 2025)]
    for start, end in intervals:
        summary = sim.run(start_year=start, end_year=end)
        expected_years = list(range(start, end + 1))
        assert summary.years_evaluated == expected_years
        assert len(summary.yearly_results) == len(expected_years)

        # Checa que cada ano corresponde exatamente
        for y, r in zip(expected_years, summary.yearly_results):
            assert r.year == y
            assert 1 <= r.rank_of_official <= 10


def test_adversarial_invalid_boundary_ranges():
    """ADVERSARIAL: Rejeição estrita de intervalos temporais anômalos e anos inexistentes."""
    sim = WalkForwardSimulator(min_train_year=2003, random_state=42)

    # Invertido: start > end
    with pytest.raises(ValueError, match="não pode ser maior que end_year"):
        sim.run(start_year=2025, end_year=2015)

    # Anterior ao ENEM pioneiro (1998 requer histórico prévio)
    with pytest.raises(ValueError, match="deve ser >= 1999"):
        sim.run(start_year=1998, end_year=2005)

    # Ano além do catálogo histórico registrado
    with pytest.raises(KeyError, match="não encontrado no histórico oficial"):
        sim.run(start_year=2024, end_year=2027)


# ==============================================================================
# SEÇÃO 3: VETO DETERMINÍSTICO E PRESERVAÇÃO DE PROBABILIDADES
# ==============================================================================

def test_adversarial_enforce_hard_veto_mechanism():
    """ADVERSARIAL: Quando enforce_hard_veto=True, o eixo do ano anterior DEVE ter probabilidade 0.0

    e os outros 9 eixos devem somar exatamente 1.0 com ordenação preservada.
    """
    sim = WalkForwardSimulator(min_train_year=2003, random_state=42, enforce_hard_veto=True)
    summary = sim.run(start_year=2016, end_year=2025)

    for res in summary.yearly_results:
        target_year = res.year
        prev_year = target_year - 1
        prev_official_axis = sim.history_map[prev_year]["axis"]

        # Localiza o eixo do ano anterior no ranking de probabilidades
        ranking_dict = dict(res.predicted_ranking)
        vetoed_prob = ranking_dict[prev_official_axis]

        # 1. A probabilidade do eixo anterior DEVE ser estritamente zero
        assert vetoed_prob == 0.0, (
            f"Veto consecutivo falhou no ano {target_year}: eixo anterior '{prev_official_axis}' "
            f"obteve probabilidade {vetoed_prob} (esperava 0.0)!"
        )

        # 2. O eixo vetado deve ocupar a última posição no ranking (10º lugar)
        last_axis, last_prob = res.predicted_ranking[-1]
        assert last_prob == 0.0

        # 3. A soma das probabilidades dos 10 eixos deve ser 1.0
        total_p = sum(p for _, p in res.predicted_ranking)
        assert abs(total_p - 1.0) < 1e-4

        # 4. As probabilidades devem estar estritamente não-crescentes
        probs = [p for _, p in res.predicted_ranking]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1] - 1e-7


# ==============================================================================
# SEÇÃO 4: RESILIÊNCIA NUMÉRICA, IMUTABILIDADE E ORÁCULOS DE CALIBRAÇÃO
# ==============================================================================

def test_adversarial_dataclass_immutability():
    """ADVERSARIAL: BacktestYearResult e BacktestSummary devem ser estritamente congelados (frozen).

    Qualquer tentativa de adulterar métricas após a criação deve gerar FrozenInstanceError.
    """
    ranking = make_valid_ranking(target_axis="Educação", target_rank=1)
    res = BacktestYearResult(
        year=2020,
        official_axis="Educação",
        official_theme="Tema Teste",
        predicted_ranking=ranking,
        rank_of_official=1,
        top1_hit=True,
        top3_hit=True,
        reciprocal_rank=1.0,
        brier_score=0.1,
        log_loss=0.5,
    )

    with pytest.raises(FrozenInstanceError):
        res.top1_hit = False

    with pytest.raises(FrozenInstanceError):
        res.rank_of_official = 5

    summary = BacktestSummary(
        years_evaluated=[2020],
        top1_accuracy=1.0,
        top3_hit_rate=1.0,
        mrr=1.0,
        mean_brier_score=0.1,
        mean_log_loss=0.5,
        baseline_top3_rate=0.30,
        relative_lift_top3=((1.0 - 0.3) / 0.3) * 100,
        binomial_test_p_value=0.30,
        yearly_results=[res],
    )

    with pytest.raises(FrozenInstanceError):
        summary.top1_accuracy = 0.5


def test_adversarial_brier_and_log_loss_pathological_inputs():
    """ADVERSARIAL: Avalia funções de calibração sob distribuições patológicas e extremas:

    - Probabilidades infinitesimais (1e-30)
    - Probabilidades zero absolutas
    - Epsilon clipping contra estouro numérico
    """
    target_axis = "Cultura"

    # 1. Probabilidade infinitesimal próxima de zero absoluto
    probs_infinitesimal = {ax: 1e-30 for ax in OFFICIAL_AXES}
    probs_infinitesimal[target_axis] = 1.0 - (9 * 1e-30)
    # Brier score deve ser praticamente 0.0
    brier = compute_brier_score(probs_infinitesimal, target_axis)
    assert abs(brier) < 1e-10

    # 2. Log loss com probabilidade zero absoluto no alvo (clipping ativado)
    probs_zero = {ax: 0.11111111 for ax in OFFICIAL_AXES if ax != target_axis}
    probs_zero[target_axis] = 0.0
    loss = compute_log_loss(probs_zero, target_axis, eps=1e-15)
    assert not math.isinf(loss)
    assert not math.isnan(loss)
    assert pytest.approx(loss, rel=1e-3) == -math.log(1e-15)


def test_adversarial_binomial_pvalue_extreme_parameters():
    """ADVERSARIAL: Estressa compute_exact_binomial_pvalue com valores de fronteira extrema:

    k=0, k=n, k > n, n grande, p0 nos limites [0.0, 1.0].
    """
    # k <= 0 deve ser sempre 1.0
    assert compute_exact_binomial_pvalue(k=0, n=10, p0=0.30) == 1.0
    assert compute_exact_binomial_pvalue(k=-5, n=10, p0=0.30) == 1.0

    # k > n deve ser sempre 0.0
    assert compute_exact_binomial_pvalue(k=11, n=10, p0=0.30) == 0.0
    assert compute_exact_binomial_pvalue(k=100, n=10, p0=0.30) == 0.0

    # p0 = 0.0
    assert compute_exact_binomial_pvalue(k=1, n=10, p0=0.0) == 0.0
    assert compute_exact_binomial_pvalue(k=0, n=10, p0=0.0) == 1.0

    # p0 = 1.0
    assert compute_exact_binomial_pvalue(k=10, n=10, p0=1.0) == 1.0
    assert compute_exact_binomial_pvalue(k=11, n=10, p0=1.0) == 0.0

    # Parâmetros inválidos devem lançar ValueError
    with pytest.raises(ValueError, match="total de ensaios.*deve ser maior que zero"):
        compute_exact_binomial_pvalue(k=1, n=0, p0=0.30)

    with pytest.raises(ValueError, match="p0 deve estar no intervalo"):
        compute_exact_binomial_pvalue(k=1, n=10, p0=-0.1)

    with pytest.raises(ValueError, match="p0 deve estar no intervalo"):
        compute_exact_binomial_pvalue(k=1, n=10, p0=1.1)


def test_adversarial_simulator_reproducibility():
    """ADVERSARIAL: Duas execuções com a mesma semente aleatória DEVEM produzir

    resultados idênticos bit a bit para todas as métricas em 11 anos.
    """
    sim1 = WalkForwardSimulator(min_train_year=2003, random_state=42)
    summary1 = sim1.run(start_year=2015, end_year=2025)

    sim2 = WalkForwardSimulator(min_train_year=2003, random_state=42)
    summary2 = sim2.run(start_year=2015, end_year=2025)

    assert summary1.top1_accuracy == summary2.top1_accuracy
    assert summary1.top3_hit_rate == summary2.top3_hit_rate
    assert summary1.mrr == summary2.mrr
    assert summary1.mean_brier_score == summary2.mean_brier_score
    assert summary1.mean_log_loss == summary2.mean_log_loss
    assert summary1.binomial_test_p_value == summary2.binomial_test_p_value

    for r1, r2 in zip(summary1.yearly_results, summary2.yearly_results):
        assert r1.year == r2.year
        assert r1.rank_of_official == r2.rank_of_official
        assert r1.brier_score == r2.brier_score
        assert r1.log_loss == r2.log_loss
        for (a1, p1), (a2, p2) in zip(r1.predicted_ranking, r2.predicted_ranking):
            assert a1 == a2
            assert p1 == p2
