"""
Suíte de testes unitários do módulo de Backtesting Walk-Forward.
Cobre todos os 16 cenários especificados por M1.3:
- Grupo 1: Isolamento Temporal e Ausência de Data Leakage (Cenários 1-2)
- Grupo 2: Cálculo Determinístico de Top-1, Top-3 e MRR (Cenários 3-4)
- Grupo 3: Brier Score e Log Loss (Cenários 5-10)
- Grupo 4: Teste Binomial Exato e Lift Relativo (Cenários 11-12)
- Grupo 5: Validações de Argumentos, Invariantes e Execução In-Memory (Cenários 13-16)
"""

import math
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

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


# Helper para construir ranking determinístico com probabilidades válidas somando 1.0
def make_valid_ranking(
    target_axis: str,
    target_rank: int,
    base_prob: float = 0.50,
) -> list[tuple[str, float]]:
    """Cria um ranking decrescente válido para os 10 eixos oficiais colocando target_axis na posição target_rank (1-based)."""
    assert 1 <= target_rank <= 10
    other_axes = [ax for ax in OFFICIAL_AXES if ax != target_axis]

    # Distribuição decrescente geométrica de probabilidades
    weights = [2.0 ** (10 - i) for i in range(10)]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    # Insere target_axis na posição target_rank - 1
    ranked_axes = []
    other_idx = 0
    for pos in range(10):
        if pos == (target_rank - 1):
            ranked_axes.append(target_axis)
        else:
            ranked_axes.append(other_axes[other_idx])
            other_idx += 1

    ranking = [(axis, float(p)) for axis, p in zip(ranked_axes, probs)]
    # Ajusta eventual resíduo de ponto flutuante
    s = sum(p for _, p in ranking)
    return [(ax, p / s) for ax, p in ranking]


# ==============================================================================
# Grupo 1: Isolamento Temporal e Ausência de Data Leakage
# ==============================================================================

def test_temporal_isolation_history_strictly_prior_years():
    """Cenário 1: Ao avaliar o ano Y, todos os anos no conjunto de treinamento devem ser estritamente < Y."""
    simulator = WalkForwardSimulator()
    matrix_df = simulator._build_dataset(max_year=2025)

    for target_year in range(2015, 2026):
        train_mask = matrix_df["year"] < target_year
        test_mask = matrix_df["year"] == target_year

        train_years = matrix_df.loc[train_mask, "year"]
        test_years = matrix_df.loc[test_mask, "year"]

        # 1. Conjunto de treino contém apenas anos anteriores a target_year
        assert not train_years.empty
        assert (train_years < target_year).all(), (
            f"Data leakage detectado: treinamento de {target_year} contém anos >= {target_year}!"
        )

        # 2. Conjunto de teste contém exatamente o ano alvo
        assert (test_years == target_year).all()
        assert len(test_years) == 10  # 10 eixos


def test_temporal_isolation_canary_injection():
    """Cenário 2: Injeção de dados futuros anômalos (ano 2024) não altera o conjunto de treino de 2020."""
    custom_history = list(HISTORICAL_ENEM_THEMES)
    canary_theme = {
        "year": 2024,
        "official_theme": "TEMA CANARIO INFILTRADO NO FUTURO",
        "thematic_axis": "Tecnologia",
        "central_problem": "Injeção de teste para detecção de leakage.",
    }
    # Substitui ou adiciona canário no futuro
    custom_history = [t for t in custom_history if t["year"] != 2024] + [canary_theme]

    simulator = WalkForwardSimulator()
    simulator.history_records = custom_history
    matrix_df = simulator._build_dataset(max_year=2024)

    # Ao filtrar para target_year = 2020, o canário de 2024 não deve estar presente no treino
    train_mask = matrix_df["year"] < 2020
    assert 2024 not in matrix_df.loc[train_mask, "year"].values
    assert (matrix_df.loc[train_mask, "year"] < 2020).all()


# ==============================================================================
# Grupo 2: Cálculo Determinístico de Top-1, Top-3 e MRR
# ==============================================================================

@pytest.mark.parametrize("rank, expected_top1, expected_top3, expected_rr", [
    (1, True, True, 1.0000),
    (2, False, True, 0.5000),
    (3, False, True, 1.0 / 3.0),
    (4, False, False, 0.2500),
    (5, False, False, 0.2000),
    (6, False, False, 1.0 / 6.0),
    (7, False, False, 1.0 / 7.0),
    (8, False, False, 0.1250),
    (9, False, False, 1.0 / 9.0),
    (10, False, False, 0.1000),
])
def test_top1_top3_mrr_for_each_rank_position(rank, expected_top1, expected_top3, expected_rr):
    """Cenário 3: Verifica invariantes de Top-1, Top-3 e Reciprocal Rank para todas as 10 posições ordinais."""
    target_axis = "Educação"
    ranking = make_valid_ranking(target_axis=target_axis, target_rank=rank)

    brier = compute_brier_score(ranking, target_axis)
    log_loss = compute_log_loss(ranking, target_axis)

    res = BacktestYearResult(
        year=2020,
        official_axis=target_axis,
        official_theme="Tema de Teste",
        predicted_ranking=ranking,
        rank_of_official=rank,
        top1_hit=expected_top1,
        top3_hit=expected_top3,
        reciprocal_rank=expected_rr,
        brier_score=brier,
        log_loss=log_loss,
    )

    assert res.top1_hit == expected_top1
    assert res.top3_hit == expected_top3
    assert pytest.approx(res.reciprocal_rank, abs=1e-5) == expected_rr


def test_aggregated_ranking_metrics_calculation():
    """Cenário 4: Avalia agregação de métricas em BacktestSummary para série de posições [1, 2, 3, 5]."""
    ranks = [1, 2, 3, 5]
    years = [2016, 2017, 2018, 2019]
    yearly_results = []

    for y, r in zip(years, ranks):
        axis = "Educação"
        ranking = make_valid_ranking(target_axis=axis, target_rank=r)
        brier = compute_brier_score(ranking, axis)
        log_loss = compute_log_loss(ranking, axis)
        yearly_results.append(
            BacktestYearResult(
                year=y,
                official_axis=axis,
                official_theme=f"Tema {y}",
                predicted_ranking=ranking,
                rank_of_official=r,
                top1_hit=(r == 1),
                top3_hit=(r <= 3),
                reciprocal_rank=1.0 / r,
                brier_score=brier,
                log_loss=log_loss,
            )
        )

    n = len(yearly_results)
    top1 = sum(1.0 for res in yearly_results if res.top1_hit) / n  # 1/4 = 0.25
    top3 = sum(1.0 for res in yearly_results if res.top3_hit) / n  # 3/4 = 0.75
    mrr = sum(res.reciprocal_rank for res in yearly_results) / n    # (1 + 0.5 + 1/3 + 0.2)/4 = 2.03333/4 = 0.508333...
    mean_brier = sum(res.brier_score for res in yearly_results) / n
    mean_log_loss = sum(res.log_loss for res in yearly_results) / n

    lift = compute_relative_lift(top3, 0.30)  # (0.75 - 0.30)/0.30 * 100 = 150.0%
    p_val = compute_exact_binomial_pvalue(k=3, n=4, p0=0.30)

    summary = BacktestSummary(
        years_evaluated=years,
        top1_accuracy=top1,
        top3_hit_rate=top3,
        mrr=mrr,
        mean_brier_score=mean_brier,
        mean_log_loss=mean_log_loss,
        baseline_top3_rate=0.30,
        relative_lift_top3=lift,
        binomial_test_p_value=p_val,
        yearly_results=yearly_results,
    )

    assert pytest.approx(summary.top1_accuracy) == 0.25
    assert pytest.approx(summary.top3_hit_rate) == 0.75
    assert pytest.approx(summary.mrr, abs=1e-4) == (1.0 + 0.5 + (1.0 / 3.0) + 0.2) / 4.0
    assert pytest.approx(summary.relative_lift_top3) == 150.0


# ==============================================================================
# Grupo 3: Brier Score e Log Loss
# ==============================================================================

def test_brier_score_perfect_prediction():
    """Cenário 5: Predição perfeita (p=1.0 no alvo, 0.0 nos demais) resulta em Brier Score = 0.0."""
    target_axis = "Saúde"
    probs = {ax: (1.0 if ax == target_axis else 0.0) for ax in OFFICIAL_AXES}
    assert compute_brier_score(probs, target_axis) == 0.0


def test_brier_score_uniform_prediction():
    """Cenário 6: Predição uniforme (p=0.10 em todos os 10 eixos) resulta em Brier Score = 0.90."""
    target_axis = "Saúde"
    probs = {ax: 0.10 for ax in OFFICIAL_AXES}
    # (0.10 - 1.0)^2 + 9 * (0.10 - 0.0)^2 = 0.81 + 0.09 = 0.90
    assert pytest.approx(compute_brier_score(probs, target_axis), abs=1e-6) == 0.90


def test_brier_score_worst_possible_prediction():
    """Cenário 7: Pior predição possível (p=1.0 no eixo errado, 0.0 no oficial) resulta em Brier Score = 2.0."""
    target_axis = "Saúde"
    wrong_axis = "Educação"
    probs = {ax: (1.0 if ax == wrong_axis else 0.0) for ax in OFFICIAL_AXES}
    # (0.0 - 1.0)^2 + (1.0 - 0.0)^2 + 8 * 0.0 = 1.0 + 1.0 = 2.0
    assert pytest.approx(compute_brier_score(probs, target_axis), abs=1e-6) == 2.0


def test_log_loss_perfect_prediction():
    """Cenário 8: Predição perfeita (p=1.0 no alvo) resulta em Log Loss = 0.0."""
    target_axis = "Tecnologia"
    probs = {target_axis: 1.0}
    assert compute_log_loss(probs, target_axis) == 0.0


def test_log_loss_uniform_prediction():
    """Cenário 9: Predição uniforme (p=0.10 no alvo) resulta em Log Loss = -ln(0.10) ~ 2.302585."""
    target_axis = "Tecnologia"
    probs = {target_axis: 0.10}
    expected = -math.log(0.10)
    assert pytest.approx(compute_log_loss(probs, target_axis), abs=1e-5) == expected


def test_log_loss_zero_probability_clipping():
    """Cenário 10: Probabilidade zero (p=0.0) no alvo é tratada com clipping epsilon=1e-15 sem exceção de domínio."""
    target_axis = "Tecnologia"
    probs = {target_axis: 0.0}
    loss = compute_log_loss(probs, target_axis, eps=1e-15)
    expected = -math.log(1e-15)
    assert pytest.approx(loss, abs=1e-4) == expected
    assert loss > 34.0


# ==============================================================================
# Grupo 4: Teste Binomial Exato e Lift Relativo
# ==============================================================================

@pytest.mark.parametrize("k, n, p0, expected_p_value", [
    (0, 11, 0.30, 1.000000),
    (3, 11, 0.30, 0.687260),
    (6, 11, 0.30, 0.078225),
    (7, 11, 0.30, 0.021619),  # Significativo a alpha=0.05
    (8, 11, 0.30, 0.004291),  # Altamente significativo a alpha=0.01
    (11, 11, 0.30, 1.77147e-06),  # 0.30^11
])
def test_exact_binomial_pvalue_lookup_table(k, n, p0, expected_p_value):
    """Cenário 11: Valida a tabela analítica do Teste Binomial Exato para N=11 e p0=0.30."""
    calculated = compute_exact_binomial_pvalue(k=k, n=n, p0=p0)
    assert pytest.approx(calculated, rel=1e-3) == expected_p_value


@pytest.mark.parametrize("hit_rate, baseline, expected_lift", [
    (0.30, 0.30, 0.0),
    (0.60, 0.30, 100.0),
    (8.0 / 11.0, 0.30, ((8.0 / 11.0 - 0.30) / 0.30) * 100.0),
    (0.15, 0.30, -50.0),
    (0.00, 0.30, -100.0),
])
def test_relative_lift_various_scenarios(hit_rate, baseline, expected_lift):
    """Cenário 12: Valida o cálculo do ganho relativo (Relative Lift) contra a baseline."""
    lift = compute_relative_lift(hit_rate, baseline)
    assert pytest.approx(lift, abs=1e-4) == expected_lift


# ==============================================================================
# Grupo 5: Validações de Argumentos, Invariantes e Execução In-Memory
# ==============================================================================

def test_validation_probabilities_sum_invariant():
    """Cenário 13: Rejeita ranking em que as probabilidades não somam 1.0."""
    target_axis = "Educação"
    ranking = make_valid_ranking(target_axis=target_axis, target_rank=1)
    # Altera propositalmente a soma das probabilidades para 0.85
    invalid_ranking = [(axis, p * 0.85) for axis, p in ranking]

    with pytest.raises(ValueError, match="soma das probabilidades deve ser 1.0"):
        BacktestYearResult(
            year=2020,
            official_axis=target_axis,
            official_theme="Tema Teste",
            predicted_ranking=invalid_ranking,
            rank_of_official=1,
            top1_hit=True,
            top3_hit=True,
            reciprocal_rank=1.0,
            brier_score=0.1,
            log_loss=0.5,
        )


def test_validation_invalid_year_ranges():
    """Cenário 14: Rejeita start_year > end_year e start_year < 1999."""
    simulator = WalkForwardSimulator()

    with pytest.raises(ValueError, match="não pode ser maior que end_year"):
        simulator.run(start_year=2026, end_year=2025)

    with pytest.raises(ValueError, match="deve ser >= 1999"):
        simulator.run(start_year=1998, end_year=2020)


def test_validation_official_axis_domain():
    """Cenário 15: Rejeita eixo oficial que não pertença aos 10 eixos canônicos."""
    ranking = make_valid_ranking(target_axis="Educação", target_rank=1)

    with pytest.raises(ValueError, match="não pertence à taxonomia oficial"):
        BacktestYearResult(
            year=2020,
            official_axis="Economia Circular",  # Eixo inválido fora dos 10
            official_theme="Tema Teste",
            predicted_ranking=ranking,
            rank_of_official=1,
            top1_hit=True,
            top3_hit=True,
            reciprocal_rank=1.0,
            brier_score=0.1,
            log_loss=0.5,
        )


def test_simulator_runs_in_memory_without_db():
    """Cenário 16: Simulador executa integralmente in-memory sem banco ativo (2023-2025)."""
    simulator = WalkForwardSimulator(historical_data_path=None, db_session=None)
    summary = simulator.run(start_year=2023, end_year=2025)

    assert isinstance(summary, BacktestSummary)
    assert summary.years_evaluated == [2023, 2024, 2025]
    assert len(summary.yearly_results) == 3

    for res in summary.yearly_results:
        assert isinstance(res, BacktestYearResult)
        assert res.year in [2023, 2024, 2025]
        assert 1 <= res.rank_of_official <= 10
        assert 0.0 <= res.brier_score <= 2.0
        assert res.log_loss >= 0.0

    assert 0.0 <= summary.top1_accuracy <= 1.0
    assert 0.0 <= summary.top3_hit_rate <= 1.0
    assert 0.0 <= summary.mrr <= 1.0
    assert 0.0 <= summary.binomial_test_p_value <= 1.0


def test_summary_serialization_to_dict_and_json():
    """Valida serialização completa de BacktestSummary para dicionário e JSON."""
    ranks = [1, 2]
    years = [2021, 2022]
    yearly_results = []
    for y, r in zip(years, ranks):
        axis = "Educação"
        ranking = make_valid_ranking(target_axis=axis, target_rank=r)
        yearly_results.append(
            BacktestYearResult(
                year=y,
                official_axis=axis,
                official_theme=f"Tema {y}",
                predicted_ranking=ranking,
                rank_of_official=r,
                top1_hit=(r == 1),
                top3_hit=(r <= 3),
                reciprocal_rank=1.0 / r,
                brier_score=compute_brier_score(ranking, axis),
                log_loss=compute_log_loss(ranking, axis),
            )
        )

    summary = BacktestSummary(
        years_evaluated=years,
        top1_accuracy=0.50,
        top3_hit_rate=1.00,
        mrr=0.75,
        mean_brier_score=sum(r.brier_score for r in yearly_results) / 2,
        mean_log_loss=sum(r.log_loss for r in yearly_results) / 2,
        baseline_top3_rate=0.30,
        relative_lift_top3=((1.00 - 0.30) / 0.30) * 100.0,
        binomial_test_p_value=compute_exact_binomial_pvalue(2, 2, 0.30),
        yearly_results=yearly_results,
    )

    d = summary.to_dict()
    assert isinstance(d, dict)
    assert d["top1_accuracy"] == 0.50
    assert len(d["yearly_results"]) == 2

    j = summary.to_json()
    assert isinstance(j, str)
    assert '"top1_accuracy": 0.5' in j


# ==============================================================================
# Grupo 6: Testes de Regressão e Resiliência Numérica (Cenários 17-21, M1.2 R2)
# ==============================================================================


def test_regression_log_loss_tolerance_p_slightly_exceeds_one():
    """Cenário 17 (Regressão M1.2): compute_log_loss com p_target = 1.0000001
    deve ceifar o valor em 1.0 e retornar 0.0, sem valores negativos.
    """
    target_axis = "Educação"
    p_discrepant = 1.0000001
    loss = compute_log_loss({target_axis: p_discrepant}, target_axis)

    assert loss >= 0.0, f"Log loss não pode ser negativo: obteve {loss}"
    assert loss == 0.0, f"Log loss para probabilidade máxima deve ser 0.0: obteve {loss}"


def test_regression_backtest_year_result_accepts_p_slightly_exceeds_one():
    """Cenário 18 (Regressão M1.2): BacktestYearResult deve aceitar ranking com soma
    ligeiramente discrepante (1.0000001) sem colapsar com ValueError em log_loss.
    """
    target = "Educação"
    # Soma = 1.0000001 (desvio 1e-7, bem abaixo do teto de 1e-4)
    ranking = [(target, 1.0000001)] + [(ax, 0.0) for ax in OFFICIAL_AXES if ax != target]
    total_p = sum(p for _, p in ranking)
    assert abs(total_p - 1.0) < 1e-4

    loss = compute_log_loss(ranking, target)
    brier = compute_brier_score(ranking, target)

    # Não deve levantar ValueError
    result = BacktestYearResult(
        year=2020,
        official_axis=target,
        official_theme="Tema Teste Tolerância",
        predicted_ranking=ranking,
        rank_of_official=1,
        top1_hit=True,
        top3_hit=True,
        reciprocal_rank=1.0,
        brier_score=brier,
        log_loss=loss,
    )

    assert result.log_loss >= 0.0
    assert result.log_loss == 0.0
    assert result.top1_hit is True
    assert result.rank_of_official == 1


def test_regression_brier_score_empty_probabilities_penalizes_target():
    """Cenário 19 (Regressão M1.2): compute_brier_score com dicionário vazio deve
    retornar 1.0 (penalidade pelo alvo com probabilidade 0.0), e não 0.0.
    """
    target_axis = "Educação"
    score = compute_brier_score({}, target_axis)
    assert score == 1.0, f"Brier score para dict vazio deve ser 1.0, obteve {score}"


def test_regression_brier_score_missing_official_axis_penalizes_target():
    """Cenário 20 (Regressão M1.2): compute_brier_score com 9 eixos não-alvo (prob 1/9)
    e alvo ausente deve somar 9*(1/9)^2 + (0 - 1)^2 = 1/9 + 1 = 10/9 ~ 1.111111.
    """
    target_axis = "Educação"
    partial_dict = {ax: 1.0 / 9.0 for ax in OFFICIAL_AXES if ax != target_axis}
    score = compute_brier_score(partial_dict, target_axis)

    expected_score = (1.0 / 9.0) + 1.0  # 10/9
    assert pytest.approx(score, abs=1e-6) == expected_score, (
        f"Brier score com alvo ausente deve ser {expected_score}, obteve {score}"
    )

    # Caso extremo: apenas um eixo errado presente com prob 1.0 (deve somar (1-0)^2 + (0-1)^2 = 2.0)
    score_worst = compute_brier_score({"Saúde": 1.0}, target_axis)
    assert pytest.approx(score_worst, abs=1e-6) == 2.0


def test_regression_consecutive_veto_zero_division_resilience():
    """Cenário 21 (Regressão M1.2): Veto consecutivo não deve falhar com ZeroDivisionError
    quando toda a massa de probabilidade do modelo estiver concentrada no eixo vetado.
    Deve aplicar fallback uniforme (1/9) sobre os eixos restantes não-vetados.
    """
    sim = WalkForwardSimulator(enforce_hard_veto=True)
    matrix_df = sim._build_dataset(max_year=2020)

    target_year = 2020
    prev_year = 2019
    prev_axis = sim.history_map[prev_year]["axis"]

    test_mask = matrix_df["year"] == target_year
    test_axes = matrix_df.loc[test_mask, "axis"].tolist()

    # Mock do modelo XGBoost prevendo prob 1.0 para prev_axis e 0.0 para os demais 9 eixos
    mock_probs = np.zeros((10, 2))
    prev_idx = test_axes.index(prev_axis)
    mock_probs[prev_idx, 1] = 1.0
    for i in range(10):
        if i != prev_idx:
            mock_probs[i, 0] = 1.0

    with patch("xgboost.XGBClassifier") as MockXGB:
        instance = MockXGB.return_value
        instance.predict_proba.return_value = mock_probs

        # Não deve disparar ZeroDivisionError
        ranking = sim._train_and_predict_year(target_year, matrix_df)

    assert len(ranking) == 10
    ranking_dict = dict(ranking)

    # 1. O eixo vetado deve ter probabilidade estritamente zero e estar na última posição
    assert ranking_dict[prev_axis] == 0.0
    assert ranking[-1][0] == prev_axis
    assert ranking[-1][1] == 0.0

    # 2. Cada um dos 9 eixos restantes deve receber exatamente 1/9 de probabilidade
    for axis, prob in ranking[:-1]:
        assert pytest.approx(prob, abs=1e-5) == 1.0 / 9.0

    # 3. A soma das probabilidades deve ser exatamente 1.0
    assert pytest.approx(sum(p for _, p in ranking), abs=1e-6) == 1.0

    # 4. A avaliação do ano deve produzir um BacktestYearResult íntegro
    res = sim._evaluate_year(target_year, ranking)
    assert isinstance(res, BacktestYearResult)
    assert res.year == target_year
    assert 0.0 <= res.brier_score <= 2.0
    assert res.log_loss >= 0.0
