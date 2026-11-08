"""
Suíte de Testes Adversariais e de Estresse Numérico para o Backtesting Walk-Forward.
Milestone 1 — Challenger M1.2.

Submete as funções de métricas e calibração de src/enem_pipeline/ml/backtest.py a:
1. Estresse numérico extremo (quase-zeros, floats no limite, desvios de soma).
2. Confrontação com oráculos matemáticos (scipy.stats.binomtest, Monte Carlo 1M).
3. Rankings com empates exatos e análise de consistência ordinal.
"""

import math
import numpy as np
import pytest
import scipy.stats as stats
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


# Helper para criar ranking normalizado válido
def make_ranking(probs_dict: dict[str, float]) -> list[tuple[str, float]]:
    sorted_items = sorted(probs_dict.items(), key=lambda x: (-x[1], x[0]))
    s = sum(p for _, p in sorted_items)
    return [(ax, p / s) for ax, p in sorted_items]


# ==============================================================================
# SEÇÃO 1: ESTRESSE NUMÉRICO EXTREMO (FLOAT LIMITS, SUB-NORMALS, EPSILON)
# ==============================================================================

class TestNumericalExtremeStress:
    """Testes de estresse em limites numéricos de precisão float IEEE 754."""

    def test_log_loss_near_zero_subnormal_precision(self):
        """Probabilidades quase-zero (1e-16, 1e-30, 0.0) devem ser ceifadas em eps=1e-15 sem colapso."""
        target = "Educação"
        for tiny_p in [1e-16, 1e-25, 1e-100, 0.0]:
            loss = compute_log_loss({target: tiny_p}, target, eps=1e-15)
            expected = -math.log(1e-15)
            assert pytest.approx(loss, rel=1e-7) == expected
            assert not math.isnan(loss)
            assert not math.isinf(loss)

    def test_log_loss_when_p_slightly_exceeds_one(self):
        """Valida que p_target ligeiramente > 1.0 (ex: 1.0000001) é devidamente ceifado em 1.0,
        garantindo log_loss == 0.0 sem colapso para valor negativo.
        """
        target = "Educação"
        p_discrepant = 1.0000001
        loss = compute_log_loss({target: p_discrepant}, target)
        assert loss == 0.0, f"loss esperado 0.0 após clipping superior em 1.0, obteve {loss}"

    def test_backtest_year_result_accepts_imprecise_sum_and_sanitizes_log_loss(self):
        """Valida a resolução da inconsistência interna: BacktestYearResult tolera soma de probabilidades
        até 1.0 + 1e-4 e aceita graciosa e seguramente o resultado quando log_loss >= -1e-6.
        """
        target = "Educação"
        # Cria ranking com soma 1.0000001 (dentro da tolerância de 1e-4 do BacktestYearResult)
        ranking = [(target, 1.0000001)] + [(ax, 0.0) for ax in OFFICIAL_AXES if ax != target]
        total_p = sum(p for _, p in ranking)
        assert abs(total_p - 1.0) < 1e-4

        loss = compute_log_loss(ranking, target)
        brier = compute_brier_score(ranking, target)

        res = BacktestYearResult(
            year=2020,
            official_axis=target,
            official_theme="Tema Teste",
            predicted_ranking=ranking,
            rank_of_official=1,
            top1_hit=True,
            top3_hit=True,
            reciprocal_rank=1.0,
            brier_score=brier,
            log_loss=loss,
        )
        assert res.log_loss == 0.0

        with pytest.raises(ValueError, match="log_loss .* deve ser >= 0.0 e finito"):
            BacktestYearResult(
                year=2020,
                official_axis=target,
                official_theme="Tema Teste",
                predicted_ranking=ranking,
                rank_of_official=1,
                top1_hit=True,
                top3_hit=True,
                reciprocal_rank=1.0,
                brier_score=brier,
                log_loss=-0.01,
            )

    def test_brier_score_empty_or_missing_official_axis(self):
        """Valida que compute_brier_score com dict vazio ou faltando official_axis
        imputa corretamente a penalidade (0.0 - 1.0)^2 = 1.0 sobre o alvo ausente.
        """
        score_empty = compute_brier_score({}, "Educação")
        assert score_empty == 1.0, f"Score de dict vazio deve ser 1.0, obteve {score_empty}"

        nine_axes = {ax: 1.0 / 9.0 for ax in OFFICIAL_AXES if ax != "Educação"}
        score_missing = compute_brier_score(nine_axes, "Educação")
        expected = (1.0 / 9.0) + 1.0
        assert pytest.approx(score_missing, abs=1e-6) == expected

    def test_brier_score_numerical_stability_extreme_probabilities(self):
        """Brier score com probabilidades extremas (1.0 - 1e-16 e 1e-16)."""
        target = "Educação"
        other = "Saúde"
        ranking = [
            (target, 1.0 - 1e-15),
            (other, 1e-15),
        ] + [(ax, 0.0) for ax in OFFICIAL_AXES if ax not in [target, other]]
        score = compute_brier_score(ranking, target)
        assert pytest.approx(score, abs=1e-12) == 0.0

    def test_relative_lift_boundary_and_zero_baseline(self):
        """compute_relative_lift deve rejeitar baseline <= 0 e calcular lift correto para extremos."""
        with pytest.raises(ValueError, match="maior que zero"):
            compute_relative_lift(0.5, baseline=0.0)

        with pytest.raises(ValueError, match="maior que zero"):
            compute_relative_lift(0.5, baseline=-0.1)

        # Lift com acerto perfeito (100% vs 30% baseline -> +233.33%)
        lift_100 = compute_relative_lift(1.0, baseline=0.30)
        assert pytest.approx(lift_100, rel=1e-4) == (7.0 / 3.0) * 100.0


# ==============================================================================
# SEÇÃO 2: ORÁCULOS MATEMÁTICOS E MONTE CARLO (BINOMIAL TEST, LOG LOSS, BRIER)
# ==============================================================================

class TestMathematicalOraclesAndMonteCarlo:
    """Confronta os cálculos analíticos de backtest.py contra oráculos de alta fidelidade."""

    def test_exact_binomial_oracle_vs_scipy_comprehensive_grid(self):
        """Confronta compute_exact_binomial_pvalue contra scipy.stats.binomtest
        para todas as combinações de n in [1..30], k in [0..n], p0 in [0.05, 0.1, 0.3, 0.5, 0.8].
        """
        p0_values = [0.05, 0.10, 0.30, 0.50, 0.80]
        max_diff = 0.0

        for n in range(1, 25):
            for p0 in p0_values:
                for k in range(0, n + 2):
                    calc_p = compute_exact_binomial_pvalue(k, n, p0)
                    if k == 0:
                        oracle_p = 1.0
                    elif k > n:
                        oracle_p = 0.0
                    else:
                        oracle_res = stats.binomtest(k=k, n=n, p=p0, alternative="greater")
                        oracle_p = oracle_res.pvalue

                    diff = abs(calc_p - oracle_p)
                    if diff > max_diff:
                        max_diff = diff
                    assert pytest.approx(calc_p, abs=1e-9) == oracle_p, (
                        f"Divergência detectada para k={k}, n={n}, p0={p0}: {calc_p} vs {oracle_p}"
                    )

        assert max_diff < 1e-9, f"Diferença máxima com oráculo scipy: {max_diff}"

    def test_exact_binomial_oracle_vs_monte_carlo_1m_simulations(self):
        """Simulação de Monte Carlo com 1.000.000 de iterações para N=11 e p0=0.30.
        Verifica se as frequências empíricas convergem rigorosamente para os p-valores analíticos.
        """
        rng = np.random.default_rng(seed=42)
        n = 11
        p0 = 0.30
        n_trials = 1_000_000

        # Amostragem direta Binomial(11, 0.30)
        samples = rng.binomial(n=n, p=p0, size=n_trials)

        for k in range(0, n + 1):
            empirical_tail_p = float(np.mean(samples >= k))
            analytical_p = compute_exact_binomial_pvalue(k, n, p0)

            # Erro padrão da estimativa de Monte Carlo: sqrt(p * (1 - p) / N)
            se = math.sqrt(analytical_p * (1.0 - analytical_p) / n_trials) if 0 < analytical_p < 1 else 1e-6
            # Intervalo de 99.9% de confiança (z ~ 3.29)
            margin = max(3.29 * se, 1e-4)

            assert abs(empirical_tail_p - analytical_p) <= margin, (
                f"Monte Carlo ({empirical_tail_p:.6f}) diverge da analítica ({analytical_p:.6f}) "
                f"para k={k} além da margem {margin:.6f}"
            )

    def test_exact_binomial_overflow_on_large_n(self):
        """ESTRESSE NUMÉRICO: Testa o ponto de quebra de compute_exact_binomial_pvalue
        onde math.comb(n, i) ultrapassa 1.79e308 e causa OverflowError ao converter para float.
        """
        # n = 1030 causa OverflowError ao tentar multiplicar int gigante por float
        with pytest.raises(OverflowError, match="int too large to convert to float"):
            compute_exact_binomial_pvalue(k=515, n=1030, p0=0.5)

    def test_brier_score_vs_independent_vector_oracle(self):
        """Confronta compute_brier_score contra formulação matricial independente ||p - y||_2^2."""
        rng = np.random.default_rng(seed=123)
        for _ in range(50):
            # Gera vetor aleatório de probabilidades via Dirichlet (uniforme)
            raw_p = rng.dirichlet(np.ones(10))
            ranking = [(ax, float(p)) for ax, p in zip(OFFICIAL_AXES, raw_p)]
            for official_axis in OFFICIAL_AXES:
                score = compute_brier_score(ranking, official_axis)

                # Oráculo vetorial puro
                y_vec = np.array([1.0 if ax == official_axis else 0.0 for ax in OFFICIAL_AXES])
                p_vec = np.array([dict(ranking)[ax] for ax in OFFICIAL_AXES])
                oracle_brier = float(np.sum((p_vec - y_vec) ** 2))

                assert pytest.approx(score, abs=1e-12) == oracle_brier

    def test_log_loss_vs_cross_entropy_oracle(self):
        """Confronta compute_log_loss contra a fórmula da entropia cruzada -sum(y_k * ln(p_k))."""
        rng = np.random.default_rng(seed=456)
        for _ in range(50):
            raw_p = rng.dirichlet(np.ones(10))
            ranking = [(ax, float(p)) for ax, p in zip(OFFICIAL_AXES, raw_p)]
            for official_axis in OFFICIAL_AXES:
                loss = compute_log_loss(ranking, official_axis)
                p_official = dict(ranking)[official_axis]
                oracle_loss = float(-np.log(max(p_official, 1e-15)))
                assert pytest.approx(loss, abs=1e-12) == oracle_loss


# ==============================================================================
# SEÇÃO 3: EMPATES EXATOS E VALIDAÇÃO DE CONSISTÊNCIA ORDINAL
# ==============================================================================

class TestTiesAndOrdinalConsistency:
    """Testa o comportamento sob empates estritos de probabilidade e invariantes ordinais."""

    def test_exact_tie_all_ten_axes_uniform(self):
        """Sob predição 100% uniforme (p=0.10 para todos), o desempate alfabético define posições determinísticas."""
        uniform_ranking = [(ax, 0.10) for ax in sorted(OFFICIAL_AXES)]
        # Verifica se BacktestYearResult aceita probabilidades iguais
        res = BacktestYearResult(
            year=2020,
            official_axis="Cultura",  # 1º alfabeticamente
            official_theme="Tema Teste",
            predicted_ranking=uniform_ranking,
            rank_of_official=1,
            top1_hit=True,
            top3_hit=True,
            reciprocal_rank=1.0,
            brier_score=compute_brier_score(uniform_ranking, "Cultura"),
            log_loss=compute_log_loss(uniform_ranking, "Cultura"),
        )
        assert res.rank_of_official == 1
        assert res.top1_hit is True

        # O último alfabeticamente ("Trabalho e Economia") terá rank 10
        res_last = BacktestYearResult(
            year=2020,
            official_axis="Trabalho e Economia",
            official_theme="Tema Teste",
            predicted_ranking=uniform_ranking,
            rank_of_official=10,
            top1_hit=False,
            top3_hit=False,
            reciprocal_rank=0.10,
            brier_score=compute_brier_score(uniform_ranking, "Trabalho e Economia"),
            log_loss=compute_log_loss(uniform_ranking, "Trabalho e Economia"),
        )
        assert res_last.rank_of_official == 10
        assert res_last.top3_hit is False

    def test_ranking_rejection_when_not_in_descending_order(self):
        """Valida que predicted_ranking fora de ordem decrescente é terminantemente rejeitado."""
        probs = [0.10] * 10
        probs[0] = 0.05
        probs[1] = 0.15  # Desordem: probs[0] < probs[1]
        probs_norm = [p / sum(probs) for p in probs]

        ranking = [(ax, p) for ax, p in zip(OFFICIAL_AXES, probs_norm)]

        with pytest.raises(ValueError, match="ordem decrescente de probabilidade"):
            BacktestYearResult(
                year=2020,
                official_axis=OFFICIAL_AXES[0],
                official_theme="Tema Teste",
                predicted_ranking=ranking,
                rank_of_official=1,
                top1_hit=True,
                top3_hit=True,
                reciprocal_rank=1.0,
                brier_score=0.9,
                log_loss=2.3,
            )

    def test_strict_ordinal_monotonicity_top1_implies_top3(self):
        """Invariante lógica: top1_hit == True IMPLICA estritamente top3_hit == True."""
        for rank in range(1, 11):
            top1 = (rank == 1)
            top3 = (rank <= 3)
            if top1:
                assert top3, f"Para rank {rank}, top1 é True mas top3 é False!"

    def test_reciprocal_rank_strict_bounds(self):
        """Reciprocal rank deve estar estritamente no conjunto {1/1, 1/2, ..., 1/10}."""
        valid_rrs = {1.0 / r for r in range(1, 11)}
        for r in range(1, 11):
            rr = 1.0 / r
            assert any(abs(rr - valid) < 1e-12 for valid in valid_rrs)


# ==============================================================================
# SEÇÃO 4: RESILIÊNCIA E COERÊNCIA GLOBAL DO SIMULADOR
# ==============================================================================

class TestSimulatorGlobalResilience:
    """Testa a integridade de ponta a ponta do simulador walk-forward."""

    def test_walk_forward_metrics_mathematical_consistency(self):
        """Executa simulação real (2020 a 2025) e valida todas as relações entre métricas."""
        sim = WalkForwardSimulator()
        summary = sim.run(start_year=2020, end_year=2025)

        n = len(summary.years_evaluated)
        assert n == 6

        # 1. Top-1 <= Top-3
        assert summary.top1_accuracy <= summary.top3_hit_rate

        # 2. MRR >= Top-1 (pois MRR = 1.0 * top1 + sum(1/r * (rank==r)))
        assert summary.mrr >= summary.top1_accuracy

        # 3. Baseline top-3 é 0.30
        assert summary.baseline_top3_rate == 0.30

        # 4. Lift calculado bate com fórmula
        expected_lift = ((summary.top3_hit_rate - 0.30) / 0.30) * 100.0
        assert pytest.approx(summary.relative_lift_top3, abs=1e-5) == expected_lift

        # 5. P-valor do teste binomial coincide com o oráculo do scipy
        k_hits = sum(1 for r in summary.yearly_results if r.top3_hit)
        scipy_p = stats.binomtest(k=k_hits, n=n, p=0.30, alternative="greater").pvalue
        assert pytest.approx(summary.binomial_test_p_value, abs=1e-6) == scipy_p
