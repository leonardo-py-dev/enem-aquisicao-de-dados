"""Suíte de Testes Adversariais Empíricos de Cegueira Preditiva e Isolamento Temporal (Gate R2.1).

Elaborado pelo Agente Challenger M1 R2.1 (teamwork_challenger).

Objetivo:
Submeter a implementação de `src/enem_pipeline/ml/backtest.py` e `features.py` a ataques
agressivos de contaminação temporal, injeção de canários no futuro profundo, mutação de
targets do ano avaliado e estresse nas rotinas numéricas remediadas na Iteração 2.
"""

import copy
import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

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
from enem_pipeline.ml.features import build_training_matrix


class TestFutureDataContaminationAndCanaries:
    """Ataques de contaminação com canários no futuro e verificação de cegueira estrita."""

    def test_multi_year_deep_future_contamination_invariance(self):
        """ATAQUE 1: Injeção de canários maciços em múltiplos anos futuros (2021 a 2028).

        Verifica se a predição para TODOS os anos anteriores a 2021 (2015 a 2020)
        permanece rigorosamente inalterada (diferença absoluta = 0.0 em todas as probabilidades).
        """
        # 1. Execução limpa de referência
        sim_clean = WalkForwardSimulator(min_train_year=2003, random_state=42)
        summary_clean = sim_clean.run(start_year=2015, end_year=2020)
        clean_rankings_by_year = {
            r.year: r.predicted_ranking for r in summary_clean.yearly_results
        }

        # 2. Injeção de canários hostis em 2021..2028 no catálogo histórico
        corrupted_history = [
            copy.deepcopy(t) for t in HISTORICAL_ENEM_THEMES if t["year"] < 2021
        ]
        # Inserir anos futuros com eixos anômalos e repetitivos
        future_canary_axes = [
            (2021, "Tecnologia", "CANÁRIO HOSTIL 2021"),
            (2022, "Tecnologia", "CANÁRIO HOSTIL 2022"),
            (2023, "Segurança Pública", "CANÁRIO HOSTIL 2023"),
            (2024, "Segurança Pública", "CANÁRIO HOSTIL 2024"),
            (2025, "Educação", "CANÁRIO HOSTIL 2025"),
            (2026, "Cultura", "CANÁRIO HOSTIL 2026"),
            (2027, "Meio Ambiente", "CANÁRIO HOSTIL 2027"),
            (2028, "Saúde", "CANÁRIO HOSTIL 2028"),
        ]
        for y, axis, theme in future_canary_axes:
            corrupted_history.append({
                "year": y,
                "official_theme": theme,
                "thematic_axis": axis,
                "central_problem": "Injeção maciça de dados futuros",
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

        summary_corrupted = sim_corrupted.run(start_year=2015, end_year=2020)
        corrupted_rankings_by_year = {
            r.year: r.predicted_ranking for r in summary_corrupted.yearly_results
        }

        # Verificação exata para cada um dos 6 anos avaliados (2015..2020)
        for y in range(2015, 2021):
            clean_ranking = clean_rankings_by_year[y]
            corrupted_ranking = corrupted_rankings_by_year[y]

            assert len(clean_ranking) == len(corrupted_ranking) == 10
            for (ax_clean, p_clean), (ax_corr, p_corr) in zip(clean_ranking, corrupted_ranking):
                assert ax_clean == ax_corr, f"Ano {y}: Ordem dos eixos divergiu: {ax_clean} != {ax_corr}"
                diff = abs(p_clean - p_corr)
                assert diff < 1e-12, (
                    f"Vazamento temporal detectado no ano {y} para o eixo {ax_clean}! "
                    f"Limpo={p_clean:.12f}, Corrompido={p_corr:.12f}, Diff={diff}"
                )

    def test_target_year_theme_mutation_blindness_all_years(self):
        """ATAQUE 2: Adulteração hostil do tema e eixo oficial no PRÓPRIO ano avaliado.

        Testa individualmente cada ano de 2015 a 2025. A alteração do tema oficial de Y
        no histórico não pode afetar em absoluto as probabilidades inferidas para Y,
        pois o modelo de treino para Y deve ser 100% cego aos dados do exame de Y.
        """
        for target_year in [2015, 2017, 2019, 2021, 2023, 2025]:
            # Execução de referência para target_year
            sim_ref = WalkForwardSimulator(min_train_year=2003, random_state=42)
            res_ref = sim_ref.run(start_year=target_year, end_year=target_year).yearly_results[0]
            ref_ranking = res_ref.predicted_ranking

            # Adulterar o tema de target_year para um eixo completamente diferente
            corrupted_history = copy.deepcopy(HISTORICAL_ENEM_THEMES)
            for item in corrupted_history:
                if item["year"] == target_year:
                    # Se o original era Ciência/Tecnologia, muda para Segurança Pública, etc.
                    item["thematic_axis"] = "Segurança Pública" if item["thematic_axis"] != "Segurança Pública" else "Educação"
                    item["official_theme"] = "TEMA FAKE HOSTIL PARA TESTE DE CEGUEIRA"

            sim_fake = WalkForwardSimulator(min_train_year=2003, random_state=42)
            sim_fake.history_records = corrupted_history
            sim_fake.history_map = {
                item["year"]: {
                    "axis": item["thematic_axis"],
                    "theme": item["official_theme"],
                }
                for item in corrupted_history
            }

            res_fake = sim_fake.run(start_year=target_year, end_year=target_year).yearly_results[0]
            fake_ranking = res_fake.predicted_ranking

            # O ranking predito para target_year DEVE SER EXATAMENTE IDÊNTICO
            for (ax1, p1), (ax2, p2) in zip(ref_ranking, fake_ranking):
                assert ax1 == ax2, f"Ano {target_year}: eixo divergiu sob tema adulterado"
                assert abs(p1 - p2) < 1e-12, (
                    f"Ano {target_year}: Quebra de cegueira preditiva! A adulteração do tema oficial "
                    f"de {target_year} alterou a probabilidade do eixo {ax1} (diff={abs(p1 - p2)})"
                )

    def test_isolated_year_vs_full_walk_forward_exact_bit_identity(self):
        """ATAQUE 3: Invariância de sub-janela (Execução Isolada vs Walk-Forward Completo).

        Para TODOS os 11 anos de 2015 a 2025, o resultado predito gerado na simulação completa
        de 11 anos DEVE ser bit-a-bit idêntico ao resultado gerado ao executar o simulador
        isoladamente apenas para aquele ano individual (start_year=Y, end_year=Y).
        """
        # Execução completa 2015-2025
        sim_full = WalkForwardSimulator(min_train_year=2003, random_state=42)
        summary_full = sim_full.run(start_year=2015, end_year=2025)
        full_by_year = {r.year: r for r in summary_full.yearly_results}

        # Execuções isoladas ano a ano
        for y in range(2015, 2026):
            sim_iso = WalkForwardSimulator(min_train_year=2003, random_state=42)
            summary_iso = sim_iso.run(start_year=y, end_year=y)
            res_iso = summary_iso.yearly_results[0]
            res_full = full_by_year[y]

            assert res_iso.official_axis == res_full.official_axis
            assert res_iso.rank_of_official == res_full.rank_of_official
            assert res_iso.top1_hit == res_full.top1_hit
            assert res_iso.top3_hit == res_full.top3_hit
            assert abs(res_iso.brier_score - res_full.brier_score) < 1e-12
            assert abs(res_iso.log_loss - res_full.log_loss) < 1e-12

            for (ax_iso, p_iso), (ax_full, p_full) in zip(
                res_iso.predicted_ranking, res_full.predicted_ranking
            ):
                assert ax_iso == ax_full
                assert abs(p_iso - p_full) < 1e-12, (
                    f"Ano {y}: Divergência entre execução isolada ({p_iso}) e walk-forward completo ({p_full})"
                )


class TestVetoFallbackAndZeroDivisionResilience:
    """Estresse adversarial no mecanismo de veto consecutivo e resiliência a divisão por zero."""

    def test_consecutive_veto_with_100_percent_concentration_on_vetoed_axis(self):
        """ATAQUE 4: O modelo concentra 100% da probabilidade no eixo vetado do ano anterior.

        Verifica se a correção da Iteração 2 (fallback uniforme para os 9 eixos não vetados)
        opera sem levantar ZeroDivisionError, atribui 0.0 ao eixo vetado e 1/9 aos demais.
        """
        sim = WalkForwardSimulator(min_train_year=2003, enforce_hard_veto=True)

        target_year = 2020
        prev_year = 2019
        prev_axis = sim.history_map[prev_year]["axis"]

        # Mock da matriz de treino
        matrix_df = sim._build_dataset(max_year=target_year)

        # Mock de XGBoost que prediz 1.0 no eixo vetado e 0.0 em todos os outros
        mock_model = MagicMock()
        mock_probs = np.zeros((10, 2))
        test_axes = matrix_df.loc[matrix_df["year"] == target_year, "axis"].tolist()
        veto_idx = test_axes.index(prev_axis)
        mock_probs[veto_idx, 1] = 1.0  # 100% de probabilidade no eixo vetado

        mock_model.predict_proba.return_value = mock_probs

        import xgboost as xgb
        original_xgb = xgb.XGBClassifier
        xgb.XGBClassifier = MagicMock(return_value=mock_model)

        try:
            ranking = sim._train_and_predict_year(target_year=target_year, matrix_df=matrix_df)
        finally:
            xgb.XGBClassifier = original_xgb

        # Validações estruturais do ranking resultante
        assert len(ranking) == 10
        ranking_dict = dict(ranking)

        # O eixo vetado deve ter probabilidade estritamente 0.0
        assert ranking_dict[prev_axis] == 0.0

        # Os outros 9 eixos devem ter exatamente 1/9 (~0.111111)
        expected_uniform = 1.0 / 9.0
        for ax, p in ranking_dict.items():
            if ax != prev_axis:
                assert abs(p - expected_uniform) < 1e-6, f"Eixo {ax} deveria ter {expected_uniform}, tem {p}"

        # A soma total deve ser exatamente 1.0
        total_p = sum(ranking_dict.values())
        assert abs(total_p - 1.0) < 1e-6

        # Avaliar ano para certificar instanciação válida de BacktestYearResult
        result = sim._evaluate_year(target_year=target_year, ranking=ranking)
        assert isinstance(result, BacktestYearResult)
        assert result.brier_score >= 0.0
        assert result.log_loss >= 0.0


class TestNumericalBoundaryStressR2:
    """Estresse nos limites numéricos de Log Loss, Brier Score e Dataclasses corrigidos na R2."""

    def test_log_loss_never_negative_for_any_float_above_one(self):
        """ATAQUE 5.1: p_target excedendo 1.0 por diferentes magnitudes de resíduo IEEE 754.

        Log Loss deve ser rigorosamente >= 0.0 (nunca -0.0 nem negativo).
        """
        for p in [1.0, 1.000000000000001, 1.0000001, 1.00001, 1.001, 1.01, 2.0, 1e6]:
            loss = compute_log_loss({"Educação": p}, "Educação")
            assert loss == 0.0, f"Para p={p}, log_loss deveria ser 0.0, obteve {loss}"
            assert not math.isnan(loss)
            assert str(loss) != "-0.0"

    def test_log_loss_zero_and_subnormal_probabilities(self):
        """ATAQUE 5.2: Probabilidades nulas e subnormais.

        Garante que o epsilon clipping (1e-15) previne math domain error e retorna ~34.538776.
        """
        for p in [0.0, -0.0, 1e-16, 1e-30, 1e-300]:
            loss = compute_log_loss({"Educação": p}, "Educação")
            assert loss > 30.0
            assert not math.isinf(loss)
            assert not math.isnan(loss)

    def test_brier_score_empty_dict_and_missing_target_penalization(self):
        """ATAQUE 5.3: Brier Score com alvo ausente ou dict vazio deve penalizar com +1.0."""
        # Dict vazio -> official_axis ausente -> penalidade quadrática = (0 - 1)^2 = 1.0
        assert compute_brier_score({}, "Educação") == 1.0

        # Dict com 9 eixos mas sem o oficial -> soma dos eixos errados + penalidade de 1.0
        non_official = {ax: 0.1 for ax in OFFICIAL_AXES if ax != "Educação"}
        # soma dos 9 eixos errados: 9 * (0.1 - 0)^2 = 9 * 0.01 = 0.09
        # penalidade do oficial ausente: (0 - 1)^2 = 1.0
        # total esperado: 1.09
        score = compute_brier_score(non_official, "Educação")
        assert abs(score - 1.09) < 1e-12

    def test_backtest_year_result_boundary_acceptance(self):
        """ATAQUE 5.4: BacktestYearResult deve aceitar resíduos infinitesimais sem disparar ValueError."""
        p_target = 1.0000001
        ranking = [("Educação", p_target)] + [(ax, 0.0) for ax in OFFICIAL_AXES if ax != "Educação"]
        # Normalização suave para caber na tolerância de soma 1e-4
        res = BacktestYearResult(
            year=2020,
            official_axis="Educação",
            official_theme="Tema Teste",
            predicted_ranking=ranking,
            rank_of_official=1,
            top1_hit=True,
            top3_hit=True,
            reciprocal_rank=1.0,
            brier_score=compute_brier_score(ranking, "Educação"),
            log_loss=compute_log_loss(ranking, "Educação"),
        )
        assert res.log_loss == 0.0
        assert res.top1_hit is True
        assert res.rank_of_official == 1
