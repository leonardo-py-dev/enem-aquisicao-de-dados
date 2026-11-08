"""
Suíte de testes unitários para a interface de linha de comando (CLI) Typer dos módulos de ML.
Testa:
- Subcomando 'backtest' com opções de intervalo de anos, saída JSON e renderização Rich.
- Validação de erros temporais em 'backtest'.
- Subcomando 'predict-2026' com renderização Rich dos 5 elementos, veto a 2025, flag --offline e --json-output.
- Tratamento resiliente de erros em 'predict-2026'.
- Textos de ajuda (--help) de todos os comandos novos.
"""

import json
from unittest.mock import patch, MagicMock
import pytest
from typer.testing import CliRunner

from enem_pipeline.cli import app
from enem_pipeline.ml.backtest import (
    OFFICIAL_AXES,
    BacktestSummary,
    BacktestYearResult,
)
from enem_pipeline.ml.predictor import (
    InterventionProposal,
    Prediction2026Result,
)


@pytest.fixture
def mock_backtest_summary() -> BacktestSummary:
    """Gera um BacktestSummary sintético válido para 2015-2025."""
    years = list(range(2015, 2026))
    yearly = []
    for y in years:
        target_axis = "Educação" if y % 2 == 0 else "Tecnologia"
        target_rank = 1 if y % 2 == 0 else 2
        other_axes = [ax for ax in OFFICIAL_AXES if ax != target_axis]
        ranked_axes = []
        for pos in range(1, 11):
            if pos == target_rank:
                ranked_axes.append(target_axis)
            else:
                ranked_axes.append(other_axes.pop(0))

        weights = [float(11 - i) for i in range(1, 11)]
        s = sum(weights)
        ranking = [(ax, w / s) for ax, w in zip(ranked_axes, weights)]

        top1 = (target_rank == 1)
        top3 = (target_rank <= 3)
        rr = 1.0 / target_rank

        yearly.append(
            BacktestYearResult(
                year=y,
                official_axis=target_axis,
                official_theme=f"Tema Oficial do ENEM {y}",
                predicted_ranking=ranking,
                rank_of_official=target_rank,
                top1_hit=top1,
                top3_hit=top3,
                reciprocal_rank=rr,
                brier_score=0.3500,
                log_loss=1.1200,
            )
        )
    return BacktestSummary(
        years_evaluated=years,
        top1_accuracy=sum(1.0 for r in yearly if r.top1_hit) / len(years),
        top3_hit_rate=sum(1.0 for r in yearly if r.top3_hit) / len(years),
        mrr=sum(r.reciprocal_rank for r in yearly) / len(years),
        mean_brier_score=0.3500,
        mean_log_loss=1.1200,
        baseline_top3_rate=0.30,
        relative_lift_top3=((1.0 - 0.30) / 0.30) * 100.0,
        binomial_test_p_value=0.0000018,
        yearly_results=yearly,
    )


@pytest.fixture
def mock_prediction_2026_result() -> Prediction2026Result:
    """Gera um Prediction2026Result sintético válido com veto determinístico a 2025."""
    probs = [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01]
    top_axes = [
        "Meio Ambiente",
        "Tecnologia",
        "Educação",
        "Saúde",
        "Cultura",
        "Trabalho e Economia",
        "Direitos Humanos e Cidadania",
        "Desigualdade Social",
        "Segurança Pública",
    ]
    ranking = [(ax, p) for ax, p in zip(top_axes, probs)] + [("Demografia e Envelhecimento", 0.0)]

    proposals = [
        InterventionProposal(
            axis="Meio Ambiente",
            candidate_theme="Racismo ambiental e proteção a desastres climáticos",
            agente="Ministério do Meio Ambiente e Mudança do Clima",
            acao="Implementar o Plano Nacional de Infraestrutura Resiliente",
            meio_modo="Aporte do Fundo Clima e convênios municipais",
            detalhamento="Instalação de alarmes pluviométricos e contenção geotécnica",
            efeito="Garantir moradia digna e mitigar o racismo ambiental",
        ),
        InterventionProposal(
            axis="Tecnologia",
            candidate_theme="Regulação algorítmica e combate à desinformação",
            agente="Ministério das Comunicações e CGI.br",
            acao="Instituir diretrizes de transparência de algoritmos e rotulagem de IA",
            meio_modo="Auditorias públicas periódicas e APIs para pesquisadores",
            detalhamento="Marcas d'água criptográficas obrigatórias em mídias sintéticas",
            efeito="Preservar a soberania informacional e a integridade democrática",
        ),
        InterventionProposal(
            axis="Educação",
            candidate_theme="Letramento científico e erradicação do analfabetismo funcional",
            agente="Ministério da Educação em colaboração com o Consed",
            acao="Criar polos de iniciação científica no contraturno escolar",
            meio_modo="Parcerias com universidades federais e laboratórios",
            detalhamento="Bolsas de estímulo à pesquisa júnior para estudantes vulneráveis",
            efeito="Superar o fosso educacional e ampliar a capacidade crítica",
        ),
    ]
    return Prediction2026Result(
        ranking=ranking,
        vetoed_axis="Demografia e Envelhecimento",
        top3_proposals=proposals,
    )


def test_cli_backtest_default_invocation(mock_backtest_summary):
    """Testa a execução padrão de 'enem-pipeline backtest' com renderização Rich."""
    runner = CliRunner()
    with patch("enem_pipeline.cli.WalkForwardSimulator") as mock_sim_cls:
        instance = mock_sim_cls.return_value
        instance.run.return_value = mock_backtest_summary

        result = runner.invoke(app, ["backtest"])

        assert result.exit_code == 0
        assert "Validação Walk-Forward" in result.stdout
        assert "Top-1 Accuracy" in result.stdout
        assert "Top-3 Hit Rate" in result.stdout
        assert "Teste Binomial Exato" in result.stdout
        assert "30.0%" in result.stdout
        instance.run.assert_called_once_with(start_year=2015, end_year=2025)


def test_cli_backtest_custom_years(mock_backtest_summary):
    """Testa 'enem-pipeline backtest --start-year 2018 --end-year 2022'."""
    runner = CliRunner()
    with patch("enem_pipeline.cli.WalkForwardSimulator") as mock_sim_cls:
        instance = mock_sim_cls.return_value
        instance.run.return_value = mock_backtest_summary

        result = runner.invoke(app, ["backtest", "--start-year", "2018", "--end-year", "2022"])

        assert result.exit_code == 0
        instance.run.assert_called_once_with(start_year=2018, end_year=2022)


def test_cli_backtest_invalid_year_range():
    """Testa que passar start_year > end_year falha com exit code 1 e mensagem explicativa."""
    runner = CliRunner()
    result = runner.invoke(app, ["backtest", "--start-year", "2025", "--end-year", "2015"])

    assert result.exit_code == 1
    assert "não pode ser maior" in result.stdout


def test_cli_backtest_json_output(mock_backtest_summary):
    """Testa 'enem-pipeline backtest --json-output' e valida se o output é um JSON parseável válido."""
    runner = CliRunner()
    with patch("enem_pipeline.cli.WalkForwardSimulator") as mock_sim_cls:
        mock_sim_cls.return_value.run.return_value = mock_backtest_summary

        result = runner.invoke(app, ["backtest", "--json-output"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "years_evaluated" in data
        assert "top1_accuracy" in data
        assert "top3_hit_rate" in data
        assert "relative_lift_top3" in data
        assert "binomial_test_p_value" in data
        assert len(data["yearly_results"]) == len(mock_backtest_summary.years_evaluated)


def test_cli_predict_2026_default_invocation(mock_prediction_2026_result):
    """Testa a execução padrão de 'enem-pipeline predict-2026'."""
    runner = CliRunner()
    with patch("enem_pipeline.cli.ENEM2026Predictor") as mock_pred_cls:
        instance = mock_pred_cls.return_value
        instance.predict.return_value = mock_prediction_2026_result

        result = runner.invoke(app, ["predict-2026"])

        assert result.exit_code == 0
        assert "REGRA DE VETO DETERMINÍSTICO" in result.stdout
        assert "Demografia e Envelhecimento" in result.stdout
        assert "0.0%" in result.stdout
        assert "PROPOSTA NOTA 1000" in result.stdout
        assert "AGENTE:" in result.stdout
        assert "AÇÃO:" in result.stdout
        assert "MEIO / MODO:" in result.stdout
        assert "DETALHAMENTO:" in result.stdout
        assert "EFEITO / FINALIDADE:" in result.stdout


def test_cli_predict_2026_json_output(mock_prediction_2026_result):
    """Testa 'enem-pipeline predict-2026 --json-output'."""
    runner = CliRunner()
    with patch("enem_pipeline.cli.ENEM2026Predictor") as mock_pred_cls:
        mock_pred_cls.return_value.predict.return_value = mock_prediction_2026_result

        result = runner.invoke(app, ["predict-2026", "--json-output"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "ranking" in data
        assert data["vetoed_axis"] == "Demografia e Envelhecimento"
        assert len(data["top3_proposals"]) == 3
        assert data["top3_proposals"][0]["axis"] == "Meio Ambiente"


def test_cli_predict_2026_offline_flag(mock_prediction_2026_result):
    """Testa que a flag --offline repassa offline=True para o construtor de ENEM2026Predictor."""
    runner = CliRunner()
    with patch("enem_pipeline.cli.ENEM2026Predictor") as mock_pred_cls:
        mock_pred_cls.return_value.predict.return_value = mock_prediction_2026_result

        result = runner.invoke(app, ["predict-2026", "--offline"])

        assert result.exit_code == 0
        mock_pred_cls.assert_called_once_with(offline=True)


def test_cli_predict_2026_error_handling():
    """Testa que falhas internas não tratadas no modelo resultam em exit code 1 e mensagem limpa."""
    runner = CliRunner()
    with patch("enem_pipeline.cli.ENEM2026Predictor") as mock_pred_cls:
        mock_pred_cls.return_value.predict.side_effect = RuntimeError("Falha de I/O nos pesos do modelo")

        result = runner.invoke(app, ["predict-2026"])

        assert result.exit_code == 1
        assert "Falha" in result.stdout or "Erro" in result.stdout


def test_cli_help_texts():
    """Valida a formatação e presença de documentação em --help."""
    import re
    ansi_re = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    runner = CliRunner()

    res_backtest = runner.invoke(app, ["backtest", "--help"])
    clean_backtest = ansi_re.sub("", res_backtest.stdout)
    assert res_backtest.exit_code == 0
    assert "--start-year" in clean_backtest
    assert "--end-year" in clean_backtest
    assert "--json-output" in clean_backtest

    res_predict = runner.invoke(app, ["predict-2026", "--help"])
    clean_predict = ansi_re.sub("", res_predict.stdout)
    assert res_predict.exit_code == 0
    assert "--json-output" in clean_predict
    assert "--offline" in clean_predict
