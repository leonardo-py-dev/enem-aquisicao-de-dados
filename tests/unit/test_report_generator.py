"""
Suíte de testes unitários para o módulo ReportGenerator (Milestone 4 — R3).

Valida:
1. Geração do arquivo HTML5 autocontido e criação automática de diretórios pais.
2. Ausência estrita de dependências externas quebráveis (CDNs, scripts ou stylesheets HTTP/HTTPS).
3. Presença da tabela ano a ano do backtesting walk-forward (2015–2025) e das 5 métricas estatísticas.
4. Comparação empírica contra a linha de base aleatória (30.0%) e p-valor do Teste Binomial Exato.
5. Presença do ranking oficial para o ENEM 2026 com veto determinístico a "Demografia e Envelhecimento" (0.00%).
6. Presença explícita dos 3 temas candidatos e dos 5 elementos obrigatórios da Competência 5.
7. Renderização dos 4 gráficos vetoriais procedurais em SVG nativo (<svg>).
8. Integridade do payload JSON embutido (<script id="enem-report-data">) e sanitização anti-XSS.
"""

import json
from pathlib import Path
import re
import pytest
from typer.testing import CliRunner

from enem_pipeline.cli import app
from enem_pipeline.ml.backtest import BacktestSummary, WalkForwardSimulator
from enem_pipeline.ml.predictor import ENEM2026Predictor, InterventionProposal, Prediction2026Result
from enem_pipeline.ml.report_generator import ReportGenerator


@pytest.fixture(scope="module")
def real_backtest_summary() -> BacktestSummary:
    simulator = WalkForwardSimulator()
    return simulator.run(start_year=2015, end_year=2025)


@pytest.fixture(scope="module")
def real_prediction_2026() -> Prediction2026Result:
    predictor = ENEM2026Predictor(offline=True)
    return predictor.predict()


def test_report_generator_creates_valid_self_contained_html(
    tmp_path: Path,
    real_backtest_summary: BacktestSummary,
    real_prediction_2026: Prediction2026Result,
):
    """Verifica que o arquivo HTML é criado, não possui CDNs externas e contém todas as seções R1, R2 e R3."""
    out_file = tmp_path / "subdir" / "enem_predictor_report.html"
    gen = ReportGenerator()
    result_path = gen.generate_html(real_backtest_summary, real_prediction_2026, output_path=str(out_file))

    assert Path(result_path).is_file()
    content = Path(result_path).read_text(encoding="utf-8")

    # 1. Estrutura HTML5 válida
    assert content.startswith("<!DOCTYPE html>")
    assert "<html lang=\"pt-BR\">" in content
    assert "</html>" in content

    # 2. Autocontenção estrita (zero scripts/links externos via http/https)
    assert not re.search(r'<script[^>]+src=["\']https?://', content, re.IGNORECASE)
    assert not re.search(r'<link[^>]+href=["\']https?://', content, re.IGNORECASE)
    assert not re.search(r'@import\s+url\(["\']?https?://', content, re.IGNORECASE)

    # 3. Presença de todos os 11 anos de backtesting (2015 a 2025)
    for yr in range(2015, 2026):
        assert f'data-year="{yr}"' in content
        assert str(yr) in content

    # 4. Presença das métricas consolidadas e comparativo com baseline de 30.0%
    assert "30.0%" in content
    assert "Top-3 Hit Rate" in content
    assert "Top-1 Accuracy" in content
    assert "Mean Reciprocal Rank" in content
    assert "Brier Score" in content
    assert "Log Loss" in content
    assert "Teste Binomial Exato" in content

    # 5. Presença do veto determinístico ao tema de 2025 para o ENEM 2026
    assert "Demografia e Envelhecimento" in content
    assert "0.00%" in content
    assert "VETADO" in content

    # 6. Presença dos 4 gráficos SVG procedurais embutidos
    svg_count = len(re.findall(r"<svg\b", content))
    assert svg_count >= 4

    # 7. Presença dos 3 temas candidatos e dos 5 elementos da Competência 5
    assert "1. AGENTE" in content
    assert "2. AÇÃO" in content
    assert "3. MEIO / MODO" in content
    assert "4. DETALHAMENTO" in content
    assert "5. EFEITO / FINALIDADE" in content
    for prop in real_prediction_2026.top3_proposals:
        assert prop.axis in content
        assert prop.candidate_theme in content
        assert prop.agente in content
        assert prop.acao in content
        assert prop.meio_modo in content
        assert prop.detalhamento in content
        assert prop.efeito in content


def test_report_generator_embedded_json_integrity(
    tmp_path: Path,
    real_backtest_summary: BacktestSummary,
    real_prediction_2026: Prediction2026Result,
):
    """Verifica que o bloco JSON embutido em #enem-report-data é perfeitamente parseável e fiel aos objetos."""
    out_file = tmp_path / "report_json_check.html"
    gen = ReportGenerator()
    gen.generate_html(real_backtest_summary, real_prediction_2026, output_path=str(out_file))

    content = out_file.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="enem-report-data" type="application/json">(.*?)</script>',
        content,
        re.DOTALL,
    )
    assert match is not None, "Bloco JSON #enem-report-data não encontrado no HTML."
    payload = json.loads(match.group(1))

    assert "metadata" in payload
    assert "backtest_summary" in payload
    assert "prediction_2026" in payload
    assert payload["backtest_summary"]["baseline_top3_rate"] == 0.30
    assert len(payload["backtest_summary"]["yearly_results"]) == 11
    assert payload["prediction_2026"]["vetoed_axis"] == "Demografia e Envelhecimento"
    assert len(payload["prediction_2026"]["ranking"]) == 10
    assert len(payload["prediction_2026"]["top3_proposals"]) == 3


def test_report_generator_xss_escaping(
    tmp_path: Path,
    real_backtest_summary: BacktestSummary,
    real_prediction_2026: Prediction2026Result,
):
    """Garante que strings contendo tags HTML maliciosas são devidamente escapadas."""
    malicious_prop = InterventionProposal(
        axis=real_prediction_2026.top3_proposals[0].axis,
        candidate_theme='<script>alert("xss")</script> Tema Teste',
        agente="Ministério <img src=x onerror=alert(1)>",
        acao="Implementar ação segura",
        meio_modo="Por meio de verbas federais",
        detalhamento="Detalhamento técnico seguro",
        efeito="Efeito social positivo",
    )
    custom_pred = Prediction2026Result(
        ranking=real_prediction_2026.ranking,
        vetoed_axis=real_prediction_2026.vetoed_axis,
        top3_proposals=[
            malicious_prop,
            real_prediction_2026.top3_proposals[1],
            real_prediction_2026.top3_proposals[2],
        ],
    )

    out_file = tmp_path / "xss_test.html"
    ReportGenerator().generate_html(real_backtest_summary, custom_pred, output_path=str(out_file))
    content = out_file.read_text(encoding="utf-8")

    assert '<script>alert("xss")</script>' not in content
    assert '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;' in content
    assert "<img src=x onerror=alert(1)>" not in content


def test_report_generator_type_validation():
    """Garante que passar tipos inválidos levanta TypeError explícito."""
    gen = ReportGenerator()
    with pytest.raises(TypeError):
        gen.generate_html("invalid", "invalid")  # type: ignore[arg-type]


def test_cli_report_command_generates_html_file(tmp_path: Path):
    """Testa o subcomando 'enem-pipeline report --output ...' via CliRunner."""
    out_file = tmp_path / "cli_generated_report.html"
    runner = CliRunner()
    result = runner.invoke(app, ["report", "--output", str(out_file)])

    assert result.exit_code == 0
    assert out_file.is_file()
    assert out_file.stat().st_size > 10_000
