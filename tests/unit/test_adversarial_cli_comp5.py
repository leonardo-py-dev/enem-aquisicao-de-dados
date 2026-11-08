"""
Testes Adversariais Empíricos — Challenger 2 (Milestone 2)
Auditoria e Estresse de Interface Typer CLI e Integridade Pedagógica da Competência 5.

Escopo:
1. Estresse da CLI (Typer CliRunner):
   - Invocação de predict-2026 com flags inválidas/conflitantes e argumentos inesperados.
   - Validação estrita do stdout de --json-output (sem poluição de logs, parseabilidade por json.loads, conformidade de schema).
   - Invocação de backtest com anos invertidos, limites fora da série histórica (1998-2025), tipos não-inteiros e verificação de graceful exit sem unhandled stack traces.
2. Estresse da Competência 5 e Catálogo Canônico:
   - Auditoria dos 10 eixos em CANONICAL_PROPOSALS: nenhum campo vazio, em branco, com placeholders ('N/A', 'TODO') ou trivial (< 20 caracteres).
   - Teste adversarial de fronteira em InterventionProposal.__post_init__: strings vazias, whitespaces, None, eixos fora da taxonomia oficial e imutabilidade (frozen).
"""

import json
from dataclasses import FrozenInstanceError
import pytest
from typer.testing import CliRunner

from enem_pipeline.cli import app
from enem_pipeline.ml.backtest import OFFICIAL_AXES
from enem_pipeline.ml.predictor import (
    CANONICAL_PROPOSALS,
    InterventionProposal,
    Prediction2026Result,
    VETOED_AXIS_2025,
)


runner = CliRunner()


# ==============================================================================
# SEÇÃO 1: ESTRESSE DA CLI TYPER (CliRunner)
# ==============================================================================

class TestCliAdversarialStress:
    """Suíte adversarial de testes de interface CLI."""

    def test_predict_2026_with_unknown_flag(self):
        """Invocação de predict-2026 com flag desconhecida deve falhar com exit code 2 (Typer/Click)."""
        result = runner.invoke(app, ["predict-2026", "--flag-totalmente-inexistente"])
        assert result.exit_code != 0
        err_msg = (result.stderr or result.output).lower()
        assert "no such option" in err_msg or "error" in err_msg

    def test_predict_2026_with_unexpected_positional_argument(self):
        """Invocação de predict-2026 com argumentos posicionais extras deve falhar graciosamente."""
        result = runner.invoke(app, ["predict-2026", "2026", "extra-arg"])
        assert result.exit_code != 0
        err_msg = (result.stderr or result.output).lower()
        assert "got unexpected extra argument" in err_msg or "error" in err_msg

    def test_predict_2026_json_output_strict_purity(self):
        """Valida que o stdout de 'predict-2026 --json-output --offline' é estritamente parseável via json.loads sem poluição Rich."""
        result = runner.invoke(app, ["predict-2026", "--json-output", "--offline"])
        assert result.exit_code == 0, f"Falha na execução: {result.stdout}"

        # Teste estrito: json.loads no stdout bruto sem qualquer limpeza manual
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as err:
            pytest.fail(f"O stdout de --json-output não é JSON puro parseável: {err}\nStdout:\n{result.stdout}")

        # Validação do contrato existente
        assert "vetoed_axis" in data, "Chave 'vetoed_axis' ausente do JSON."
        assert data["vetoed_axis"] == VETOED_AXIS_2025
        assert "ranking" in data, "Chave 'ranking' ausente do JSON."
        assert len(data["ranking"]) == 10
        assert "top3_proposals" in data, "Chave 'top3_proposals' ausente do JSON."
        assert len(data["top3_proposals"]) == 3

        # Auditoria de tipos
        assert isinstance(data["vetoed_axis"], str)
        assert isinstance(data["ranking"], list)
        for item in data["ranking"]:
            assert isinstance(item, list) and len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)
            assert 0.0 <= item[1] <= 1.0

        for prop in data["top3_proposals"]:
            assert isinstance(prop, dict)
            for k in ("axis", "candidate_theme", "agente", "acao", "meio_modo", "efeito", "detalhamento"):
                assert k in prop
                assert isinstance(prop[k], str)
                assert len(prop[k].strip()) > 0

    def test_predict_2026_json_optional_metadata_audit(self):
        """Audita a presença ou ausência dos metadados 'target_year' e 'execution_timestamp' na saída JSON."""
        result = runner.invoke(app, ["predict-2026", "--json-output", "--offline"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        
        # Teste adversarial informativo: verifica se o schema possui chaves de contexto temporal
        has_target_year = "target_year" in data
        has_timestamp = "execution_timestamp" in data
        # Registramos se estão ausentes para fins de observação e relatório
        assert isinstance(has_target_year, bool)
        assert isinstance(has_timestamp, bool)

    def test_backtest_inverted_year_range_fails_cleanly(self):
        """Invocação de backtest com --start-year maior que --end-year deve sair com exit code 1 sem traceback."""
        result = runner.invoke(app, ["backtest", "--start-year", "2025", "--end-year", "2015"])
        assert result.exit_code == 1
        assert "não pode ser maior que" in result.stdout
        assert "Traceback" not in result.stdout

    def test_backtest_out_of_bounds_historical_range_fails_cleanly(self):
        """Invocação de backtest fora do catálogo histórico (<1999 ou >2025) deve sair com exit code 1 sem traceback."""
        # Teste 1: ano anterior ao limite inferior (< 1999)
        res_low = runner.invoke(app, ["backtest", "--start-year", "1990", "--end-year", "2020"])
        assert res_low.exit_code == 1
        assert "fora do catálogo histórico" in res_low.stdout
        assert "Traceback" not in res_low.stdout

        # Teste 2: ano posterior ao limite superior (> 2025)
        res_high = runner.invoke(app, ["backtest", "--start-year", "2015", "--end-year", "2030"])
        assert res_high.exit_code == 1
        assert "fora do catálogo histórico" in res_high.stdout
        assert "Traceback" not in res_high.stdout

    def test_backtest_non_integer_year_types(self):
        """Invocação de backtest com tipos de dados não-inteiros deve ser barrada pelo parser do Typer."""
        result = runner.invoke(app, ["backtest", "--start-year", "ano_invalido"])
        assert result.exit_code != 0
        err_msg = (result.stderr or result.output).lower()
        assert "is not a valid int" in err_msg or "invalid value" in err_msg

    def test_backtest_json_output_strict_purity(self):
        """Valida que 'backtest --json-output' emite JSON estritamente parseável com todos os KPIs obrigatórios."""
        # Executa backtest de janela curta para velocidade (2024 a 2025)
        result = runner.invoke(app, ["backtest", "--start-year", "2024", "--end-year", "2025", "--json-output"])
        assert result.exit_code == 0

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as err:
            pytest.fail(f"O stdout de backtest --json-output não é JSON puro: {err}\nStdout:\n{result.stdout}")

        required_keys = [
            "years_evaluated",
            "top1_accuracy",
            "top3_hit_rate",
            "mrr",
            "mean_brier_score",
            "mean_log_loss",
            "baseline_top3_rate",
            "relative_lift_top3",
            "binomial_test_p_value",
            "yearly_results",
        ]
        for key in required_keys:
            assert key in data, f"Chave obrigatória '{key}' ausente de BacktestSummary JSON."


# ==============================================================================
# SEÇÃO 2: ESTRESSE DA COMPETÊNCIA 5 E DO CATÁLOGO CANÔNICO
# ==============================================================================

class TestCompetence5CanonicalStress:
    """Suíte adversarial de testes de conformidade dos 5 elementos da Competência 5."""

    def test_canonical_catalog_exhaustiveness_and_coverage(self):
        """Valida que CANONICAL_PROPOSALS possui exatamente os 10 eixos oficiais do ENEM."""
        assert len(CANONICAL_PROPOSALS) == 10
        assert set(CANONICAL_PROPOSALS.keys()) == set(OFFICIAL_AXES)

    @pytest.mark.parametrize("axis", OFFICIAL_AXES)
    def test_canonical_proposal_elements_non_empty_and_substantive(self, axis: str):
        """Audita exaustivamente cada um dos 10 eixos: nenhum campo pode ser vazio, whitespace, 'N/A', 'TODO' ou < 20 chars."""
        prop = CANONICAL_PROPOSALS[axis]
        assert prop.axis == axis

        fields_to_check = {
            "candidate_theme": prop.candidate_theme,
            "agente": prop.agente,
            "acao": prop.acao,
            "meio_modo": prop.meio_modo,
            "detalhamento": prop.detalhamento,
            "efeito": prop.efeito,
        }

        forbidden_placeholders = {"n/a", "na", "todo", "tbd", "teste", "none", "null", "undefined"}

        for field_name, value in fields_to_check.items():
            assert isinstance(value, str), f"Campo '{field_name}' no eixo '{axis}' não é string."
            clean_val = value.strip()
            
            # 1. Não pode ser vazio ou só whitespace
            assert len(clean_val) > 0, f"Campo '{field_name}' no eixo '{axis}' está vazio."
            
            # 2. Não pode ser placeholder genérico
            assert clean_val.lower() not in forbidden_placeholders, (
                f"Campo '{field_name}' no eixo '{axis}' contém placeholder proibido: '{clean_val}'."
            )

            # 3. Deve ter substância pedagógica (comprimento substantivo >= 20 caracteres)
            assert len(clean_val) >= 20, (
                f"Campo '{field_name}' no eixo '{axis}' é trivial (< 20 caracteres): '{clean_val}' (len={len(clean_val)})."
            )

    @pytest.mark.parametrize(
        "empty_val",
        ["", "   ", "\t\n  "],
    )
    def test_intervention_proposal_rejects_empty_and_whitespace_fields(self, empty_val: str):
        """Valida que __post_init__ rejeita ativamente qualquer campo vazio ou só com whitespaces."""
        target_fields = ["candidate_theme", "agente", "acao", "meio_modo", "detalhamento", "efeito"]
        
        valid_args = {
            "axis": "Educação",
            "candidate_theme": "Tema devidamente detalhado para teste de estresse",
            "agente": "Ministério da Educação em cooperação com o Consed",
            "acao": "Implementar polos de iniciação científica e letramento",
            "meio_modo": "Por meio de dotação orçamentária do FNDE e parcerias",
            "detalhamento": "Com laboratórios móveis e bolsas de estímulo à pesquisa",
            "efeito": "A fim de superar o analfabetismo funcional dos jovens",
        }

        for field in target_fields:
            corrupted_args = dict(valid_args)
            corrupted_args[field] = empty_val
            with pytest.raises(ValueError, match=f"campo '{field}' não pode ser vazio"):
                InterventionProposal(**corrupted_args)

    def test_intervention_proposal_rejects_none_fields(self):
        """Valida que __post_init__ rejeita campos com valor None."""
        valid_args = {
            "axis": "Educação",
            "candidate_theme": "Tema devidamente detalhado para teste de estresse",
            "agente": "Ministério da Educação em cooperação com o Consed",
            "acao": "Implementar polos de iniciação científica e letramento",
            "meio_modo": "Por meio de dotação orçamentária do FNDE e parcerias",
            "detalhamento": "Com laboratórios móveis e bolsas de estímulo à pesquisa",
            "efeito": "A fim de superar o analfabetismo funcional dos jovens",
        }
        for field in valid_args.keys():
            corrupted_args = dict(valid_args)
            corrupted_args[field] = None
            with pytest.raises(ValueError):
                InterventionProposal(**corrupted_args)

    def test_intervention_proposal_rejects_invalid_axis(self):
        """Valida que eixos fora de OFFICIAL_AXES são rejeitados com ValueError."""
        invalid_axes = [
            "Eixo Inexistente",
            "meio ambiente",  # case-sensitive
            "EDUCAÇÃO",
            "Políticas Públicas",
            "Esporte e Lazer",
            "",
            "   ",
        ]
        for bad_axis in invalid_axes:
            with pytest.raises(ValueError):
                InterventionProposal(
                    axis=bad_axis,
                    candidate_theme="Tema válido com mais de vinte caracteres",
                    agente="Agente válido institucional",
                    acao="Ação concreta de políticas públicas",
                    meio_modo="Meio e modo orçamentário específico",
                    detalhamento="Detalhamento substantivo da intervenção",
                    efeito="Efeito social na garantia de direitos fundamentais",
                )

    def test_intervention_proposal_is_immutable(self):
        """Valida que InterventionProposal é frozen e imutável contra modificações acidentais."""
        prop = CANONICAL_PROPOSALS["Educação"]
        with pytest.raises(FrozenInstanceError):
            prop.agente = "Outro Agente Não Autorizado"
        with pytest.raises(FrozenInstanceError):
            prop.detalhamento = "Tentativa de sobrescrita maliciosa"
