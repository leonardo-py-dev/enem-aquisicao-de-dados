"""Testes Adversariais e de Estresse do Milestone 3 (R4 - Infraestrutura Remota em Nuvem / CI/CD).

Valida comportamentos sob falha remota, ausência de artefatos, flags if-no-files-found e if: always(),
integridade de parsing JSON sob variáveis de ambiente headless, e resiliência do script PowerShell.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import pytest
import yaml

WORKFLOW_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".github", "workflows", "data_pipeline.yml")
)
SCRIPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "run_remote_ml.ps1")
)


class TestWorkflowYamlIntegrity:
    """Validação adversarial e estrutural do arquivo .github/workflows/data_pipeline.yml."""

    @pytest.fixture(scope="class")
    @classmethod
    def workflow_data(cls):
        assert os.path.exists(WORKFLOW_PATH), f"Workflow não encontrado em {WORKFLOW_PATH}"
        with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data

    def test_workflow_top_level_keys(self, workflow_data):
        """Garante que as chaves obrigatórias do GitHub Actions estão presentes."""
        assert "name" in workflow_data
        on_clause = workflow_data.get("on") or workflow_data.get(True)
        assert on_clause is not None
        assert "jobs" in workflow_data
        assert "pipeline" in workflow_data["jobs"]

    def test_workflow_dispatch_ml_actions_exhaustiveness(self, workflow_data):
        """Verifica se todas as ações de ML e dados estão explicitadas no input choice."""
        on_clause = workflow_data.get("on") or workflow_data.get(True)
        dispatch = on_clause.get("workflow_dispatch")
        assert dispatch is not None, "workflow_dispatch não configurado"
        inputs = dispatch.get("inputs", {})
        action_input = inputs.get("action", {})
        options = action_input.get("options", [])

        expected_options = [
            "run-all",
            "ml-pipeline",
            "backtest",
            "predict-2026",
            "report",
            "seed",
            "classify",
            "embed",
            "status",
        ]
        for opt in expected_options:
            assert opt in options, f"Ação obrigatória '{opt}' ausente das opções do workflow_dispatch"

    def test_workflow_headless_environment_variables(self, workflow_data):
        """Verifica a configuração estrita do ambiente headless para evitar travamento em X11/buffer."""
        steps = workflow_data["jobs"]["pipeline"]["steps"]
        pipeline_step = next(
            (s for s in steps if "Executar Pipeline" in s.get("name", "")), None
        )
        assert pipeline_step is not None, "Step de execução do pipeline não encontrado"
        env = pipeline_step.get("env", {})

        # Validação das flags headless e sem buffer (R4)
        assert env.get("MPLBACKEND") == "Agg"
        assert str(env.get("PYTHONUNBUFFERED")) == "1"
        assert str(env.get("FORCE_COLOR")) == "1"
        assert env.get("TERM") == "xterm-256color"
        assert env.get("QT_QPA_PLATFORM") == "offscreen"

    def test_upload_artifact_v4_resilience_configuration(self, workflow_data):
        """Verifica a configuração de resiliência do step de upload de artefatos."""
        steps = workflow_data["jobs"]["pipeline"]["steps"]
        upload_step = next(
            (s for s in steps if "actions/upload-artifact@v4" in s.get("uses", "")), None
        )
        assert upload_step is not None, "Step actions/upload-artifact@v4 ausente"

        # 1. Preservação de artefatos mesmo após falhas prévias
        assert upload_step.get("if") == "always()", (
            "Step de upload DEVE ter 'if: always()' para salvar artefatos parciais/diagnósticos"
        )

        with_clause = upload_step.get("with", {})
        # 2. Tolerância a arquivos faltantes (essencial para transição M3 -> M4)
        assert with_clause.get("if-no-files-found") == "warn", (
            "if-no-files-found DEVE ser 'warn' para não falhar o workflow caso o relatório HTML ainda não exista"
        )

        # 3. Retenção conforme governança
        assert with_clause.get("retention-days") == 30

        # 4. Todos os 3 artefatos essenciais listados no path
        path_str = with_clause.get("path", "")
        assert "enem_predictor_report.html" in path_str
        assert "backtest_summary.json" in path_str
        assert "prediction_2026.json" in path_str


class TestJsonPurityAndRedirectionStress:
    """Testes de estresse da pureza e integridade de parsing do JSON gerado pela CLI sob redirecionamento."""

    def test_backtest_json_redirection_purity_under_headless_env(self):
        """Verifica que a saída de `backtest --json-output` redirecionada a arquivo é JSON 100% puro."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "backtest_summary.json")
            env = os.environ.copy()
            env["FORCE_COLOR"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            env["TERM"] = "xterm-256color"
            env["COLUMNS"] = "120"

            with open(out_file, "w", encoding="utf-8") as f:
                res = subprocess.run(
                    [sys.executable, "-m", "enem_pipeline.cli", "backtest", "--json-output"],
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )

            assert res.returncode == 0, f"Falha no comando backtest: {res.stderr}"

            with open(out_file, "r", encoding="utf-8") as f:
                raw_content = f.read()

            # Verificação contra sequências ANSI de escape
            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            assert not ansi_escape.search(raw_content), "Sequências de cor ANSI vazaram no arquivo JSON!"

            # Verificação de parseabilidade estrita
            data = json.loads(raw_content)
            assert data["baseline_top3_rate"] == 0.30
            assert len(data["years_evaluated"]) == 11
            assert len(data["yearly_results"]) == 11
            for yr in data["yearly_results"]:
                assert "year" in yr
                assert "predicted_ranking" in yr
                assert len(yr["predicted_ranking"]) == 10

    def test_predict_2026_json_redirection_purity_with_offline_fallback(self):
        """Verifica que a saída de `predict-2026 --json-output` redirecionada a arquivo é JSON puro sem API key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "prediction_2026.json")
            env = os.environ.copy()
            env.pop("GEMINI_API_KEY", None)  # Garante ausência de chave
            env["FORCE_COLOR"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            env["TERM"] = "xterm-256color"

            with open(out_file, "w", encoding="utf-8") as f:
                res = subprocess.run(
                    [sys.executable, "-m", "enem_pipeline.cli", "predict-2026", "--json-output"],
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                )

            assert res.returncode == 0, f"Falha no comando predict-2026: {res.stderr}"

            with open(out_file, "r", encoding="utf-8") as f:
                raw_content = f.read()

            # Verificação contra sequências ANSI de escape
            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            assert not ansi_escape.search(raw_content), "Sequências de cor ANSI vazaram no arquivo JSON!"

            # Verificação de parseabilidade estrita
            data = json.loads(raw_content)
            assert data["vetoed_axis"] == "Demografia e Envelhecimento"
            assert len(data["ranking"]) == 10
            assert len(data["top3_proposals"]) == 3

            # Eixo de 2025 DEVE ter probabilidade exatamente 0.0
            vetoed_prob = next(p for a, p in data["ranking"] if a == data["vetoed_axis"])
            assert vetoed_prob == 0.0

    def test_report_command_graceful_transition_code_zero(self, tmp_path):
        """Verifica se o comando `report` sai com código 0 e gera o relatório via report_generator."""
        dummy_out = str(tmp_path / "dummy_report.html")
        res = subprocess.run(
            [sys.executable, "-m", "enem_pipeline.cli", "report", "--output", dummy_out],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert "Milestone 4" in res.stdout or "Milestone 4" in res.stderr or "report_generator" in res.stdout


class TestRemoteScriptStressAndFailureScenarios:
    """Testes de estresse do script PowerShell `scripts/run_remote_ml.ps1` com mock da GitHub CLI."""

    def test_script_compiles_in_powershell_ast(self):
        """Garante que a sintaxe do PowerShell é válida e compila perfeitamente."""
        cmd = f"[scriptblock]::Create((Get-Content '{SCRIPT_PATH}' -Raw))"
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"Erro de compilação no PowerShell: {res.stderr}"

    def test_script_parameter_validation_rejects_unsupported_action(self):
        """Verifica se a validação de parâmetros [ValidateSet] intercepta ações inválidas."""
        res = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                SCRIPT_PATH,
                "-Action",
                "acao_invalida_adversarial",
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode != 0
        assert (
            "Cannot validate argument on parameter 'Action'" in res.stderr
            or "ValidateSet" in res.stderr
            or "acao_invalida_adversarial" in res.stderr
        )

    def test_script_handles_remote_workflow_failure_with_log_failed(self):
        """Simula falha remota na nuvem (gh run watch saindo com erro) e verifica auto-diagnóstico."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_gh_bat = os.path.join(tmpdir, "gh.bat")
            mock_script = (
                "@echo off\n"
                "if \"%1\"==\"auth\" exit 0\n"
                "if \"%1\"==\"workflow\" (\n"
                "  type nul > \"%~dp0disparado.flag\"\n"
                "  exit 0\n"
                ")\n"
                "if \"%1\"==\"run\" (\n"
                "  if \"%2\"==\"list\" (\n"
                "    if not exist \"%~dp0disparado.flag\" (\n"
                "      echo 9988776650\n"
                "      exit 0\n"
                "    ) else (\n"
                "      echo 9988776655\n"
                "      exit 0\n"
                "    )\n"
                "  )\n"
                "  if \"%2\"==\"watch\" (\n"
                "    echo [CI/CD] Simulando cancelamento ou erro na nuvem...\n"
                "    exit 1\n"
                "  )\n"
                "  if \"%2\"==\"view\" (\n"
                "    echo [MOCK-LOG-FAILED] Erro capturado via --log-failed: Falha simulada no runner ubuntu-latest\n"
                "    exit 0\n"
                "  )\n"
                ")\n"
                "exit 0\n"
            )
            with open(mock_gh_bat, "w", encoding="utf-8") as f:
                f.write(mock_script)

            env = os.environ.copy()
            env["PATH"] = tmpdir + os.pathsep + env["PATH"]

            res = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    SCRIPT_PATH,
                    "-Action",
                    "ml-pipeline",
                ],
                capture_output=True,
                text=True,
                env=env,
            )

            # O script DEVE terminar com código de erro correspondente ao WATCH_EXIT (1)
            assert res.returncode != 0, f"Script deveria ter falhado sob erro remoto: {res.stdout}"
            # O script DEVE ter acionado a inspeção de diagnóstico (--log-failed)
            assert "[ALERTA DE FALHA]" in res.stdout
            assert "[MOCK-LOG-FAILED]" in res.stdout

    def test_script_handles_missing_artifacts_gracefully(self):
        """Simula workflow com sucesso na nuvem mas ausência de artefatos baixados."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_out = os.path.join(tmpdir, "out")
            os.makedirs(empty_out, exist_ok=True)

            mock_gh_bat = os.path.join(tmpdir, "gh.bat")
            mock_script = (
                "@echo off\n"
                "if \"%1\"==\"auth\" exit 0\n"
                "if \"%1\"==\"workflow\" (\n"
                "  type nul > \"%~dp0disparado.flag\"\n"
                "  exit 0\n"
                ")\n"
                "if \"%1\"==\"run\" (\n"
                "  if \"%2\"==\"list\" (\n"
                "    if not exist \"%~dp0disparado.flag\" (\n"
                "      echo 1122334450\n"
                "      exit 0\n"
                "    ) else (\n"
                "      echo 1122334455\n"
                "      exit 0\n"
                "    )\n"
                "  )\n"
                "  if \"%2\"==\"watch\" exit 0\n"
                "  if \"%2\"==\"download\" exit 1\n"
                ")\n"
                "exit 0\n"
            )
            with open(mock_gh_bat, "w", encoding="utf-8") as f:
                f.write(mock_script)

            env = os.environ.copy()
            env["PATH"] = tmpdir + os.pathsep + env["PATH"]

            res = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    SCRIPT_PATH,
                    "-Action",
                    "ml-pipeline",
                    "-OutputDir",
                    empty_out,
                ],
                capture_output=True,
                text=True,
                env=env,
            )

            assert res.returncode == 0, f"Script falhou inesperadamente: {res.stderr}\n{res.stdout}"
            # Deve relatar de forma limpa como pendente/aviso para cada arquivo ausente
            assert "[PENDENTE/AVISO] Arquivo nao encontrado: enem_predictor_report.html" in res.stdout
            assert "[PENDENTE/AVISO] Arquivo nao encontrado: backtest_summary.json" in res.stdout
            assert "[PENDENTE/AVISO] Arquivo nao encontrado: prediction_2026.json" in res.stdout
