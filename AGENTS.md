# Diretrizes para Agentes de IA — ENEM Predictor

Este documento orienta os agentes autônomos do Google Antigravity no desenvolvimento, teste, monitoramento de CI/CD e revisão de código neste repositório.

---

## 1. Contexto do Projeto
- **Nome**: `enem-predictor-pipeline` (`enem-aquisicao-de-dados`)
- **Objetivo**: Pipeline de engenharia de dados para aquisição, processamento e classificação temática de dados do ENEM.
- **Stack**: Python 3.12, SQLAlchemy 2.0, Alembic, PostgreSQL + pgvector (Supabase), Typer, Loguru, Pytest.
- **CLI Principal**: `python -m enem_pipeline.cli` (ou `enem-pipeline`).

---

## 2. Monitoramento de CI/CD e GitHub Actions

Os agentes têm permissão para monitorar e auditar as execuções do GitHub Actions via `gh`:

### Comandos de Monitoramento
- **Listar execuções recentes**:
  ```powershell
  gh run list --workflow="data_pipeline.yml" --limit 5
  ```
- **Inspecionar falhas**:
  ```powershell
  gh run view <run-id> --log-failed
  ```
- **Acompanhar execução ao vivo**:
  ```powershell
  gh run watch <run-id>
  ```
- **Disparar workflow manualmente (quando solicitado)**:
  ```powershell
  gh workflow run data_pipeline.yml -f action=status
  ```

### Procedimento de Auto-Remediação em Falhas de CI
1. Ao detectar uma falha (`conclusion: failure`), execute `gh run view <run-id> --log-failed` para extrair o stack trace exato.
2. Identifique se o erro é decorrente de:
   - **Ambiente/Secret**: Falta de variáveis de ambiente (`DATABASE_URL`, `SUPABASE_*`, etc.).
   - **Dependência/Build**: Incompatibilidade em `requirements.txt` ou `pyproject.toml`.
   - **Código/Lógica**: Falha em testes de integração ou no script `enem_pipeline.cli`.
3. Reproduza o erro localmente executando os testes unitários (`pytest`).
4. Crie uma branch de correção (`fix/...`), realize o commit explicativo e abra ou atualize o Pull Request.

---

## 3. Gestão e Revisão de Pull Requests (PRs)

- **Listar PRs abertos**:
  ```powershell
  gh pr list --state open
  ```
- **Verificar status de checks de um PR**:
  ```powershell
  gh pr checks <pr-number>
  ```
- **Analisar diff de código**:
  ```powershell
  gh pr diff <pr-number>
  ```
- **Adicionar revisão ou comentário**:
  ```powershell
  gh pr comment <pr-number> --body "Resumo da análise..."
  ```

---

## 4. Testes e Validação Local Pré-Commit

Antes de commitar qualquer alteração ou marcar uma tarefa como concluída:
1. Rodar os testes unitários:
   ```powershell
   pytest
   ```
2. Garantir que as migrações e o schema do banco não quebrem a tipagem do Pydantic / SQLAlchemy.
3. Não versionar arquivos `.env` ou dados sensíveis.

---

## 5. Autonomia e Decisões
- O agente pode ler e editar arquivos do projeto e rodar testes de forma autônoma.
- Sempre documentar o **porquê** de cada decisão técnica nos commits e mensagens ao usuário.
