# Guia Operacional do Pipeline — ENEM Predictor

Este documento descreve como operar, executar, auditar e agendar os fluxos de aquisição e processamento de dados do pipeline.

---

## 1. Comandos Básicos da CLI (`enem-pipeline`)

A ferramenta de linha de comando (`enem-pipeline`) oferece uma interface intuitiva construída com Typer e Rich.

### 1.1. Inicialização e Semeio do Banco (`seed`)
Popula as fontes oficiais, a taxonomia dos 10 eixos do ENEM e o histórico de temas de redação de 1998 a 2024:
```bash
python -m enem_pipeline.cli seed
```

### 1.2. Execução de Coletores Individuais (`run`)
Permite disparar um coletor isolado com controle do volume máximo de registros:
```bash
# Executa apenas o coletor do INEP
python -m enem_pipeline.cli run --source inep --limit 10

# Executa apenas o coletor de notícias e indicadores do IBGE
python -m enem_pipeline.cli run --source ibge --limit 20

# Executa apenas o coletor de proposições legislativas da Câmara dos Deputados
python -m enem_pipeline.cli run --source camara --limit 30

# Executa apenas o coletor de pesquisas acadêmicas do OpenAlex
python -m enem_pipeline.cli run --source openalex --limit 25
```

### 1.3. Execução Integrada Completa (`run-all`)
Executa sequencialmente todos os 4 coletores prioritários, persiste os dados na camada RAW e banco, normaliza os textos e aplica a classificação temática automática nos 10 eixos:
```bash
python -m enem_pipeline.cli run-all --limit 25
```

### 1.4. Higienização e Classificação Temática Avulsa (`classify`)
Processa o texto de documentos pendentes, remove tags, normaliza acentos e recalcula a pertinência temática:
```bash
python -m enem_pipeline.cli classify --limit 200
```

### 1.5. Visualização do Painel de Status (`status`)
Exibe o volume consolidado de documentos, fontes, indicadores, classificações e as últimas execuções de coleta com seus respectivos status e contadores de deduplicação:
```bash
python -m enem_pipeline.cli status
```

---

## 2. Automação e Agendamento em Segundo Plano (`schedule`)

Para executar o pipeline em modo contínuo (daemon), utilize o agendador integrado (APScheduler):
```bash
python -m enem_pipeline.cli schedule
```

### Grade de Agendamento Padrão:
- **Notícias do IBGE e Câmara dos Deputados:** A cada 6 horas.
- **Produção Científica (OpenAlex):** A cada 24 horas (diário).
- **Classificação Temática Automática:** Disparada imediatamente após cada ciclo de ingestão.

---

## 3. Monitoramento e Recuperação de Falhas

### 3.1. Rastreamento por Execução (`collection_runs`)
Cada execução recebe um ID único e registra:
- `started_at` e `finished_at`.
- `status`: `running`, `success`, `failed` ou `partial`.
- `records_found`, `records_inserted`, `records_updated`, `records_rejected`.
- `error_details`: Stacktrace completo em caso de falha.

### 3.2. Recuperação de Falha de Conexão ou Rate Limit
- Se uma API externa retornar HTTP 429 ou falha de timeout, o coletor aplica **retries automáticos com backoff exponencial** (1s, 2s, 4s).
- Se a API falhar completamente após todas as tentativas, a execução é marcada como `failed` no banco e o erro é logado sem abortar os demais coletores do lote.
- Ao reexecutar o pipeline, a **deduplicação por SHA-256** garante que nenhum documento previamente salvo seja duplicado.

---

## 4. Testes Automatizados

Para rodar a suíte completa de testes unitários e de integração:
```bash
python -m pytest -v
```
