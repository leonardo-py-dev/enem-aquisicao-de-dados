# Project: ENEM Predictor — Validação Estatística Walk-Forward, Predição 2026 e Relatório Comprobatório

## Architecture
O projeto é estruturado em uma arquitetura modular orientada a dados e aprendizado de máquina, com execução híbrida e processamento pesado remoto em nuvem (GitHub Actions + Supabase):
- **Camada de Dados & Taxonomia (`src/enem_pipeline/db/`):**
  - PostgreSQL + pgvector via SQLAlchemy 2.0.
  - Taxonomia de 10 eixos temáticos oficiais e catálogo histórico do ENEM de 1998 a 2025.
- **Camada de Machine Learning & Backtesting (`src/enem_pipeline/ml/`):**
  - Engenharia de atributos históricos e conjunturais (`features.py`).
  - Simulador walk-forward cego ano a ano para 2015–2025 com cálculo de 5 métricas estatísticas e teste binomial exato contra baseline 30% (`backtest.py`).
  - Módulo preditivo para o ENEM 2026 com veto determinístico estrito ao eixo de 2025 (*Demografia e Envelhecimento*) e formulação de intervenções nota 1000 nos 5 elementos da Competência 5 via Gemini (`predictor.py`).
  - Gerador do relatório comprobatório final em HTML interativo autocontido com gráficos procedurais em SVG e interatividade Vanilla JS (`report_generator.py`).
- **Camada de Interface & CLI (`src/enem_pipeline/cli.py`):**
  - Comandos Typer estruturados: `seed`, `run`, `classify`, `embed`, `backtest`, `predict-2026`, `report`.
- **Camada de Infraestrutura em Nuvem (R4) (`.github/workflows/data_pipeline.yml`):**
  - Execução remota pesada no GitHub Actions em runners `ubuntu-latest`.
  - Publicação de artefatos via `actions/upload-artifact@v4` e download local via GitHub CLI `gh`.
- **Trilha de Qualidade & Testes E2E (`tests/e2e/`):**
  - Suíte completa de testes de regressão, integridade matemática, veto determinístico, conformidade dos 5 elementos da Competência 5 e geração do relatório HTML.

## Feature Inventory
| # | Feature | Descrição | Milestone | Source |
|---|---------|-----------|-----------|--------|
| 1 | Simulador Walk-Forward (2015-2025) | Simulação estritamente cega com janela expansiva ano a ano, garantindo zero vazamento temporal ($y < Y$) | M1 | Survey / R1 |
| 2 | Métricas Estatísticas do Modelo | Top-1 Accuracy, Top-3 Hit Rate, MRR, Brier Score multi-classe e Log Loss ano a ano e consolidadas | M1 | Survey / R1 |
| 3 | Comparação com Baseline Aleatório | Cálculo quantitativo do Lift relativo e Teste Binomial Exato contra a linha de base ingênua de 30% | M1 | Survey / R1 |
| 4 | Regra de Veto Consecutivo ENEM 2026 | Post-processing determinístico forçando $P(\text{Demografia e Envelhecimento}) = 0.0\%$ e renormalização dos 9 eixos para 100% | M2 | Survey / R2 |
| 5 | Ranking e Matriz Oficial 2026 | Distribuição ordenada de probabilidades para todos os 10 eixos do ENEM 2026 | M2 | Survey / R2 |
| 6 | Propostas Nota 1000 (Competência 5) | Geração de temas candidatos e propostas com Agente, Ação, Meio/Modo, Efeito e Detalhamento para os 3 eixos líderes | M2 | Survey / R2 |
| 7 | Entrypoints na CLI para ML | Subcomandos `backtest`, `predict-2026` e `report` integrados ao Typer em `src/enem_pipeline/cli.py` | M2 | Survey / R1, R2 |
| 8 | Suporte a ML no GitHub Actions | Input `ml-pipeline` e execução dos módulos de ML no runner `ubuntu-latest` sem uso local de CPU/RAM | M3 | Survey / R4 |
| 9 | Upload de Artefatos no Workflow | Step `actions/upload-artifact@v4` salvando `enem_predictor_report.html` e dados de predição/backtesting | M3 | Survey / R4 |
| 10 | Automação Remota via GitHub CLI | Script/rotina de acionamento (`gh workflow run`), monitoramento (`gh run watch`) e download (`gh run download`) | M3 | Survey / R4 |
| 11 | Relatório HTML Interativo Autocontido | Arquivo HTML único, design executivo moderno, SVG procedural nativo e zero dependências externas quebradiças | M4 | Survey / R3 |
| 12 | Visualizações Interativas em HTML | Scorecards de KPIs, tabela ano a ano com filtros, gráficos SVG temporais, ranking com veto e feature importance | M4 | Survey / R3 |
| 13 | Apresentação das Intervenções no HTML | Exibição destacada dos 3 eixos líderes com detalhamento explícito dos 5 elementos da Competência 5 | M4 | Survey / R3 |
| 14 | Execução Remota E2E e Auditoria Final | Execução de ponta a ponta na nuvem, validação contra a suíte E2E (`TEST_READY.md`) e auditoria forense de integridade | M5 | Survey / R1, R2, R3, R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|--------------|--------|
| M1 | Backtesting Walk-Forward & Estatística | Módulo `src/enem_pipeline/ml/backtest.py` com simulador cego (2015-2025), métricas (Top-1, Top-3, MRR, Brier Score, Log Loss) e teste binomial | none | DONE (92 testes verdes) |
| M2 | Predição Oficial 2026 & Intervenções | Módulo `src/enem_pipeline/ml/predictor.py` com veto estrito a 2025, ranking 2026, geração das propostas nota 1000 nos 5 elementos e CLI | M1 | DONE (114 testes verdes) |
| M3 | Infraestrutura de Execução Remota em Nuvem | Parametrização de `.github/workflows/data_pipeline.yml` com ação `ml-pipeline`, upload de artefatos e orquestração remota via `gh` | M2 | DONE (166 testes verdes) |
| M4 | Relatório Interativo HTML Autocontido | Módulo `src/enem_pipeline/ml/report_generator.py` gerando `enem_predictor_report.html` com gráficos SVG, tabelas e propostas | M2 | DONE (67 testes verdes) |
| M5 | Execução Remota em Nuvem & Validação Final | Execução na nuvem via GitHub Actions (`Run #42432048145`), download do relatório final e artefatos JSON, e validação de integridade | M3, M4 | DONE (100% sucesso em nuvem) |

## Interface Contracts

### `src/enem_pipeline/ml/backtest.py`
```python
@dataclass
class BacktestYearResult:
    year: int
    official_axis: str
    official_theme: str
    predicted_ranking: list[tuple[str, float]]  # [(axis, prob), ...] ordenado decrescente
    rank_of_official: int  # 1-based
    top1_hit: bool
    top3_hit: bool
    reciprocal_rank: float
    brier_score: float
    log_loss: float

@dataclass
class BacktestSummary:
    years_evaluated: list[int]  # [2015, ..., 2025]
    top1_accuracy: float
    top3_hit_rate: float
    mrr: float
    mean_brier_score: float
    mean_log_loss: float
    baseline_top3_rate: float  # 0.30
    relative_lift_top3: float  # (top3_hit_rate - 0.30) / 0.30 * 100
    binomial_test_p_value: float
    yearly_results: list[BacktestYearResult]

class WalkForwardSimulator:
    def __init__(self, historical_data_path: str | None = None): ...
    def run(self, start_year: int = 2015, end_year: int = 2025) -> BacktestSummary: ...
```

### `src/enem_pipeline/ml/predictor.py`
```python
@dataclass
class InterventionProposal:
    axis: str
    candidate_theme: str
    agente: str
    acao: str
    meio_modo: str
    efeito: str
    detalhamento: str

@dataclass
class Prediction2026Result:
    ranking: list[tuple[str, float]]  # 10 eixos ordenados por prob desc, veto de 2025 zerado (0.0%)
    vetoed_axis: str  # "Demografia e Envelhecimento"
    top3_proposals: list[InterventionProposal]

class ENEM2026Predictor:
    def predict(self) -> Prediction2026Result: ...
```

### `src/enem_pipeline/ml/report_generator.py`
```python
class ReportGenerator:
    def generate_html(
        self,
        backtest_summary: BacktestSummary,
        prediction_2026: Prediction2026Result,
        output_path: str = "enem_predictor_report.html"
    ) -> str: ...
```

## Code Layout
```
enem-predictor/
├── .github/workflows/
│   └── data_pipeline.yml         # Workflow de nuvem atualizado com ação ml-pipeline e upload de artefatos
├── src/enem_pipeline/
│   ├── cli.py                    # Entrypoints Typer atualizados (backtest, predict-2026, report)
│   ├── db/
│   │   ├── models.py             # Modelos SQLAlchemy
│   │   └── seed.py               # Taxonomia dos 10 eixos e histórico 1998-2025
│   └── ml/
│       ├── features.py           # Matriz temporal de features existente
│       ├── backtest.py           # Simulador walk-forward e cálculo estatístico (M1)
│       ├── predictor.py          # Predição 2026 com veto e Competência 5 (M2)
│       └── report_generator.py   # Gerador do relatório HTML autocontido (M4)
├── tests/
│   ├── unit/                     # Testes unitários dos módulos de ML
│   └── e2e/                      # Suíte de testes E2E opacos baseados em requisitos
└── enem_predictor_report.html    # Relatório comprobatório final
```
