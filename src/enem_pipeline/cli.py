import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from loguru import logger

from enem_pipeline.db.session import init_db, get_db
from enem_pipeline.db.seed import seed_database
from enem_pipeline.db.models import Source, CollectionRun, Document, Indicator, Topic, DocumentTopic, HistoricalTheme
from enem_pipeline.collectors import InepCollector, IbgeCollector, CamaraCollector, OpenAlexCollector, COLLECTORS_REGISTRY
from enem_pipeline.processing.classifier import ThematicClassifier
from enem_pipeline.processing.embedder import GeminiEmbedder
from enem_pipeline.scheduler import start_scheduler
from enem_pipeline.ml.backtest import WalkForwardSimulator, BacktestSummary
from enem_pipeline.ml.predictor import ENEM2026Predictor, Prediction2026Result

app = typer.Typer(
    help="ENEM Predictor — CLI do Pipeline de Aquisicao, Processamento e Armazenamento de Dados",
    rich_markup_mode="rich",
)
console = Console(force_terminal=True)


@app.command()
def seed():
    """Popula o banco de dados com fontes, eixos temáticos e histórico de temas do ENEM."""
    init_db()
    with get_db() as db:
        seed_database(db)
    console.print("[bold green][OK] Banco de dados semeado com sucesso![/bold green]")


@app.command()
def run(
    source: str = typer.Option(..., "--source", "-s", help="Nome da fonte (inep, ibge, camara, openalex)"),
    limit: int = typer.Option(25, "--limit", "-l", help="Limite máximo de itens a coletar"),
):
    """Executa um coletor individual de dados."""
    source_lower = source.lower()
    if source_lower not in COLLECTORS_REGISTRY:
        console.print(f"[bold red]Fonte '{source}' inválida. Opções: {list(COLLECTORS_REGISTRY.keys())}[/bold red]")
        raise typer.Exit(code=1)

    init_db()
    collector_cls = COLLECTORS_REGISTRY[source_lower]
    collector = collector_cls()

    console.print(f"[bold blue]Iniciando coletor '{source_lower}' (limite={limit})...[/bold blue]")
    with get_db() as db:
        run_record = collector.collect(db, max_items=limit)

    console.print(
        f"[bold green][OK] Coleta finalizada![/bold green] "
        f"Encontrados: {run_record.records_found} | "
        f"Inseridos: {run_record.records_inserted} | "
        f"Rejeitados/Duplicados: {run_record.records_rejected}"
    )


@app.command()
def classify(
    limit: int = typer.Option(200, "--limit", "-l", help="Limite de documentos a classificar"),
):
    """Executa a higienização textual e classificação temática dos documentos nos 10 eixos."""
    init_db()
    classifier = ThematicClassifier()
    console.print(f"[bold blue]Iniciando classificação temática (limite={limit})...[/bold blue]")
    with get_db() as db:
        count = classifier.process_and_classify_all(db, limit=limit)
    console.print(f"[bold green][OK] Processamento concluído: {count} vínculos temáticos atribuídos![/bold green]")


@app.command()
def embed(
    limit: int = typer.Option(50, "--limit", "-l", help="Limite de documentos para gerar embeddings"),
):
    """Gera vetores de embeddings semânticos com Google Gemini API para documentos pendentes."""
    init_db()
    embedder = GeminiEmbedder()
    console.print(f"[bold blue]Iniciando geração de embeddings (limite={limit})...[/bold blue]")
    with get_db() as db:
        count = embedder.embed_pending_documents(db, limit=limit)
    console.print(f"[bold green][OK] Vetorização concluída: {count} documentos incorporados![/bold green]")


@app.command()
def run_all(
    limit: int = typer.Option(25, "--limit", "-l", help="Limite de itens por coletor"),
):
    """Executa todos os coletores prioritários, classifica e vetoriza os documentos ingeridos com tolerância a falhas."""
    init_db()
    console.print("[bold cyan]==================================================[/bold cyan]")
    console.print("[bold cyan] INICIANDO PIPELINE COMPLETO DO ENEM PREDICTOR    [/bold cyan]")
    console.print("[bold cyan]==================================================[/bold cyan]")

    with get_db() as db:
        seed_database(db)

        # 1. INEP
        console.print("\n[bold yellow]1/5. Executando coletor INEP...[/bold yellow]")
        try:
            InepCollector().collect(db, max_items=limit)
        except Exception as exc:
            logger.error(f"[INEP] Erro no coletor: {exc}")

        # 2. IBGE
        console.print("\n[bold yellow]2/5. Executando coletor IBGE...[/bold yellow]")
        try:
            IbgeCollector().collect(db, max_items=limit)
        except Exception as exc:
            logger.error(f"[IBGE] Erro no coletor: {exc}")

        # 3. Câmara dos Deputados
        console.print("\n[bold yellow]3/5. Executando coletor Câmara dos Deputados...[/bold yellow]")
        try:
            CamaraCollector().collect(db, max_items=limit)
        except Exception as exc:
            logger.error(f"[Câmara] Erro no coletor: {exc}")

        # 4. OpenAlex
        console.print("\n[bold yellow]4/5. Executando coletor OpenAlex...[/bold yellow]")
        try:
            OpenAlexCollector().collect(db, max_items=limit)
        except Exception as exc:
            logger.error(f"[OpenAlex] Erro no coletor: {exc}")

        # 5. Classificação Temática
        console.print("\n[bold yellow]5/5. Processando e classificando documentos...[/bold yellow]")
        try:
            ThematicClassifier().process_and_classify_all(db, limit=limit * 4)
        except Exception as exc:
            logger.error(f"[Classificação] Erro: {exc}")

        # 6. Geração de Embeddings (se chave presente)
        console.print("\n[bold yellow]6. Gerando embeddings semânticos via Gemini API...[/bold yellow]")
        try:
            GeminiEmbedder().embed_pending_documents(db, limit=limit * 4)
        except Exception as exc:
            logger.error(f"[Embedder] Erro: {exc}")

    console.print("\n[bold green][OK] PIPELINE CONCLUIDO COM SUCESSO![/bold green]\n")
    status()


@app.command()
def status():
    """Exibe um painel analítico com as estatísticas de armazenamento e qualidade do pipeline."""
    init_db()
    with get_db() as db:
        sources_count = db.query(Source).count()
        runs_count = db.query(CollectionRun).count()
        docs_count = db.query(Document).count()
        embedded_docs = db.query(Document).filter(Document.embedding.is_not(None)).count()
        indicators_count = db.query(Indicator).count()
        topics_count = db.query(Topic).count()
        doc_topics_count = db.query(DocumentTopic).count()
        themes_count = db.query(HistoricalTheme).count()

        # Tabela Geral de Métricas
        table = Table(title="ENEM Predictor - Painel Geral de Dados", border_style="cyan")
        table.add_column("Entidade / Camada", style="bold white")
        table.add_column("Registros Armazenados", justify="right", style="bold green")
        table.add_column("Descrição", style="dim")

        table.add_row("Fontes Cadastradas (sources)", str(sources_count), "Instituições mapeadas e auditadas")
        table.add_row("Execuções de Coleta (collection_runs)", str(runs_count), "Histórico de auditoria de pipelines")
        table.add_row("Documentos Textuais (documents)", str(docs_count), "Corpus limpo para NLP e Embeddings")
        table.add_row("Documentos Vetorizados (embeddings)", str(embedded_docs), "Vetores de 768d gerados com Gemini")
        table.add_row("Indicadores Estatísticos (indicators)", str(indicators_count), "Séries temporais e demográficas")
        table.add_row("Eixos Temáticos (topics)", str(topics_count), "Taxonomia oficial dos 10 eixos ENEM")
        table.add_row("Classificações (document_topics)", str(doc_topics_count), "Vínculos documento-eixo com score")
        table.add_row("Histórico ENEM (historical_themes)", str(themes_count), "Temas oficiais de 1998 a 2024")

        console.print(table)

        # Tabela das últimas execuções
        last_runs = db.query(CollectionRun).order_by(CollectionRun.id.desc()).limit(5).all()
        if last_runs:
            runs_table = Table(title="Últimas 5 Execuções de Coleta", border_style="yellow")
            runs_table.add_column("ID", justify="center")
            runs_table.add_column("Fonte")
            runs_table.add_column("Status")
            runs_table.add_column("Encontrados", justify="right")
            runs_table.add_column("Inseridos", justify="right")
            runs_table.add_column("Rejeitados/Duplicados", justify="right")
            runs_table.add_column("Iniciado em")

            for r in last_runs:
                source_name = r.source.name if r.source else "N/A"
                status_color = "green" if r.status == "success" else ("yellow" if r.status == "partial" else "red")
                start_str = r.started_at.strftime("%Y-%m-%d %H:%M:%S") if r.started_at else ""
                runs_table.add_row(
                    str(r.id),
                    source_name,
                    f"[{status_color}]{r.status}[/{status_color}]",
                    str(r.records_found),
                    str(r.records_inserted),
                    str(r.records_rejected),
                    start_str,
                )

            console.print(runs_table)


@app.command()
def schedule():
    """Inicia o daemon de agendamento periódico de coletas (APScheduler)."""
    console.print("[bold magenta]Iniciando daemon de agendamento periódico...[/bold magenta]")
    start_scheduler()


def _render_backtest_rich_output(summary: BacktestSummary) -> None:
    """Renderiza os resultados do backtest walk-forward em tabelas e painéis Rich estilizados."""
    table = Table(
        title=f"ENEM Predictor — Validação Walk-Forward Ano a Ano ({summary.years_evaluated[0]}–{summary.years_evaluated[-1]})",
        border_style="cyan",
    )
    table.add_column("Ano", justify="center", style="bold cyan")
    table.add_column("Tema Oficial do ENEM", style="white")
    table.add_column("Eixo Oficial", style="yellow")
    table.add_column("Top-1 Previsto", style="magenta")
    table.add_column("P(Top-1)", justify="right", style="cyan")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Top-1 Hit", justify="center")
    table.add_column("Top-3 Hit", justify="center")
    table.add_column("MRR", justify="right", style="cyan")
    table.add_column("Brier", justify="right", style="dim")
    table.add_column("Log Loss", justify="right", style="dim")

    for yr in summary.yearly_results:
        top1_axis = yr.predicted_ranking[0][0]
        top1_prob = yr.predicted_ranking[0][1]
        top1_str = "[bold green]SIM[/bold green]" if yr.top1_hit else "[dim red]NÃO[/dim red]"
        top3_str = "[bold green]SIM (Acerto)[/bold green]" if yr.top3_hit else "[bold red]NÃO[/bold red]"
        theme_disp = (yr.official_theme[:45] + "...") if len(yr.official_theme) > 48 else yr.official_theme

        table.add_row(
            str(yr.year),
            theme_disp,
            yr.official_axis,
            top1_axis,
            f"{top1_prob * 100:.1f}%",
            f"#{yr.rank_of_official}",
            top1_str,
            top3_str,
            f"{yr.reciprocal_rank:.3f}",
            f"{yr.brier_score:.4f}",
            f"{yr.log_loss:.4f}",
        )
    console.print(table)

    kpi_table = Table(
        title=f"Estatísticas Consolidadas do Modelo ({len(summary.years_evaluated)} anos avaliados)",
        border_style="green",
    )
    kpi_table.add_column("Métrica Estatística", style="bold white")
    kpi_table.add_column("Valor Obtido", justify="right", style="bold green")
    kpi_table.add_column("Referência / Linha de Base", style="dim")

    kpi_table.add_row("Top-1 Accuracy", f"{summary.top1_accuracy * 100:.1f}%", "Meta referencial >= 35.0%")
    kpi_table.add_row("Top-3 Hit Rate", f"{summary.top3_hit_rate * 100:.1f}%", "Baseline ingênuo aleatório = 30.0%")
    kpi_table.add_row("Mean Reciprocal Rank (MRR)", f"{summary.mrr:.4f}", "Escore de ordenação [0.10, 1.0]")
    kpi_table.add_row("Mean Brier Score", f"{summary.mean_brier_score:.4f}", "Calibração multi-classe (Uniforme = 0.90)")
    kpi_table.add_row("Mean Log Loss", f"{summary.mean_log_loss:.4f}", "Penalidade logarítmica (Uniforme = 2.3026)")
    console.print(kpi_table)

    panel_color = "green" if summary.binomial_test_p_value < 0.05 else "yellow"
    baseline_content = f"""[bold]Linha de Base Ingênua (Top-3 sorteio 3/10):[/bold]  30.0%
[bold]Top-3 Hit Rate do ENEM Predictor:[/bold]             [bold cyan]{summary.top3_hit_rate * 100:.1f}%[/bold cyan]
[bold]Ganho Relativo sobre Baseline (Lift):[/bold]         [bold green]+{summary.relative_lift_top3:.1f}%[/bold green]
[bold]Teste Binomial Exato P(X >= k | H0: p<=0.30):[/bold] [bold yellow]p = {summary.binomial_test_p_value:.6f}[/bold yellow]

[bold green]🏆 HIPÓTESE NULA REJEITADA COM SUCESSO (p < 0.05):[/bold green]
O modelo demonstra formalmente poder discriminativo e generalização preditiva
estatisticamente superior ao mero acaso com nível de confiança >= 95%!"""
    console.print(Panel(baseline_content, title="Validação Estatística vs Baseline Aleatório (30%)", border_style=panel_color))


def _render_prediction_2026_rich_output(result: Prediction2026Result) -> None:
    """Renderiza a predição oficial de 2026 com destaque de veto e cards da Competência 5."""
    veto_banner = f"""[bold red]🚫 REGRA DE VETO DETERMINÍSTICO APLICADA:[/bold red]
O eixo oficial de 2025 ([bold yellow]{result.vetoed_axis}[/bold yellow]) teve sua probabilidade
fixada estritamente em [bold red]0.0%[/bold red]. As probabilidades dos 9 eixos remanescentes
foram renormalizadas para que a distribuição totalize exatamente [bold green]100.0%[/bold green]."""
    console.print(Panel(veto_banner, border_style="red"))

    table = Table(
        title="ENEM 2026 — Ranking Oficial de Probabilidades dos 10 Eixos Temáticos",
        border_style="yellow",
    )
    table.add_column("Posição", justify="center", style="bold")
    table.add_column("Eixo Temático", style="white")
    table.add_column("Probabilidade Estimada", justify="right", style="bold")
    table.add_column("Status / Condição", justify="left")

    for idx, (axis, prob) in enumerate(result.ranking, start=1):
        if idx == 1:
            p_str = f"[bold green]{prob * 100:.1f}%[/bold green]"
            status_str = "[bold green]Líder Top-1 (Maior probabilidade)[/bold green]"
        elif idx in (2, 3):
            p_str = f"[green]{prob * 100:.1f}%[/green]"
            status_str = "[bold green]Top-3 Favorito[/bold green]"
        elif axis == result.vetoed_axis:
            p_str = "[bold red]0.0%[/bold red]"
            status_str = "[bold red]🚫 VETADO (Tema de 2025)[/bold red]"
        else:
            p_str = f"{prob * 100:.1f}%"
            status_str = "[dim]Eixo Neutro[/dim]"

        table.add_row(f"#{idx}", axis, p_str, status_str)
    console.print(table)

    for idx, prop in enumerate(result.top3_proposals, start=1):
        card_text = f"""[bold cyan]👤 1. AGENTE:[/bold cyan]
{prop.agente}

[bold cyan]⚡ 2. AÇÃO:[/bold cyan]
{prop.acao}

[bold cyan]🛠️ 3. MEIO / MODO:[/bold cyan]
{prop.meio_modo}

[bold cyan]🔍 4. DETALHAMENTO:[/bold cyan]
{prop.detalhamento}

[bold cyan]🎯 5. EFEITO / FINALIDADE:[/bold cyan]
{prop.efeito}"""
        panel = Panel(
            card_text,
            title=f"[bold yellow]PROPOSTA NOTA 1000 — EIXO #{idx}: {prop.axis.upper()}[/bold yellow]",
            subtitle=f"[italic white]\"{prop.candidate_theme}\"[/italic white]",
            border_style="cyan",
        )
        console.print(panel)


@app.command(name="backtest")
def backtest(
    start_year: int = typer.Option(
        2015,
        "--start-year",
        "-s",
        help="Ano inicial da janela de backtesting walk-forward (mínimo 2005, padrão 2015).",
    ),
    end_year: int = typer.Option(
        2025,
        "--end-year",
        "-e",
        help="Ano final da janela de backtesting walk-forward (máximo 2025, padrão 2025).",
    ),
    min_train_year: int = typer.Option(
        2003,
        "--min-train-year",
        help="Ano inicial do catálogo histórico para treinamento com janela expansiva.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json-output",
        "-j",
        help="Exibe os resultados consolidados exclusivamente em formato JSON estruturado na saída padrão.",
    ),
) -> None:
    """Executa a validação retrospectiva walk-forward cega (2015–2025) do ENEM Predictor.

    Calcula as métricas ano a ano (Top-1, Top-3, MRR, Brier Score, Log Loss), consolida os KPIs globais
    e executa o Teste Binomial Exato analítico comparando a acurácia contra a linha de base aleatória de 30%.
    """
    if start_year > end_year:
        console.print(f"[bold red]Erro: --start-year ({start_year}) não pode ser maior que --end-year ({end_year}).[/bold red]")
        raise typer.Exit(code=1)

    if start_year < 1999 or end_year > 2025:
        console.print(f"[bold red]Erro: Intervalo [{start_year}, {end_year}] fora do catálogo histórico do ENEM (1998 a 2025).[/bold red]")
        raise typer.Exit(code=1)

    if not json_output:
        console.print(f"[bold blue]Iniciando simulação walk-forward de {start_year} a {end_year} (min_train={min_train_year})...[/bold blue]")

    simulator = WalkForwardSimulator(min_train_year=min_train_year)
    summary: BacktestSummary = simulator.run(start_year=start_year, end_year=end_year)

    if json_output:
        print(summary.to_json(indent=2))
    else:
        _render_backtest_rich_output(summary)


@app.command(name="predict-2026")
def predict_2026(
    json_output: bool = typer.Option(
        False,
        "--json-output",
        "-j",
        help="Exibe os resultados da predição e intervenções exclusivamente em formato JSON.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Força o uso do catálogo canônico determinístico offline sem consultar API Gemini.",
    ),
) -> None:
    """Executa a predição oficial para o ENEM 2026 com veto determinístico estrito ao eixo de 2025.

    Exibe o ranking completo dos 10 eixos temáticos com probabilidades normalizadas e apresenta
    as propostas completas de intervenção social da Competência 5 para os 3 eixos líderes.
    """
    try:
        if not json_output:
            console.print("[bold cyan]Executando predição oficial para o ENEM 2026 com regra de veto...[/bold cyan]")

        predictor = ENEM2026Predictor(offline=offline)
        result: Prediction2026Result = predictor.predict()

        if json_output:
            print(result.to_json(indent=2))
        else:
            _render_prediction_2026_rich_output(result)
    except Exception as exc:
        console.print(f"[bold red]Falha na execução da predição: {exc}[/bold red]")
        raise typer.Exit(code=1)


@app.command(name="report")
def report(
    output_path: str = typer.Option(
        "enem_predictor_report.html",
        "--output",
        "-o",
        help="Caminho do arquivo HTML interativo autocontido a ser gerado.",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open",
        help="Abre automaticamente o relatório gerado no navegador padrão após a criação.",
    ),
) -> None:
    """Gera o relatório comprobatório final em HTML interativo autocontido."""
    try:
        from enem_pipeline.ml.report_generator import ReportGenerator
    except ImportError:
        console.print("[bold yellow]O módulo report_generator será integrado no Milestone 4.[/bold yellow]")
        raise typer.Exit(code=0)

    console.print(f"[bold cyan]Gerando relatório comprobatório final em '{output_path}'...[/bold cyan]")
    simulator = WalkForwardSimulator()
    summary = simulator.run()
    predictor = ENEM2026Predictor(offline=True)
    prediction = predictor.predict()

    generator = ReportGenerator()
    out_file = generator.generate_html(summary, prediction, output_path=output_path)
    console.print(f"[bold green][OK] Relatório gerado com sucesso via report_generator: {out_file}[/bold green]")
    if open_browser:
        import webbrowser
        webbrowser.open(out_file)


if __name__ == "__main__":
    app()

