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
from loguru import logger

from enem_pipeline.db.session import init_db, get_db
from enem_pipeline.db.seed import seed_database
from enem_pipeline.db.models import Source, CollectionRun, Document, Indicator, Topic, DocumentTopic, HistoricalTheme
from enem_pipeline.collectors import InepCollector, IbgeCollector, CamaraCollector, OpenAlexCollector, COLLECTORS_REGISTRY
from enem_pipeline.processing.classifier import ThematicClassifier
from enem_pipeline.scheduler import start_scheduler

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
def run_all(
    limit: int = typer.Option(25, "--limit", "-l", help="Limite de itens por coletor"),
):
    """Executa todos os coletores prioritários e classifica os documentos ingeridos."""
    init_db()
    console.print("[bold cyan]==================================================[/bold cyan]")
    console.print("[bold cyan] INICIANDO PIPELINE COMPLETO DO ENEM PREDICTOR    [/bold cyan]")
    console.print("[bold cyan]==================================================[/bold cyan]")

    with get_db() as db:
        seed_database(db)

        # 1. INEP
        console.print("\n[bold yellow]1/4. Executando coletor INEP...[/bold yellow]")
        InepCollector().collect(db, max_items=limit)

        # 2. IBGE
        console.print("\n[bold yellow]2/4. Executando coletor IBGE...[/bold yellow]")
        IbgeCollector().collect(db, max_items=limit)

        # 3. Câmara dos Deputados
        console.print("\n[bold yellow]3/4. Executando coletor Câmara dos Deputados...[/bold yellow]")
        CamaraCollector().collect(db, max_items=limit)

        # 4. OpenAlex
        console.print("\n[bold yellow]4/4. Executando coletor OpenAlex...[/bold yellow]")
        OpenAlexCollector().collect(db, max_items=limit)

        # 5. Classificação Temática
        console.print("\n[bold yellow]5/5. Processando e classificando documentos...[/bold yellow]")
        count = ThematicClassifier().process_and_classify_all(db, limit=limit * 4)

    console.print("\n[bold green][OK] PIPELINE CONCLUIDO COM SUCESSO TOTAL![/bold green]\n")
    status()


@app.command()
def status():
    """Exibe um painel analítico com as estatísticas de armazenamento e qualidade do pipeline."""
    init_db()
    with get_db() as db:
        sources_count = db.query(Source).count()
        runs_count = db.query(CollectionRun).count()
        docs_count = db.query(Document).count()
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
                status_color = "green" if r.status == "success" else "red"
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


if __name__ == "__main__":
    app()
