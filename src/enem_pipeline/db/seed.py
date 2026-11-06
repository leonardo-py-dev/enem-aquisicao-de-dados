from sqlalchemy.orm import Session
from enem_pipeline.db.session import get_db
from enem_pipeline.db.models import Source, Topic, HistoricalTheme
from loguru import logger

INITIAL_SOURCES = [
    {
        "name": "inep",
        "category": "education",
        "base_url": "https://www.gov.br/inep/",
        "access_type": "structured_json",
        "reliability_level": "high",
        "active": True,
    },
    {
        "name": "ibge",
        "category": "statistics",
        "base_url": "https://servicodados.ibge.gov.br/api",
        "access_type": "api_rest",
        "reliability_level": "high",
        "active": True,
    },
    {
        "name": "camara",
        "category": "legislative",
        "base_url": "https://dadosabertos.camara.leg.br/api/v2",
        "access_type": "api_rest",
        "reliability_level": "high",
        "active": True,
    },
    {
        "name": "openalex",
        "category": "academic",
        "base_url": "https://api.openalex.org",
        "access_type": "api_rest",
        "reliability_level": "high",
        "active": True,
    },
    {
        "name": "ipea",
        "category": "statistics",
        "base_url": "http://www.ipeadata.gov.br/api/odata4",
        "access_type": "odata",
        "reliability_level": "high",
        "active": True,
    },
    {
        "name": "gdelt",
        "category": "news",
        "base_url": "https://api.gdeltproject.org/api/v2",
        "access_type": "api_rest",
        "reliability_level": "medium",
        "active": False,
    },
    {
        "name": "google_trends",
        "category": "trends",
        "base_url": "https://trends.google.com.br",
        "access_type": "scraping",
        "reliability_level": "medium",
        "active": False,
    },
]

INITIAL_TOPICS = [
    {"name": "Educação", "thematic_axis": "Educação", "description": "Acesso, qualidade, permanência, formação docente, analfabetismo, evasão escolar e inclusão educacional."},
    {"name": "Saúde", "thematic_axis": "Saúde", "description": "Sistema Único de Saúde (SUS), saúde mental, saneamento básico, pandemias, vacinação e atenção primária."},
    {"name": "Tecnologia", "thematic_axis": "Tecnologia", "description": "Inteligência artificial, inclusão digital, privacidade de dados, algoritmos, desinformação e automação do trabalho."},
    {"name": "Meio Ambiente", "thematic_axis": "Meio Ambiente", "description": "Mudanças climáticas, transição energética, desmatamento, recursos hídricos, sustentabilidade e povos da floresta."},
    {"name": "Cultura", "thematic_axis": "Cultura", "description": "Democratização cultural, patrimônio histórico, memória nacional, herança afro-brasileira e indígena e linguagens artísticas."},
    {"name": "Trabalho e Economia", "thematic_axis": "Trabalho e Economia", "description": "Trabalho digno, informalidade, precarização, trabalho de cuidado, produtividade, inflação e desenvolvimento sustentável."},
    {"name": "Direitos Humanos e Cidadania", "thematic_axis": "Direitos Humanos e Cidadania", "description": "Garantia de direitos fundamentais, combate a preconceitos, registro civil, cidadania plena e liberdade civil."},
    {"name": "Desigualdade Social", "thematic_axis": "Desigualdade Social", "description": "Distribuição de renda, fome, moradia, vulnerabilidade social, concentração de terras e disparidades regionais."},
    {"name": "Segurança Pública", "thematic_axis": "Segurança Pública", "description": "Violência urbana e rural, sistema prisional, segurança no trânsito, combate a facções e políticas preventivas."},
    {"name": "Demografia e Envelhecimento", "thematic_axis": "Demografia e Envelhecimento", "description": "Transição demográfica, previdência social, cuidado com idosos, migrações internas e internacionais e urbanização."},
]

HISTORICAL_ENEM_THEMES = [
    {"year": 1998, "official_theme": "Viver e aprender", "thematic_axis": "Educação", "central_problem": "O papel contínuo da educação e da aprendizagem ao longo da vida para o indivíduo e a coletividade."},
    {"year": 1999, "official_theme": "Cidadania e participação na comunidade", "thematic_axis": "Direitos Humanos e Cidadania", "central_problem": "O exercício ativo da cidadania através do envolvimento comunitário e da solidariedade."},
    {"year": 2000, "official_theme": "Direitos da criança e do adolescente: como enfrentar esse desafio nacional?", "thematic_axis": "Direitos Humanos e Cidadania", "central_problem": "Garantia e cumprimento das diretrizes do Estatuto da Criança e do Adolescente frente à negligência e exploração."},
    {"year": 2001, "official_theme": "Desenvolvimento e preservação ambiental: como conciliar os interesses em jogo?", "thematic_axis": "Meio Ambiente", "central_problem": "O equilíbrio sustentável entre crescimento produtivo econômico e preservação dos biomas e recursos naturais."},
    {"year": 2002, "official_theme": "O direito de votar: como fazer dessa conquista um meio para promover as transformações sociais que o Brasil necessita?", "thematic_axis": "Direitos Humanos e Cidadania", "central_problem": "A conscientização política e o voto como instrumento transformador contra a desigualdade e a corrupção."},
    {"year": 2003, "official_theme": "A violência na sociedade brasileira: como mudar as regras desse jogo?", "thematic_axis": "Segurança Pública", "central_problem": "O avanço da criminalidade urbana e a urgência de políticas de prevenção social e segurança integrada."},
    {"year": 2004, "official_theme": "Como garantir a liberdade de informação e evitar abusos nos meios de comunicação?", "thematic_axis": "Cultura", "central_problem": "O conflito entre liberdade de imprensa, ética e responsabilidade social nos meios de comunicação."},
    {"year": 2005, "official_theme": "O trabalho infantil na realidade brasileira", "thematic_axis": "Trabalho e Economia", "central_problem": "A erradicação da exploração laboral de crianças e adolescentes e a quebra do ciclo da pobreza."},
    {"year": 2006, "official_theme": "O poder de transformação da leitura", "thematic_axis": "Educação", "central_problem": "O baixo índice de hábitos de leitura no Brasil e seu impacto direto na formação crítica e na emancipação cidadã."},
    {"year": 2007, "official_theme": "O desafio de se conviver com a diferença", "thematic_axis": "Direitos Humanos e Cidadania", "central_problem": "A tolerância, o respeito à diversidade sociocultural e a superação de preconceitos estruturais."},
    {"year": 2008, "official_theme": "Como preservar a floresta Amazônica?", "thematic_axis": "Meio Ambiente", "central_problem": "O combate ao desmatamento predatório na Amazônia garantindo alternativas socioeconômicas às populações locais."},
    {"year": 2009, "official_theme": "O indivíduo frente à ética nacional", "thematic_axis": "Direitos Humanos e Cidadania", "central_problem": "A responsabilidade ética de cada cidadão na fiscalização e na integridade das instituições públicas."},
    {"year": 2010, "official_theme": "O trabalho na construção da dignidade humana", "thematic_axis": "Trabalho e Economia", "central_problem": "A valorização do trabalho decente e os impactos da precarização e do desemprego na dignidade individual."},
    {"year": 2011, "official_theme": "Viver em rede no século XXI: os limites entre o público e o privado", "thematic_axis": "Tecnologia", "central_problem": "A exposição excessiva, a invasão de privacidade e os limites da convivência nas redes sociais virtuais."},
    {"year": 2012, "official_theme": "O movimento imigratório para o Brasil no século XXI", "thematic_axis": "Demografia e Envelhecimento", "central_problem": "O acolhimento, inserção laboral e respeito aos direitos humanos de imigrantes e refugiados no país."},
    {"year": 2013, "official_theme": "Efeitos da implantação da Lei Seca no Brasil", "thematic_axis": "Segurança Pública", "central_problem": "A eficácia e conscientização social quanto à redução de mortes no trânsito associadas ao consumo de álcool."},
    {"year": 2014, "official_theme": "Publicidade infantil em questão no Brasil", "thematic_axis": "Cultura", "central_problem": "O apelo comercial desmedido voltado às crianças e os riscos para os hábitos de consumo e saúde infantil."},
    {"year": 2015, "official_theme": "A persistência da violência contra a mulher na sociedade brasileira", "thematic_axis": "Direitos Humanos e Cidadania", "central_problem": "O machismo estrutural e a violência física, psicológica e doméstica contra a mulher apesar da Lei Maria da Penha."},
    {"year": 2016, "official_theme": "Caminhos para combater a intolerância religiosa no Brasil", "thematic_axis": "Direitos Humanos e Cidadania", "central_problem": "A perseguição a religiões minoritárias (especialmente de matriz africana) e o desrespeito ao Estado laico."},
    {"year": 2017, "official_theme": "Desafios para a formação educacional de surdos no Brasil", "thematic_axis": "Educação", "central_problem": "A carência de recursos, intérpretes de Libras e metodologias pedagógicas inclusivas para estudantes com deficiência auditiva."},
    {"year": 2018, "official_theme": "Manipulação do comportamento do usuário pelo controle de dados na internet", "thematic_axis": "Tecnologia", "central_problem": "A formação de bolhas informacionais, algoritmos de direcionamento comportamental e riscos à autonomia do usuário."},
    {"year": 2019, "official_theme": "Democratização do acesso ao cinema no Brasil", "thematic_axis": "Cultura", "central_problem": "A concentração geográfica das salas de exibição e os preços elevados que excluem as classes populares do cinema."},
    {"year": 2020, "official_theme": "O estigma associado às doenças mentais na sociedade brasileira", "thematic_axis": "Saúde", "central_problem": "O preconceito, a desinformação e a escassez de atendimento psicológico e psiquiátrico público adequado."},
    {"year": 2021, "official_theme": "Invisibilidade e registro civil: garantia de acesso à cidadania no Brasil", "thematic_axis": "Direitos Humanos e Cidadania", "central_problem": "Milhares de brasileiros sem certidão de nascimento, impedidos de ter acesso a programas sociais, saúde e educação básica."},
    {"year": 2022, "official_theme": "Desafios para a valorização de comunidades e povos tradicionais no Brasil", "thematic_axis": "Cultura", "central_problem": "A marginalização cultural, conflitos fundiários e falta de reconhecimento institucional de quilombolas, ribeirinhos e povos indígenas."},
    {"year": 2023, "official_theme": "Desafios para o enfrentamento da invisibilidade do trabalho de cuidado realizado pela mulher no Brasil", "thematic_axis": "Trabalho e Economia", "central_problem": "A sobrecarga não remunerada e não reconhecida das mulheres na assistência de dependentes e tarefas domésticas."},
    {"year": 2024, "official_theme": "Desafios para a valorização da herança africana no Brasil", "thematic_axis": "Cultura", "central_problem": "O apagamento histórico e o racismo estrutural que desvalorizam a contribuição fundamental dos povos africanos para a identidade nacional."},
]


def seed_database(db: Session = None):
    """Popula o banco com os dados iniciais de fontes, tópicos e histórico do ENEM."""
    if db is None:
        with get_db() as session:
            _run_seed(session)
    else:
        _run_seed(db)


def _run_seed(db: Session):
    try:
        # 1. Fontes
        for s_data in INITIAL_SOURCES:
            existing = db.query(Source).filter_by(name=s_data["name"]).first()
            if not existing:
                db.add(Source(**s_data))
                logger.info(f"Fonte cadastrada: {s_data['name']}")

        # 2. Tópicos
        for t_data in INITIAL_TOPICS:
            existing = db.query(Topic).filter_by(name=t_data["name"]).first()
            if not existing:
                db.add(Topic(**t_data))
                logger.info(f"Tópico cadastrado: {t_data['name']}")

        db.flush()

        inep_source = db.query(Source).filter_by(name="inep").first()
        inep_id = inep_source.id if inep_source else None

        # 3. Histórico do ENEM
        for h_data in HISTORICAL_ENEM_THEMES:
            existing = db.query(HistoricalTheme).filter_by(year=h_data["year"]).first()
            if not existing:
                db.add(HistoricalTheme(
                    year=h_data["year"],
                    official_theme=h_data["official_theme"],
                    thematic_axis=h_data["thematic_axis"],
                    central_problem=h_data["central_problem"],
                    source_id=inep_id,
                    source_url="https://www.gov.br/inep/pt-br/areas-de-atuacao/avaliacao-e-exames-educacionais/enem",
                ))
                logger.info(f"Tema histórico cadastrado: {h_data['year']} - {h_data['official_theme']}")

        db.commit()
        logger.success("Seed do banco de dados concluído com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao popular banco de dados: {e}")
        raise


if __name__ == "__main__":
    seed_database()
