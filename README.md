# ENEM Predictor — Pipeline de Aquisição e Armazenamento de Dados

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-blue.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-supported-green.svg)](https://github.com/pgvector/pgvector)
[![Supabase](https://img.shields.io/badge/Supabase-Integrated-3ECF8E.svg)](https://supabase.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Sistema de engenharia de dados em Python para coleta, validação, auditoria, processamento, organização e armazenamento de dados abertos para o projeto **ENEM Predictor** — cujo objetivo é prever potenciais eixos temáticos e propostas de intervenção para a prova de redação do ENEM.

---

## 1. Visão Geral do Sistema

O pipeline reúne informações multidisciplinares provenientes de fontes oficiais brasileiras e globais:
1. **Histórico Oficial do ENEM:** Provas anteriores e temas de redação (1998–2024) catalogados pelo INEP.
2. **Dados Sociais e Demográficos:** Notícias, pesquisas e indicadores de desigualdade do IBGE.
3. **Debates Legislativos:** Projetos de Lei (PL) e Propostas de Emenda à Constituição (PEC) da Câmara dos Deputados.
4. **Pesquisas Científicas:** Produção acadêmica brasileira catalogada na API aberta do OpenAlex.
5. **Classificação Temática:** Motor léxico ponderado nos 10 eixos oficiais do ENEM com scoring de relevância.

---

## 2. Tecnologias Utilizadas

- **Linguagem:** Python 3.13+ / 3.11+
- **Banco de Dados:** PostgreSQL 17 gerenciado no **Supabase** (com extensão nativa `pgvector` habilitada) + fallback local transparente em SQLite (`enem_pipeline.db`).
- **ORM & Migrations:** SQLAlchemy 2.0 e Alembic.
- **Camada RAW (Data Lake):** Armazenamento imutável de payloads brutos em disco particionado por data (`data/raw/...`).
- **Validação & Configurações:** Pydantic v2 e Pydantic-Settings.
- **Requisições HTTP:** HTTPX com suporte nativo a certificados Windows via `truststore`.
- **Interface CLI:** Typer e Rich para formatação visual de tabelas e logs.
- **Agendamento:** APScheduler para orquestração periódica contínua.
- **Testes:** Pytest com mocks e fixtures de banco em memória.

---

## 3. Estrutura do Repositório

```text
enem-predictor-pipeline/
├── data/
│   ├── raw/                 # Payloads brutos imutáveis por fonte/data
│   │   ├── inep/
│   │   ├── ibge/
│   │   ├── camara/
│   │   └── openalex/
│   └── processed/           # Snapshots processados
├── docs/                    # Documentação técnica detalhada
│   ├── ARCHITECTURE.md      # Desenho de arquitetura e decisões (ADRs)
│   ├── DATA_SOURCES.md      # Auditoria e matriz de viabilidade das 7 fontes
│   ├── DATA_DICTIONARY.md   # Dicionário de dados e modelo relacional (ERD)
│   └── PIPELINE.md          # Guia operacional de execução e comandos CLI
├── migrations/              # Migrações relacionais controladas pelo Alembic
├── src/
│   └── enem_pipeline/
│       ├── collectors/      # Coletores modulares (INEP, IBGE, Câmara, OpenAlex)
│       ├── db/              # Modelos SQLAlchemy, sessões e seeders
│       ├── processing/      # Normalizador textual e Classificador dos 10 eixos
│       ├── storage/         # Armazenador imutável da camada RAW
│       ├── cli.py           # Interface de linha de comando (CLI)
│       ├── config.py        # Configurações globais
│       └── scheduler.py     # Daemon de agendamento em background
├── tests/                   # Suíte de testes automatizados com pytest
├── pyproject.toml           # Metadados e dependências do projeto
├── requirements.txt         # Lista estrita de dependências Python
├── .env.example             # Variáveis de ambiente de exemplo
└── README.md
```

---

## 4. Instalação e Configuração

### 4.1. Clonar e Criar Ambiente Virtual
```bash
# Navegar até o diretório do projeto
cd enem-predictor-pipeline

# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
# No Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# No Linux/macOS:
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
pip install -e .
```

### 4.2. Configurar Variáveis de Ambiente
Copie o arquivo `.env.example` para `.env`:
```bash
cp .env.example .env
```

Para conectar diretamente ao banco PostgreSQL com `pgvector` no Supabase:
```env
DATABASE_URL=postgresql+psycopg2://postgres.[REF]:[SENHA]@aws-0-sa-east-1.pooler.supabase.com:6543/postgres
```
*Se a variável for omitida ou mantida com `sqlite:///./data/enem_pipeline.db`, o pipeline funcionará de forma 100% autônoma e local no seu computador.*

---

## 5. Como Executar

### 5.1. Inicializar o Banco e Semeio Inicial
```bash
python -m enem_pipeline.cli seed
```

### 5.2. Executar o Pipeline Completo
```bash
python -m enem_pipeline.cli run-all --limit 25
```

### 5.3. Executar um Coletor Isolado
```bash
python -m enem_pipeline.cli run --source inep --limit 10
python -m enem_pipeline.cli run --source ibge --limit 20
python -m enem_pipeline.cli run --source camara --limit 20
python -m enem_pipeline.cli run --source openalex --limit 20
```

### 5.4. Exibir Painel de Status e Auditoria
```bash
python -m enem_pipeline.cli status
```

### 5.5. Iniciar Agendamento Periódico (Segundo Plano)
```bash
python -m enem_pipeline.cli schedule
```

---

## 6. Testes Automatizados

Execute a suíte com `pytest`:
```bash
python -m pytest -v
```

---

## 7. Documentação Técnica Completa

- [Arquitetura Geral e Decisões de Projeto](docs/ARCHITECTURE.md)
- [Auditoria e Viabilidade das 7 Fontes](docs/DATA_SOURCES.md)
- [Dicionário de Dados e Esquema de Tabelas](docs/DATA_DICTIONARY.md)
- [Guia Operacional e Execução do Pipeline](docs/PIPELINE.md)
