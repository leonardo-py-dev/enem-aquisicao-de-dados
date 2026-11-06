# Auditoria e Classificação das Fontes de Dados — ENEM Predictor

Este documento apresenta a análise técnica detalhada, classificação de prioridade e matriz de viabilidade para as 7 fontes de dados investigadas para alimentar os modelos do ENEM Predictor.

---

## 1. Matriz de Auditoria e Classificação

| Instituição / Plataforma | Categoria de Dados | Tipo de Acesso | Autenticação | Rate Limit | Histórico | Confiabilidade | Prioridade | Status Técnico |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **INEP** | Provas e Temas Oficiais do ENEM, Microdados | Estruturado (JSON/CSV) + Portal Web | Nenhuma | Tolerante | 1998–2024 | Máxima (Oficial) | **Alta** | Validado (200 OK) |
| **Câmara dos Deputados** | Proposições, Projetos de Lei, Temas de Debate | REST API (JSON) | Nenhuma | Não restritivo | 1988–Hoje | Máxima (Oficial) | **Alta** | Validado (200 OK) |
| **IBGE** | Notícias, Estatísticas Sociais, Demografia, SIDRA | REST API (JSON) | Nenhuma | Alto limite | Séries decenais | Máxima (Oficial) | **Alta** | Validado (200 OK) |
| **OpenAlex** | Artigos Científicos, Pesquisas Acadêmicas no Brasil | REST API (JSON) | Opcional (Polite pool via email) | 10 req/s | Amplo | Alta (Acadêmica) | **Alta** | Validado (200 OK) |
| **IPEA (Ipeadata)** | Séries Temporais, Vulnerabilidade Social, Trabalho | REST API (OData v4 / JSON) | Nenhuma | Moderado | 1970–Hoje | Alta (Oficial) | **Média** | Validado (200 OK, payload denso) |
| **GDELT Project** | Notícias Globais e Monitoramento de Mídia | REST API (JSON) | Nenhuma | Agressivo (Timeouts frequentes) | 2015–Hoje | Média | **Baixa** | Instável (Handshake Timeout) |
| **Google Trends** | Frequência de buscas e interesse do público | Scraping não oficial (pytrends) | Não | Bloqueio rápido (HTTP 429) | 2004–Hoje | Média | **Baixa** | Frágil (Requer proxies / rotação) |

---

## 2. Detalhamento Técnico das Fontes Prioritárias

### 2.1. INEP (Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira)
- **URL Base:** `https://www.gov.br/inep/`
- **Endpoints & Recursos:** Catálogo oficial de provas, temas históricos de redação e links de microdados.
- **Estratégia de Coleta:** Mapeamento estruturado inicial de todos os 26+ anos de temas do ENEM, extraindo o problema social central, o eixo temático oficial e o ano de aplicação. Atualização anual automatizada.
- **Formato:** JSON estruturado / Base local consolidada.

### 2.2. Câmara dos Deputados (Dados Abertos)
- **URL Base:** `https://dadosabertos.camara.leg.br/api/v2`
- **Endpoints Chave:**
  - `/proposicoes`: Lista e detalhes de projetos de lei (PL, PEC, PLP) com filtros por data e tema.
  - `/temas`: Taxonomia legislativa oficial (Educação, Meio Ambiente, Direitos Humanos, etc.).
- **Estratégia de Coleta:** Consulta periódica incremental buscando proposições apresentadas nos últimos anos relacionadas aos eixos temáticos do ENEM.
- **Vantagens:** API pública estável, paginação consistente (`links`), campos padronizados (`ementa`, `dataApresentacao`, `urlInteiroTeor`).

### 2.3. IBGE (Serviço de Dados e SIDRA)
- **URL Base:** `https://servicodados.ibge.gov.br/api`
- **Endpoints Chave:**
  - `/v3/noticias/`: Releases, notas estatísticas e análises conjunturais sobre pobreza, mercado de trabalho, saneamento e educação.
  - `/v3/agregados/`: Séries temporais de censos e PNAD Contínua.
  - `/v1/localidades/estados`: Metadados geográficos oficiais.
- **Estratégia de Coleta:** Monitoramento de notícias e relatórios conjunturais + ingestão de indicadores-chave de desigualdade e educação.

### 2.4. OpenAlex
- **URL Base:** `https://api.openalex.org/works`
- **Filtros Chave:** `default.search`, `institutions.country_code:BR`, `concepts.id`, `publication_year`.
- **Estratégia de Coleta:** Busca semântica por trabalhos acadêmicos brasileiros focados em desafios contemporâneos da sociedade brasileira.
- **Vantagens:** Metadados acadêmicos ricos (título, abstract invertido, conceitos, autores, DOIs).

---

## 3. Fontes de Menor Prioridade e Mitigação Técnica

1. **IPEA (Ipeadata):**
   - *Limitação:* Os endpoints de metadados retornam JSONs com mais de 7MB de uma só vez, o que exige paginação estrita via `$top` e `$skip` no padrão OData v4.
   - *Mitigação:* Desenvolver coletor com filtros específicos por temas sociais (`SERNOME` contendo 'educacao', 'renda', 'pobreza').
2. **GDELT Project:**
   - *Limitação:* A API do GDELT 2.0 Doc sofre frequentemente de lentidão e timeouts de SSL quando consultada a partir do Brasil.
   - *Mitigação:* Timeouts estendidos (30s+), retries com backoff exponencial e execução em lotes assíncronos não bloqueantes.
3. **Google Trends:**
   - *Limitação:* Ausência de API REST oficial para desenvolvedores. O uso de `pytrends` dispara `429 Too Many Requests` rapidamente sem proxy residencial.
   - *Mitigação:* Reservar a coleta do Google Trends para lote semanal com delays controlados entre consultas de palavras-chave.
