"""
Módulo de Predição Oficial para o ENEM 2026.

Implementa a classe ENEM2026Predictor e as dataclasses InterventionProposal e Prediction2026Result:
1. Treinamento supervisionado com modelo tabular XGBoost sobre o histórico completo (1998 a 2025).
2. Regra determinística de veto consecutivo ao eixo de 2025 ("Demografia e Envelhecimento").
3. Renormalização conservativa das 9 probabilidades restantes para 100%.
4. Ordenação estável decrescente com desempate determinístico lexicográfico.
5. Formulação completa de propostas de intervenção para os 3 eixos líderes contemplando os 5 elementos da Competência 5.
6. Catálogo canônico offline determinístico + fallback automático se chamada ao Gemini falhar.
"""

from dataclasses import dataclass, asdict
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from enem_pipeline.config import settings
from enem_pipeline.db.seed import HISTORICAL_ENEM_THEMES, INITIAL_TOPICS
from enem_pipeline.ml.backtest import OFFICIAL_AXES
from enem_pipeline.ml.features import build_training_matrix

VETOED_AXIS_2025: str = "Demografia e Envelhecimento"


@dataclass(frozen=True)
class InterventionProposal:
    """Representa uma proposta completa de intervenção social para a Competência 5 da redação do ENEM."""
    axis: str
    candidate_theme: str
    agente: str
    acao: str
    meio_modo: str
    efeito: str
    detalhamento: str

    def __post_init__(self):
        for field in ("axis", "candidate_theme", "agente", "acao", "meio_modo", "efeito", "detalhamento"):
            val = getattr(self, field)
            if not val or not str(val).strip():
                raise ValueError(
                    f"Invariante da Competência 5 violada: o campo '{field}' não pode ser vazio."
                )
        if self.axis not in OFFICIAL_AXES:
            raise ValueError(f"Eixo '{self.axis}' desconhecido na taxonomia oficial do ENEM.")

    def to_dict(self) -> dict:
        return asdict(self)


# Catálogo Canônico Determinístico Offline cobrindo todos os 10 eixos oficiais do ENEM
CANONICAL_PROPOSALS: Dict[str, InterventionProposal] = {
    "Meio Ambiente": InterventionProposal(
        axis="Meio Ambiente",
        candidate_theme="Desafios para a mitigação do racismo ambiental e a proteção de populações vulneráveis a eventos climáticos extremos no Brasil",
        agente="Ministério do Meio Ambiente e Mudança do Clima (MMA), em atuação conjunta com o Ministério das Cidades e a Defesa Civil Nacional",
        acao="Implementar o Plano Nacional de Infraestrutura Resiliente e Adaptação Climática Periférica",
        meio_modo="Mediante a destinação de recursos vinculados do Fundo Clima (FNMC) e a formalização de convênios federativos com consórcios intermunicipais de áreas vulneráveis",
        detalhamento="Essa descentralização orçamentária priorizará obras de contenção de encostas por bioengenharia, drenagem sustentável e instalação de sistemas comunitários de alarme pluviométrico operados por lideranças locais treinadas",
        efeito="A fim de salvaguardar a integridade física de grupos marginalizados, reverter os impactos do racismo ambiental e assegurar o direito constitucional à moradia digna e segura frente à emergência climática",
    ),
    "Tecnologia": InterventionProposal(
        axis="Tecnologia",
        candidate_theme="Caminhos para a regulação algorítmica e o combate à desinformação automatizada na sociedade brasileira",
        agente="Ministério das Comunicações (MCom), em articulação técnica com a Anatel e o Comitê Gestor da Internet no Brasil (CGI.br)",
        acao="Instituir o Marco Nacional de Transparência Algorítmica e Rastreabilidade de Conteúdos Sintéticos",
        meio_modo="Por meio de diretrizes regulatórias compulsórias para grandes plataformas digitais e exigência de auditorias externas periódicas e independentes",
        detalhamento="Tal regulação obrigará a rotulagem criptográfica visível em mídias geradas por inteligência artificial e disponibilizará APIs seguras com dados agregados a pesquisadores acadêmicos credenciados",
        efeito="Com o intuito de mitigar a proliferação de notícias falsas e discursos de ódio, preservando a estabilidade das instituições democráticas e garantindo a soberania informacional da sociedade",
    ),
    "Educação": InterventionProposal(
        axis="Educação",
        candidate_theme="Desafios para a erradicação do analfabetismo funcional e o letramento científico da juventude no Brasil contemporâneo",
        agente="Ministério da Educação (MEC), em cooperação federativa com o Consed (Conselho Nacional de Secretários de Educação) e a Undime",
        acao="Criar o Programa Nacional de Polos de Iniciação Científica e Letramento Crítico no contraturno da rede pública de educação básica",
        meio_modo="Por intermédio da descentralização de recursos do Fundo Nacional de Desenvolvimento da Educação (FNDE) e concessão de bolsas pedagógicas a graduandos de licenciatura de universidades públicas",
        detalhamento="Essa iniciativa dotará as unidades escolares de bibliotecas laboratoriais com acervo científico interdisciplinar e oferecerá tutorias diagnósticas em pequenos grupos aos estudantes com defasagens registradas no SAEB",
        efeito="Com o fito de erradicar o analfabetismo funcional, capacitar a juventude para a interpretação crítica de dados e argumentos científicos e romper o ciclo histórico de exclusão socioeducacional",
    ),
    "Saúde": InterventionProposal(
        axis="Saúde",
        candidate_theme="Caminhos para o fortalecimento da atenção primária e o enfrentamento às doenças crônicas não transmissíveis no SUS",
        agente="Ministério da Saúde (MS), em articulação com o Conselho Nacional de Secretarias Municipais de Saúde (Conasems)",
        acao="Ampliar o Programa Saúde da Família e modernizar a infraestrutura diagnóstica das Unidades Básicas de Saúde",
        meio_modo="Por meio de investimentos estratégicos em telemedicina pública e destinação vinculada de verbas do Fundo Nacional de Saúde",
        detalhamento="Garantindo atendimento multidisciplinar contínuo com equipes de nutrição, psicologia clínica e medicina de família para populações periféricas e ribeirinhas",
        efeito="A fim de prevenir agravos cardiovasculares e oncológicos precoces, descentralizar o atendimento de média complexidade e consolidar a universalidade resolutiva do SUS",
    ),
    "Cultura": InterventionProposal(
        axis="Cultura",
        candidate_theme="Desafios para a salvaguarda do patrimônio imaterial e valorização da diversidade linguística dos povos originários no Brasil",
        agente="Ministério da Cultura (MinC), em parceria institucional com o IPHAN e o Ministério dos Povos Indígenas",
        acao="Instituir centros regionais de registro oral, difusão e preservação dos saberes tradicionais e línguas autóctones",
        meio_modo="Mediante editais descentralizados da Política Nacional Aldir Blanc de Fomento à Cultura e termos de fomento comunitário",
        detalhamento="Financiando a documentação audiovisual sistemática e oficinas intergeracionais de contação de histórias conduzidas por mestres tradicionais nas aldeias e quilombos",
        efeito="Com o objetivo de combater o etnocídio linguístico, fortalecer a identidade pluricultural brasileira e assegurar o respeito à memória coletiva",
    ),
    "Trabalho e Economia": InterventionProposal(
        axis="Trabalho e Economia",
        candidate_theme="Caminhos para a formalização e proteção previdenciária dos trabalhadores na era da economia de plataformas",
        agente="Ministério do Trabalho e Emprego (MTE), com suporte técnico do Ministério Público do Trabalho (MPT)",
        acao="Estabelecer o Estatuto dos Trabalhadores sob Demanda com garantias mínimas de saúde e remuneração condigna por hora logada",
        meio_modo="Por meio de regulamentação negociada em mesa tripartite permanente entre governo federal, centrais sindicais e empresas gestoras de aplicativos",
        detalhamento="Exigindo o recolhimento patronal obrigatório ao regime geral da previdência e a disponibilização de apólices coletivas de seguro contra acidentes de percurso",
        efeito="A fim de coibir a precarização uberizada da força de trabalho, garantir amparo social a entregadores e motoristas e manter o equilíbrio fiscal da previdência pública",
    ),
    "Direitos Humanos e Cidadania": InterventionProposal(
        axis="Direitos Humanos e Cidadania",
        candidate_theme="Desafios para a erradicação da subnotificação e garantia da cidadania plena aos grupos historicamente invisibilizados",
        agente="Ministério dos Direitos Humanos e da Cidadania (MDHC), em cooperação federativa com a Associação Nacional dos Defensores Públicos",
        acao="Implantar caravanas itinerantes permanentes de registro civil unificado e emissão gratuita de documentação básica",
        meio_modo="Por intermédio de postos móveis de atendimento dotados de tecnologia biométrica e conexão satelital que percorram periferias urbanas e sertões",
        detalhamento="Integrando cartórios civis, tribunais de justiça e assistência social no mesmo local para emissão instantânea da CIN e cadastramento no CadÚnico",
        efeito="Com a finalidade de restabelecer a dignidade individual, combater a morte civil e facultar o acesso a todos os serviços públicos essenciais do Estado democrático",
    ),
    "Desigualdade Social": InterventionProposal(
        axis="Desigualdade Social",
        candidate_theme="Caminhos para a inclusão socioeconômica e redução das disparidades de renda nas periferias urbanas brasileiras",
        agente="Ministério do Desenvolvimento e Assistência Social, Família e Combate à Fome (MDS), em parceria financeira com o BNDES",
        acao="Criar o Programa Nacional de Microcrédito Produtivo Orientado e Fomento à Economia Circular Comunitária",
        meio_modo="Mediante abertura de linhas de crédito a juros subsidiados geridas por bancos comunitários de desenvolvimento e cooperativas populares",
        detalhamento="Aliando o aporte financeiro à mentoria técnica contínua em gestão financeira e inovação para empreendimentos encabeçados por mulheres chefes de família",
        efeito="A fim de romper a armadilha intergeracional da pobreza, dinamizar os fluxos econômicos locais e reduzir a concentração de renda territorial",
    ),
    "Segurança Pública": InterventionProposal(
        axis="Segurança Pública",
        candidate_theme="Desafios para o fortalecimento do policiamento comunitário e a valorização profissional nas forças de segurança pública",
        agente="Ministério da Justiça e Segurança Pública (MJSP), em articulação estratégica com o Conselho Nacional de Comandantes-Gerais",
        acao="Implementar a Matriz Curricular Nacional de Policiamento Cidadão e Mediação Conflitiva",
        meio_modo="Por intermédio de repasses condicionados do Fundo Nacional de Segurança Pública (FNSP) vinculados à modernização doutrinária e psicológica",
        detalhamento="Incluindo acompanhamento psicológico preventivo permanente, uso disseminado de câmeras corporais com custódia de dados auditada e programas habitacionais exclusivos para agentes",
        efeito="Com o propósito de resguardar a integridade física tanto dos policiais quanto da comunidade, reduzindo índices de letalidade e aumentando a confiança social nas forças policiais",
    ),
    "Demografia e Envelhecimento": InterventionProposal(
        axis="Demografia e Envelhecimento",
        candidate_theme="Caminhos para a universalização do cuidado continuado e a consolidação do envelhecimento ativo no Brasil contemporâneo",
        agente="Ministério dos Direitos Humanos e da Cidadania, em parceria com o Ministério do Desenvolvimento e Assistência Social",
        acao="Estruturar o Sistema Nacional de Cuidados Continuados e Ampliação dos Centros-Dia para Pessoas Idosas",
        meio_modo="Por meio de coparticipação orçamentária tripartite com consórcios intermunicipais e organização da sociedade civil especializada",
        detalhamento="Capacitando cuidadores comunitários remunerados pelo poder público e provendo suporte multiprofissional domiciliar para idosos dependentes de baixa renda",
        efeito="A fim de assegurar a dignidade humana na longevidade, prevenir a vulnerabilidade asilar forçada e valorizar a autonomia dos cidadãos seniores",
    ),
}


def _call_gemini_for_proposals(axis: str, api_key: Optional[str] = None) -> Optional[InterventionProposal]:
    """Tenta consultar a Google Gemini API para formular proposta de intervenção social personalizada.
    
    Em caso de sucesso, retorna um InterventionProposal estritamente validado.
    Em caso de falha de conexão, exceção de API ou ausência de chave, propaga a exceção para que
    o chamador execute o fallback transparente para o catálogo canônico.
    """
    key = api_key or getattr(settings, "GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        prompt = f"""
        Você é o especialista pedagógico oficial em redação do ENEM do INEP.
        Gere uma proposta de intervenção social NOTA 1000 com nota máxima na Competência 5 para o eixo temático: '{axis}'.
        Você DEVE retornar estritamente um JSON com as seguintes chaves exatas:
        {{
          "axis": "{axis}",
          "candidate_theme": "Frase-tema formulada no padrão do INEP (ex: 'Desafios para...')",
          "agente": "Quem executará a ação (órgão ou ministério específico e institucional)",
          "acao": "O que será feito (verbo afirmativo no infinitivo)",
          "meio_modo": "Como será viabilizado (canais, fundos orçamentários, parcerias)",
          "detalhamento": "Informação técnica complementar substantiva detalhando o meio/modo ou a ação",
          "efeito": "Finalidade e impacto social pretendido na garantia de direitos fundamentais"
        }}
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        if response.text:
            data = json.loads(response.text)
            return InterventionProposal(
                axis=axis,
                candidate_theme=data["candidate_theme"],
                agente=data["agente"],
                acao=data["acao"],
                meio_modo=data["meio_modo"],
                detalhamento=data["detalhamento"],
                efeito=data["efeito"],
            )
    except Exception as e:
        logger.warning(
            f"[ENEM2026Predictor] Geração dinâmica via Gemini para '{axis}' indisponível: {e}."
        )
        raise e

    return None


def _generate_proposal_for_axis(
    axis: str,
    offline: bool = False,
    use_llm: bool = True,
    api_key: Optional[str] = None,
) -> InterventionProposal:
    """Recupera ou gera proposta de intervenção para um eixo, priorizando Gemini se habilitado com fallback canônico."""
    if not offline and use_llm:
        try:
            proposal = _call_gemini_for_proposals(axis, api_key=api_key)
            if proposal is not None:
                return proposal
        except Exception as exc:
            logger.warning(
                f"[ENEM2026Predictor] Recorrendo ao Catálogo Canônico Offline para o eixo '{axis}' devido a: {exc}"
            )

    if axis in CANONICAL_PROPOSALS:
        return CANONICAL_PROPOSALS[axis]

    # Fallback genérico caso surja algum eixo inesperado fora do padrão
    return InterventionProposal(
        axis=axis,
        candidate_theme=f"Desafios para a efetivação de políticas públicas integradas no eixo {axis} no Brasil",
        agente="Poder Executivo Federal, em atuação conjunta com a sociedade civil organizada",
        acao="Implementar plano setorial de garantia de direitos e modernização estrutural",
        meio_modo="Mediante dotação orçamentária vinculada e formalização de convênios federativos intermunicipais",
        detalhamento="Com priorização estratégica das populações em situação de acentuada vulnerabilidade social",
        efeito="A fim de consolidar a cidadania plena e salvaguardar os preceitos constitucionais fundamentais",
    )


@dataclass(frozen=True)
class Prediction2026Result:
    """Resultado oficial da predição para o ENEM 2026 com ranking dos 10 eixos e propostas da Competência 5."""
    ranking: List[Tuple[str, float]]
    vetoed_axis: str
    top3_proposals: List[InterventionProposal]

    def __post_init__(self):
        # 1. Validação do eixo vetado
        if self.vetoed_axis != VETOED_AXIS_2025:
            raise ValueError(
                f"vetoed_axis deve ser estritamente '{VETOED_AXIS_2025}', obteve '{self.vetoed_axis}'."
            )

        # 2. Validação da cardinalidade do ranking
        if len(self.ranking) != 10:
            raise ValueError(
                f"ranking deve conter exatamente 10 eixos, obteve {len(self.ranking)}."
            )

        # 3. Validação da taxonomia oficial
        ranked_axes = [ax for ax, _ in self.ranking]
        if set(ranked_axes) != set(OFFICIAL_AXES):
            raise ValueError(
                "ranking deve conter todos os 10 eixos oficiais do ENEM sem repetições."
            )

        # 4. Probabilidade do eixo vetado deve ser estritamente 0.0
        prob_dict = dict(self.ranking)
        vetoed_p = prob_dict.get(self.vetoed_axis, -1.0)
        if vetoed_p != 0.0:
            raise ValueError(
                f"A probabilidade do eixo vetado deve ser estritamente 0.0, obteve {vetoed_p}."
            )

        # 5. O eixo vetado deve ocupar a última posição no ranking ordenado
        if self.ranking[-1][0] != self.vetoed_axis:
            raise ValueError(
                f"O eixo vetado '{self.vetoed_axis}' (P=0.0) deve ocupar a última posição (10º lugar), "
                f"mas a última posição é ocupada por '{self.ranking[-1][0]}'."
            )

        # 6. Validação numérica de sanidade das probabilidades
        probs = [p for _, p in self.ranking]
        if any(p < 0.0 or math.isnan(p) or math.isinf(p) for p in probs):
            raise ValueError("As probabilidades devem ser valores numéricos finitos e não-negativos.")

        # 7. Soma das probabilidades deve ser exatamente 1.0 (100.0%)
        total_p = sum(probs)
        if abs(total_p - 1.0) > 1e-4:
            raise ValueError(
                f"A soma das probabilidades deve ser exatamente 1.0 (observada: {total_p:.6f})."
            )

        # 8. Monotonicidade decrescente com desempate determinístico
        for i in range(len(self.ranking) - 1):
            curr_ax, curr_p = self.ranking[i]
            next_ax, next_p = self.ranking[i + 1]
            if curr_p < next_p - 1e-7:
                raise ValueError(
                    f"Ranking não monotônico na posição {i+1} ({curr_ax}: {curr_p}) vs {i+2} ({next_ax}: {next_p})."
                )
            if abs(curr_p - next_p) < 1e-7 and curr_ax > next_ax and curr_ax != self.vetoed_axis:
                raise ValueError(
                    f"Desempate de probabilidades idênticas deve ser alfabético: '{curr_ax}' vs '{next_ax}'."
                )

        # 9. Consistência do Top-3 com as propostas da Competência 5
        if len(self.top3_proposals) != 3:
            raise ValueError(
                f"top3_proposals deve conter exatamente 3 propostas, obteve {len(self.top3_proposals)}."
            )

        expected_top3_axes = ranked_axes[:3]
        proposal_axes = [p.axis for p in self.top3_proposals]
        if proposal_axes != expected_top3_axes:
            raise ValueError(
                f"Os eixos de top3_proposals ({proposal_axes}) não coincidem com o Top-3 do ranking ({expected_top3_axes})."
            )

    def to_dict(self) -> dict:
        return {
            "vetoed_axis": self.vetoed_axis,
            "ranking": self.ranking,
            "top3_proposals": [p.to_dict() for p in self.top3_proposals],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def get_top_axis(self) -> str:
        return self.ranking[0][0]

    def get_top3_axes(self) -> List[str]:
        return [ax for ax, _ in self.ranking[:3]]

    def get_axis_probability(self, axis: str) -> float:
        d = dict(self.ranking)
        if axis not in d:
            raise KeyError(f"Eixo '{axis}' não consta no ranking oficial de 2026.")
        return d[axis]


class ENEM2026Predictor:
    """Preditor oficial para o ENEM 2026 com veto determinístico estrito ao eixo de 2025."""

    FEATURE_COLS = [
        "years_since_last",
        "occurred_previous_year",
        "occurred_last_3_years",
        "cumulative_frequency",
        "vol_ibge",
        "vol_camara",
        "vol_openalex",
        "score_total",
        "debate_acceleration",
    ]

    def __init__(
        self,
        historical_data_path: Optional[str] = None,
        db_session=None,
        min_train_year: int = 2003,
        random_state: int = 42,
        offline: bool = False,
        use_llm: bool = True,
    ):
        self.historical_data_path = historical_data_path
        self.db_session = db_session
        self.min_train_year = min_train_year
        self.random_state = random_state
        self.offline = offline
        self.use_llm = use_llm and not offline

        self.history_records = self._load_historical_data()
        self.history_map: Dict[int, Dict[str, str]] = {
            item["year"]: {
                "axis": item["thematic_axis"],
                "theme": item["official_theme"],
            }
            for item in self.history_records
        }

    def _load_historical_data(self) -> List[Dict]:
        """Carrega o catálogo histórico de temas oficiais do ENEM com fallback gracioso em cascata."""
        if self.historical_data_path:
            logger.info(f"[ENEM2026Predictor] Carregando histórico de {self.historical_data_path}")
            if self.historical_data_path.endswith(".json"):
                with open(self.historical_data_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            elif self.historical_data_path.endswith(".csv"):
                df = pd.read_csv(self.historical_data_path)
                return df.to_dict(orient="records")

        if self.db_session is not None:
            try:
                from enem_pipeline.db.models import HistoricalTheme
                db_themes = (
                    self.db_session.query(HistoricalTheme)
                    .order_by(HistoricalTheme.year.asc())
                    .all()
                )
                if db_themes:
                    return [
                        {
                            "year": t.year,
                            "official_theme": t.official_theme,
                            "thematic_axis": t.thematic_axis,
                        }
                        for t in db_themes
                    ]
            except Exception as exc:
                logger.warning(f"[ENEM2026Predictor] Falha ao consultar banco, usando fallback seed: {exc}")

        return list(HISTORICAL_ENEM_THEMES)

    def _build_dataset(self, target_year: int = 2026) -> pd.DataFrame:
        """Constrói a matriz de features tabular temporal até o ano-alvo."""
        return build_training_matrix(
            db=self.db_session,
            min_year=self.min_train_year,
            max_year=target_year,
            historical_themes=self.history_records,
        )

    def _train_and_predict_raw(
        self, matrix_df: pd.DataFrame, target_year: int = 2026
    ) -> Dict[str, float]:
        """Treina modelo tabular XGBoost sobre anos históricos e infere probabilidades para target_year."""
        import xgboost as xgb

        train_mask = matrix_df["year"] < target_year
        test_mask = matrix_df["year"] == target_year

        # Auditoria ativa contra vazamento temporal (data leakage)
        train_years = matrix_df.loc[train_mask, "year"]
        if not train_years.empty and train_years.max() >= target_year:
            raise RuntimeError(
                f"[DATA LEAKAGE DETECTED] O conjunto de treino de {target_year} contém anos futuros: {train_years.max()}!"
            )

        X_train = matrix_df.loc[train_mask, self.FEATURE_COLS]
        y_train = matrix_df.loc[train_mask, "target"]

        X_test = matrix_df.loc[test_mask, self.FEATURE_COLS]
        test_axes = matrix_df.loc[test_mask, "axis"].tolist()

        if len(test_axes) != 10:
            raise RuntimeError(
                f"O conjunto de inferência para o ano {target_year} deve conter exatamente 10 eixos, obteve {len(test_axes)}."
            )

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)

        raw_probs = model.predict_proba(X_test)[:, 1]
        sum_raw = float(raw_probs.sum())
        norm_probs = (raw_probs / sum_raw) if sum_raw > 0 else np.ones_like(raw_probs) / len(raw_probs)

        return {axis: float(p) for axis, p in zip(test_axes, norm_probs)}

    def _apply_deterministic_veto(
        self, raw_probs: Dict[str, float], target_year: int = 2026
    ) -> Tuple[List[Tuple[str, float]], str]:
        """Aplica a regra de veto consecutivo: fixa P(eixo_anterior)=0.0, renormaliza os 9 eixos e ordena de forma estável."""
        prev_year = target_year - 1
        vetoed_axis = self.history_map.get(prev_year, {}).get("axis", VETOED_AXIS_2025)

        prob_dict = dict(raw_probs)
        # 1. Força estritamente 0.0 no eixo do ano imediatamente anterior
        prob_dict[vetoed_axis] = 0.0

        # 2. Renormaliza os 9 eixos restantes
        remaining_sum = sum(v for k, v in prob_dict.items() if k != vetoed_axis)
        if remaining_sum > 0:
            for k in prob_dict:
                if k != vetoed_axis:
                    prob_dict[k] = prob_dict[k] / remaining_sum
        else:
            uniform_val = 1.0 / 9.0
            for k in prob_dict:
                if k != vetoed_axis:
                    prob_dict[k] = uniform_val

        # 3. Ordenação estável: eixos não-vetados ordenados por probabilidade decrescente e nome alfabético;
        # o eixo vetado recebe prioridade 1 para ser posicionado obrigatoriamente na última colocação (#10).
        sorted_items = sorted(
            prob_dict.items(),
            key=lambda x: (1 if x[0] == vetoed_axis else 0, -x[1], x[0]),
        )

        # 4. Ajuste final de ponto flutuante para garantir soma unitária perfeita
        total_non_vetoed = sum(p for ax, p in sorted_items if ax != vetoed_axis)
        ranking = []
        for axis, p in sorted_items:
            if axis == vetoed_axis:
                ranking.append((axis, 0.0))
            else:
                ranking.append((axis, float(p / total_non_vetoed)))

        return ranking, vetoed_axis

    def predict(self, target_year: int = 2026) -> Prediction2026Result:
        """Executa a predição oficial do ENEM para o ano especificado (padrão 2026)."""
        logger.info(f"[ENEM2026Predictor] Iniciando predição oficial para o ano {target_year}...")
        matrix_df = self._build_dataset(target_year=target_year)

        raw_probs = self._train_and_predict_raw(matrix_df=matrix_df, target_year=target_year)
        ranking, vetoed_axis = self._apply_deterministic_veto(raw_probs=raw_probs, target_year=target_year)

        top3_axes = [ax for ax, _ in ranking[:3]]
        logger.info(f"[ENEM2026Predictor] Top-3 Eixos para {target_year}: {top3_axes}")
        logger.info(f"[ENEM2026Predictor] Eixo Vetado: '{vetoed_axis}' com probabilidade {dict(ranking)[vetoed_axis]:.1f}%")

        top3_proposals = [
            _generate_proposal_for_axis(
                ax,
                offline=self.offline,
                use_llm=self.use_llm,
            )
            for ax in top3_axes
        ]

        result = Prediction2026Result(
            ranking=ranking,
            vetoed_axis=vetoed_axis,
            top3_proposals=top3_proposals,
        )
        return result
