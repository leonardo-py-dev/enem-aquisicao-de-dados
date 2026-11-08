"""
Módulo de Backtesting Walk-Forward para o ENEM Predictor.

Realiza a validação preditiva retrospectiva estritamente cega ano a ano (2015–2025)
com janela temporal expansiva, garantindo zero vazamento temporal (data leakage).
Calcula as 5 métricas estatísticas obrigatórias (Top-1, Top-3, MRR, Brier Score e Log Loss),
executa o Teste Binomial Exato puro (zero-dependency via combinatória analítica) e
computa o Lift relativo sobre a linha de base ingênua de 30%.
"""

from dataclasses import dataclass, asdict
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from enem_pipeline.db.seed import HISTORICAL_ENEM_THEMES, INITIAL_TOPICS
from enem_pipeline.ml.features import build_training_matrix

# Taxonomia oficial canônica dos 10 eixos temáticos do ENEM
OFFICIAL_AXES: List[str] = [
    "Educação",
    "Saúde",
    "Tecnologia",
    "Meio Ambiente",
    "Cultura",
    "Trabalho e Economia",
    "Direitos Humanos e Cidadania",
    "Desigualdade Social",
    "Segurança Pública",
    "Demografia e Envelhecimento",
]


def compute_brier_score(
    probabilities: Dict[str, float] | Sequence[Tuple[str, float]],
    official_axis: str,
) -> float:
    """Calcula o Brier Score multi-classe: sum_{k=1}^K (p_k - y_k)^2.

    Mede a calibração probabilística no intervalo [0.0, 2.0].
    Para predição perfeita: 0.0.
    Para baseline uniforme (p=0.10): 0.90.
    Para pior predição possível: 2.0.
    """
    prob_dict = dict(probabilities) if not isinstance(probabilities, dict) else probabilities
    score = 0.0
    for axis, prob in prob_dict.items():
        y = 1.0 if axis == official_axis else 0.0
        score += (prob - y) ** 2
    if official_axis not in prob_dict:
        score += 1.0
    return float(score)


def compute_log_loss(
    probabilities: Dict[str, float] | Sequence[Tuple[str, float]],
    official_axis: str,
    eps: float = 1e-15,
) -> float:
    """Calcula o Log Loss multi-classe: -ln(max(p_oficial, eps)).

    Aplica amortecimento numérico Epsilon Clipping (eps = 1e-15) para imunidade
    contra erro de domínio em log(0).
    Para predição perfeita (p=1.0): 0.0.
    Para baseline uniforme (p=0.10): ~2.302585.
    Para probabilidade zero (p=0.0): ~34.538776.
    """
    prob_dict = dict(probabilities) if not isinstance(probabilities, dict) else probabilities
    p_target = prob_dict.get(official_axis, 0.0)
    clipped_prob = min(max(float(p_target), eps), 1.0)
    return max(0.0, float(-math.log(clipped_prob)))


def compute_relative_lift(top3_hit_rate: float, baseline: float = 0.30) -> float:
    """Calcula o ganho percentual relativo sobre a baseline ingênua de 30%:

    Lift = ((top3_hit_rate - baseline) / baseline) * 100.
    """
    if baseline <= 0.0:
        raise ValueError("A baseline deve ser estritamente maior que zero.")
    return float(((top3_hit_rate - baseline) / baseline) * 100.0)


def compute_exact_binomial_pvalue(k: int, n: int, p0: float = 0.30) -> float:
    """Calcula o p-valor exato da cauda superior P(X >= k) sob H0: p <= p0.

    Implementação analítica pura baseada no coeficiente binomial nativo (math.comb).
    Disponibiliza portabilidade zero-dependency e estabilidade sem depender do scipy.
    """
    if n <= 0:
        raise ValueError("n (total de ensaios) deve ser maior que zero.")
    if not (0.0 <= p0 <= 1.0):
        raise ValueError("p0 deve estar no intervalo [0.0, 1.0].")

    if k <= 0:
        return 1.0
    if k > n:
        return 0.0

    p_val = sum(
        math.comb(n, i) * (p0 ** i) * ((1.0 - p0) ** (n - i))
        for i in range(k, n + 1)
    )
    return float(min(max(p_val, 0.0), 1.0))


@dataclass(frozen=True)
class BacktestYearResult:
    """Armazena o resultado preditivo e as métricas estatísticas para uma única edição do ENEM."""

    year: int
    official_axis: str
    official_theme: str
    predicted_ranking: list[tuple[str, float]]  # [(axis, prob), ...] ordenado decrescente
    rank_of_official: int  # 1-based (1 a 10)
    top1_hit: bool
    top3_hit: bool
    reciprocal_rank: float
    brier_score: float
    log_loss: float

    def __post_init__(self):
        # 1. Validação do ano
        if not (1998 <= self.year <= 2030):
            raise ValueError(f"year ({self.year}) fora do intervalo permitido [1998, 2030].")

        # 2. Validação do eixo oficial
        if self.official_axis not in OFFICIAL_AXES:
            raise ValueError(
                f"official_axis '{self.official_axis}' não pertence à taxonomia oficial de 10 eixos do ENEM."
            )

        # 3. Validação do tema oficial
        if not self.official_theme or not str(self.official_theme).strip():
            raise ValueError("official_theme não pode ser vazio.")

        # 4. Validação do ranking predito
        if len(self.predicted_ranking) != 10:
            raise ValueError(
                f"predicted_ranking deve conter exatamente 10 eixos temáticos, obteve {len(self.predicted_ranking)}."
            )

        ranked_axes = [axis for axis, _ in self.predicted_ranking]
        if set(ranked_axes) != set(OFFICIAL_AXES):
            raise ValueError("predicted_ranking deve conter exatamente os 10 eixos oficiais sem repetições.")

        probs = [p for _, p in self.predicted_ranking]
        if any(p < 0.0 or math.isnan(p) or math.isinf(p) for p in probs):
            raise ValueError("Probabilidades em predicted_ranking devem ser valores finitos e não-negativos.")

        total_p = sum(probs)
        if abs(total_p - 1.0) > 1e-4:
            raise ValueError(f"A soma das probabilidades deve ser 1.0 (soma observada: {total_p:.6f}).")

        # Validação de ordenação decrescente de probabilidade
        for i in range(len(probs) - 1):
            if probs[i] < probs[i + 1] - 1e-7:
                raise ValueError("predicted_ranking deve estar ordenado em ordem decrescente de probabilidade.")

        # 5. Validação da posição do eixo oficial
        expected_rank = ranked_axes.index(self.official_axis) + 1
        if self.rank_of_official != expected_rank:
            raise ValueError(
                f"rank_of_official informado ({self.rank_of_official}) diverge da posição real no ranking ({expected_rank})."
            )

        if not (1 <= self.rank_of_official <= 10):
            raise ValueError(f"rank_of_official ({self.rank_of_official}) deve estar no intervalo [1, 10].")

        # 6. Invariantes de métricas ordinais (Top-1, Top-3, Reciprocal Rank)
        if self.top1_hit != (self.rank_of_official == 1):
            raise ValueError(f"Invariante violada: top1_hit ({self.top1_hit}) != (rank_of_official == 1).")

        if self.top3_hit != (self.rank_of_official in [1, 2, 3]):
            raise ValueError(f"Invariante violada: top3_hit ({self.top3_hit}) != (rank_of_official in [1, 2, 3]).")

        if abs(self.reciprocal_rank - (1.0 / self.rank_of_official)) > 1e-5:
            raise ValueError(f"Invariante violada: reciprocal_rank ({self.reciprocal_rank}) != 1.0 / rank.")

        # 7. Invariantes de calibração (Brier Score, Log Loss)
        if not (0.0 <= self.brier_score <= 2.0001):
            raise ValueError(f"brier_score ({self.brier_score}) fora do intervalo teórico [0.0, 2.0].")

        if self.log_loss < -1e-6 or math.isnan(self.log_loss):
            raise ValueError(f"log_loss ({self.log_loss}) deve ser >= 0.0 e finito.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BacktestSummary:
    """Consolida as métricas globais e o teste de hipótese para todo o horizonte de validação."""

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

    def __post_init__(self):
        n = len(self.years_evaluated)
        if n == 0:
            raise ValueError("years_evaluated não pode ser vazio.")
        if len(self.yearly_results) != n:
            raise ValueError(
                f"yearly_results (len {len(self.yearly_results)}) deve ter o mesmo tamanho de years_evaluated ({n})."
            )

        # Consistência referencial dos anos
        result_years = [r.year for r in self.yearly_results]
        if result_years != self.years_evaluated:
            raise ValueError("Anos em yearly_results não coincidem com years_evaluated.")

        # Consistência das médias agregadas
        calc_top1 = sum(1.0 for r in self.yearly_results if r.top1_hit) / n
        calc_top3 = sum(1.0 for r in self.yearly_results if r.top3_hit) / n
        calc_mrr = sum(r.reciprocal_rank for r in self.yearly_results) / n
        calc_brier = sum(r.brier_score for r in self.yearly_results) / n
        calc_log_loss = sum(r.log_loss for r in self.yearly_results) / n

        if abs(self.top1_accuracy - calc_top1) > 1e-4:
            raise ValueError(f"top1_accuracy ({self.top1_accuracy}) difere da média calculada ({calc_top1}).")
        if abs(self.top3_hit_rate - calc_top3) > 1e-4:
            raise ValueError(f"top3_hit_rate ({self.top3_hit_rate}) difere da média calculada ({calc_top3}).")
        if abs(self.mrr - calc_mrr) > 1e-4:
            raise ValueError(f"mrr ({self.mrr}) difere da média calculada ({calc_mrr}).")
        if abs(self.mean_brier_score - calc_brier) > 1e-4:
            raise ValueError(f"mean_brier_score ({self.mean_brier_score}) difere da média calculada ({calc_brier}).")
        if abs(self.mean_log_loss - calc_log_loss) > 1e-4:
            raise ValueError(f"mean_log_loss ({self.mean_log_loss}) difere da média calculada ({calc_log_loss}).")

        if abs(self.baseline_top3_rate - 0.30) > 1e-6:
            raise ValueError(f"baseline_top3_rate ({self.baseline_top3_rate}) deve ser exatamente 0.30.")

        expected_lift = ((self.top3_hit_rate - 0.30) / 0.30) * 100.0
        if abs(self.relative_lift_top3 - expected_lift) > 1e-4:
            raise ValueError(
                f"relative_lift_top3 ({self.relative_lift_top3}) difere do esperado ({expected_lift})."
            )

        if not (0.0 <= self.binomial_test_p_value <= 1.0):
            raise ValueError(f"binomial_test_p_value ({self.binomial_test_p_value}) deve estar em [0.0, 1.0].")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class WalkForwardSimulator:
    """Simulador preditivo walk-forward estritamente cego com janela expansiva ano a ano."""

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

    calculate_binomial_pvalue = staticmethod(compute_exact_binomial_pvalue)
    calculate_relative_lift = staticmethod(compute_relative_lift)

    def __init__(
        self,
        historical_data_path: Optional[str] = None,
        db_session=None,
        min_train_year: int = 2003,
        random_state: int = 42,
        enforce_hard_veto: bool = False,
    ):
        """Inicializa o simulador.

        Suporta carregamento via caminho customizado, sessão SQLAlchemy ou
        fallback in-memory determinístico para HISTORICAL_ENEM_THEMES de seed.py.
        """
        self.historical_data_path = historical_data_path
        self.db_session = db_session
        self.min_train_year = min_train_year
        self.random_state = random_state
        self.enforce_hard_veto = enforce_hard_veto

        self.history_records = self._load_historical_data()
        self.history_map: Dict[int, Dict[str, str]] = {
            item["year"]: {
                "axis": item["thematic_axis"],
                "theme": item["official_theme"],
            }
            for item in self.history_records
        }
        self.all_axes: List[str] = list(OFFICIAL_AXES)

    def _load_historical_data(self) -> List[Dict]:
        """Carrega o catálogo de temas históricos com suporte hierárquico a arquivo, DB ou seed."""
        if self.historical_data_path:
            logger.info(f"[WalkForward] Carregando temas históricos de {self.historical_data_path}")
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
            except Exception as e:
                logger.warning(f"[WalkForward] Falha ao consultar DB, usando fallback de seed: {e}")

        return list(HISTORICAL_ENEM_THEMES)

    def _build_dataset(self, max_year: int) -> pd.DataFrame:
        """Constrói a matriz temporal de features garantindo integridade e consistência."""
        return build_training_matrix(
            db=self.db_session,
            min_year=self.min_train_year,
            max_year=max_year,
            historical_themes=self.history_records,
        )

    def _train_and_predict_year(
        self, target_year: int, matrix_df: pd.DataFrame
    ) -> List[Tuple[str, float]]:
        """Treina modelo fresco em y < target_year e infere probabilidades para target_year.

        Garante zero vazamento temporal (data leakage).
        """
        import xgboost as xgb

        # Isolamento estrito de treino: dados estritamente anteriores a target_year
        train_mask = matrix_df["year"] < target_year
        test_mask = matrix_df["year"] == target_year

        # Auditoria em runtime de data leakage
        train_years = matrix_df.loc[train_mask, "year"]
        if not train_years.empty and train_years.max() >= target_year:
            raise RuntimeError(
                f"[DATA LEAKAGE DETECTED] Treinamento para o ano {target_year} "
                f"contém dados do ano {train_years.max()}!"
            )

        X_train = matrix_df.loc[train_mask, self.FEATURE_COLS]
        y_train = matrix_df.loc[train_mask, "target"]

        X_test = matrix_df.loc[test_mask, self.FEATURE_COLS]
        test_axes = matrix_df.loc[test_mask, "axis"].tolist()

        if len(test_axes) != 10:
            raise RuntimeError(
                f"Conjunto de teste para o ano {target_year} deve conter exatamente 10 eixos, "
                f"obteve {len(test_axes)}."
            )

        # Instanciação isolada do modelo (sem reaproveitamento de estado/árvores)
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)

        raw_probs = model.predict_proba(X_test)[:, 1]

        # Normalização de probabilidades para soma exata 1.0
        prob_sum = float(raw_probs.sum())
        probs = (raw_probs / prob_sum) if prob_sum > 0 else np.ones_like(raw_probs) / len(raw_probs)

        prob_dict = {axis: float(prob) for axis, prob in zip(test_axes, probs)}

        # Veto consecutivo opcional (se habilitado)
        if self.enforce_hard_veto:
            prev_year = target_year - 1
            if prev_year in self.history_map:
                prev_axis = self.history_map[prev_year]["axis"]
                prob_dict[prev_axis] = 0.0
                remaining_sum = sum(prob_dict.values())
                if remaining_sum > 0:
                    prob_dict = {k: v / remaining_sum for k, v in prob_dict.items()}
                else:
                    non_vetoed = [ax for ax in prob_dict if ax != prev_axis]
                    fallback_p = 1.0 / len(non_vetoed) if non_vetoed else 0.0
                    for ax in non_vetoed:
                        prob_dict[ax] = fallback_p

        # Ordenação por probabilidade decrescente e desempate alfabético estável
        sorted_items = sorted(prob_dict.items(), key=lambda x: (-x[1], x[0]))

        # Re-normalização precisa de arredondamento
        sum_p = sum(p for _, p in sorted_items)
        if sum_p > 0:
            ranking = [(axis, float(p / sum_p)) for axis, p in sorted_items]
        else:
            uniform_p = 1.0 / len(sorted_items)
            ranking = [(axis, float(uniform_p)) for axis, _ in sorted_items]

        return ranking

    def _evaluate_year(
        self, target_year: int, ranking: List[Tuple[str, float]]
    ) -> BacktestYearResult:
        """Avalia o ranking inferido contra o tema oficial do exame no ano avaliado."""
        official_data = self.history_map[target_year]
        official_axis = official_data["axis"]
        official_theme = official_data["theme"]

        ranked_axes = [axis for axis, _ in ranking]
        rank = ranked_axes.index(official_axis) + 1  # 1-based (1 a 10)

        top1_hit = (rank == 1)
        top3_hit = (rank <= 3)
        rr = 1.0 / rank

        brier = compute_brier_score(ranking, official_axis)
        log_loss = compute_log_loss(ranking, official_axis)

        return BacktestYearResult(
            year=target_year,
            official_axis=official_axis,
            official_theme=official_theme,
            predicted_ranking=ranking,
            rank_of_official=rank,
            top1_hit=top1_hit,
            top3_hit=top3_hit,
            reciprocal_rank=float(rr),
            brier_score=float(brier),
            log_loss=float(log_loss),
        )

    def run(self, start_year: int = 2015, end_year: int = 2025) -> BacktestSummary:
        """Executa a simulação walk-forward estritamente cega ano a ano no horizonte especificado."""
        if start_year > end_year:
            raise ValueError(f"start_year ({start_year}) não pode ser maior que end_year ({end_year}).")

        if start_year < 1999:
            raise ValueError(
                f"start_year ({start_year}) deve ser >= 1999 pois o exame pioneiro de 1998 requer histórico prévio."
            )

        logger.info(f"[WalkForward] Iniciando simulação retrospectiva ({start_year} a {end_year})...")
        matrix_df = self._build_dataset(max_year=end_year)

        yearly_results: List[BacktestYearResult] = []

        for y in range(start_year, end_year + 1):
            if y not in self.history_map:
                raise KeyError(f"Ano {y} não encontrado no histórico oficial do ENEM.")

            ranking = self._train_and_predict_year(target_year=y, matrix_df=matrix_df)
            res = self._evaluate_year(target_year=y, ranking=ranking)
            yearly_results.append(res)
            logger.info(
                f"[WalkForward] Ano {y}: Oficial='{res.official_axis}' | "
                f"Rank={res.rank_of_official}º | Top1={res.top1_hit} | Top3={res.top3_hit} | "
                f"Brier={res.brier_score:.4f} | LogLoss={res.log_loss:.4f}"
            )

        n = len(yearly_results)
        top1_acc = sum(1.0 for r in yearly_results if r.top1_hit) / n
        top3_rate = sum(1.0 for r in yearly_results if r.top3_hit) / n
        mrr = sum(r.reciprocal_rank for r in yearly_results) / n
        mean_brier = sum(r.brier_score for r in yearly_results) / n
        mean_log_loss = sum(r.log_loss for r in yearly_results) / n

        baseline_top3 = 0.30
        relative_lift = compute_relative_lift(top3_rate, baseline_top3)

        k_top3_hits = sum(1 for r in yearly_results if r.top3_hit)
        p_value = compute_exact_binomial_pvalue(k=k_top3_hits, n=n, p0=baseline_top3)

        summary = BacktestSummary(
            years_evaluated=list(range(start_year, end_year + 1)),
            top1_accuracy=float(top1_acc),
            top3_hit_rate=float(top3_rate),
            mrr=float(mrr),
            mean_brier_score=float(mean_brier),
            mean_log_loss=float(mean_log_loss),
            baseline_top3_rate=baseline_top3,
            relative_lift_top3=float(relative_lift),
            binomial_test_p_value=float(p_value),
            yearly_results=yearly_results,
        )

        logger.success(
            f"[WalkForward] Concluído! Top-1: {top1_acc*100:.1f}% | "
            f"Top-3: {top3_rate*100:.1f}% (Lift: {relative_lift:+.1f}%) | "
            f"MRR: {mrr:.3f} | Brier: {mean_brier:.4f} | LogLoss: {mean_log_loss:.4f} | "
            f"p-valor: {p_value:.5f}"
        )
        return summary
