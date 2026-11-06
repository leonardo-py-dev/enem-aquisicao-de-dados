import re
import unicodedata
from typing import Dict, List, Tuple
from sqlalchemy.orm import Session
from loguru import logger

from enem_pipeline.db.models import Document, Topic, DocumentTopic
from enem_pipeline.processing.cleaner import TextCleaner


TAXONOMY_KEYWORDS: Dict[str, List[str]] = {
    "Educação": [
        "educacao", "escola", "escolar", "professor", "docente", "aluno", "estudante",
        "analfabetismo", "evasao", "universidade", "pedagogico", "letramento", "leitura",
        "ensino", "academico", "livro", "aprendizagem", "enem", "surdos",
    ],
    "Saúde": [
        "saude", "sus", "hospital", "vacina", "vacinacao", "doenca", "mental",
        "saneamento", "epidemia", "medico", "psiquiatrico", "psicologico", "terapeutico",
        "leito", "enfermagem", "pandemia", "tratamento", "estigma",
    ],
    "Tecnologia": [
        "tecnologia", "inteligencia artificial", "algoritmo", "dados", "internet",
        "rede social", "digital", "computacao", "virtual", "privacidade", "desinformacao",
        "fake news", "ciber", "automacao", "software", "plataforma",
    ],
    "Meio Ambiente": [
        "meio ambiente", "clima", "climatico", "desmatamento", "amazonia", "bioma",
        "sustentavel", "sustentabilidade", "carbono", "poluicao", "floresta", "hidrico",
        "agua", "fauna", "flora", "aquecimento global", "biodiversidade", "recursos naturais",
    ],
    "Cultura": [
        "cultura", "cultural", "cinema", "arte", "artistico", "literatura", "patrimonio",
        "memoria", "museu", "teatro", "indigena", "quilombola", "matriz africana",
        "afro", "ancestralidade", "tradicionais", "comunidades tradicionais",
    ],
    "Trabalho e Economia": [
        "trabalho", "trabalhador", "emprego", "desemprego", "renda", "salario", "clt",
        "informalidade", "precarizacao", "trabalho de cuidado", "produtividade", "inflacao",
        "economia", "economico", "trabalho infantil", "previdencia",
    ],
    "Direitos Humanos e Cidadania": [
        "direitos humanos", "cidadania", "cidadao", "direito fundamental", "discriminacao",
        "preconceito", "racismo", "voto", "registro civil", "intolerancia", "liberdade",
        "inclusao", "acessibilidade", "certidao de nascimento", "mulher", "violencia contra a mulher",
    ],
    "Desigualdade Social": [
        "desigualdade", "desigualdade social", "pobreza", "miseria", "fome", "periferia",
        "concentracao de renda", "favela", "vulnerabilidade", "habitacao", "moradia",
        "sem teto", "segregacao", "exclusao social", "distribuicao de renda",
    ],
    "Segurança Pública": [
        "seguranca publica", "violencia", "crime", "criminalidade", "policia", "policial",
        "prisao", "sistema prisional", "encarceramento", "homicidio", "trafico",
        "faccao", "armas", "transito", "lei seca", "delito",
    ],
    "Demografia e Envelhecimento": [
        "demografia", "demografico", "idoso", "envelhecimento", "longevidade", "censo",
        "expectativa de vida", "piramide etaria", "natalidade", "populacao", "migracao",
        "imigrante", "refugiado", "emigracao", "urbanizacao",
    ],
}


class ThematicClassifier:
    """Classificador Temático determinístico baseado em léxico ponderado e regras auditáveis."""

    CLASSIFIER_VERSION = "1.0.0"
    METHOD_NAME = "lexical_rules_v1"

    @staticmethod
    def _strip_accents(text: str) -> str:
        """Remove acentos e converte para minúsculas para correspondência fonética/ortográfica."""
        return "".join(
            c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn"
        )

    def classify_text(self, title: str, text: str) -> List[Tuple[str, float]]:
        """Avalia a aderência do texto a cada um dos 10 eixos temáticos, retornando tuplas (eixo, score)."""
        clean_title = self._strip_accents(title or "")
        clean_text = self._strip_accents(text or "")

        full_content = f"{clean_title} {clean_text}"
        results = []

        for axis, keywords in TAXONOMY_KEYWORDS.items():
            score = 0.0
            matches = 0

            for kw in keywords:
                clean_kw = self._strip_accents(kw)
                # Ocorrência no título tem peso 3.0
                if re.search(r"\b" + re.escape(clean_kw) + r"\b", clean_title):
                    score += 3.0
                    matches += 1
                # Ocorrência no corpo do texto tem peso 1.0
                body_matches = len(re.findall(r"\b" + re.escape(clean_kw) + r"\b", clean_text))
                if body_matches > 0:
                    score += min(body_matches, 5) * 1.0
                    matches += 1

            if matches > 0 and score >= 2.0:
                # Normaliza para uma escala de relevância entre 0.20 e 1.0
                normalized_score = min(1.0, 0.20 + (score / 15.0) * 0.80)
                results.append((axis, round(normalized_score, 4)))

        # Ordena do mais relevante para o menos relevante
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def process_and_classify_all(self, db: Session, limit: int = 200) -> int:
        """Processa a higienização de texto e classifica documentos pendentes."""
        topics = {t.name: t for t in db.query(Topic).all()}
        if not topics:
            logger.warning("Nenhum tópico encontrado no banco de dados para classificação.")
            return 0

        # Busca documentos sem processamento ou sem tópicos associados
        documents = db.query(Document).limit(limit).all()
        logger.info(f"Classificando lote de {len(documents)} documentos...")

        classified_count = 0

        for doc in documents:
            # 1. Higienização e Normalização Textual
            if not doc.processed_content:
                doc.processed_content = TextCleaner.normalize_text(doc.original_content)

            # 2. Classificação Temática
            predictions = self.classify_text(doc.title or "", doc.processed_content or doc.original_content)

            for axis_name, score in predictions:
                topic = topics.get(axis_name)
                if not topic:
                    continue

                existing_assoc = (
                    db.query(DocumentTopic)
                    .filter_by(
                        document_id=doc.id,
                        topic_id=topic.id,
                        classification_method=self.METHOD_NAME,
                    )
                    .first()
                )

                if not existing_assoc:
                    assoc = DocumentTopic(
                        document_id=doc.id,
                        topic_id=topic.id,
                        relevance_score=score,
                        classification_method=self.METHOD_NAME,
                        classifier_version=self.CLASSIFIER_VERSION,
                    )
                    db.add(assoc)
                    classified_count += 1
                else:
                    existing_assoc.relevance_score = score

        db.commit()
        logger.success(f"Classificação concluída: {classified_count} vínculos documento-eixo criados.")
        return classified_count
