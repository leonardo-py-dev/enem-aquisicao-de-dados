import re
import unicodedata
from typing import List, Set


PORTUGUESE_STOPWORDS: Set[str] = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos", "pelas", "para",
    "com", "sem", "sob", "sobre", "entre", "contra", "ate", "desde", "e", "ou", "mas",
    "se", "porque", "como", "que", "quando", "onde", "quem", "qual", "quais", "cujo",
    "eu", "tu", "ele", "ela", "nos", "vos", "eles", "elas", "meu", "minha", "seu", "sua",
    "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas", "aquele", "aquela",
    "isso", "isto", "aquilo", "ser", "foi", "era", "foram", "sao", "esta", "estao",
    "ter", "tem", "tinha", "mais", "menos", "muito", "pouco", "ja", "nao", "sim", "tambem",
}


class TextCleaner:
    """Mecanismos de normalização textual e higienização para NLP."""

    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove tags HTML mantendo quebras de linha."""
        if not text:
            return ""
        clean = re.sub(r"<[^>]+>", " ", text)
        return clean

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Normaliza caracteres, espaços em branco e formato Unicode."""
        if not text:
            return ""

        # Remove HTML se houver
        text = cls.remove_html_tags(text)

        # Normaliza Unicode (NFKC)
        text = unicodedata.normalize("NFKC", text)

        # Padroniza quebras de linha
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Limpa espaços horizontais de cada linha
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Remove múltiplas linhas vazias consecutivas
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        return text.strip()

    @classmethod
    def extract_keywords(cls, text: str, top_n: int = 10) -> List[str]:
        """Extrai termos mais relevantes baseando-se em frequência e remoção de stopwords."""
        if not text:
            return []

        # Remove acentos para comparação léxica uniforme
        normalized_str = "".join(
            c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn"
        )

        words = re.findall(r"\b[a-z]{3,}\b", normalized_str)
        filtered_words = [w for w in words if w not in PORTUGUESE_STOPWORDS]

        # Contagem de frequência simples
        counts = {}
        for w in filtered_words:
            counts[w] = counts.get(w, 0) + 1

        sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_n]]
