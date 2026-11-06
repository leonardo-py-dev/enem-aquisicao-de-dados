from enem_pipeline.processing.cleaner import TextCleaner
from enem_pipeline.processing.classifier import ThematicClassifier


def test_cleaner_html_removal():
    """Testa se tags HTML são removidas mantendo o conteúdo íntegro."""
    raw_html = "<p>O <b>ENEM</b> aborda temas como a <i>desigualdade social</i>.</p>"
    clean = TextCleaner.remove_html_tags(raw_html)
    assert "<p>" not in clean
    assert "<b>" not in clean
    assert "ENEM" in clean
    assert "desigualdade social" in clean


def test_cleaner_normalization():
    """Testa normalização de espaços múltiplos e quebras de linha."""
    messy_text = "  Texto   com   muitos   espaços   \n\n\n\ne linhas extras.  "
    norm = TextCleaner.normalize_text(messy_text)
    assert "  " not in norm
    assert "\n\n\n" not in norm
    assert norm == "Texto com muitos espaços\n\ne linhas extras."


def test_cleaner_keyword_extraction():
    """Testa extração de palavras-chave eliminando stopwords em português."""
    text = "A inteligência artificial e os algoritmos estão transformando a tecnologia e o trabalho no Brasil."
    keywords = TextCleaner.extract_keywords(text, top_n=5)
    assert "inteligencia" in keywords or "algoritmos" in keywords or "tecnologia" in keywords
    assert "e" not in keywords
    assert "os" not in keywords
    assert "no" not in keywords


def test_thematic_classifier_matching():
    """Testa a classificação temática nos 10 eixos do ENEM."""
    classifier = ThematicClassifier()

    # Caso 1: Educação
    text_edu = "Os desafios da evasão escolar no ensino médio público e a valorização docente dos professores."
    predictions_edu = classifier.classify_text("Evasão Escolar no Brasil", text_edu)
    axes_edu = [p[0] for p in predictions_edu]
    assert "Educação" in axes_edu

    # Caso 2: Meio Ambiente
    text_amb = "O desmatamento na Amazônia e a crise climática afetam a sustentabilidade dos recursos hídricos."
    predictions_amb = classifier.classify_text("Preservação da Amazônia", text_amb)
    axes_amb = [p[0] for p in predictions_amb]
    assert "Meio Ambiente" in axes_amb

    # Caso 3: Multi-eixo (Tecnologia + Trabalho)
    text_tech_work = "A automação com inteligência artificial altera o emprego, o salário e a renda do trabalhador."
    predictions_multi = classifier.classify_text("Inteligência Artificial e Mercado de Trabalho", text_tech_work)
    axes_multi = [p[0] for p in predictions_multi]
    assert "Tecnologia" in axes_multi
    assert "Trabalho e Economia" in axes_multi
