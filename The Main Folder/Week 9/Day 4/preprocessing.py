import re
import html

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# Lemmatizer
lemmatizer = WordNetLemmatizer()


def to_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


# Reusable resources for the final pipeline
BASE_STOPWORDS = set(stopwords.words("english"))
NEGATIONS_TO_KEEP = {"not", "no", "nor", "never"}


def normalize_negation_contractions(text):
    return re.sub(r"n['’]t\b", " not", text)


def preprocess_text(
    text,
    *,
    remove_stopwords=True,
    keep_negations=True,
    lemmatize=True,
    keep_numbers=False,
):
    # 1) Ensure string input
    text = str(text)

    # 2) Decode HTML entities and remove HTML tags
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)

    # 3) Lowercase
    text = text.lower()

    # 4) Preserve negation
    text = normalize_negation_contractions(text)

    # 5) Tokenize
    tokens = word_tokenize(text)

    # 6) Keep alphabetic tokens
    if keep_numbers:
        tokens = [t for t in tokens if t.isalpha() or t.isdigit()]
    else:
        tokens = [t for t in tokens if t.isalpha()]

    # 7) Stop-word removal
    if remove_stopwords:
        active_stopwords = BASE_STOPWORDS.copy()

        if keep_negations:
            active_stopwords -= NEGATIONS_TO_KEEP

        tokens = [t for t in tokens if t not in active_stopwords]

    # 8) POS-aware lemmatization
    if lemmatize and tokens:
        tagged_tokens = pos_tag(tokens)

        tokens = [
            lemmatizer.lemmatize(
                token,
                to_wordnet_pos(tag)
            )
            for token, tag in tagged_tokens
        ]

    return tokens


def preprocess_to_string(text, **kwargs):
    return " ".join(
        preprocess_text(text, **kwargs)
    )