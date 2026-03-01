"""
NLP Service – NLTK-based text preprocessing for user queries.
"""
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data on first run
for pkg in ("punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"):
    nltk.download(pkg, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))


def preprocess_query(text: str) -> str:
    """
    Tokenize → lowercase → remove stopwords → lemmatize.
    Returns a cleaned string suitable for feeding into the model.
    """
    tokens   = word_tokenize(text.lower())
    filtered = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalnum() and tok not in _stop_words
    ]
    return " ".join(filtered)


def extract_medicine_names(text: str) -> list[str]:
    """
    Naïve NER: return capitalised tokens that are likely medicine names.
    Replace with spaCy / a dedicated NER model for production.
    """
    tokens = word_tokenize(text)
    return [tok for tok in tokens if tok[0].isupper() and len(tok) > 3]


def split_sentences(text: str) -> list[str]:
    return sent_tokenize(text)
