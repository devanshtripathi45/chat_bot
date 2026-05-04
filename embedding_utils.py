import unicodedata

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SYMBOL_REPLACEMENTS = str.maketrans(
    {
        "\u2022": "-",
        "\u2113": "l",
        "\u2225": "||",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2192": "->",
        "\u2190": "<-",
        "\u2212": "-",
        "\u2014": "-",
        "\u00d7": "x",
        "\u00f7": "/",
    }
)


def normalize_text(text):
    cleaned = (text or "").translate(SYMBOL_REPLACEMENTS)
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    return cleaned.strip()


class NormalizingHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        cleaned_texts = [normalize_text(text) for text in texts]
        return super().embed_documents(cleaned_texts)

    def embed_query(self, text):
        return super().embed_query(normalize_text(text))


def build_embeddings():
    try:
        return NormalizingHuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as exc:
        message = str(exc)
        if (
            "No connection could be made" in message
            or "actively refused it" in message
            or "client has been closed" in message
        ):
            raise RuntimeError(
                "Embedding model download nahi ho pa raha. Internet/proxy allow karke phir command dubara chalao."
            ) from exc
        raise
