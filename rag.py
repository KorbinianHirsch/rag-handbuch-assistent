"""Gemeinsame Bausteine für Ingestion, Chat-App und Agent: Modelle, Vektordatenbank, Prompt."""

import warnings

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Das Modell darf nur aus dem gefundenen Kontext antworten, sonst halluziniert es Anleitungen.
PROMPT = PromptTemplate.from_template(
    """Du bist ein technischer Assistent für Maschinen.
    Beantworte die Frage AUSSCHLIESSLICH basierend auf dem folgenden Text.
    Nenne am Ende deiner Antwort immer die Seitenzahl aus dem Text.
    Wenn die Antwort nicht im Text steht, antworte: "Das weiß ich leider nicht, das steht nicht im Handbuch."

    Kontext (aus dem Handbuch):
    {kontext}

    Frage:
    {frage}

    Antwort:"""
)


def embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def llm() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


def load_db() -> Chroma:
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings())


def search_context(db: Chroma, frage: str, k: int = 4) -> str:
    """Holt die k ähnlichsten Abschnitte und schreibt die Seitenzahl davor, damit das Modell sie zitieren kann."""
    bloecke = db.similarity_search(frage, k=k)
    return "\n\n".join(f"--- Seite {b.metadata.get('page', '?')} ---\n{b.page_content}" for b in bloecke)
