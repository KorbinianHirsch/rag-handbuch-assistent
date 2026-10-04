# KI-Handbuch-Assistent (RAG + Agent)

Beantwortet Fragen zu technischen PDF-Handbüchern (z.B. Maschinen-Bedienungsanleitungen) mit
Retrieval-Augmented Generation: Das Sprachmodell antwortet nur auf Basis der passenden Handbuchseiten
und nennt die Seitenzahl. Ein zusätzlicher Agent kombiniert die Handbuchsuche mit einer
(simulierten) ERP-Lagerabfrage.

## Funktionsweise

```
PDF ──► Seiten laden ──► Chunks (1000 Zeichen, 200 Overlap) ──► OpenAI-Embeddings ──► Chroma (lokal)

Frage ──► Ähnlichkeitssuche in Chroma (top-k) ──► Kontext mit Seitenzahlen ──► GPT-4o-mini ──► Antwort + Seite
```

- **Gegen Halluzinationen:** Der Prompt erlaubt nur Antworten aus dem gefundenen Kontext. Steht die
  Antwort nicht im Handbuch, sagt der Assistent das ausdrücklich, statt etwas zu erfinden.
- **Quellenangabe:** Jeder Kontextabschnitt trägt seine Seitenzahl, das Modell muss sie zitieren.
- **Agent (ReAct, LangGraph):** Entscheidet selbst, welche Tools eine Frage braucht. Beispiel:
  *„Haben wir noch eine Pumpe auf Lager? Und wie setze ich das Gerät zurück?“* führt zu einer
  Lagerabfrage **und** einer Handbuchsuche, deren Ergebnisse zu einer Antwort zusammengefasst werden.

## Dateien

| Datei | Inhalt |
|---|---|
| `rag.py` | Gemeinsame Bausteine: Modelle, Vektordatenbank, Prompt, Kontextsuche |
| `ingest.py` | PDF einlesen, zerlegen, Embeddings in `chroma_db/` speichern |
| `app.py` | Chat-Oberfläche (Streamlit) |
| `agent.py` | ReAct-Agent mit Handbuchsuche und ERP-Lagerabfrage |

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env          # OpenAI-API-Key eintragen
```

## Verwendung

```powershell
python ingest.py handbuch.pdf   # einmalig: Vektordatenbank aufbauen
streamlit run app.py            # Chat im Browser
python agent.py "Haben wir noch eine Pumpe auf Lager?"
```

Das Handbuch selbst ist nicht im Repository enthalten (Urheberrecht des Herstellers). Jedes
Text-PDF funktioniert, z.B. die Bedienungsanleitung eines Haushaltsgeräts.

## Tech-Stack

Python · LangChain / LangGraph · OpenAI (`text-embedding-3-small`, `gpt-4o-mini`) · Chroma · Streamlit
