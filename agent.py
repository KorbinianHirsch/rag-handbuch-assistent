"""ReAct-Agent, der Handbuchsuche und eine (simulierte) ERP-Lagerabfrage kombiniert.

Usage:
    python agent.py "Haben wir noch eine Pumpe auf Lager? Und wie setze ich das Gerät zurück?"
"""

import sys

from langchain.agents import create_agent
from langchain_core.tools import tool

from rag import llm, load_db, search_context

vektor_datenbank = load_db()


# Die Docstrings der Tools sind wichtig: Das Modell liest sie, um zu entscheiden, WANN es ein Tool benutzt.
@tool
def suche_im_handbuch(frage: str) -> str:
    """Nutze dieses Werkzeug, um in technischen Handbüchern nach Anleitungen,
    Fehlercodes oder Wartungsintervallen zu suchen."""
    return search_context(vektor_datenbank, frage, k=3)


@tool
def pruefe_lagerbestand(artikel_name: str) -> str:
    """Nutze dieses Werkzeug, um den aktuellen Lagerbestand von Ersatzteilen oder Maschinen abzufragen."""
    # Platzhalter für einen echten API-Aufruf an ein ERP-System (z.B. SAP)
    lager_datenbank = {"magnetventil": 12, "dichtungsring": 0, "pumpe": 3}
    bestand = lager_datenbank.get(artikel_name.lower(), "Artikel im System nicht gefunden")
    return f"Der Lagerbestand für '{artikel_name}' beträgt: {bestand} Stück."


SYSTEM_PROMPT = """Du bist ein technischer Service-Agent.
Löse die Probleme des Nutzers, indem du deine Werkzeuge klug kombinierst.
Nenne bei Informationen aus dem Handbuch immer die Seitenzahl.
Zähle Dinge logisch auf und bleibe sachlich."""

DEFAULT_FRAGE = "Haben wir noch eine Pumpe auf Lager? Und wie setze ich das Gerät auf Werkseinstellungen zurück?"


def main():
    frage = " ".join(sys.argv[1:]) or DEFAULT_FRAGE
    # LangGraph baut die ReAct-Schleife (denken -> Tool aufrufen -> beobachten) automatisch
    agent = create_agent(model=llm(), tools=[suche_im_handbuch, pruefe_lagerbestand], system_prompt=SYSTEM_PROMPT)

    print(f"Nutzer: {frage}\n" + "-" * 50)
    ergebnis = agent.invoke({"messages": [("user", frage)]})
    print("Agent:", ergebnis["messages"][-1].content)


if __name__ == "__main__":
    main()
