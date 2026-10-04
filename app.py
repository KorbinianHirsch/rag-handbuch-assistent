"""Chat-Oberfläche für Fragen an die Handbücher.

Run:
    streamlit run app.py
"""

import streamlit as st

from rag import PROMPT, llm, load_db, search_context

st.set_page_config(page_title="Handbuch-Assistent", page_icon="🤖")
st.title("🤖 Handbuch-Assistent")
st.markdown("Frag mich alles zu den Handbüchern. Ich liefere dir die Antwort inklusive Seitenzahl.")


@st.cache_resource
def load_system():
    return load_db(), PROMPT | llm()


vektor_datenbank, chain = load_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if frage := st.chat_input("Was möchtest du aus dem Handbuch wissen?"):
    st.session_state.messages.append({"role": "user", "content": frage})
    with st.chat_message("user"):
        st.markdown(frage)

    with st.chat_message("assistant"):
        with st.spinner("Durchsuche Handbücher..."):
            kontext_text = search_context(vektor_datenbank, frage, k=4)
            antwort = chain.invoke({"kontext": kontext_text, "frage": frage})
            st.markdown(antwort.content)

    st.session_state.messages.append({"role": "assistant", "content": antwort.content})
