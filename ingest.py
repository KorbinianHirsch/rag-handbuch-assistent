"""Liest ein PDF-Handbuch ein, zerlegt es in Abschnitte und speichert die Embeddings in Chroma.

Usage:
    python ingest.py handbuch.pdf
"""

import argparse

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import DB_DIR, embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", help="Pfad zum PDF-Handbuch")
    args = parser.parse_args()

    dokumente = PyPDFLoader(args.pdf).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(dokumente)

    Chroma.from_documents(documents=chunks, embedding=embeddings(), persist_directory=DB_DIR)
    print(f"{len(dokumente)} Seiten in {len(chunks)} Abschnitte zerlegt und nach {DB_DIR} gespeichert.")


if __name__ == "__main__":
    main()
