"""Dump the current state of the vector store for manual verification.
Usage: python3 scripts/inspect_store.py [ticket_id]
"""
import sys
import os

# Add project root (parent of this script's directory) to the path,
# so `app` resolves regardless of where this script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vectorstore import get_collection, collection_count


def summary():
    col = get_collection()
    total = collection_count()
    print(f"Collection metadata: {col.metadata}")
    print(f"Total chunks in store: {total}\n")
    result = col.get(include=["metadatas"])
    for id_, meta in zip(result["ids"], result["metadatas"]):
        print(f"{id_:12} | {meta.get('category', ''):12} | {meta.get('priority', ''):15} | {meta.get('location', '')}")

def show_one(ticket_id: str):
    col = get_collection()
    result = col.get(ids=[ticket_id], include=["documents", "metadatas"])
    if not result["ids"]:
        print(f"No chunk found for id={ticket_id}")
        return
    print("ID:", result["ids"][0])
    print("METADATA:", result["metadatas"][0])
    print("TEXT:\n" + result["documents"][0])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show_one(sys.argv[1])
    else:
        summary()