# import sys
# from pathlib import Path

# ROOT = Path(__file__).resolve()

# sys.path.append(str(ROOT))

# from pipeline.config import cfg
# import chromadb


# print("Chroma Path:")
# print(cfg.CHROMA_DB_PATH)

# client = chromadb.PersistentClient(
#     path=str(cfg.CHROMA_DB_PATH)
# )

# print("\nAvailable collections:")

# collections = client.list_collections()

# if not collections:
#     print("No collections found!")

# for c in collections:
#     print(
#         f"- {c.name} | Count: {c.count()}"
#     )

import chromadb
from sentence_transformers import SentenceTransformer
from pipeline.config import cfg


print("Loading embedding model...")
model = SentenceTransformer(cfg.EMBEDDING_MODEL)


client = chromadb.PersistentClient(
    path=str(cfg.CHROMA_DB_PATH)
)


collection = client.get_collection(
    cfg.HADITH_COLLECTION
)


query = "The best of people are those who are most beneficial to people"


embedding = model.encode(
    query,
    normalize_embeddings=True
).tolist()


result = collection.query(
    query_embeddings=[embedding],
    n_results=3,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)


for i in range(len(result["documents"][0])):

    print("\n====================")
    print("DOCUMENT:")
    print(result["documents"][0][i])

    print("\nMETADATA:")
    print(result["metadatas"][0][i])

    print("\nDISTANCE:")
    print(result["distances"][0][i])