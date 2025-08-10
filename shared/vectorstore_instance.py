from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3")

def get_vectorstore_client():
    client = QdrantClient(
        url="http://localhost:6333",
        api_key=None
    )

    collection_name = "semantic_search"
    
    ensure_collection(client, collection_name, models, embeddings)

    vector_store = QdrantVectorStore(
        client = client,
        collection_name=collection_name,
        embedding=embeddings
    )

    return vector_store

def ensure_collection(client, collection_name, models, embeddings):
    try:
        sample_vec = embeddings.embed_query("determine vector size")
    except Exception:
        # fallback if embed_query fails for any reason
        sample_vec = embeddings.embed_documents(["determine vector size"])[0]

    dim = len(sample_vec)
    print(f"Detected embedding dimension = {dim}")

    # 2) Create collection if missing
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Created collection '{collection_name}' with size={dim}")