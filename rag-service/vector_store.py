import weaviate
from langchain_weaviate import WeaviateVectorStore

from config import settings
from embeddings import get_embeddings

#_COLLECTION_NAME = "helikon_docs"

# lazy singletons
#_client: chromadb.ClientAPI | None = None
#_collection: chromadb.Collection | None = None
#_vectorstore: Chroma | None = None


_INDEX_NAME = "HelikonDocs"

_client: weaviate.WeaviateClient | None = None
_vectorstore: WeaviateVectorStore | None = None

def _get_client() -> weaviate.WeaviateClient:
    global _client
    if _client is None:
        if not settings.weaviate_api_key == "":
            auth_credentials = weaviate.auth.AuthApiKey(settings.weaviate_api_key)
            _client = weaviate.connect_to_custom(
                http_host=settings.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
                http_port=8080,
                http_secure=False,
                grpc_port=50051,
                grpc_secure=False,
                auth_credentials=auth_credentials,
            )
        else:
            _client = weaviate.connect_to_local(
                host=settings.weaviate_url.replace("http://", "").replace("https://", "").split(":")[0],
                port=8080,
                grpc_port=50051
            )
    return _client


def get_collection():
    client  = _get_client()
    if not client.is_ready():
        raise RuntimeError("Weaviate client is not ready. Please check the connection.")
    return _client


def get_vectorstore():
    """LangChain chroma wrapper used for add/similarity-search."""
    global _vectorstore
    if _vectorstore is None:
        client = _get_client()
        _vectorstore = WeaviateVectorStore(
            client=client,
            index_name=_INDEX_NAME,
            text_key="text",
            embedding=get_embeddings(),
        )
    return _vectorstore
