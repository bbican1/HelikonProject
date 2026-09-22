from fastapi import APIRouter, Query
from pydantic import BaseModel
from collections import defaultdict

import weaviate.classes.query as wq

from vector_store import _get_client, _INDEX_NAME

router = APIRouter()


@router.get("/documents")
def list_documents(user_id: str = Query(...)):
    """List all documents (collapsed by doc_id) belonging to *user_id*."""
    client = _get_client()
    collection = client.collections.get(_INDEX_NAME)

    # Weavaite get() with where filter returns all matching chunks
    response = collection.query.fetch_objects(
        filters=wq.Filter.by_property("user_id").equal(user_id),
        limit=10000  # Ausreichendes Limit für alle Chunks des Users
    )

    # Collapse chunks → one entry per doc_id
    docs: dict[str, dict] = {}

    for obj in response.objects:
        meta = obj.properties
        doc_id =  meta.get("doc_id", "")

        if not doc_id:
            continue

        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("filename", ""),
                "uploaded_at": meta.get("uploaded_at", ""),
                "subject": meta.get("subject", ""),
                "chunk_count": 0,
            }
        docs[doc_id]["chunk_count"] += 1

    return list(docs.values())


class DeleteRequest(BaseModel):
    doc_ids: list[str]


@router.delete("/documents")
def delete_documents(req: DeleteRequest):
    """Delete all chunks belonging to the given doc_ids."""
    client = _get_client()
    collection = client.collections.get(_INDEX_NAME)

    deleted = 0
    for doc_id in req.doc_ids:
        result = collection.data.delete_many(
            where=wq.Filter.by_property("doc_id").equal(doc_id),
        )
        deleted += result.successful

    return {"deleted": deleted}
