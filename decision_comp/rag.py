"""
RAG (Retrieval-Augmented Generation) for Decision Companion.
Builds an ephemeral Chroma collection from uploaded PDFs and returns retrieved context
for the research LLM. Uses a local Hugging Face embedding model (sentence-transformers)
for vector search. First request may be slower while the model is downloaded.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import CriterionSchema, OptionSchema


def _get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Return the local Hugging Face embedding function for Chroma."""
    model_name = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    device = (os.getenv("RAG_EMBEDDING_DEVICE") or "cpu").lower()
    if device not in ("cpu", "cuda"):
        device = "cpu"
    return SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device=device,
        normalize_embeddings=False,
    )


def build_rag_context(
    documents_per_option: List[List[Tuple[str, bytes]]],
    options: List[OptionSchema],
    criteria: List[CriterionSchema],
    problem_description: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 12,
) -> str:
    """
    Load PDFs per option, chunk, embed into an ephemeral Chroma collection, and retrieve
    relevant context per (option, criterion). Returns a single string with one section per
    (option_index, criterion_index), e.g. [Context for Option i (name) - Criterion: name].
    Similarity search is criterion-specific so chunks most relevant to each criterion
    (e.g. Communication, Technical skills) are retrieved for scoring.
    """
    num_options = len(options)
    if num_options == 0:
        return ""

    # (chunk_text, filename, doc_index) with doc_index = option_index
    all_chunks: List[Tuple[str, str, int]] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    for option_index, doc_list in enumerate(documents_per_option):
        for (filename, content) in (doc_list or []):
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".pdf", delete=False, prefix="decision_rag_"
                ) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    for doc in docs:
                        text = doc.page_content.strip()
                        if text:
                            chunks = splitter.split_text(text)
                            for c in chunks:
                                all_chunks.append((c, filename, option_index))
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                continue

    # Ephemeral Chroma with local Hugging Face embedding.
    client = chromadb.EphemeralClient()
    collection_name = f"rag_{uuid.uuid4().hex[:16]}"
    collection = client.create_collection(
        name=collection_name,
        embedding_function=_get_embedding_function(),
    )

    if all_chunks:
        ids = [f"chunk_{i}" for i in range(len(all_chunks))]
        texts = [t for t, _, _ in all_chunks]
        metadatas = [
            {"doc_index": d, "option_name": options[d].name}
            for _, _, d in all_chunks
        ]
        collection.add(documents=texts, ids=ids, metadatas=metadatas)

    # Build output: one section per (option, criterion) with criterion-specific similarity search
    num_criteria = len(criteria)
    out_parts = []
    for option_index in range(num_options):
        opt = options[option_index]
        doc_chunk_count = sum(1 for _, _, d in all_chunks if d == option_index)
        for criterion_index in range(num_criteria):
            crit = criteria[criterion_index]
            section_header = (
                f"[Context for Option {option_index} ({opt.name}) - Criterion: {crit.name}]"
            )
            if doc_chunk_count == 0:
                out_parts.append(f"{section_header}\nNo uploaded document for this option.")
                continue
            # Put criterion first so chunks containing the criterion or synonyms rank higher
            query_parts = [
                f"Criterion: {crit.name}. {crit.description or ''}",
                f"Option: {opt.name}. {opt.description or ''}",
                problem_description,
            ]
            query = " ".join(query_parts).strip()
            n_results = min(top_k, doc_chunk_count)
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"option_name": opt.name},
            )
            if not results or not results.get("documents") or not results["documents"][0]:
                out_parts.append(
                    f"{section_header}\n(No chunks retrieved for this option and criterion.)"
                )
                continue
            chunk_texts = results["documents"][0]
            out_parts.append(f"{section_header}\n" + "\n\n".join(chunk_texts))

    return "\n\n---\n\n".join(out_parts)
