"""
Tests for RAG (build_rag_context): per-option documents, option_name metadata,
and criterion-specific similarity search.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import TestCase

from decision_comp.models import CriterionSchema, OptionSchema
from decision_comp.rag import build_rag_context


class MockDocument:
    def __init__(self, page_content: str):
        self.page_content = page_content


def _make_mock_chroma_collection() -> MagicMock:
    """Fake Chroma collection: stores docs by option_name, query returns filtered by where."""
    store: list[tuple[list[str], list[str], list[dict]]] = []  # (ids, documents, metadatas)

    def add(ids=None, documents=None, metadatas=None, **kwargs):
        store.append((list(ids or []), list(documents or []), list(metadatas or [])))

    def query(query_texts=None, n_results=10, where=None, **kwargs):
        if not store:
            return {"documents": [[]]}
        all_ids, all_docs, all_metas = store[0]
        want_name = (where or {}).get("option_name")
        if want_name is None:
            matched = all_docs[:n_results]
        else:
            matched = [d for d, m in zip(all_docs, all_metas) if m.get("option_name") == want_name]
            matched = matched[:n_results]
        return {"documents": [matched] if matched else [[]]}

    coll = MagicMock()
    coll.add = add
    coll.query = query
    return coll


class RAGTests(TestCase):
    def setUp(self) -> None:
        self.options = [
            OptionSchema(name="Candidate A", description=""),
            OptionSchema(name="Candidate B", description=""),
        ]
        self.criteria = [
            CriterionSchema(name="Communication", weight=5, description="Verbal and written"),
            CriterionSchema(name="Technical skills", weight=8, description="Programming and systems"),
        ]
        self.problem = "Choose the best candidate."

    def test_empty_options_returns_empty_string(self) -> None:
        result = build_rag_context(
            documents_per_option=[],
            options=[],
            criteria=self.criteria,
            problem_description=self.problem,
        )
        self.assertEqual(result, "")

    def test_no_documents_all_placeholders(self) -> None:
        # Two options, no PDFs for either
        documents_per_option: list = [[], []]
        result = build_rag_context(
            documents_per_option=documents_per_option,
            options=self.options,
            criteria=self.criteria,
            problem_description=self.problem,
        )
        self.assertIn("[Context for Option 0 (Candidate A) - Criterion: Communication]", result)
        self.assertIn("[Context for Option 0 (Candidate A) - Criterion: Technical skills]", result)
        self.assertIn("[Context for Option 1 (Candidate B) - Criterion: Communication]", result)
        self.assertIn("[Context for Option 1 (Candidate B) - Criterion: Technical skills]", result)
        self.assertIn("No uploaded document for this option.", result)
        # Should have 4 sections (2 options x 2 criteria)
        self.assertEqual(result.count("No uploaded document for this option."), 4)

    @patch("decision_comp.rag.PyPDFLoader")
    @patch("decision_comp.rag._get_embedding_function", return_value=MagicMock())
    @patch("decision_comp.rag.chromadb")
    def test_criterion_specific_sections_and_option_filter(
        self,
        mock_chromadb: MagicMock,
        _mock_embed: MagicMock,
        mock_pdf_loader_class: MagicMock,
    ) -> None:
        mock_coll = _make_mock_chroma_collection()
        mock_client = MagicMock()
        mock_client.create_collection.return_value = mock_coll
        mock_chromadb.EphemeralClient.return_value = mock_client

        # Mock PDF loader: option 0 content about communication, option 1 about technical
        opt0_text = "Led talk sessions and communication workshops. Presented at conferences."
        opt1_text = "Built RAG pipelines in Python. Completed NPTEL course on algorithms."
        mock_load = MagicMock(side_effect=[
            [MockDocument(opt0_text)],
            [MockDocument(opt1_text)],
        ])
        mock_loader = MagicMock()
        mock_loader.load = mock_load
        mock_pdf_loader_class.return_value = mock_loader

        # One "PDF" per option (dummy bytes; loader is mocked so file content not read)
        documents_per_option = [
            [("resume_a.pdf", b"dummy_pdf_bytes_a")],
            [("resume_b.pdf", b"dummy_pdf_bytes_b")],
        ]
        result = build_rag_context(
            documents_per_option=documents_per_option,
            options=self.options,
            criteria=self.criteria,
            problem_description=self.problem,
        )

        # All four section headers must be present
        self.assertIn("[Context for Option 0 (Candidate A) - Criterion: Communication]", result)
        self.assertIn("[Context for Option 0 (Candidate A) - Criterion: Technical skills]", result)
        self.assertIn("[Context for Option 1 (Candidate B) - Criterion: Communication]", result)
        self.assertIn("[Context for Option 1 (Candidate B) - Criterion: Technical skills]", result)

        # Option 0's content should appear only in Option 0 sections (option_name filter)
        self.assertIn("talk sessions", result)
        self.assertIn("communication workshops", result)
        # Option 1's content should appear only in Option 1 sections
        self.assertIn("RAG pipelines", result)
        self.assertIn("NPTEL", result)

        # Candidate A text should not appear in a section labeled Candidate B
        idx_b_comm = result.find("[Context for Option 1 (Candidate B) - Criterion: Communication]")
        idx_b_tech = result.find("[Context for Option 1 (Candidate B) - Criterion: Technical skills]")
        after_b_sections = result[idx_b_comm:]
        self.assertIn("RAG pipelines", after_b_sections)
        self.assertIn("NPTEL", after_b_sections)
        # Candidate A's section (before B) should contain A's text
        before_b = result[:idx_b_comm]
        self.assertIn("talk sessions", before_b)
        self.assertIn("Candidate A", before_b)

    @patch("decision_comp.rag.PyPDFLoader")
    @patch("decision_comp.rag._get_embedding_function", return_value=MagicMock())
    @patch("decision_comp.rag.chromadb")
    def test_one_option_with_docs_one_without(
        self, mock_chromadb: MagicMock, _mock_embed: MagicMock, mock_pdf_loader_class: MagicMock
    ) -> None:
        mock_coll = _make_mock_chroma_collection()
        mock_client = MagicMock()
        mock_client.create_collection.return_value = mock_coll
        mock_chromadb.EphemeralClient.return_value = mock_client

        mock_loader = MagicMock()
        mock_loader.load.return_value = [MockDocument("Only option 0 has content here.")]
        mock_pdf_loader_class.return_value = mock_loader

        # Option 0 has one doc, option 1 has none
        documents_per_option = [
            [("a.pdf", b"dummy")],
            [],
        ]
        result = build_rag_context(
            documents_per_option=documents_per_option,
            options=self.options,
            criteria=self.criteria,
            problem_description=self.problem,
        )

        # Option 0 sections should have retrieved content
        self.assertIn("Only option 0 has content here.", result)
        self.assertIn("[Context for Option 0 (Candidate A) - Criterion: Communication]", result)
        # Option 1 sections should be placeholders
        self.assertIn("[Context for Option 1 (Candidate B) - Criterion: Communication]", result)
        self.assertIn("No uploaded document for this option.", result)
