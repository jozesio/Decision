from __future__ import annotations

from unittest.mock import patch

from django.test import Client, TestCase

from decision_comp.models import (
    AIResearchResultState,
    OptionCriterionScore,
    TriangularFuzzyNumber,
)


class ApiResearchTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()

    def _build_payload(self) -> dict:
        return {
            "problem_description": "Test decision",
            "options": [
                {"name": "Option A", "description": ""},
                {"name": "Option B", "description": ""},
            ],
            "criteria": [
                {
                    "name": "Cost",
                    "weight": 5,
                    "description": "Total cost",
                    "kind": "cost",
                }
            ],
        }

    @patch("decision_comp.views.run_ai_research")
    @patch("decision_comp.views.classify_criteria_nature")
    def test_api_research_success_with_stubbed_llm(
        self,
        mock_classify_criteria_nature,
        mock_run_ai_research,
    ) -> None:
        payload = self._build_payload()

        class NatureItem:
            def __init__(self, name: str, kind: str, rationale: str) -> None:
                self.criterion_name = name
                self.kind = kind
                self.rationale = rationale

        class NatureBatch:
            def __init__(self, items) -> None:
                self.items = items

        mock_classify_criteria_nature.return_value = NatureBatch(
            [NatureItem("Cost", "cost", "Lower is better.")]
        )

        scores = {
            ("Option A", "Cost"): OptionCriterionScore(
                option_name="Option A",
                criterion_name="Cost",
                score_tfn=TriangularFuzzyNumber(l=4.0, m=5.0, u=6.0),
                justification="Stub score.",
            ),
            ("Option B", "Cost"): OptionCriterionScore(
                option_name="Option B",
                criterion_name="Cost",
                score_tfn=TriangularFuzzyNumber(l=7.0, m=8.0, u=9.0),
                justification="Stub score.",
            ),
        }
        mock_run_ai_research.return_value = AIResearchResultState(scores=scores)

        response = self.client.post(
            "/api/research/",
            data=payload,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("scores", data)
        self.assertIn("criteria", data)
        self.assertEqual(len(data["scores"]), 2)

        first = data["scores"][0]
        for key in ("option_name", "criterion_name", "l", "m", "u", "justification"):
            self.assertIn(key, first)

