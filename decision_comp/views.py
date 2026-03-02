"""
Django API endpoints for the Decision Companion (headless backend).
Streamlit (or other clients) call these to run research and calculation.
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .graph import run_ai_research, run_calculation_and_synthesis
from .llm_services import classify_criteria_nature
from .rag import build_rag_context
from .models import (
    CriterionSchema,
    DecisionInputState,
    FinalScoresState,
    OptionCriterionScore,
    OptionSchema,
    TriangularFuzzyNumber,
)


def _parse_json_body(request) -> dict:
    """Parse JSON from request body; return 400 on error."""
    try:
        return json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return None


def _research_payload_to_inputs(data: dict) -> DecisionInputState:
    """Build DecisionInputState from API payload for research."""
    problem = data.get("problem_description") or ""
    options_data = data.get("options") or []
    criteria_data = data.get("criteria") or []

    if len(options_data) < 2:
        raise ValueError("At least 2 options are required.")
    if len(criteria_data) < 1:
        raise ValueError("At least 1 criterion is required.")

    options = [
        OptionSchema(name=o["name"], description=o.get("description"))
        for o in options_data
    ]
    criteria = []
    for c in criteria_data:
        weight = int(c.get("weight", 5))
        if not (1 <= weight <= 10):
            raise ValueError(f"Criterion weight must be 1–10, got {weight}.")
        kind = (c.get("kind") or "benefit").lower()
        if kind not in ("benefit", "cost"):
            kind = "benefit"
        criteria.append(
            CriterionSchema(
                name=c["name"],
                weight=weight,
                description=c.get("description"),
                kind=kind,
            )
        )

    return DecisionInputState(
        problem_description=problem,
        options=options,
        criteria=criteria,
    )


def _scores_list_to_final_scores(
    inputs: DecisionInputState,
    scores_list: List[Dict[str, Any]],
) -> FinalScoresState:
    """Build FinalScoresState from a list of score objects (e.g. from Streamlit grid)."""
    scores: Dict[Tuple[str, str], OptionCriterionScore] = {}
    for row in scores_list:
        opt_name = row["option_name"]
        crit_name = row["criterion_name"]
        l_val = float(row["l"])
        m_val = float(row["m"])
        u_val = float(row["u"])
        justification = (row.get("justification") or "").strip() or "No justification provided."
        tfn = TriangularFuzzyNumber(l=l_val, m=m_val, u=u_val)
        key = (opt_name, crit_name)
        scores[key] = OptionCriterionScore(
            option_name=opt_name,
            criterion_name=crit_name,
            score_tfn=tfn,
            justification=justification,
        )

    return FinalScoresState(scores=scores)


@csrf_exempt
@require_http_methods(["POST"])
def api_research(request):
    """
    POST /api/research/
    Body: { problem_description, options: [{name, description?, documents?: [{filename, content_base64}]}], criteria: [...] }
    Returns:
      {
        "scores": [{ option_name, criterion_name, l, m, u, justification }],
        "criteria": [{ name, weight, description, kind, rationale }]
      }
    """
    try:
        data = _parse_json_body(request)
        if data is None:
            return JsonResponse({"error": "Invalid JSON body."}, status=400)

        try:
            inputs = _research_payload_to_inputs(data)
        except (KeyError, ValueError, TypeError) as e:
            return JsonResponse({"error": str(e)}, status=400)

        try:
            # First, classify criteria as benefit/cost using the LLM.
            nature_batch = classify_criteria_nature(inputs)
        except Exception as e:
            return JsonResponse({"error": f"Criterion classification failed: {e}"}, status=500)

        # Map classification results by criterion name.
        nature_by_name: Dict[str, Any] = {}
        for item in nature_batch.items:
            nature_by_name[item.criterion_name] = item

        # Optional RAG: parse documents per option (PDFs as base64), build context, pass to research.
        rag_context = ""
        options_data = data.get("options") or []
        documents_per_option: List[List[Tuple[str, bytes]]] = []
        for i in range(len(inputs.options)):
            opt_data = options_data[i] if i < len(options_data) else {}
            doc_list: List[Tuple[str, bytes]] = []
            for item in opt_data.get("documents") or []:
                if not isinstance(item, dict):
                    continue
                name = item.get("filename") or "document.pdf"
                b64 = item.get("content_base64")
                if not b64:
                    continue
                try:
                    doc_bytes = base64.b64decode(b64)
                    if doc_bytes:
                        doc_list.append((name, doc_bytes))
                except Exception:
                    continue
            documents_per_option.append(doc_list)
        try:
            rag_context = build_rag_context(
                documents_per_option,
                inputs.options,
                inputs.criteria,
                inputs.problem_description,
            )
        except Exception as e:
            logger.exception("RAG context build failed: %s", e)
            rag_context = ""

        # Run AI research to get fuzzy scores (with optional RAG context).
        try:
            ai_result = run_ai_research(inputs, rag_context=rag_context or None)
        except Exception as e:
            return JsonResponse({"error": f"AI research failed: {e}"}, status=500)

        # If user uploaded docs but RAG produced no context, add a warning so UI can inform them.
        had_docs = any(doc_list for doc_list in documents_per_option)
        rag_warning = (
            "PDF context could not be generated; scores may not reflect uploaded documents."
            if (had_docs and not (rag_context and rag_context.strip()))
            else None
        )

        # Serialize scores to a list (JSON does not support tuple keys).
        scores_list = []
        for (opt_name, crit_name), score_obj in ai_result.scores.items():
            scores_list.append(
                {
                    "option_name": opt_name,
                    "criterion_name": crit_name,
                    "l": score_obj.score_tfn.l,
                    "m": score_obj.score_tfn.m,
                    "u": score_obj.score_tfn.u,
                    "justification": score_obj.justification,
                }
            )

        # Serialize criteria with LLM-derived kinds and rationales.
        criteria_list = []
        for crit in inputs.criteria:
            nature = nature_by_name.get(crit.name)
            kind = crit.kind
            rationale = ""
            if nature is not None:
                kind = nature.kind
                rationale = nature.rationale
            criteria_list.append(
                {
                    "name": crit.name,
                    "weight": crit.weight,
                    "description": crit.description,
                    "kind": kind,
                    "rationale": rationale,
                }
            )

        payload = {"scores": scores_list, "criteria": criteria_list}
        if rag_warning:
            payload["rag_warning"] = rag_warning
        return JsonResponse(payload)
    except Exception as e:
        # Last-resort guard: surface unexpected errors as structured JSON
        # instead of a generic HTML 500 page.
        return JsonResponse(
            {"error": f"Unexpected server error in /api/research/: {e}"},
            status=500,
        )


@csrf_exempt
@require_http_methods(["POST"])
def api_calculate(request):
    """
    POST /api/calculate/
    Body: { problem_description, options: [...], criteria: [...], scores: [{ option_name, criterion_name, l, m, u, justification }] }
    Returns: { winner, loser, explanation, options: [{ option_name, closeness_coefficient, distance_to_fpis, distance_to_fnis }] }
    """
    data = _parse_json_body(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    try:
        inputs = _research_payload_to_inputs(data)
    except (KeyError, ValueError, TypeError) as e:
        return JsonResponse({"error": str(e)}, status=400)

    scores_list = data.get("scores")
    if not scores_list:
        return JsonResponse({"error": "Missing or empty 'scores' array."}, status=400)

    try:
        final_scores = _scores_list_to_final_scores(inputs, scores_list)
    except (KeyError, ValueError, TypeError) as e:
        return JsonResponse({"error": f"Invalid score data: {e}"}, status=400)

    try:
        topsis_result, explanation, intermediates = run_calculation_and_synthesis(inputs, final_scores)
    except Exception as e:
        logger.exception("Calculation and synthesis failed")
        return JsonResponse({"error": f"Calculation and synthesis failed: {e}"}, status=500)

    options_out = [
        {
            "option_name": o.option_name,
            "closeness_coefficient": o.closeness_coefficient,
            "distance_to_fpis": o.distance_to_fpis,
            "distance_to_fnis": o.distance_to_fnis,
        }
        for o in topsis_result.options
    ]

    return JsonResponse({
        "winner": topsis_result.winner,
        "loser": topsis_result.loser,
        "explanation": explanation,
        "options": options_out,
        "intermediates": intermediates,
    })
