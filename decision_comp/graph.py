from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langgraph.graph import StateGraph, END

from .llm_services import run_research_llm, SynthesisOutput
from .models import (
    AIResearchResultState,
    DecisionInputState,
    FinalScoresState,
    FuzzyOptionResult,
    FuzzyTopsisResult,
    GraphState,
    NormalizedTriangularFuzzyNumber,
    OptionCriterionScore,
    TriangularFuzzyNumber,
    WeightedTriangularFuzzyNumber,
)


def ai_research_node(state: GraphState) -> GraphState:
    if state.inputs is None:
        raise ValueError("GraphState.inputs must be set before calling ai_research_node.")

    research_output = run_research_llm(state.inputs, rag_context=state.rag_context)

    scores: Dict[Tuple[str, str], OptionCriterionScore] = {}

    options = state.inputs.options
    criteria = state.inputs.criteria
    num_options = len(options)
    num_criteria = len(criteria)

    for item in research_output.items:
        if not (0 <= item.option_index < num_options):
            raise ValueError(f"AI returned invalid option_index {item.option_index}.")
        if not (0 <= item.criterion_index < num_criteria):
            raise ValueError(f"AI returned invalid criterion_index {item.criterion_index}.")

        opt_name = options[item.option_index].name
        crit_name = criteria[item.criterion_index].name
        key = (opt_name, crit_name)
        scores[key] = OptionCriterionScore(
            option_name=opt_name,
            criterion_name=crit_name,
            score_tfn=item.score,
            justification=item.justification,
        )

    state.ai_scores = AIResearchResultState(scores=scores)
    return state


def _normalize_fuzzy_matrix(
    inputs: DecisionInputState,
    final_scores: FinalScoresState,
) -> Dict[Tuple[str, str], NormalizedTriangularFuzzyNumber]:
    """
    Normalize the fuzzy decision matrix so all values lie in [0, 1].

    - Benefit criteria: divide each TFN (l, m, u) by the maximum upper bound u
      across all options for that criterion.
    - Cost criteria: flip each TFN using the minimum lower bound, then scale
      the flipped TFNs by their maximum upper bound so the result is in [0, 1].
    """
    normalized: Dict[Tuple[str, str], NormalizedTriangularFuzzyNumber] = {}

    for crit in inputs.criteria:
        # Collect TFNs for this criterion across all options.
        tfn_by_option: Dict[str, TriangularFuzzyNumber] = {}
        for opt in inputs.options:
            key = (opt.name, crit.name)
            if key not in final_scores.scores:
                raise ValueError(f"Missing score for option '{opt.name}' and criterion '{crit.name}'.")
            tfn_by_option[opt.name] = final_scores.scores[key].score_tfn

        if crit.kind == "benefit":
            # Benefit criterion: divide each TFN by the maximum upper bound u across all options.
            max_u = max(t.u for t in tfn_by_option.values())
            if max_u <= 0.0:
                raise ValueError(f"Maximum upper bound for criterion '{crit.name}' must be positive.")

            for opt in inputs.options:
                tfn = tfn_by_option[opt.name]
                normalized[(opt.name, crit.name)] = NormalizedTriangularFuzzyNumber(
                    l=tfn.l / max_u,
                    m=tfn.m / max_u,
                    u=tfn.u / max_u,
                )
        else:
            # Cost criterion: use the requested formula (l_min/u, l_min/m, l_min/l).
            min_l = min(t.l for t in tfn_by_option.values())
            if min_l <= 0.0:
                raise ValueError(f"Minimum lower bound for cost criterion '{crit.name}' must be positive.")

            for opt in inputs.options:
                tfn = tfn_by_option[opt.name]
                if tfn.l <= 0.0 or tfn.m <= 0.0 or tfn.u <= 0.0:
                    raise ValueError(
                        f"Cost criterion '{crit.name}' has non-positive TFN components "
                        f"for option '{opt.name}', cannot flip."
                    )
                normalized[(opt.name, crit.name)] = NormalizedTriangularFuzzyNumber(
                    l=min_l / tfn.u,
                    m=min_l / tfn.m,
                    u=min_l / tfn.l,
                )

    return normalized


def _apply_weights(
    normalized: Dict[Tuple[str, str], NormalizedTriangularFuzzyNumber],
    inputs: DecisionInputState,
) -> Dict[Tuple[str, str], WeightedTriangularFuzzyNumber]:
    """
    Multiply each normalized TFN by the corresponding criterion's crisp weight.
    """
    weighted: Dict[Tuple[str, str], WeightedTriangularFuzzyNumber] = {}

    weight_by_criterion: Dict[str, int] = {c.name: c.weight for c in inputs.criteria}

    for opt in inputs.options:
        for crit in inputs.criteria:
            key = (opt.name, crit.name)
            tfn = normalized[key]
            w = float(weight_by_criterion[crit.name])
            weighted[key] = WeightedTriangularFuzzyNumber(
                l=tfn.l * w,
                m=tfn.m * w,
                u=tfn.u * w,
            )

    return weighted


def _compute_fpis_fnis(
    weighted: Dict[Tuple[str, str], WeightedTriangularFuzzyNumber],
    inputs: DecisionInputState,
) -> Tuple[Dict[str, WeightedTriangularFuzzyNumber], Dict[str, WeightedTriangularFuzzyNumber]]:
    """
    Compute the Fuzzy Positive Ideal Solution (FPIS) and
    Fuzzy Negative Ideal Solution (FNIS) per criterion in the
    weighted fuzzy space.
    """
    fpis: Dict[str, WeightedTriangularFuzzyNumber] = {}
    fnis: Dict[str, WeightedTriangularFuzzyNumber] = {}

    for crit in inputs.criteria:
        l_vals = []
        m_vals = []
        u_vals = []
        for opt in inputs.options:
            key = (opt.name, crit.name)
            tfn = weighted[key]
            l_vals.append(tfn.l)
            m_vals.append(tfn.m)
            u_vals.append(tfn.u)

        fpis[crit.name] = WeightedTriangularFuzzyNumber(
            l=max(l_vals),
            m=max(m_vals),
            u=max(u_vals),
        )
        fnis[crit.name] = WeightedTriangularFuzzyNumber(
            l=min(l_vals),
            m=min(m_vals),
            u=min(u_vals),
        )

    return fpis, fnis


def _distance_between_tfn(
    a: WeightedTriangularFuzzyNumber,
    b: WeightedTriangularFuzzyNumber,
) -> float:
    """
    Euclidean distance between two TFNs using the vertex method.
    d(a,b) = sqrt( ( (al-bl)^2 + (am-bm)^2 + (au-bu)^2 ) / 3 ).
    """
    return (
        (
            (a.l - b.l) ** 2
            + (a.m - b.m) ** 2
            + (a.u - b.u) ** 2
        )
        / 3.0
    ) ** 0.5


def _serialize_intermediates(
    inputs: DecisionInputState,
    normalized: Dict[Tuple[str, str], NormalizedTriangularFuzzyNumber],
    weighted: Dict[Tuple[str, str], WeightedTriangularFuzzyNumber],
    fpis: Dict[str, WeightedTriangularFuzzyNumber],
    fnis: Dict[str, WeightedTriangularFuzzyNumber],
    option_results: List[FuzzyOptionResult],
) -> Dict[str, Any]:
    """Build a JSON-serializable dict of TOPSIS intermediates for the algorithm explanation page."""
    normalized_matrix = []
    for opt in inputs.options:
        for crit in inputs.criteria:
            key = (opt.name, crit.name)
            if key in normalized:
                n = normalized[key]
                normalized_matrix.append({"option_name": opt.name, "criterion_name": crit.name, "l": n.l, "m": n.m, "u": n.u})
    weighted_matrix = []
    for opt in inputs.options:
        for crit in inputs.criteria:
            key = (opt.name, crit.name)
            if key in weighted:
                w = weighted[key]
                weighted_matrix.append({"option_name": opt.name, "criterion_name": crit.name, "l": w.l, "m": w.m, "u": w.u})
    fpis_out = {crit.name: {"l": fpis[crit.name].l, "m": fpis[crit.name].m, "u": fpis[crit.name].u} for crit in inputs.criteria}
    fnis_out = {crit.name: {"l": fnis[crit.name].l, "m": fnis[crit.name].m, "u": fnis[crit.name].u} for crit in inputs.criteria}
    options_out = [
        {
            "option_name": o.option_name,
            "closeness_coefficient": o.closeness_coefficient,
            "distance_to_fpis": o.distance_to_fpis,
            "distance_to_fnis": o.distance_to_fnis,
        }
        for o in option_results
    ]
    return {
        "normalized_matrix": normalized_matrix,
        "weighted_matrix": weighted_matrix,
        "fpis": fpis_out,
        "fnis": fnis_out,
        "options": options_out,
    }


def compute_fuzzy_topsis(
    inputs: DecisionInputState,
    final_scores: FinalScoresState,
) -> Tuple[FuzzyTopsisResult, Dict[str, Any]]:
    """
    Deterministic Fuzzy TOPSIS implementation using triangular fuzzy numbers.
    Returns (result, intermediates_dict) for the algorithm explanation page.
    """
    # Ensure all required scores are present.
    for opt in inputs.options:
        for crit in inputs.criteria:
            key = (opt.name, crit.name)
            if key not in final_scores.scores:
                raise ValueError(f"Missing score for option '{opt.name}' and criterion '{crit.name}'.")

    normalized = _normalize_fuzzy_matrix(inputs, final_scores)
    weighted = _apply_weights(normalized, inputs)
    fpis, fnis = _compute_fpis_fnis(weighted, inputs)

    option_results_list: List[FuzzyOptionResult] = []

    for opt in inputs.options:
        d_plus = 0.0
        d_minus = 0.0
        for crit in inputs.criteria:
            key = (opt.name, crit.name)
            v_ij = weighted[key]
            d_plus += _distance_between_tfn(v_ij, fpis[crit.name])
            d_minus += _distance_between_tfn(v_ij, fnis[crit.name])

        denom = d_plus + d_minus
        if denom == 0.0:
            cc = 0.0
        else:
            cc = d_minus / denom

        option_results_list.append(
            FuzzyOptionResult(
                option_name=opt.name,
                distance_to_fpis=d_plus,
                distance_to_fnis=d_minus,
                closeness_coefficient=cc,
            )
        )

    sorted_options = sorted(
        option_results_list,
        key=lambda o: (-o.closeness_coefficient, o.option_name),
    )

    winner = sorted_options[0].option_name
    loser = sorted_options[-1].option_name if len(sorted_options) > 1 else None
    result = FuzzyTopsisResult(options=sorted_options, winner=winner, loser=loser)

    intermediates = _serialize_intermediates(inputs, normalized, weighted, fpis, fnis, sorted_options)
    return result, intermediates


def fuzzy_topsis_calculation_node(state: GraphState) -> GraphState:
    if state.inputs is None or state.final_scores is None:
        raise ValueError(
            "GraphState.inputs and GraphState.final_scores must be set before calling fuzzy_topsis_calculation_node."
        )

    result, _ = compute_fuzzy_topsis(state.inputs, state.final_scores)
    state.topsis_result = result
    return state


def synthesis_node(state: GraphState) -> GraphState:
    from .llm_services import run_synthesis_llm

    if state.inputs is None or state.topsis_result is None:
        raise ValueError("GraphState.inputs and GraphState.topsis_result must be set before calling synthesis_node.")

    synthesis: SynthesisOutput = run_synthesis_llm(state.inputs, state.topsis_result)
    state.explanation = synthesis.explanation
    return state


def build_decision_graph() -> StateGraph:
    """
    Build and return a LangGraph StateGraph for the automated parts of the workflow.
    Human-in-the-loop editing of scores happens outside of this graph in the CLI.
    """
    graph = StateGraph(GraphState)

    graph.add_node("ai_research", ai_research_node)
    graph.add_node("fuzzy_topsis_calculation", fuzzy_topsis_calculation_node)
    graph.add_node("synthesis", synthesis_node)

    graph.set_entry_point("ai_research")
    graph.add_edge("ai_research", "fuzzy_topsis_calculation")
    graph.add_edge("fuzzy_topsis_calculation", "synthesis")
    graph.add_edge("synthesis", END)

    return graph


def run_ai_research(
    inputs: DecisionInputState,
    rag_context: str | None = None,
) -> AIResearchResultState:
    """
    Convenience function: run only the AI research node.
    Intended to be called from the CLI or API before the human-in-the-loop step.
    If rag_context is provided (e.g. from RAG over uploaded PDFs), the research LLM
    will base fuzzy scores strictly on that context.
    """
    state = GraphState(inputs=inputs, rag_context=rag_context)
    updated_state = ai_research_node(state)
    if updated_state.ai_scores is None:
        raise RuntimeError("AI research did not produce any scores.")
    return updated_state.ai_scores


def run_calculation_and_synthesis(
    inputs: DecisionInputState,
    final_scores: FinalScoresState,
) -> Tuple[FuzzyTopsisResult, str, Dict[str, Any]]:
    """
    Convenience function: run deterministic Fuzzy TOPSIS calculation and synthesis explanation.
    Returns (topsis_result, explanation, intermediates) for the algorithm explanation page.
    """
    topsis_result, intermediates = compute_fuzzy_topsis(inputs, final_scores)
    state = GraphState(inputs=inputs, final_scores=final_scores, topsis_result=topsis_result)
    state = synthesis_node(state)
    if state.explanation is None:
        raise RuntimeError("Calculation and synthesis did not complete successfully.")
    return topsis_result, state.explanation, intermediates

