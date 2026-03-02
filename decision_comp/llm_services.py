from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Literal, Optional, Set, Tuple

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .models import DecisionInputState, FuzzyTopsisResult, TriangularFuzzyNumber

logger = logging.getLogger(__name__)


class StrictBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        validate_assignment = True


class OptionCriterionAIResearch(StrictBaseModel):
    """
    AI-proposed score for a specific (option, criterion) pair, referenced by index.
    The indices must correspond exactly to the order of options and criteria
    provided in the decision input.
    """

    option_index: int = Field(..., ge=0, description="Zero-based index into the options list.")
    criterion_index: int = Field(..., ge=0, description="Zero-based index into the criteria list.")
    score: TriangularFuzzyNumber = Field(
        ...,
        description="Triangular fuzzy score (l, m, u) representing worst-case, most-likely, and best-case values.",
    )
    justification: str = Field(..., min_length=1)

    @field_validator("justification")
    def justification_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Justification must not be empty.")
        return v


class ResearchBatchOutput(StrictBaseModel):
    items: List[OptionCriterionAIResearch]


class ResearchItemLLMOutput(StrictBaseModel):
    """
    Raw LLM output for one (option, criterion) pair: requires chain-of-thought
    extracted_evidence and forbids fallback phrasing. Converted to OptionCriterionAIResearch
    before returning from run_research_llm.
    """
    option_index: int = Field(..., ge=0)
    criterion_index: int = Field(..., ge=0)
    extracted_evidence: str = Field(
        ...,
        min_length=1,
        description="1-2 exact phrases or concepts from the context that relate to the criterion.",
    )
    lower_bound: float = Field(..., ge=1.0, le=10.0, description="Worst-case score.")
    most_likely: float = Field(..., ge=1.0, le=10.0, description="Most realistic score.")
    upper_bound: float = Field(..., ge=1.0, le=10.0, description="Best-case score.")
    justification: str = Field(..., min_length=1)

    @field_validator("extracted_evidence", "justification")
    def strip_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Must not be empty.")
        return v.strip()

    @model_validator(mode="after")
    def bounds_ordered(self) -> "ResearchItemLLMOutput":
        if not (self.lower_bound <= self.most_likely <= self.upper_bound):
            raise ValueError("Require lower_bound <= most_likely <= upper_bound.")
        return self


class ResearchBatchOutputLLM(StrictBaseModel):
    """LLM-structured output with extracted_evidence per item; converted to ResearchBatchOutput."""
    items: List[ResearchItemLLMOutput]


class SynthesisOutput(StrictBaseModel):
    explanation: str = Field(..., min_length=1)


class CriterionNature(StrictBaseModel):
    criterion_name: str
    kind: Literal["benefit", "cost"]
    rationale: str = Field(..., min_length=1)


class CriterionNatureBatchOutput(StrictBaseModel):
    items: List[CriterionNature]


def get_llm(model_name: str = "llama-3.3-70b-versatile") -> ChatGroq:
    """
    Instantiate a ChatGroq LLM client.
    Raises a clear error if GROQ_API_KEY is not configured.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Please set it before running the decision companion."
        )

    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.1,
        max_retries=2,
        timeout=60,
    )


_SECTION_HEADER_RE = re.compile(
    r"^\[Context for Option (?P<option_index>\d+) \(.*?\) - Criterion: (?P<criterion>[^\]]+)\]$"
)
_CAUTIOUS_PHRASES = (
    "lack of information",
    "no information",
    "not mentioned",
    "insufficient data",
)
_WEAK_TERMS = {
    "skill",
    "skills",
    "ability",
    "abilities",
    "criterion",
    "performance",
    "general",
}


def _extract_section_map(rag_context: Optional[str]) -> Dict[Tuple[int, str], str]:
    """
    Parse RAG context into {(option_index, criterion_name_lower): section_text}.
    """
    out: Dict[Tuple[int, str], str] = {}
    if not rag_context or not rag_context.strip():
        return out
    for part in rag_context.split("\n\n---\n\n"):
        if not part.strip():
            continue
        lines = part.split("\n", 1)
        header = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        m = _SECTION_HEADER_RE.match(header)
        if not m:
            logger.debug("RAG section header did not match regex: %s", header)
            continue
        out[(int(m.group("option_index")), m.group("criterion").strip().lower())] = body
    logger.debug("Parsed %d RAG sections from context", len(out))
    return out


def _criterion_terms(criterion_name: str, criterion_description: Optional[str]) -> Set[str]:
    """Extract search terms from criterion name/description, including truncated
    stems (first 6 chars of words with 6+ chars) for broader matching."""
    terms: Set[str] = set()
    full_name = (criterion_name or "").strip().lower()
    if full_name:
        terms.add(full_name)
    combined = f"{criterion_name or ''} {criterion_description or ''}".lower()
    for tok in re.findall(r"[a-zA-Z]{4,}", combined):
        if tok in _WEAK_TERMS:
            continue
        terms.add(tok)
        if len(tok) >= 6:
            terms.add(tok[:6])
    return terms


def _count_matching_terms(section_text: str, terms: Set[str]) -> int:
    """Return how many distinct terms from *terms* appear in section_text."""
    if not section_text.strip():
        return 0
    lower = section_text.lower()
    return sum(1 for t in terms if t in lower)


def _section_has_direct_evidence(section_text: str, terms: Set[str]) -> bool:
    if not section_text.strip():
        return False
    lower = section_text.lower()
    if "no uploaded document for this option." in lower:
        return False
    if "(no chunks retrieved for this option and criterion.)" in lower:
        return False
    return any(t in lower for t in terms)


def _pick_evidence_snippet(section_text: str, terms: Set[str]) -> str:
    for line in section_text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        lower = clean.lower()
        if any(t in lower for t in terms):
            return clean[:180]
    return section_text.strip().splitlines()[0][:180] if section_text.strip() else "relevant evidence"


def _contains_cautious_phrase(text: str) -> bool:
    lower = (text or "").lower()
    return any(p in lower for p in _CAUTIOUS_PHRASES)


def _annotate_rag_context(
    rag_context: str,
    criteria: List,
) -> str:
    """Prepend keyword-match markers to each RAG section so the LLM cannot
    overlook evidence.  Returns the annotated context string."""
    if not rag_context or not rag_context.strip():
        return rag_context

    criteria_terms_by_name: Dict[str, Set[str]] = {}
    for crit in criteria:
        cname = (crit.name if hasattr(crit, "name") else crit.get("name", "")).strip().lower()
        cdesc = (crit.description if hasattr(crit, "description") else crit.get("description")) or ""
        if cname:
            criteria_terms_by_name[cname] = _criterion_terms(cname, cdesc)

    annotated_parts: List[str] = []
    for part in rag_context.split("\n\n---\n\n"):
        if not part.strip():
            annotated_parts.append(part)
            continue
        lines = part.split("\n", 1)
        header = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        m = _SECTION_HEADER_RE.match(header)
        if not m or not body:
            annotated_parts.append(part)
            continue

        crit_key = m.group("criterion").strip().lower()
        terms = criteria_terms_by_name.get(crit_key, set())
        if not terms:
            annotated_parts.append(part)
            continue

        lower_body = body.lower()
        matched = sorted(t for t in terms if t in lower_body)
        if matched:
            marker = (
                f'>>> KEYWORD MATCH: {", ".join(repr(t) for t in matched)} '
                "found in this section. This is strong evidence for this "
                "criterion — score accordingly.\n"
            )
            annotated_parts.append(f"{header}\n{marker}{body}")
        else:
            annotated_parts.append(part)

    return "\n\n---\n\n".join(annotated_parts)


def run_research_llm(
    decision_input: DecisionInputState,
    rag_context: Optional[str] = None,
) -> ResearchBatchOutput:
    """
    Call the Groq Llama model to research each option against each criterion.
    Uses a universal Expert Analyst persona: infers scores from document context and
    broad knowledge; forbids \"not known\" / \"insufficient data\"; injects the user's
    criterion description so scoring matches what the user cares about. Returns a
    strictly validated ResearchBatchOutput with triangular fuzzy scores (l, m, u).
    """
    llm = get_llm()

    system_prompt = (
        "You are a highly perceptive Expert Analyst and Evaluator. Your job is to score options "
        "based on provided text. You are evaluating every (option, criterion) pair. For each pair "
        "use the option name, the criterion name, and the User's Definition of that criterion "
        "(provided below). The provided context is grouped by option; use ONLY the section for "
        "that option when scoring that option.\n\n"
        "CRITICAL RULES - READ CAREFULLY:\n"
        "1. EXPLICIT AUTHORIZATION TO INFER: You are STRICTLY FORBIDDEN from outputting \"not known\", "
        "\"implies average\", or \"insufficient data\". Resumes and contracts rarely state soft skills "
        "directly. You MUST use your expert industry knowledge to infer the option's proficiency "
        "based on projects, roles, and achievements.\n"
        "2. MANDATORY PROXY MAPPING (EXAMPLES):\n"
        "   - If evaluating 'Communication' and the context mentions roles like \"Public Relations "
        "Core Member\" or coordinating media events, you MUST score them very high.\n"
        "   - If evaluating 'Technical Skills' and the context mentions completing an \"NPTEL online "
        "course\" on algorithms or building complex RAG pipelines, you MUST infer a solid technical "
        "baseline and score them high.\n"
        "3. BENEFIT OF THE DOUBT: Do not default to the middle. If you see evidence of teamwork, "
        "complex system design, or leadership, aggressively reward it in your fuzzy scores.\n"
        "4. MISSING DATA FALLBACK: If the context block for an option is completely empty (no files "
        "uploaded), you must use your broad industry knowledge regarding that option name to infer "
        "a reasonable score. Do NOT crash or output \"not known\".\n"
        "5. CONTEXT CONTAINS EVIDENCE: If the context section for this option and criterion contains "
        "the criterion name, obvious synonyms, or related phrases (e.g. \"communication skills\", "
        "\"communication\", \"presentations\", \"client relations\"), you MUST use that as extracted_evidence "
        "and assign HIGHER fuzzy scores. You are STRICTLY FORBIDDEN from saying \"lack of information\", "
        "\"no information\", or \"not mentioned\" when the context block for that option and criterion "
        "actually contains such text. Read the context section carefully before scoring.\n"
        "6. FILE CONTENT WEIGHT: When an option has uploaded file context (non-empty section), you MUST "
        "heavily weight that content for scoring. If ANY keyword or phrase in the uploaded file matches "
        "or relates to the criterion (e.g. criterion is 'Communication' and the file mentions "
        "\"communication\", \"communication skills\", \"presented\", \"spoke\", \"wrote\"), assign a HIGHER "
        "fuzzy score (typically most_likely >= 7). Keyword match in the file is strong evidence; reward it.\n\n"
        "OUTPUT SCHEMA (CHAIN OF THOUGHT): You must think step-by-step. Return one JSON object per "
        "(option_index, criterion_index) with: extracted_evidence (1-2 exact phrases or proxy "
        "concepts from the context that relate to the criterion), lower_bound, most_likely, "
        "upper_bound (floats 1-10, l <= m <= u), and justification (explain how the extracted "
        "evidence proves capability; do NOT say the information is missing).\n"
    )

    # Flatten options and criteria into an explicit, index-based description
    # that the model can follow without renaming anything.
    options_block_lines = []
    for idx, opt in enumerate(decision_input.options):
        options_block_lines.append(
            f"{idx}: {opt.name} - {opt.description or 'No additional description.'}"
        )
    options_block = "\n".join(options_block_lines)

    criteria_block_lines = []
    for idx, crit in enumerate(decision_input.criteria):
        user_def = (crit.description or "").strip() or "Not specified; use the criterion name and context."
        criteria_block_lines.append(
            f"Criterion {idx}: **{crit.name}** (weight {crit.weight}/10). "
            f"**User's description of this criterion:** {user_def}"
        )
    criteria_block = "\n".join(criteria_block_lines)

    # Annotate RAG sections with keyword-match markers before the LLM sees them.
    annotated_context = _annotate_rag_context(rag_context, decision_input.criteria) if rag_context else rag_context

    user_prompt = (
        f"Decision problem:\n{decision_input.problem_description}\n\n"
        f"Options (indexed by option_index):\n{options_block}\n\n"
        f"Criteria (indexed by criterion_index):\n{criteria_block}\n\n"
    )
    if annotated_context and annotated_context.strip():
        user_prompt += (
            "Provided Context (from uploaded files):\n"
            "<context>\n"
            f"{annotated_context.strip()}\n"
            "</context>\n\n"
            "Context sections are labeled [Context for Option i (name) - Criterion: criterion_name]. "
            "When scoring option_index i and criterion_index j, use ONLY the section for Option i "
            "and Criterion j. HEAVILY WEIGHT the contents of the uploaded file: if ANY keyword or "
            "phrase in that section matches or relates to the criterion (e.g. criterion 'Communication' "
            "and file contains 'communication', 'communication skills', 'presentations'), you MUST quote "
            "it in extracted_evidence and assign HIGHER fuzzy scores (most_likely >= 7). Even a single "
            "keyword match in the file is strong evidence — reward it. Never say 'lack of information' "
            "when the context contains such a match. Options with no or weak evidence in that section "
            "must receive lower scores.\n\n"
        )
    user_prompt += (
        "You MUST create one item for EVERY possible combination of option_index and criterion_index. "
        "Think step-by-step. For each item return a JSON object with:\n"
        "- option_index (integer)\n"
        "- criterion_index (integer)\n"
        "- extracted_evidence: List 1-2 exact phrases or proxy concepts from the context that "
        "relate to the criterion.\n"
        "- lower_bound: float 1-10 (worst-case)\n"
        "- most_likely: float 1-10 (most realistic)\n"
        "- upper_bound: float 1-10 (best-case); lower_bound <= most_likely <= upper_bound\n"
        "- justification: Explain how the extracted evidence proves their capability. Do NOT say "
        "the information is missing.\n\n"
        "Example item:\n"
        "{\n"
        '  \"option_index\": 0,\n'
        '  \"criterion_index\": 1,\n'
        '  \"extracted_evidence\": \"Public Relations Core Member; coordinated media events\",\n'
        '  \"lower_bound\": 7.0,\n'
        '  \"most_likely\": 8.5,\n'
        '  \"upper_bound\": 9.0,\n'
        '  \"justification\": \"PR role and media coordination demonstrate strong communication.\"\n'
        "}\n\n"
        "Do NOT omit any combinations. Do NOT output \"not known\", \"implies average\", or "
        "\"insufficient data\". Always produce a defensible score and justification.\n"
    )

    try:
        structured_llm = llm.with_structured_output(ResearchBatchOutputLLM)
        raw_result: ResearchBatchOutputLLM = structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except ValidationError as ve:
        raise RuntimeError(f"Failed to validate AI research output: {ve}") from ve
    except Exception as exc:
        raise RuntimeError(f"Error while calling research LLM: {exc}") from exc

    # Convert LLM output (extracted_evidence, lower_bound, most_likely, upper_bound) to pipeline
    # format (score as TriangularFuzzyNumber, justification).
    section_map = _extract_section_map(rag_context)
    evidence_flags: Dict[Tuple[int, int], bool] = {}
    items: List[OptionCriterionAIResearch] = []
    for raw in raw_result.items or []:
        lower_bound = raw.lower_bound
        most_likely = raw.most_likely
        upper_bound = raw.upper_bound
        justification = raw.justification.strip()

        # Deterministic guardrail: if retrieved section contains criterion evidence,
        # force evidence-led justification and avoid cautious/low scoring drift.
        # Uses tiered boosting: strong evidence (2+ term matches) gets higher floors.
        has_evidence = False
        if (
            0 <= raw.option_index < len(decision_input.options)
            and 0 <= raw.criterion_index < len(decision_input.criteria)
        ):
            crit = decision_input.criteria[raw.criterion_index]
            section_text = section_map.get(
                (raw.option_index, crit.name.strip().lower()),
                "",
            )
            terms = _criterion_terms(crit.name, crit.description)
            if _section_has_direct_evidence(section_text, terms):
                has_evidence = True
                match_count = _count_matching_terms(section_text, terms)
                if match_count >= 2:
                    lower_bound = max(lower_bound, 7.0)
                    most_likely = max(most_likely, 8.0)
                    upper_bound = max(upper_bound, 9.0)
                else:
                    lower_bound = max(lower_bound, 6.0)
                    most_likely = max(most_likely, 7.0)
                    upper_bound = max(upper_bound, 8.0)
                snippet = _pick_evidence_snippet(section_text, terms)
                needs_rewrite = (
                    _contains_cautious_phrase(justification)
                    or not any(t in justification.lower() for t in terms)
                )
                if needs_rewrite:
                    justification = (
                        f'Uploaded document evidence includes "{snippet}", '
                        f"which directly supports stronger {crit.name.lower()} and "
                        "justifies a higher fuzzy score."
                    )
                # Preserve TFN ordering after enforced floors.
                most_likely = max(most_likely, lower_bound)
                upper_bound = max(upper_bound, most_likely)
                lower_bound = min(lower_bound, 10.0)
                most_likely = min(most_likely, 10.0)
                upper_bound = min(upper_bound, 10.0)

        evidence_flags[(raw.option_index, raw.criterion_index)] = has_evidence
        score_tfn = TriangularFuzzyNumber(
            l=lower_bound,
            m=most_likely,
            u=upper_bound,
        )
        items.append(
            OptionCriterionAIResearch(
                option_index=raw.option_index,
                criterion_index=raw.criterion_index,
                score=score_tfn,
                justification=justification,
            )
        )

    # Comparative post-processing: for each criterion, if some options have
    # direct evidence and others do not, ensure evidence-bearing options outscore
    # non-evidence options by at least 1.0 on m.
    for crit_idx in range(len(decision_input.criteria)):
        crit_items_with = [it for it in items if it.criterion_index == crit_idx and evidence_flags.get((it.option_index, crit_idx))]
        crit_items_without = [it for it in items if it.criterion_index == crit_idx and not evidence_flags.get((it.option_index, crit_idx))]
        if crit_items_with and crit_items_without:
            max_no_evidence_m = max(it.score.m for it in crit_items_without)
            required_min_m = min(max_no_evidence_m + 1.0, 10.0)
            for it in crit_items_with:
                if it.score.m < required_min_m:
                    gap = required_min_m - it.score.m
                    new_l = min(it.score.l + gap, 10.0)
                    new_m = min(it.score.m + gap, 10.0)
                    new_u = min(it.score.u + gap, 10.0)
                    new_m = max(new_m, new_l)
                    new_u = max(new_u, new_m)
                    it.score = TriangularFuzzyNumber(l=new_l, m=new_m, u=new_u)

    result = ResearchBatchOutput(items=items)

    # Additional safety: ensure we have exactly one item per (option, criterion)
    # pair, with no duplicates and all indices in range.
    expected_items = len(decision_input.options) * len(decision_input.criteria)

    if len(items) != expected_items:
        raise RuntimeError(
            f"AI returned {len(items)} score items, but expected {expected_items} "
            f"(number_of_options x number_of_criteria)."
        )

    seen_pairs: set = set()
    for item in items:
        if not (0 <= item.option_index < len(decision_input.options)):
            raise RuntimeError(f"AI returned invalid option_index {item.option_index}.")
        if not (0 <= item.criterion_index < len(decision_input.criteria)):
            raise RuntimeError(f"AI returned invalid criterion_index {item.criterion_index}.")

        pair = (item.option_index, item.criterion_index)
        if pair in seen_pairs:
            raise RuntimeError(
                "AI returned duplicate scores for the same (option_index, criterion_index) pair."
            )
        seen_pairs.add(pair)

    return result


def classify_criteria_nature(decision_input: DecisionInputState) -> CriterionNatureBatchOutput:
    """
    Use the LLM to classify each criterion as 'benefit' (higher is better)
    or 'cost' (lower is better), with a short rationale.
    """
    llm = get_llm()

    system_prompt = (
        "You classify decision criteria as either 'benefit' (higher values are better) "
        "or 'cost' (lower values are better).\n"
        "Examples:\n"
        "- 'Battery life' -> benefit\n"
        "- 'Price', 'Latency', 'Response time', 'CO2 emissions' -> cost\n"
        "Base your decision only on the criterion names and descriptions provided."
    )

    criteria_block = "\n".join(
        f"- {idx}: {c.name} — {c.description or 'No additional description.'}"
        for idx, c in enumerate(decision_input.criteria)
    )

    user_prompt = (
        "Classify each of the following criteria:\n"
        f"{criteria_block}\n\n"
        "For each criterion, output:\n"
        "- criterion_name: exactly as given\n"
        "- kind: 'benefit' or 'cost'\n"
        "- rationale: ONE sentence justifying your choice"
    )

    try:
        structured_llm = llm.with_structured_output(CriterionNatureBatchOutput)
        result: CriterionNatureBatchOutput = structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except ValidationError as ve:
        raise RuntimeError(f"Failed to validate criterion nature output: {ve}") from ve
    except Exception as exc:
        raise RuntimeError(f"Error while classifying criteria nature: {exc}") from exc

    return result


def run_synthesis_llm(
    decision_input: DecisionInputState,
    topsis_result: FuzzyTopsisResult,
) -> SynthesisOutput:
    """
    Call the Groq Llama model to synthesize a two-part markdown explanation:
    1) Algorithmic Breakdown of why the winner achieved the highest closeness
       coefficient based on fuzzy scores, weights, and distances.
    2) Contextual Fit explaining how the winner aligns with the user's overall
       goal and the criterion descriptions.
    """
    llm = get_llm()

    system_prompt = (
        "You are an explanation engine. You are given a decision problem, "
        "options, criteria with weights, fuzzy scores (as triangular fuzzy numbers), "
        "and the results of a Fuzzy TOPSIS calculation.\n\n"
        "You MUST produce a markdown explanation with TWO sections, in this order:\n\n"
        "## Algorithmic Breakdown\n"
        "Explain step-by-step how the algorithm reached the selected winner. Use numbered steps:\n"
        "1. **Normalization**: How the raw fuzzy scores were normalized per criterion type "
        "(Benefit: higher is better, divide by max; Cost: lower is better, flip and scale). "
        "Mention that values end up in [0, 1].\n"
        "2. **Weighting**: How each criterion's weight (1–10) was applied to the normalized "
        "values to get the weighted matrix.\n"
        "3. **Ideal solutions**: How the Positive Ideal (FPIS) and Negative Ideal (FNIS) "
        "were determined from the weighted matrix (best and worst value per criterion).\n"
        "4. **Distances**: How each option's distance to FPIS and to FNIS was computed "
        "(vertex method: square root of average of squared differences of l, m, u). "
        "Use the actual distance numbers from the results where available.\n"
        "5. **Closeness coefficient (CC)**: How CC = distance_to_FNIS / (distance_to_FPIS + "
        "distance_to_FNIS) for each option; higher CC means closer to the positive ideal.\n"
        "6. **Why this winner**: Explain concretely why the chosen option has the highest CC "
        "and is therefore selected (refer to its distances and CC vs. the others).\n"
        "Stay strictly grounded in the math; use concrete numbers from the provided results.\n\n"
        "## Contextual Fit\n"
        "- Explain how this mathematically winning option fits the user's overall "
        "goal and the criterion descriptions.\n"
        "- Use only the provided goal/context and criterion descriptions; do NOT "
        "introduce new facts or speculative scenarios.\n"
        "Do NOT reference Fuzzy TOPSIS by name; just explain the reasoning."
    )

    options_text = "\n".join(
        f"- {opt.name}: {opt.description or 'No additional description.'}"
        for opt in decision_input.options
    )

    criteria_text = "\n".join(
        f"- {c.name} (weight {c.weight}/10): {c.description or 'No additional description.'}"
        for c in decision_input.criteria
    )

    breakdown_lines: List[str] = []
    for opt_result in topsis_result.options:
        breakdown_lines.append(
            f"Option {opt_result.option_name}: "
            f"distance_to_fpis={opt_result.distance_to_fpis:.4f}, "
            f"distance_to_fnis={opt_result.distance_to_fnis:.4f}, "
            f"closeness_coefficient={opt_result.closeness_coefficient:.4f}"
        )

    breakdown_text = "\n".join(breakdown_lines)

    user_prompt = (
        f"Decision problem (overall goal/context):\n{decision_input.problem_description}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Criteria and weights:\n{criteria_text}\n\n"
        "Fuzzy TOPSIS results (for each option):\n"
        f"{breakdown_text}\n\n"
        f"Winner: {topsis_result.winner}\n"
        f"Loser (lowest score, if any): {topsis_result.loser or 'N/A'}\n\n"
        "Write your answer in exactly TWO markdown sections with these headings:\n"
        "## Algorithmic Breakdown\n"
        "Use numbered steps (1–6) as in the instructions: normalization, weighting, "
        "FPIS/FNIS, distances, closeness coefficient, and why this winner was selected. "
        "Include concrete numbers from the results above where relevant.\n\n"
        "## Contextual Fit\n"
        "- Explain why this mathematically winning option fits the user's stated "
        "overall goal and the criterion descriptions.\n"
        "Do not add new information beyond what is implied by the numbers, goal, "
        "and descriptions."
    )

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        explanation_text = (response.content or "").strip()
        if not explanation_text:
            raise RuntimeError("Synthesis LLM returned an empty explanation.")
        return SynthesisOutput(explanation=explanation_text)
    except ValidationError as ve:
        raise RuntimeError(f"Failed to validate synthesis output: {ve}") from ve
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Error while calling synthesis LLM: {exc}") from exc

