"""
Decision Companion — Streamlit frontend (self-contained, no Django server needed).
Run: streamlit run streamlit_app.py
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import streamlit as st
import pandas as pd

# Set GROQ_API_KEY from Streamlit secrets before importing LLM modules.
if "GROQ_API_KEY" not in os.environ:
    _key = st.secrets.get("GROQ_API_KEY", "")
    if _key:
        os.environ["GROQ_API_KEY"] = _key

from decision_comp.graph import run_ai_research, run_calculation_and_synthesis
from decision_comp.llm_services import classify_criteria_nature
from decision_comp.rag import build_rag_context
from decision_comp.models import (
    CriterionSchema,
    DecisionInputState,
    FinalScoresState,
    OptionCriterionScore,
    OptionSchema,
    TriangularFuzzyNumber,
)
from streamlit_utils import render_footer

logger = logging.getLogger(__name__)


def _ensure_session_state():
    # Core decision definition
    if "num_options" not in st.session_state:
        st.session_state.num_options = 2
    if "num_criteria" not in st.session_state:
        st.session_state.num_criteria = 1
    if "options" not in st.session_state:
        st.session_state.options = []
    if "criteria" not in st.session_state:
        st.session_state.criteria = []
    if "problem_description" not in st.session_state:
        st.session_state.problem_description = ""

    # Research + human-in-the-loop state
    if "research_scores" not in st.session_state:
        st.session_state.research_scores = []
    if "criterion_meta" not in st.session_state:
        # List of dicts: {name, weight, description, kind, rationale}
        st.session_state.criterion_meta = []
    if "criterion_kinds" not in st.session_state:
        # Mapping criterion_name -> "benefit" / "cost" (user-overridable)
        st.session_state.criterion_kinds = {}
    if "edited_scores" not in st.session_state:
        # Mapping criterion_name -> DataFrame of rows for that criterion
        st.session_state.edited_scores = {}

    # Final result
    if "calculate_result" not in st.session_state:
        st.session_state.calculate_result = None
    if "calculation_intermediates" not in st.session_state:
        st.session_state.calculation_intermediates = None

    # Gate: show score editors only after user confirms criteria classification
    if "classification_confirmed" not in st.session_state:
        st.session_state.classification_confirmed = False


def _reset_decision_state() -> None:
    """
    Clear the current decision so the user can start a new one.
    Keeps num_options and num_criteria so counts are preserved.
    """
    # Defer actual reset until the next run, before widgets are created.
    # This avoids modifying widget-managed keys (e.g. option_pdf_{i}) after instantiation.
    st.session_state._reset_decision_requested = True


def _build_inputs(
    problem_description: str,
    options_raw: list,
    criteria_raw: list,
) -> DecisionInputState:
    """Build Pydantic DecisionInputState from UI data."""
    options = [
        OptionSchema(name=o["name"], description=o.get("description") or "")
        for o in options_raw
    ]
    criteria = []
    for c in criteria_raw:
        weight = int(c.get("weight", 5))
        kind = (c.get("kind") or "benefit").lower()
        if kind not in ("benefit", "cost"):
            kind = "benefit"
        criteria.append(
            CriterionSchema(
                name=c["name"], weight=weight,
                description=c.get("description"), kind=kind,
            )
        )
    return DecisionInputState(
        problem_description=problem_description,
        options=options, criteria=criteria,
    )


def _run_research(
    problem_description: str,
    options_with_docs: list,
    criteria_raw: list,
) -> dict:
    """Run classification, RAG, and AI research directly (no HTTP)."""
    inputs = _build_inputs(problem_description, options_with_docs, criteria_raw)

    nature_batch = classify_criteria_nature(inputs)
    nature_by_name: Dict[str, Any] = {}
    for item in nature_batch.items:
        nature_by_name[item.criterion_name] = item

    documents_per_option: List[List[Tuple[str, bytes]]] = []
    for o in options_with_docs:
        doc_list: List[Tuple[str, bytes]] = []
        for d in o.get("documents") or []:
            fname = d.get("filename") or "document.pdf"
            raw = d.get("content_bytes")
            if raw:
                doc_list.append((fname, raw))
        documents_per_option.append(doc_list)

    rag_context = ""
    try:
        rag_context = build_rag_context(
            documents_per_option, inputs.options, inputs.criteria,
            inputs.problem_description,
        )
    except Exception as e:
        logger.exception("RAG context build failed: %s", e)

    ai_result = run_ai_research(inputs, rag_context=rag_context or None)

    had_docs = any(dl for dl in documents_per_option)
    rag_warning = (
        "PDF context could not be generated; scores may not reflect uploaded documents."
        if (had_docs and not (rag_context and rag_context.strip()))
        else None
    )

    scores_list = []
    for (opt_name, crit_name), score_obj in ai_result.scores.items():
        scores_list.append({
            "option_name": opt_name,
            "criterion_name": crit_name,
            "l": score_obj.score_tfn.l,
            "m": score_obj.score_tfn.m,
            "u": score_obj.score_tfn.u,
            "justification": score_obj.justification,
        })

    criteria_list = []
    for crit in inputs.criteria:
        nature = nature_by_name.get(crit.name)
        kind = crit.kind
        rationale = ""
        if nature is not None:
            kind = nature.kind
            rationale = nature.rationale
        criteria_list.append({
            "name": crit.name, "weight": crit.weight,
            "description": crit.description, "kind": kind,
            "rationale": rationale,
        })

    result: dict = {"scores": scores_list, "criteria": criteria_list}
    if rag_warning:
        result["rag_warning"] = rag_warning
    return result


def _run_calculate(
    problem_description: str,
    options_raw: list,
    criteria_raw: list,
    scores_list: list,
) -> dict:
    """Run Fuzzy TOPSIS and synthesis directly (no HTTP)."""
    inputs = _build_inputs(problem_description, options_raw, criteria_raw)

    scores: Dict[Tuple[str, str], OptionCriterionScore] = {}
    for row in scores_list:
        key = (row["option_name"], row["criterion_name"])
        scores[key] = OptionCriterionScore(
            option_name=row["option_name"],
            criterion_name=row["criterion_name"],
            score_tfn=TriangularFuzzyNumber(
                l=float(row["l"]), m=float(row["m"]), u=float(row["u"]),
            ),
            justification=(row.get("justification") or "").strip() or "No justification provided.",
        )
    final_scores = FinalScoresState(scores=scores)

    topsis_result, explanation, intermediates = run_calculation_and_synthesis(inputs, final_scores)

    return {
        "winner": topsis_result.winner,
        "loser": topsis_result.loser,
        "explanation": explanation,
        "options": [
            {
                "option_name": o.option_name,
                "closeness_coefficient": o.closeness_coefficient,
                "distance_to_fpis": o.distance_to_fpis,
                "distance_to_fnis": o.distance_to_fnis,
            }
            for o in topsis_result.options
        ],
        "intermediates": intermediates,
    }


def main():
    _ensure_session_state()

    # Apply a pending reset (triggered from the previous run) *before* any widgets are instantiated.
    if st.session_state.get("_reset_decision_requested"):
        prev_num_options = int(st.session_state.get("num_options", 2))
        prev_num_criteria = int(st.session_state.get("num_criteria", 1))

        # Remove widget-managed keys so Streamlit recreates them fresh.
        for i in range(prev_num_options):
            for k in (f"option_name_{i}", f"option_pdf_{i}"):
                st.session_state.pop(k, None)
        for i in range(prev_num_criteria):
            for k in (f"criterion_name_{i}", f"criterion_weight_{i}", f"criterion_desc_{i}"):
                st.session_state.pop(k, None)
        st.session_state.pop("overall_goal", None)

        # Reset counts to defaults
        st.session_state.num_options = 2
        st.session_state.num_criteria = 1

        # Core decision definition
        st.session_state.problem_description = ""
        st.session_state.options = []
        st.session_state.criteria = []

        # Research + human-in-the-loop state
        st.session_state.research_scores = []
        st.session_state.criterion_meta = []
        st.session_state.criterion_kinds = {}
        st.session_state.edited_scores = {}
        st.session_state.classification_confirmed = False

        # Final result
        st.session_state.calculate_result = None
        st.session_state.calculation_intermediates = None

        st.session_state._reset_decision_requested = False

    st.set_page_config(page_title="Run Your Decision", layout="wide")
    st.title("Run Your Decision")
    st.caption("Define options and criteria, run AI research, edit fuzzy scores, then run the final decision.")

    # ----- Phase 1: Dynamic decision inputs -----
    st.header("Phase 1: Overall goal, options, and criteria")

    # Overall goal / context (required, feeds directly into problem_description)
    problem = st.text_area(
        "Overall Goal / Context",
        value=st.session_state.problem_description,
        key="overall_goal",
        height=80,
        placeholder=(
            "Example: We are evaluating three candidates for a Team Lead role. "
            "I uploaded each candidate's resume under that option's Documents field. "
            "Please evaluate them based on "
            "Leadership, System Design, and Communication skills."
        ),
        help="Describe what you are trying to decide. This is passed to the AI for research and final explanation.",
    )
    st.session_state.problem_description = problem

    col_counts = st.columns(2)
    with col_counts[0]:
        st.number_input(
            "Number of Options",
            min_value=2,
            step=1,
            key="num_options",
            help="How many options do you want to compare?",
        )
    with col_counts[1]:
        st.number_input(
            "Number of Criteria",
            min_value=1,
            step=1,
            key="num_criteria",
            help="How many criteria will you use to evaluate the options?",
        )

    num_options = int(st.session_state.num_options)
    num_criteria = int(st.session_state.num_criteria)

    # Ensure options/criteria lists match the selected counts
    if len(st.session_state.options) < num_options:
        for _ in range(num_options - len(st.session_state.options)):
            st.session_state.options.append({"name": "", "description": ""})
    elif len(st.session_state.options) > num_options:
        st.session_state.options = st.session_state.options[:num_options]

    if len(st.session_state.criteria) < num_criteria:
        for _ in range(num_criteria - len(st.session_state.criteria)):
            st.session_state.criteria.append(
                {"name": "", "weight": 5, "description": "", "kind": "benefit"}
            )
    elif len(st.session_state.criteria) > num_criteria:
        st.session_state.criteria = st.session_state.criteria[:num_criteria]

    col_opts, col_crits = st.columns(2)

    # Dynamic option inputs
    with col_opts:
        st.subheader("Options")
        st.caption("Attach each candidate's PDF to that option's **Documents (optional)** uploader.")
        for i in range(num_options):
            opt = st.session_state.options[i]
            name_key = f"option_name_{i}"
            name_val = st.text_input(
                f"Option {i + 1} name",
                value=opt.get("name", ""),
                key=name_key,
            )
            opt["name"] = name_val.strip()
            st.file_uploader(
                f"Documents for Option {i + 1} (optional)",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"option_pdf_{i}",
                help="Optional: Upload documents specifically for this option. The AI will extract facts from these files to justify its scores, preventing guesswork.",
            )

    # Dynamic criterion inputs
    with col_crits:
        st.subheader("Criteria")
        for i in range(num_criteria):
            crit = st.session_state.criteria[i]
            if "name" not in crit:
                crit["name"] = ""
            if "weight" not in crit:
                crit["weight"] = 5
            if "description" not in crit:
                crit["description"] = ""
            if "kind" not in crit:
                crit["kind"] = "benefit"

            st.markdown(f"**Criterion {i + 1}**")
            crit_name = st.text_input(
                "Name",
                value=crit["name"],
                key=f"criterion_name_{i}",
            )
            crit_weight = st.number_input(
                "Weight (1–10)",
                min_value=1,
                max_value=10,
                step=1,
                value=int(crit["weight"]),
                key=f"criterion_weight_{i}",
            )
            crit_description = st.text_area(
                "Description (optional)",
                value=crit.get("description") or "",
                key=f"criterion_desc_{i}",
                height=60,
            )

            crit["name"] = crit_name.strip()
            crit["weight"] = int(crit_weight)
            crit["description"] = crit_description.strip()

    # Validation for proceeding
    options_valid = num_options >= 2 and all(o["name"] for o in st.session_state.options)
    criteria_valid = num_criteria >= 1 and all(
        c["name"] and 1 <= int(c["weight"]) <= 10 for c in st.session_state.criteria
    )
    goal_valid = bool(st.session_state.problem_description.strip())

    if not goal_valid:
        st.info("Please describe your overall goal/context before running AI research.")
    if not options_valid or not criteria_valid:
        st.info("You need at least 2 named options and 1 named criterion (with weights 1–10).")

    # ----- Phase 2: Research -----
    st.header("Phase 2: AI research")
    research_disabled = not (goal_valid and options_valid and criteria_valid)
    if st.button("Run AI research", disabled=research_disabled):
        with st.spinner("AI agents researching and validating data..."):
            success = False
            try:
                options_with_docs = []
                for i, o in enumerate(st.session_state.options):
                    opt: dict = {"name": o["name"], "description": o.get("description") or ""}
                    files = st.session_state.get(f"option_pdf_{i}") or []
                    if files:
                        doc_list = []
                        for f in files:
                            try:
                                f.seek(0)
                            except Exception:
                                pass
                            content = f.read()
                            if content:
                                doc_list.append({
                                    "filename": getattr(f, "name", "document.pdf"),
                                    "content_bytes": content,
                                })
                        if doc_list:
                            opt["documents"] = doc_list
                    options_with_docs.append(opt)
                result = _run_research(
                    st.session_state.problem_description,
                    options_with_docs,
                    st.session_state.criteria,
                )
                if result.get("rag_warning"):
                    st.warning(result["rag_warning"])
                else:
                    uploaded_option_count = sum(
                        1 for opt in options_with_docs if opt.get("documents")
                    )
                    if uploaded_option_count > 0:
                        st.info(
                            "RAG is using uploaded PDFs for "
                            f"{uploaded_option_count} option(s)."
                        )
                st.session_state.research_scores = result.get("scores") or []
                st.session_state.criterion_meta = result.get("criteria") or []
                kinds = {}
                for crit in st.session_state.criterion_meta:
                    name = crit.get("name")
                    if not name:
                        continue
                    kinds[name] = (crit.get("kind") or "benefit").lower()
                st.session_state.criterion_kinds = kinds
                st.session_state.edited_scores = {}
                st.session_state.classification_confirmed = False
                success = True
                st.success("Research complete. Confirm criteria classification below, then review scores.")
            except Exception as e:
                st.error(f"Research failed: {e}")
        if success:
            st.rerun()

    # ----- Phase 2.5: Criteria classification confirmation -----
    scores = st.session_state.research_scores
    criterion_meta = st.session_state.criterion_meta

    if scores and criterion_meta and not st.session_state.classification_confirmed:
        st.header("Phase 2.5: Criteria classification")
        st.subheader("Are you satisfied with the criteria classification?")
        st.caption("Review the AI's classification for each criterion. Change any type below, then proceed to score review.")
        for crit in criterion_meta:
            crit_name = crit.get("name")
            if not crit_name:
                continue
            crit_kind_llm = (crit.get("kind") or "benefit").lower()
            crit_rationale = crit.get("rationale") or ""
            current_kind = st.session_state.criterion_kinds.get(crit_name, crit_kind_llm)
            kind_index = 0 if (str(current_kind).strip().lower() or "benefit") == "benefit" else 1
            selected = st.selectbox(
                f"**{crit_name}** — AI suggested: {(crit_kind_llm or 'benefit').capitalize()}",
                options=["Benefit", "Cost"],
                index=kind_index,
                key=f"criterion_kind_confirm_{crit_name}",
            )
            st.session_state.criterion_kinds[crit_name] = selected.lower()
            if crit_rationale:
                st.caption(f"Rationale: {crit_rationale}")
        if st.button("Yes, proceed to score review"):
            st.session_state.classification_confirmed = True
            st.rerun()

    # ----- Phase 3: Human-in-the-loop (score editors) -----
    if scores and criterion_meta and st.session_state.classification_confirmed:
        st.header("Phase 3: Review and edit scores (per criterion)")

        # Render one editor per criterion
        for crit in criterion_meta:
            crit_name = crit.get("name")
            if not crit_name:
                continue

            crit_desc = crit.get("description") or ""
            crit_kind_llm = (crit.get("kind") or "benefit").lower()
            crit_rationale = crit.get("rationale") or ""

            st.subheader(f"Criterion: {crit_name}")
            if crit_desc:
                st.caption(crit_desc)

            # Benefit/Cost selectbox with AI guess as default
            current_kind = st.session_state.criterion_kinds.get(crit_name, crit_kind_llm)
            kind_index = 0 if (str(current_kind).strip().lower() or "benefit") == "benefit" else 1
            selected_kind = st.selectbox(
                "Criterion type (Benefit/Cost)",
                options=["Benefit", "Cost"],
                index=kind_index,
                key=f"criterion_kind_{crit_name}",
            )
            st.session_state.criterion_kinds[crit_name] = selected_kind.lower()

            if crit_rationale:
                st.caption(f"LLM rationale: {crit_rationale}")

            # Build or reuse the DataFrame for this criterion
            if crit_name in st.session_state.edited_scores:
                df = st.session_state.edited_scores[crit_name]
            else:
                rows = [row for row in scores if row.get("criterion_name") == crit_name]
                if not rows:
                    st.warning("No scores found for this criterion.")
                    continue
                df = pd.DataFrame(rows)
                # Only keep the relevant columns for editing
                expected_cols = ["option_name", "l", "m", "u", "justification"]
                existing_cols = [c for c in expected_cols if c in df.columns]
                df = df[existing_cols]

            edited_df = st.data_editor(
                df,
                use_container_width=True,
                key=f"scores_editor_{crit_name}",
                column_config={
                    "option_name": st.column_config.TextColumn("Option", disabled=True),
                    "l": st.column_config.NumberColumn("l", min_value=1.0, max_value=10.0, format="%.2f"),
                    "m": st.column_config.NumberColumn("m", min_value=1.0, max_value=10.0, format="%.2f"),
                    "u": st.column_config.NumberColumn("u", min_value=1.0, max_value=10.0, format="%.2f"),
                    "justification": st.column_config.TextColumn("Justification"),
                },
            )
            st.session_state.edited_scores[crit_name] = edited_df
    elif not (scores and criterion_meta):
        st.info("Run AI research first to see per-criterion score tables.")

    # ----- Phase 4: Calculation -----
    st.header("Phase 4: Final decision")
    can_calculate = bool(scores and criterion_meta and st.session_state.classification_confirmed)
    if not can_calculate:
        st.info("You must run AI research, confirm criteria classification, and review scores before running the final decision.")
    else:
        if st.button("Run final decision"):
            # Flatten per-criterion edited tables into a single scores list.
            scores_payload = []
            for crit in criterion_meta:
                crit_name = crit.get("name")
                if not crit_name:
                    continue

                df = st.session_state.edited_scores.get(crit_name)
                if df is None:
                    # Fallback: build from original scores for this criterion.
                    rows = [row for row in scores if row.get("criterion_name") == crit_name]
                    if not rows:
                        continue
                    df = pd.DataFrame(rows)
                    expected_cols = ["option_name", "l", "m", "u", "justification"]
                    existing_cols = [c for c in expected_cols if c in df.columns]
                    df = df[existing_cols]

                for row in df.to_dict(orient="records"):
                    scores_payload.append(
                        {
                            "option_name": row["option_name"],
                            "criterion_name": crit_name,
                            "l": row["l"],
                            "m": row["m"],
                            "u": row["u"],
                            "justification": row.get("justification", "") or "",
                        }
                    )

            # Build criteria payload from criterion_meta (research response)
            # so names match the scores exactly.
            criteria_payload = []
            for crit in st.session_state.criterion_meta:
                name = crit.get("name")
                if not name:
                    continue
                kind_override = st.session_state.criterion_kinds.get(name, crit.get("kind") or "benefit")
                criteria_payload.append(
                    {
                        "name": name,
                        "weight": crit.get("weight", 5),
                        "description": crit.get("description") or "",
                        "kind": (kind_override or "benefit").lower(),
                    }
                )

            # Build options payload from score data so names match exactly.
            option_names_from_scores = list(dict.fromkeys(
                row["option_name"] for row in scores_payload
            ))
            options_payload = [
                {"name": n, "description": ""}
                for n in option_names_from_scores
            ]

            with st.spinner("Computing Fuzzy TOPSIS and synthesis..."):
                try:
                    calc = _run_calculate(
                        st.session_state.problem_description,
                        options_payload,
                        criteria_payload,
                        scores_payload,
                    )
                    st.session_state.calculate_result = calc
                    st.session_state.calculation_intermediates = calc.get("intermediates")
                    st.success("Calculation complete.")
                except Exception as e:
                    st.error(f"Final decision failed: {e}")
            st.rerun()

    # ----- Phase 5: Results -----
    result = st.session_state.calculate_result
    if result:
        st.header("Phase 5: Results")
        winner = result.get("winner")
        explanation = result.get("explanation", "")
        options_cc = result.get("options") or []

        if options_cc:
            cc_df = pd.DataFrame(options_cc)
            cc_df = cc_df.sort_values("closeness_coefficient", ascending=False).reset_index(drop=True)
            cc_df.index = cc_df.index + 1
            cc_df.index.name = "Rank"

            st.success(f"Recommended: **{cc_df.iloc[0]['option_name']}** (Rank 1)")

            st.subheader("Rankings")
            rank_df = cc_df[["option_name", "closeness_coefficient"]].copy()
            rank_df.columns = ["Option", "Closeness Coefficient"]
            rank_df["Closeness Coefficient"] = rank_df["Closeness Coefficient"].map(lambda x: f"{x:.4f}")
            st.table(rank_df)

        if explanation:
            st.subheader("Explanation")
            st.markdown(explanation)

        if options_cc:
            chart_df = pd.DataFrame(options_cc).sort_values("closeness_coefficient", ascending=False)
            st.subheader("Score Comparison")
            st.bar_chart(chart_df.set_index("option_name")["closeness_coefficient"])

        st.markdown("---")
        if st.session_state.calculation_intermediates:
            col_breakdown, col_new_decision = st.columns(2)
            with col_breakdown:
                if st.button("View Detailed Mathematical Breakdown"):
                    st.switch_page("pages/How_this_Algorithm_works.py")
            with col_new_decision:
                if st.button("Run another decision"):
                    _reset_decision_state()
                    st.rerun()
        else:
            if st.button("Run another decision"):
                _reset_decision_state()
                st.rerun()

    render_footer()


if __name__ == "__main__":
    main()
