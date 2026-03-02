"""
Algorithm Explanation page for Decision Companion (multipage Streamlit).
Section 1: Beginner-friendly Fuzzy TOPSIS explanation + static example.
Section 2: Dynamic breakdown from the last run (session_state).
"""
from __future__ import annotations

import streamlit as st
import pandas as pd

from streamlit_utils import render_footer

def _section1_static_explanation():
    st.header("Section 1: How Fuzzy TOPSIS Works")
    st.markdown(
        """
        **Fuzzy TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution) helps you rank
        options when criteria are evaluated with uncertainty. Instead of a single number per score, we use
        **triangular fuzzy numbers (TFNs)** — three values (l, m, u): worst-case, most-likely, and best-case.

        **Steps in brief:**
        1. **Normalize** the fuzzy matrix so every value lies between 0 and 1. For *benefit* criteria (higher is better),
           we divide by the maximum upper bound; for *cost* criteria (lower is better), we flip and scale so lower
           raw values become higher normalized values.
        2. **Weight** each normalized TFN by the criterion's importance (e.g. weight 1–10).
        3. **Ideal solutions:** The **Positive Ideal (FPIS)** is the best possible value per criterion across options;
           the **Negative Ideal (FNIS)** is the worst. We compute these in the weighted fuzzy space.
        4. **Distances:** For each option we measure its distance to FPIS and to FNIS using the vertex method
           (square root of the average of squared differences of l, m, u).
        5. **Closeness coefficient (CC):** CC = distance_to_FNIS / (distance_to_FPIS + distance_to_FNIS).
           The option with the **highest CC** is closest to the positive ideal and farthest from the negative ideal — that's the winner.
        """
    )
    st.subheader("Static example (2 options, 2 criteria)")
    st.markdown(
        """
        Suppose we have **Option A** and **Option B**, and two criteria: **Quality** (benefit, weight 8) and **Cost** (cost, weight 6).

        | Option | Quality (l, m, u) | Cost (l, m, u) |
        |--------|------------------|----------------|
        | A      | (6, 7, 8)        | (2, 3, 4)      |
        | B      | (5, 6, 7)        | (1, 2, 3)      |

        - **Normalization (Quality, benefit):** max u = 8 → A becomes (6/8, 7/8, 8/8), B becomes (5/8, 6/8, 7/8).
        - **Normalization (Cost, cost):** min l = 1. We flip: for B (low cost) (1,2,3) → (1/3, 1/2, 1/1); for A (2,3,4) → (1/4, 1/3, 1/2). Then scale so all in [0,1] if needed.
        - After **weighting** and computing **FPIS/FNIS**, we get distances and then **CC**. The option with higher CC wins.
        """
    )


def _section2_dynamic_breakdown():
    st.header("Section 2: Your Run — Step-by-Step Breakdown")
    intermediates = st.session_state.get("calculation_intermediates")
    calculate_result = st.session_state.get("calculate_result")

    if not intermediates and not calculate_result:
        st.info("Run a decision on the main page first to see the breakdown for your scenario.")
        return

    if not intermediates:
        st.warning("No intermediate math was saved for this run. Run the final decision again to see the breakdown.")
        return

    options_list = intermediates.get("options") or []
    winner = (calculate_result or {}).get("winner") or (options_list[0]["option_name"] if options_list else None)

    st.markdown(
        "Below is the step-by-step flow for your run. **Step 1:** Raw fuzzy scores were normalized "
        "so every value lies in [0, 1] (Benefit criteria: divide by max; Cost criteria: flip and scale). "
        "**Step 2:** Normalized values were multiplied by each criterion's weight. **Step 3:** The "
        "positive ideal (FPIS) and negative ideal (FNIS) were computed from the weighted matrix. "
        "**Step 4:** For each option, distances to FPIS and FNIS were calculated; then the closeness "
        "coefficient (CC) = distance_to_FNIS / (distance_to_FPIS + distance_to_FNIS). **Step 5:** "
        "The option with the highest CC is closest to the ideal and is the winner."
    )
    st.markdown("---")

    # Normalized matrix
    st.subheader("Step 1: Normalized matrix (values in [0, 1])")
    norm = intermediates.get("normalized_matrix") or []
    if norm:
        st.dataframe(pd.DataFrame(norm), use_container_width=True)
    else:
        st.caption("No normalized matrix data.")

    # Weighted matrix
    st.subheader("Step 2: Weighted matrix (normalized × criterion weight)")
    wgt = intermediates.get("weighted_matrix") or []
    if wgt:
        st.dataframe(pd.DataFrame(wgt), use_container_width=True)
    else:
        st.caption("No weighted matrix data.")

    # FPIS and FNIS
    st.subheader("Step 3: Positive ideal (FPIS) and negative ideal (FNIS)")
    fpis = intermediates.get("fpis") or {}
    fnis = intermediates.get("fnis") or {}
    if fpis or fnis:
        col_fpis, col_fnis = st.columns(2)
        with col_fpis:
            st.markdown("**FPIS (best per criterion)**")
            for c, v in fpis.items():
                st.text(f"{c}: (l={v['l']:.4f}, m={v['m']:.4f}, u={v['u']:.4f})")
        with col_fnis:
            st.markdown("**FNIS (worst per criterion)**")
            for c, v in fnis.items():
                st.text(f"{c}: (l={v['l']:.4f}, m={v['m']:.4f}, u={v['u']:.4f})")
    else:
        st.caption("No FPIS/FNIS data.")

    # Distances and closeness coefficients
    st.subheader("Step 4 & 5: Distances and closeness coefficients")
    if options_list:
        df_cc = pd.DataFrame(options_list)
        df_cc = df_cc.sort_values("closeness_coefficient", ascending=False).reset_index(drop=True)
        st.dataframe(df_cc, use_container_width=True)
        if winner:
            st.success(
                f"**Winner: {winner}** — This option has the highest closeness coefficient, meaning it is "
                "closest to the positive ideal solution and farthest from the negative ideal in the weighted fuzzy space."
            )
    else:
        st.caption("No option results.")


def main():
    st.set_page_config(page_title="How this Algorithm works?", layout="wide")
    st.title("How this Algorithm works?")
    st.caption("Understand how Fuzzy TOPSIS computed the winning option.")

    _section1_static_explanation()
    st.markdown("---")
    _section2_dynamic_breakdown()

    render_footer()


if __name__ == "__main__":
    main()
