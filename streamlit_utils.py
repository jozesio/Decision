"""
Shared Streamlit UI helpers (e.g. footer) for Decision Companion.
"""
from __future__ import annotations

import streamlit as st


def render_footer() -> None:
    """Render a centered footer with author contact details on all pages."""
    st.markdown(
        """
        <div style="text-align: center; margin-top: 3rem; padding: 1rem 0; font-size: 0.85rem; color: #6c757d;">
            Sion Jose<br/>
            <a href="mailto:sionjose2004sj@gmail.com">sionjose2004sj@gmail.com</a><br/>
            7025807399
        </div>
        """,
        unsafe_allow_html=True,
    )
