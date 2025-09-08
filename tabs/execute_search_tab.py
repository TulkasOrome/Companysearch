# tabs/execute_search_tab.py
"""
Execute Search Tab - Fixed pagination, alphabet segmentation, and tier passing
"""

import streamlit as st
import asyncio
import pandas as pd
import time
import traceback
import nest_asyncio
import concurrent.futures
from datetime import datetime
from typing import Dict, Any, List
from shared.session_state import update_cost
import logging

# Apply nest_asyncio to allow nested event loops in Streamlit
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_parallel_search_wrapper(selected_models, current_criteria, target_count, enable_recursive, tier):
    """
    Wrapper function to run parallel search in a separate thread.
    NO session state access allowed here!
    UPDATED: Added tier parameter
    """
    # Import here to avoid circular imports
    from shared.search_utils import execute_parallel_search

    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Run the async function with tier parameter
        result = loop.run_until_complete(
            execute_parallel_search(
                selected_models,
                current_criteria,
                target_count,
                serper_key=None,
                enable_recursive=enable_recursive,
                progress_callback=None,
                tier=tier  # PASS TIER
            )
        )
        return result
    finally:
        loop.close()


def render_execute_search_tab():
    """Render the Execute Search tab"""

    st.header("Execute Company Search")

    if not st.session_state.current_criteria:
        st.warning("⚠️ Please configure search criteria first.")
    else:
        # Show criteria summary
        with st.expander("📋 Current Search Criteria", expanded=True):
            criteria = st.session_state.current_criteria

            # Display current profile and tier if available
            if st.session_state.get('current_profile_name'):
                profile_names = {
                    'rmh_sydney': '🏥 RMH Sydney',
                    'guide_dogs_victoria': '🦮 Guide Dogs Victoria'
                }
                profile_display = profile_names.get(st.session_state.current_profile_name,
                                                    st.session_state.current_profile_name)
                current_tier = st.session_state.get('current_tier', 'A')
                st.info(f"**Profile:** {profile_display} - **Tier {current_tier}**")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Location:**")
                if criteria.location.countries:
                    st.write(f"Countries: {', '.join(criteria.location.countries)}")
                if criteria.location.states:
                    st.write(f"States: {', '.join(criteria.location.states)}")
                if criteria.location.cities:
                    st.write(f"Cities: {', '.join(criteria.location.cities[:5])}...")

            with col2:
                st.markdown("**Financial:**")
                if criteria.financial.revenue_categories:
                    st.write(f"Revenue: {', '.join(criteria.financial.revenue_categories)}")
                if criteria.financial.revenue_min and criteria.financial.revenue_max:
                    st.write(
                        f"Range: ${criteria.financial.revenue_min / 1e6:.0f}M-${criteria.financial.revenue_max / 1e6:.0f}M")
                if criteria.organizational.employee_count_min:
                    st.write(f"Employees: {criteria.organizational.employee_count_min}+")

            with col3:
                st.markdown("**Industries:**")
                if criteria.industries:
                    st.write(f"{len(criteria.industries)} industries")
                if criteria.behavioral.csr_focus_areas:
                    st.write(f"CSR: {', '.join(criteria.behavioral.csr_focus_areas[:3])}")

        st.divider()

        # Initialize search history if not exists
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'all_search_results' not in st.session_state:
            st.session_state.all_search_results = []

        # Search mode selection
        col1, col2 = st.columns([3, 1])

        with col1:
            if len(st.session_state.search_results) > 0:
                search_mode = st.radio(
                    "Search Mode",
                    ["🆕 New Search (Replace existing results)",
                     "➕ Add to Existing (Append and deduplicate)"],
                    index=0,
                    horizontal=True,
                    help="Choose whether to replace existing results or add to them"
                )
            else:
                search_mode = "🆕 New Search (Replace existing results)"
                st.info("First search - results will be saved")

        with col2:
            if st.session_state.search_results:
                st.metric("Current Results", f"{len(st.session_state.search_results):,} companies")

        # Initialize target count
        if 'target_count' not in st.session_state:
            st.session_state.target_count = 100

        # Number of companies selector
        col1, col2 = st.columns([3, 1])

        with col1:
            target_count_slider = st.slider(
                "🎯 Number of Companies to Find",
                min_value=10,
                max_value=10000,
                value=st.session_state.target_count,
                step=10 if st.session_state.target_count < 1000 else 50,
                help="Select how many companies you want to find",
                key="target_slider"
            )

        with col2:
            target_count_input = st.number_input(
                "Or enter exact number",
                min_value=10,
                max_value=10000,
                value=target_count_slider,
                step=10,
                help="Type exact number of companies",
                key="target_input"
            )

        target_count = target_count_input
        st.session_state.target_count = target_count

        # Recursive search option - WITH TIER-AWARE WARNING
        current_tier = st.session_state.get('current_tier', 'A')

        if current_tier in ['B', 'C']:
            st.warning(f"⚠️ **Tier {current_tier} Selected**: Relaxation is limited to preserve tier requirements")
            default_recursive = True  # Still enable but with limited relaxation
        else:
            default_recursive = True

        enable_recursive = st.checkbox(
            "🔄 Enable Recursive Search (Recommended)",
            value=default_recursive,
            help=f"Automatically relax criteria if not enough unique companies are found. {'Limited relaxation for Tier ' + current_tier if current_tier in ['B', 'C'] else 'Full relaxation available for Tier A'}"
        )

        # Show deduplication strategy
        with st.expander("🛡️ Deduplication Strategy", expanded=False):
            st.markdown("""
            **Strong Deduplication Enforcement:**
            - ✅ Each model is assigned a **hard segment** (e.g., companies A-E, F-J, etc.)
            - ✅ Models **cannot search outside their segment**
            - ✅ **Exclusion lists** prevent duplicates across models
            - ✅ **Post-processing validation** rejects non-compliant companies
            - ✅ Segments are **preserved through relaxation** levels

            **Result:** Near-zero duplicates even at scale!
            """)

        # Metrics
        col1, col2, col3 = st.columns(3)

        # Ensure session state is initialized
        if 'parallel_execution_enabled' not in st.session_state:
            st.session_state.parallel_execution_enabled = False
        if 'selected_models' not in st.session_state:
            st.session_state.selected_models = ["gpt-4.1"]

        if st.session_state.parallel_execution_enabled:
            models_count = len(st.session_state.selected_models)
            calls_per_model = (target_count // (15 * models_count)) + 1
            total_calls = calls_per_model * models_count
            estimated_time = calls_per_model * 12
        else:
            total_calls = (target_count // 15) + 1
            estimated_time = total_calls * 12

        estimated_cost = total_calls * 0.02

        with col1:
            st.caption("Target")
            st.markdown(f"**{target_count:,}**")
        with col2:
            st.caption("Est. Cost")
            st.markdown(f"**${estimated_cost:.2f}**")
        with col3:
            st.caption("Est. Time")
            minutes = estimated_time // 60
            seconds = estimated_time % 60
            if minutes > 0:
                st.markdown(f"**{minutes}m {seconds}s**")
            else:
                st.markdown(f"**{seconds}s**")

        # Check execution requirements
        if target_count > 500 and not st.session_state.parallel_execution_enabled:
            st.error(
                f"⚠️ **Parallel execution required for {target_count:,} companies!**\n\n"
                f"Please enable parallel execution in the sidebar."
            )
            execute_enabled = False
        else:
            execute_enabled = True

        # Show segmentation strategy with FIXED alphabet distribution
        if st.session_state.parallel_execution_enabled:
            with st.expander("🎯 How Models Will Be Segmented", expanded=False):
                st.write(f"Using {len(st.session_state.selected_models)} models in parallel")

                # Show alphabet distribution
                if target_count >= 100:
                    st.markdown("**Alphabet Segmentation Strategy:**")
                    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    models = st.session_state.selected_models
                    letters_per_model = len(alphabet) // len(models)
                    remainder = len(alphabet) % len(models)

                    for i, model in enumerate(models):
                        # FIXED CALCULATION - ensures no overlap and covers all letters
                        if i < remainder:
                            # Models that get an extra letter
                            start_idx = i * (letters_per_model + 1)
                            end_idx = start_idx + letters_per_model  # end_idx is inclusive
                        else:
                            # Models that get the base number of letters
                            start_idx = remainder * (letters_per_model + 1) + (i - remainder) * letters_per_model
                            end_idx = start_idx + letters_per_model - 1  # end_idx is inclusive

                        # Ensure we don't go past Z
                        end_idx = min(end_idx, len(alphabet) - 1)

                        # Extract the actual letters
                        assigned_letters = alphabet[start_idx:end_idx + 1]
                        letter_range = f"{alphabet[start_idx]}-{alphabet[end_idx]}" if start_idx != end_idx else \
                            alphabet[start_idx]

                        st.write(f"**{model}**: Companies starting with **{letter_range}** "
                                 f"({len(assigned_letters)} letters: {assigned_letters})")

                st.info