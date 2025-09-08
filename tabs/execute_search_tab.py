# tabs/execute_search_tab.py
"""
Execute Search Tab - Simplified with minimal logging
No thread interruptions, clean execution
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


def run_parallel_search_wrapper(selected_models, current_criteria, target_count, enable_recursive):
    """
    Wrapper function to run parallel search in a separate thread.
    NO session state access allowed here!
    """
    # Import here to avoid circular imports
    from shared.search_utils import execute_parallel_search

    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Run the async function
        result = loop.run_until_complete(
            execute_parallel_search(
                selected_models,
                current_criteria,
                target_count,
                serper_key=None,
                enable_recursive=enable_recursive,
                progress_callback=None
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
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Location:**")
                if criteria.location.countries:
                    st.write(f"Countries: {', '.join(criteria.location.countries)}")
                if criteria.location.cities:
                    st.write(f"Cities: {', '.join(criteria.location.cities[:5])}...")

            with col2:
                st.markdown("**Financial:**")
                if criteria.financial.revenue_categories:
                    st.write(f"Revenue: {', '.join(criteria.financial.revenue_categories)}")
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

        # Recursive search option
        enable_recursive = st.checkbox(
            "🔄 Enable Recursive Search (Recommended)",
            value=True,
            help="Automatically relax criteria if not enough unique companies are found"
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

        # Show segmentation strategy
        if st.session_state.parallel_execution_enabled:
            with st.expander("🎯 How Models Will Be Segmented", expanded=False):
                st.write(f"Using {len(st.session_state.selected_models)} models in parallel")

                # Show alphabet distribution
                if target_count >= 100:
                    st.markdown("**Alphabet Segmentation Strategy:**")
                    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    models = st.session_state.selected_models
                    letters_per_model = max(1, len(alphabet) // len(models))
                    remainder = len(alphabet) % len(models)

                    for i, model in enumerate(models):
                        if i < remainder:
                            # Models that get an extra letter
                            start_idx = i * (letters_per_model + 1)
                            end_idx = start_idx + letters_per_model  # inclusive
                        else:
                            # Models that get the base number of letters
                            start_idx = remainder * (letters_per_model + 1) + (i - remainder) * letters_per_model
                            end_idx = start_idx + letters_per_model - 1

                        # Ensure we don't go past Z
                        end_idx = min(end_idx, len(alphabet) - 1)

                        letter_range = f"{alphabet[start_idx]}-{alphabet[end_idx]}"
                        letters_list = alphabet[start_idx:end_idx + 1]
                        st.write(
                            f"**{model}**: Companies starting with **{letter_range}** ({len(letters_list)} letters: {letters_list})")

                st.info("Each model is **locked** to its segment and cannot search outside it!")

        st.divider()

        # Execution button
        if execute_enabled:
            if st.button(
                    f"🚀 Search for {target_count:,} Companies",
                    type="primary",
                    use_container_width=True
            ):
                execute_search(
                    target_count,
                    total_calls,
                    estimated_cost,
                    search_mode,
                    enable_recursive
                )
        else:
            st.button(
                f"🚀 Search Disabled - Enable Parallel Execution",
                type="secondary",
                use_container_width=True,
                disabled=True
            )

        # Display search history
        if st.session_state.search_history:
            with st.expander(f"📜 Search History ({len(st.session_state.search_history)} searches)", expanded=False):
                for i, search in enumerate(reversed(st.session_state.search_history[-5:])):
                    st.write(f"**Search {len(st.session_state.search_history) - i}:** "
                             f"{search['count']} companies, "
                             f"{search['timestamp']}, "
                             f"Found: {search['found']}, "
                             f"Duplicates prevented: {search.get('duplicates_prevented', 0)}")

        # Display results
        display_search_results()


def execute_search(target_count, total_calls, estimated_cost, search_mode, enable_recursive):
    """Execute search with clean thread handling"""

    # Initialize UI elements
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    # Ensure session state is initialized
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = ["gpt-4.1"]
    if 'parallel_execution_enabled' not in st.session_state:
        st.session_state.parallel_execution_enabled = False

    # Store existing results if adding
    existing_results = []
    if "Add to Existing" in search_mode and st.session_state.search_results:
        existing_results = st.session_state.search_results.copy()
        status_placeholder.info(f"Adding to existing {len(existing_results):,} companies...")
    else:
        st.session_state.search_results = []
        st.session_state.all_search_results = []

    # Clear model tracking
    st.session_state.model_results = {}
    st.session_state.model_execution_times = {}
    st.session_state.model_success_status = {}

    try:
        if st.session_state.parallel_execution_enabled and len(st.session_state.selected_models) > 1:
            # PARALLEL EXECUTION
            status_placeholder.info(
                f"🚀 Searching for {target_count:,} companies using {len(st.session_state.selected_models)} models..."
            )

            start_time = time.time()

            # CRITICAL: Extract ALL session state values BEFORE thread
            selected_models = st.session_state.selected_models.copy()
            current_criteria = st.session_state.current_criteria

            # Execute in thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    run_parallel_search_wrapper,
                    selected_models,
                    current_criteria,
                    target_count,
                    enable_recursive
                )

                # Show progress while waiting
                while not future.done():
                    elapsed = time.time() - start_time
                    estimated_progress = min(elapsed / 30, 0.9)
                    progress_bar.progress(estimated_progress)
                    time.sleep(0.5)

                # Get result
                result = future.result()

            execution_time = time.time() - start_time

            # Extract results
            new_companies = result.get('companies', [])
            metadata = result.get('metadata', {})

            # Show execution summary
            if metadata:
                with st.expander("📊 Execution Summary", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Companies Found", len(new_companies))
                        st.metric("Execution Time", f"{execution_time:.1f}s")
                    with col2:
                        st.metric("Duplicates Prevented", metadata.get('duplicates_prevented', 0))
                        st.metric("Strategy Used", metadata.get('strategy', 'Unknown'))
                    with col3:
                        st.metric("Relaxation Levels", metadata.get('relaxation_levels_used', 0))
                        st.metric("Models Used", len(selected_models))

                    # Show model performance if available
                    if metadata.get('model_performance'):
                        st.markdown("**Model Performance:**")
                        for model, stats in metadata['model_performance'].items():
                            st.write(
                                f"• {model}: {stats.get('found', 0)} unique, {stats.get('duplicates', 0)} duplicates blocked")

            # Process results
            if existing_results:
                all_companies = existing_results + new_companies

                # Final deduplication
                seen_names = set()
                unique_companies = []

                for company in all_companies:
                    if hasattr(company, 'name'):
                        company_name = company.name.lower().strip()
                    else:
                        company_name = company.get('name', '').lower().strip()

                    if company_name not in seen_names:
                        seen_names.add(company_name)
                        unique_companies.append(company)

                st.session_state.search_results = unique_companies
                new_unique_count = len(unique_companies) - len(existing_results)

                status_placeholder.success(
                    f"✅ Search complete! Added {new_unique_count} new unique companies!\n"
                    f"Total: {len(unique_companies):,} companies"
                )
            else:
                st.session_state.search_results = new_companies[:target_count]
                status_placeholder.success(
                    f"✅ Found {len(st.session_state.search_results):,} companies!"
                )

            # Update session state
            if metadata:
                st.session_state.model_success_status = metadata.get('model_performance', {})

        else:
            # SINGLE MODEL EXECUTION
            st.error("Single model execution not recommended for large searches. Please enable parallel execution.")
            return

        # Track search in history
        if 'new_companies' in locals():
            st.session_state.search_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'count': target_count,
                'found': len(new_companies),
                'duplicates_prevented': metadata.get('duplicates_prevented', 0) if 'metadata' in locals() else 0,
                'mode': search_mode.split()[0],
                'recursive': enable_recursive,
                'relaxation_levels': metadata.get('relaxation_levels_used', 0) if 'metadata' in locals() else 0
            })

            # Calculate actual cost
            actual_cost = estimated_cost * (1 + metadata.get('relaxation_levels_used',
                                                             0) * 0.3) if 'metadata' in locals() else estimated_cost
            update_cost(actual_cost)

    except Exception as e:
        if 'status_placeholder' in locals():
            status_placeholder.error(f"Search failed: {str(e)}")
        else:
            st.error(f"Search failed: {str(e)}")
        logger.error(f"Search error: {str(e)}")
        traceback.print_exc()

    finally:
        # Clean up progress bar
        time.sleep(1)
        progress_bar.empty()


def display_search_results():
    """Display search results with enhanced information"""

    if st.session_state.search_results:
        st.divider()
        st.subheader(f"📊 Search Results ({len(st.session_state.search_results):,} companies)")

        # Show segment distribution if available
        if st.session_state.search_results and len(st.session_state.search_results) > 0:
            sample_company = st.session_state.search_results[0]
            if hasattr(sample_company, 'search_segment') or (
                    isinstance(sample_company, dict) and 'search_segment' in sample_company):

                with st.expander("📊 Segment Distribution", expanded=False):
                    segment_counts = {}

                    for company in st.session_state.search_results:
                        if hasattr(company, 'search_segment'):
                            segment = company.search_segment
                        else:
                            segment = company.get('search_segment', 'unknown')

                        segment_counts[segment] = segment_counts.get(segment, 0) + 1

                    # Display as metrics
                    cols = st.columns(min(len(segment_counts), 4))
                    for i, (segment, count) in enumerate(segment_counts.items()):
                        with cols[i % len(cols)]:
                            st.metric(segment, count)

        # Filtering
        if len(st.session_state.search_results) > 100:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                tier_filter = st.multiselect(
                    "Filter by ICP Tier",
                    ["A", "B", "C", "D"],
                    default=["A", "B"]
                )

            with col2:
                industries = list(set([
                    c.industry_category if hasattr(c, 'industry_category') else c.get('industry_category', 'Unknown')
                    for c in st.session_state.search_results[:200]
                ]))[:20]

                industry_filter = st.multiselect(
                    "Filter by Industry",
                    industries,
                    default=[]
                )

            with col3:
                revenue_filter = st.multiselect(
                    "Revenue Category",
                    ["very_high", "high", "medium", "low", "very_low", "unknown"],
                    default=[]
                )

            with col4:
                search_filter = st.text_input(
                    "Search company names",
                    placeholder="Type to search..."
                )
        else:
            tier_filter = ["A", "B", "C", "D"]
            industry_filter = []
            revenue_filter = []
            search_filter = ""

        # Convert to dataframe
        results_data = []
        for i, company in enumerate(st.session_state.search_results):
            if hasattr(company, 'dict'):
                c = company.dict()
            else:
                c = company

            # Apply filters
            if c.get('icp_tier', 'D') not in tier_filter:
                continue
            if industry_filter and c.get('industry_category', 'Unknown') not in industry_filter:
                continue
            if revenue_filter and c.get('revenue_category', 'unknown') not in revenue_filter:
                continue
            if search_filter and search_filter.lower() not in c.get('name', '').lower():
                continue

            # Map revenue categories
            revenue_display = {
                "very_high": "$1B+",
                "high": "$100M-$1B",
                "medium": "$10M-$100M",
                "low": "$1M-$10M",
                "very_low": "<$1M",
                "unknown": "Unknown"
            }

            row_data = {
                "#": i + 1,
                "Company": c.get('name', 'Unknown'),
                "Industry": c.get('industry_category', 'Unknown'),
                "Revenue Cat.": revenue_display.get(c.get('revenue_category', 'unknown'), 'Unknown'),
                "Employees": c.get('estimated_employees', 'Unknown'),
                "ICP Score": c.get('icp_score', 0),
                "ICP Tier": c.get('icp_tier', 'D'),
                "Source": c.get('source_model', 'Unknown'),
                "Segment": c.get('search_segment', 'Unknown')[:15],
                "Relaxation": c.get('relaxation_level', 0)
            }

            results_data.append(row_data)

        if results_data:
            df = pd.DataFrame(results_data)

            if len(st.session_state.search_results) > 100:
                st.write(f"Showing {len(df)} of {len(st.session_state.search_results):,} companies")

            # Style the dataframe
            def style_tier(val):
                if val == 'A':
                    return 'background-color: #90EE90'
                elif val == 'B':
                    return 'background-color: #98FB98'
                elif val == 'C':
                    return 'background-color: #FFE4B5'
                else:
                    return ''

            def style_relaxation(val):
                if val == 0:
                    return 'color: green'
                elif val <= 2:
                    return 'color: orange'
                else:
                    return 'color: red'

            # Pagination
            if len(df) > 500:
                page_size = 100
                page = st.number_input("Page", 1, (len(df) + page_size - 1) // page_size, 1)
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, len(df))
                st.dataframe(styled_df.iloc[start_idx:end_idx], use_container_width=True, height=400)
            else:
                st.dataframe(styled_df, use_container_width=True, height=min(400, len(df) * 35 + 50))