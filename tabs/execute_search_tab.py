# tabs/execute_search_tab.py
"""
Execute Search Tab - With recursive approach to ensure target count
REMOVED: All revenue validation references
ADDED: Recursive search to meet target count
"""

import streamlit as st
import asyncio
import pandas as pd
import time
import traceback
import nest_asyncio
import concurrent.futures
import io
import sys
import json
from datetime import datetime
from typing import Dict, Any, List
from shared.session_state import update_cost
from dataclasses import asdict

# Apply nest_asyncio to allow nested event loops in Streamlit
nest_asyncio.apply()


def render_execute_search_tab():
    """Render the Execute Search tab with recursive search"""

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

        # Search mode selection (New vs Add to Existing)
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

        # Initialize target count in session state if not exists
        if 'target_count' not in st.session_state:
            st.session_state.target_count = 100

        # Number of companies selector with synchronized slider and number input
        col1, col2 = st.columns([3, 1])

        with col1:
            # Slider for visual selection
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
            # Number input for precise entry
            target_count_input = st.number_input(
                "Or enter exact number",
                min_value=10,
                max_value=10000,
                value=target_count_slider,  # Use slider value as default
                step=10,
                help="Type exact number of companies",
                key="target_input"
            )

        # Use the number input value as the final target (it updates when slider changes)
        target_count = target_count_input
        st.session_state.target_count = target_count

        # Recursive search option
        enable_recursive = st.checkbox(
            "🔄 Enable Recursive Search",
            value=True,
            help="If not enough companies are found, automatically relax criteria and search again"
        )

        # Compact metrics row
        col1, col2, col3 = st.columns(3)

        # Calculate requirements
        if st.session_state.parallel_execution_enabled:
            models_count = len(st.session_state.selected_models)
            calls_per_model = (target_count // (15 * models_count)) + 1
            total_calls = calls_per_model * models_count
            # Parallel execution: time is based on the slowest model
            estimated_time = calls_per_model * 12  # 12 seconds per call layer
        else:
            total_calls = (target_count // 15) + 1
            # Single model: sequential calls
            estimated_time = total_calls * 12  # 12 seconds per call

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

        # Show distribution info in expander
        if st.session_state.parallel_execution_enabled:
            with st.expander("🔄 How search will be distributed", expanded=False):
                st.write(f"Using {len(st.session_state.selected_models)} models in parallel")
                st.write("**True parallel execution:** All models run simultaneously")

                # Determine distribution strategy
                num_cities = len(criteria.location.cities) if criteria.location.cities else 0
                num_industries = len(criteria.industries) if criteria.industries else 0

                if num_cities >= len(st.session_state.selected_models):
                    st.write(f"✓ **Geographic Distribution**: Each model searches different cities")
                elif num_industries >= len(st.session_state.selected_models) * 2:
                    st.write(f"✓ **Industry Distribution**: Each model searches different industries")
                else:
                    st.write(f"✓ **Alphabet Distribution**: Each model searches different letter ranges")

                st.info("All models use your full criteria while focusing on their segments.")

        st.divider()

        # Single execution button
        if execute_enabled:
            if st.button(
                    f"🚀 Search for {target_count:,} Companies",
                    type="primary",
                    use_container_width=True
            ):
                execute_search(target_count, total_calls, estimated_cost, search_mode, enable_recursive)
        else:
            st.button(
                f"🚀 Search Disabled - Enable Parallel Execution",
                type="secondary",
                use_container_width=True,
                disabled=True
            )

        # Display search history if exists
        if st.session_state.search_history:
            with st.expander(f"📜 Search History ({len(st.session_state.search_history)} searches)", expanded=False):
                for i, search in enumerate(reversed(st.session_state.search_history[-5:])):  # Show last 5
                    st.write(f"**Search {len(st.session_state.search_history) - i}:** "
                             f"{search['count']} companies, "
                             f"{search['timestamp']}, "
                             f"Found: {search['found']}")

        # Display results
        display_search_results()


def relax_criteria(criteria, relaxation_level: int):
    """
    Progressively relax search criteria
    Level 1: Remove CSR requirements
    Level 2: Expand revenue categories
    Level 3: Remove industry restrictions
    Level 4: Expand geographic scope
    Level 5: Lower employee minimums
    """
    import copy
    from shared.data_models import SearchCriteria, determine_revenue_categories_from_range

    relaxed = copy.deepcopy(criteria)

    if relaxation_level >= 1:
        # Remove CSR requirements
        relaxed.behavioral.csr_focus_areas = []
        relaxed.behavioral.certifications = []
        relaxed.behavioral.recent_events = []
        relaxed.behavioral.esg_maturity = None

    if relaxation_level >= 2:
        # Expand revenue categories
        if relaxed.financial.revenue_min or relaxed.financial.revenue_max:
            # Widen by one category on each side
            if relaxed.financial.revenue_min and relaxed.financial.revenue_min > 1_000_000:
                relaxed.financial.revenue_min = max(0, relaxed.financial.revenue_min / 10)
            if relaxed.financial.revenue_max and relaxed.financial.revenue_max < 10_000_000_000:
                relaxed.financial.revenue_max = min(10_000_000_000, relaxed.financial.revenue_max * 10)
            # Recalculate categories
            relaxed.financial.revenue_categories = determine_revenue_categories_from_range(
                relaxed.financial.revenue_min,
                relaxed.financial.revenue_max
            )

    if relaxation_level >= 3:
        # Remove industry restrictions
        relaxed.industries = []
        relaxed.excluded_industries = []

    if relaxation_level >= 4:
        # Expand geographic scope
        if relaxed.location.cities:
            # Move cities to states
            relaxed.location.states = list(set(relaxed.location.states + ["Any"]))
            relaxed.location.cities = []
        elif relaxed.location.states:
            # Remove state restrictions
            relaxed.location.states = []

    if relaxation_level >= 5:
        # Lower employee minimums
        if relaxed.organizational.employee_count_min:
            relaxed.organizational.employee_count_min = max(1, relaxed.organizational.employee_count_min // 2)

    return relaxed


def run_parallel_search_wrapper(selected_models, current_criteria, target_count):
    """
    Wrapper function to run parallel search in a separate thread.
    This function runs in a thread pool and returns results without updating UI.
    """
    # Import here to avoid circular imports
    from shared.search_utils import execute_parallel_search

    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Run the async function and capture ALL output
        result = loop.run_until_complete(
            execute_parallel_search(
                selected_models,
                current_criteria,
                target_count,
                serper_key=None  # No longer needed
            )
        )
        return result
    finally:
        loop.close()


def run_single_search(agent, criteria, target_count):
    """Run a single search call in a separate event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            agent.generate_enhanced_strategy(
                criteria,
                target_count=target_count
            )
        )
    finally:
        loop.close()


def execute_search(target_count, total_calls, estimated_cost, search_mode, enable_recursive):
    """Execute the search with progress tracking and recursive approach"""

    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    # Store existing results if adding
    existing_results = []
    if "Add to Existing" in search_mode and st.session_state.search_results:
        existing_results = st.session_state.search_results.copy()
        status_placeholder.info(f"Adding to existing {len(existing_results):,} companies...")
    else:
        # Clear for new search
        st.session_state.search_results = []
        st.session_state.all_search_results = []

    # Clear model tracking
    st.session_state.model_results = {}
    st.session_state.model_execution_times = {}
    st.session_state.model_success_status = {}

    # Recursive search implementation
    all_found_companies = []
    relaxation_level = 0
    max_relaxation = 5
    search_attempts = []

    current_criteria = st.session_state.current_criteria
    remaining_target = target_count

    while remaining_target > 0 and relaxation_level <= max_relaxation:

        if relaxation_level > 0:
            status_placeholder.info(f"🔄 Relaxing criteria (Level {relaxation_level}) to find more companies...")
            current_criteria = relax_criteria(st.session_state.current_criteria, relaxation_level)

        # Execute search with current criteria
        if st.session_state.parallel_execution_enabled and len(st.session_state.selected_models) > 1:
            # PARALLEL EXECUTION
            status_placeholder.info(
                f"🚀 Searching for {remaining_target:,} companies using {len(st.session_state.selected_models)} models..."
                + (f" (Attempt {relaxation_level + 1})" if relaxation_level > 0 else "")
            )

            try:
                start_time = time.time()

                # Extract values from session state BEFORE passing to thread
                selected_models = st.session_state.selected_models.copy()

                # Execute the parallel search in a thread pool
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    # Submit the job
                    future = executor.submit(
                        run_parallel_search_wrapper,
                        selected_models,
                        current_criteria,
                        remaining_target
                    )

                    # Show progress while waiting
                    while not future.done():
                        # Update progress bar
                        elapsed = time.time() - start_time
                        estimated_progress = min(elapsed / 30, 0.9)  # Assume ~30 seconds max
                        progress_bar.progress(estimated_progress)
                        time.sleep(0.5)

                    # Get the result
                    result = future.result()

                execution_time = time.time() - start_time

                # Extract results
                new_companies = result.get('companies', [])
                metadata = result.get('metadata', {})

                # Store attempt info
                search_attempts.append({
                    'relaxation_level': relaxation_level,
                    'found': len(new_companies),
                    'time': execution_time
                })

                # Deduplicate with already found companies
                seen_names = set()
                for company in all_found_companies:
                    if hasattr(company, 'name'):
                        seen_names.add(company.name.lower().strip())
                    else:
                        seen_names.add(company.get('name', '').lower().strip())

                unique_new = []
                for company in new_companies:
                    if hasattr(company, 'name'):
                        company_name = company.name.lower().strip()
                    else:
                        company_name = company.get('name', '').lower().strip()

                    if company_name not in seen_names:
                        seen_names.add(company_name)
                        unique_new.append(company)

                all_found_companies.extend(unique_new)
                remaining_target = target_count - len(all_found_companies)

                status_placeholder.success(
                    f"Found {len(unique_new)} new unique companies. "
                    f"Total: {len(all_found_companies)}/{target_count}"
                )

                # Check if we should continue
                if not enable_recursive or remaining_target <= 0:
                    break

                # Check if this attempt was too unsuccessful
                if len(unique_new) < remaining_target * 0.1:  # Less than 10% of what we need
                    relaxation_level += 1

            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                traceback.print_exc()
                break

        else:
            # SINGLE MODEL EXECUTION
            if target_count > 500:
                st.error("Single model cannot handle more than 500 companies.")
                break
            else:
                status_placeholder.info(f"Searching with {st.session_state.selected_models[0]}...")

                try:
                    # Import here to avoid circular imports
                    from shared.search_utils import EnhancedSearchStrategistAgent

                    agent = EnhancedSearchStrategistAgent(deployment_name=st.session_state.selected_models[0])

                    companies_this_round = []
                    calls_needed = (remaining_target + 14) // 15  # Updated for 15 companies per call

                    for call_num in range(calls_needed):
                        call_target = min(15, remaining_target - len(companies_this_round))
                        progress = (call_num + 1) / calls_needed
                        progress_bar.progress(progress * 0.7)

                        status_placeholder.info(f"Processing batch {call_num + 1}/{calls_needed}...")

                        # Run async code in a new event loop
                        result = run_single_search(agent, current_criteria, call_target)

                        companies = result.get("companies", [])
                        companies_this_round.extend(companies)

                        if call_num < calls_needed - 1:
                            time.sleep(0.5)

                    # Deduplicate
                    seen_names = set()
                    for company in all_found_companies:
                        if hasattr(company, 'name'):
                            seen_names.add(company.name.lower().strip())
                        else:
                            seen_names.add(company.get('name', '').lower().strip())

                    unique_new = []
                    for company in companies_this_round:
                        if hasattr(company, 'name'):
                            company_name = company.name.lower().strip()
                        else:
                            company_name = company.get('name', '').lower().strip()

                        if company_name not in seen_names:
                            seen_names.add(company_name)
                            unique_new.append(company)

                    all_found_companies.extend(unique_new)
                    remaining_target = target_count - len(all_found_companies)

                    if not enable_recursive or remaining_target <= 0:
                        break

                    relaxation_level += 1

                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    break

    # Final processing
    if existing_results:
        # Combine with existing
        all_companies = existing_results + all_found_companies

        # Final deduplication
        seen_names = set()
        seen_name_cores = set()
        unique_companies = []

        for company in all_companies:
            if hasattr(company, 'name'):
                company_name = company.name.lower().strip()
            else:
                company_name = company.get('name', '').lower().strip()

            # Create core name
            name_core = company_name
            for suffix in ['pty ltd', 'limited', 'ltd', 'inc', 'corporation', 'corp']:
                name_core = name_core.replace(suffix, '').strip()

            if name_core not in seen_name_cores and company_name not in seen_names:
                seen_names.add(company_name)
                seen_name_cores.add(name_core)
                unique_companies.append(company)

        st.session_state.search_results = unique_companies
        new_unique_count = len(unique_companies) - len(existing_results)

        status_placeholder.success(
            f"✅ Search complete! Added {new_unique_count} new unique companies!\n"
            f"Total: {len(unique_companies):,} companies"
        )
    else:
        # New search
        st.session_state.search_results = all_found_companies[:target_count]  # Trim to target

        success_message = f"✅ Found {len(st.session_state.search_results):,} companies!"
        if enable_recursive and relaxation_level > 0:
            success_message += f"\n🔄 Used {relaxation_level} levels of criteria relaxation"

        status_placeholder.success(success_message)

    # Update session state
    if 'metadata' in locals():
        st.session_state.model_success_status = metadata.get('model_stats', {})

    # Track search in history
    st.session_state.search_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'count': target_count,
        'found': len(all_found_companies),
        'mode': search_mode.split()[0],
        'recursive': enable_recursive,
        'relaxation_levels': relaxation_level if enable_recursive else 0
    })

    # Calculate actual cost
    actual_cost = estimated_cost * (1 + relaxation_level * 0.5) if enable_recursive else estimated_cost
    update_cost(actual_cost)

    progress_bar.progress(1.0)
    time.sleep(0.5)
    progress_bar.empty()


def display_search_results():
    """Display search results with revenue categories"""

    if st.session_state.search_results:
        st.divider()
        st.subheader(f"📊 Search Results ({len(st.session_state.search_results):,} companies)")

        # Filtering for large result sets
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

            # Map revenue categories to display names
            revenue_display = {
                "very_high": "$1B+",
                "high": "$100M-$1B",
                "medium": "$10M-$100M",
                "low": "$1M-$10M",
                "very_low": "<$1M",
                "unknown": "Unknown"
            }

            # Prepare display data
            row_data = {
                "#": i + 1,
                "Company": c.get('name', 'Unknown'),
                "Industry": c.get('industry_category', 'Unknown'),
                "Revenue Cat.": revenue_display.get(c.get('revenue_category', 'unknown'), 'Unknown'),
                "Est. Revenue": c.get('estimated_revenue', 'Unknown'),
                "Employees": c.get('estimated_employees', 'Unknown'),
                "ICP Score": c.get('icp_score', 0),
                "ICP Tier": c.get('icp_tier', 'D')
            }

            results_data.append(row_data)

        if results_data:
            df = pd.DataFrame(results_data)

            if len(st.session_state.search_results) > 100:
                st.write(f"Showing {len(df)} of {len(st.session_state.search_results):,} companies")

            # Style the dataframe to highlight revenue categories
            def style_revenue_cat(val):
                if val == "$1B+":
                    return 'background-color: #90EE90'
                elif val == "$100M-$1B":
                    return 'background-color: #98FB98'
                elif val == "$10M-$100M":
                    return 'background-color: #FFE4B5'
                elif val == "$1M-$10M":
                    return 'background-color: #F0E68C'
                else:
                    return ''

            styled_df = df.style.applymap(style_revenue_cat, subset=['Revenue Cat.'])

            # Pagination for large results
            if len(df) > 500:
                page_size = 100
                page = st.number_input("Page", 1, (len(df) + page_size - 1) // page_size, 1)
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, len(df))
                st.dataframe(styled_df.iloc[start_idx:end_idx], use_container_width=True, height=400)
            else:
                st.dataframe(styled_df, use_container_width=True, height=min(400, len(df) * 35 + 50))