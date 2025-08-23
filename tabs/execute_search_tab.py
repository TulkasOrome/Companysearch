# tabs/execute_search_tab.py
"""
Execute Search Tab - Fixed timing estimation and added live logging
"""

import streamlit as st
import asyncio
import pandas as pd
import time
import traceback
from datetime import datetime
from shared.search_utils import execute_parallel_search, EnhancedSearchStrategistAgent
from shared.session_state import update_cost


def render_execute_search_tab():
    """Render the Execute Search tab with fixed timing"""

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
                if criteria.financial.revenue_min:
                    st.write(f"Revenue: ${criteria.financial.revenue_min / 1e6:.0f}M+")
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

        # Number of companies selector with both slider and number input
        col1, col2 = st.columns([3, 1])

        with col1:
            # Slider for visual selection
            target_count_slider = st.slider(
                "🎯 Number of Companies to Find",
                min_value=10,
                max_value=10000,
                value=st.session_state.get('target_count', 100),
                step=10 if st.session_state.get('target_count', 100) < 1000 else 50,
                help="Select how many companies you want to find"
            )

        with col2:
            # Number input for precise entry
            target_count = st.number_input(
                "Or enter exact number",
                min_value=10,
                max_value=10000,
                value=target_count_slider,
                step=10,
                help="Type exact number of companies"
            )

        # Update slider if number input changes
        if target_count != target_count_slider:
            st.session_state.target_count = target_count
        else:
            st.session_state.target_count = target_count_slider
            target_count = target_count_slider

        # Compact metrics row
        col1, col2, col3 = st.columns(3)

        # Calculate requirements - FIXED TIMING ESTIMATION
        if st.session_state.parallel_execution_enabled:
            models_count = len(st.session_state.selected_models)
            # Each model processes in parallel, but each needs multiple calls for large targets
            calls_per_model = (target_count // (50 * models_count)) + 1
            total_calls = calls_per_model * models_count
            # Parallel execution: time is based on the slowest model, not sum of all
            # Each call takes about 10-15 seconds, models run in parallel
            estimated_time = calls_per_model * 12  # 12 seconds per call layer
        else:
            total_calls = (target_count // 50) + 1
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
                execute_search(target_count, total_calls, estimated_cost, search_mode)
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


def execute_search(target_count, total_calls, estimated_cost, search_mode):
    """Execute the search with progress tracking and live logging"""

    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    # Create a container for live logs
    log_container = st.container()
    log_placeholder = log_container.empty()
    logs = []

    def add_log(message):
        """Add a log message to the display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        logs.append(f"[{timestamp}] {message}")
        # Keep only last 20 logs for display
        display_logs = logs[-20:]
        log_text = "\n".join(display_logs)
        log_placeholder.text_area("Search Progress Log", log_text, height=300, disabled=True)

    # Store existing results if adding
    existing_results = []
    if "Add to Existing" in search_mode and st.session_state.search_results:
        existing_results = st.session_state.search_results.copy()
        status_placeholder.info(f"Adding to existing {len(existing_results):,} companies...")
        add_log(f"Mode: Adding to existing {len(existing_results):,} companies")
    else:
        # Clear for new search
        st.session_state.search_results = []
        st.session_state.all_search_results = []
        add_log("Mode: New search - clearing existing results")

    # Clear model tracking
    st.session_state.model_results = {}
    st.session_state.model_execution_times = {}
    st.session_state.model_success_status = {}

    if st.session_state.parallel_execution_enabled and len(st.session_state.selected_models) > 1:
        # PARALLEL EXECUTION
        status_placeholder.info(
            f"🚀 Searching for {target_count:,} companies using {len(st.session_state.selected_models)} models..."
        )
        add_log(f"Starting parallel search with {len(st.session_state.selected_models)} models")
        add_log(f"Target: {target_count:,} companies")

        try:
            start_time = time.time()

            # Capture logs from the parallel search by monkey-patching print
            original_print = print

            def logged_print(*args, **kwargs):
                message = " ".join(str(arg) for arg in args)
                add_log(message)
                original_print(*args, **kwargs)

            # Temporarily replace print
            import builtins
            builtins.print = logged_print

            # Run parallel search
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            add_log("Initializing parallel execution...")
            result = loop.run_until_complete(
                execute_parallel_search(
                    st.session_state.selected_models,
                    st.session_state.current_criteria,
                    target_count
                )
            )
            loop.close()

            # Restore original print
            builtins.print = original_print

            execution_time = time.time() - start_time
            add_log(f"Search completed in {execution_time:.1f} seconds")

            # Extract results
            new_companies = result.get('companies', [])
            metadata = result.get('metadata', {})

            # Handle existing results
            if existing_results:
                # Combine and deduplicate
                all_companies = existing_results + new_companies

                # Deduplicate
                seen_names = set()
                seen_name_cores = set()
                unique_companies = []
                duplicates_with_existing = 0

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
                    elif company in existing_results:
                        # Skip existing
                        unique_companies.append(company)
                    else:
                        duplicates_with_existing += 1

                st.session_state.search_results = unique_companies
                new_unique_count = len(unique_companies) - len(existing_results)

                add_log(f"✅ Added {new_unique_count} new unique companies")
                add_log(f"Total: {len(unique_companies):,} companies")
                add_log(f"Duplicates removed: {duplicates_with_existing}")

                # Clear the log area and show success
                log_placeholder.empty()
                status_placeholder.success(
                    f"✅ Added {new_unique_count} new unique companies!\n"
                    f"Total: {len(unique_companies):,} companies\n"
                    f"Duplicates with existing: {duplicates_with_existing}"
                )
            else:
                # New search
                st.session_state.search_results = new_companies
                add_log(f"✅ Found {len(new_companies):,} unique companies")

                # Clear the log area and show success
                log_placeholder.empty()
                status_placeholder.success(
                    f"✅ Found {len(new_companies):,} unique companies in {execution_time:.1f} seconds!"
                )

            # Update session state
            st.session_state.model_success_status = metadata.get('model_stats', {})

            # Track search in history
            st.session_state.search_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'count': target_count,
                'found': len(new_companies),
                'mode': search_mode.split()[0],
                'criteria_hash': hash(str(st.session_state.current_criteria))
            })

            # Calculate actual cost
            actual_calls = metadata.get('total_api_calls', total_calls)
            actual_cost = actual_calls * 0.02
            update_cost(actual_cost)

            progress_bar.progress(1.0)

        except Exception as e:
            add_log(f"❌ Search failed: {str(e)}")
            st.error(f"Search failed: {str(e)}")
            traceback.print_exc()

    else:
        # SINGLE MODEL EXECUTION
        if target_count > 500:
            st.error("Single model cannot handle more than 500 companies.")
        else:
            status_placeholder.info(f"Searching with {st.session_state.selected_models[0]}...")
            add_log(f"Starting single model search with {st.session_state.selected_models[0]}")

            try:
                agent = EnhancedSearchStrategistAgent(deployment_name=st.session_state.selected_models[0])

                all_companies = []
                calls_needed = (target_count + 49) // 50

                for call_num in range(calls_needed):
                    call_target = min(50, target_count - len(all_companies))
                    progress = (call_num + 1) / calls_needed
                    progress_bar.progress(progress)

                    add_log(f"Call {call_num + 1}/{calls_needed}: Requesting {call_target} companies")

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(
                        agent.generate_enhanced_strategy(
                            st.session_state.current_criteria,
                            target_count=call_target
                        )
                    )
                    loop.close()

                    companies = result.get("companies", [])
                    all_companies.extend(companies)
                    add_log(f"Call {call_num + 1} returned {len(companies)} companies")

                    if call_num < calls_needed - 1:
                        time.sleep(0.5)

                # Handle existing results
                if existing_results:
                    combined = existing_results + all_companies
                    # Simple dedup
                    seen = set()
                    unique = []
                    for c in combined:
                        name = c.name if hasattr(c, 'name') else c.get('name', '')
                        if name.lower() not in seen:
                            seen.add(name.lower())
                            unique.append(c)
                    st.session_state.search_results = unique
                else:
                    st.session_state.search_results = all_companies

                update_cost(calls_needed * 0.02)
                add_log(f"✅ Found {len(all_companies)} companies")

                # Clear the log area and show success
                log_placeholder.empty()
                status_placeholder.success(f"✅ Found {len(all_companies)} companies!")

            except Exception as e:
                add_log(f"❌ Search failed: {str(e)}")
                st.error(f"Search failed: {str(e)}")

    progress_bar.empty()
    status_placeholder.empty()


def display_search_results():
    """Display search results"""

    if st.session_state.search_results:
        st.divider()
        st.subheader(f"📊 Search Results ({len(st.session_state.search_results):,} companies)")

        # Filtering for large result sets
        if len(st.session_state.search_results) > 100:
            col1, col2, col3 = st.columns(3)

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
                search_filter = st.text_input(
                    "Search company names",
                    placeholder="Type to search..."
                )
        else:
            tier_filter = ["A", "B", "C", "D"]
            industry_filter = []
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
            if search_filter and search_filter.lower() not in c.get('name', '').lower():
                continue

            results_data.append({
                "#": i + 1,
                "Company": c.get('name', 'Unknown'),
                "Industry": c.get('industry_category', 'Unknown'),
                "Revenue": c.get('estimated_revenue', 'Unknown'),
                "Employees": c.get('estimated_employees', 'Unknown'),
                "ICP Score": c.get('icp_score', 0),
                "ICP Tier": c.get('icp_tier', 'D')
            })

        if results_data:
            df = pd.DataFrame(results_data)

            if len(st.session_state.search_results) > 100:
                st.write(f"Showing {len(df)} of {len(st.session_state.search_results):,} companies")

            # Pagination for large results
            if len(df) > 500:
                page_size = 100
                page = st.number_input("Page", 1, (len(df) + page_size - 1) // page_size, 1)
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, len(df))
                st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True, height=400)
            else:
                st.dataframe(df, use_container_width=True, height=min(400, len(df) * 35 + 50))