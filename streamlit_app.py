# enhanced_streamlit_app.py

import streamlit as st
import asyncio
import pandas as pd
import json
from datetime import datetime
import time
from typing import Dict, Any, List, Optional

# Import enhanced modules
from search_strategist_agent import (
    EnhancedSearchStrategistAgent,
    SearchCriteria,
    LocationCriteria,
    FinancialCriteria,
    OrganizationalCriteria,
    BehavioralSignals,
    BusinessType,
    EnhancedCompanyEntry
)
from validation_agent import (
    EnhancedValidationAgent,
    EnhancedValidationMode,
    EnhancedValidationResult
)
from session_manager import SessionManager

# Page config
st.set_page_config(
    page_title="Advanced Company Search & Validation Platform",
    page_icon="🔍",
    layout="wide"
)

# Initialize session manager
session_mgr = SessionManager()

# Title and description
st.title("🔍 Advanced AI-Powered Company Search & Validation Platform")
st.markdown("Comprehensive company discovery with location, financial, CSR, and behavioral criteria")

# Sidebar for session management
with st.sidebar:
    with st.expander("📊 Session Management", expanded=False):
        # Session info
        session_summary = session_mgr.get_session_summary()
        st.info(f"Session: {session_summary['session_id'][:12]}...")

        # Progress indicator
        progress = session_summary['progress']
        st.progress(progress)
        st.caption(f"Progress: {progress * 100:.0f}%")

        # Session stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Companies Found", session_summary['companies_found'])
            st.metric("Total Cost", f"${session_summary['total_cost']:.2f}")
        with col2:
            st.metric("Validated", session_summary['companies_validated'])
            st.metric("Duration", session_summary['duration'].split('.')[0])

        st.divider()

        # Save/Load section
        st.markdown("### 💾 Save/Load Session")

        # Save options
        if st.button("💾 Save Session", use_container_width=True):
            session_data = session_mgr.save_session_to_file()
            st.download_button(
                "📥 Download Session File",
                data=session_data,
                file_name=f"{st.session_state.session_id}.json",
                mime="application/json",
                use_container_width=True
            )

        # Generate resume link
        if st.button("🔗 Generate Resume Link", use_container_width=True):
            resume_link = session_mgr.generate_resume_link()
            st.code(resume_link, language=None)
            st.caption("Share this link to resume your session later")

        # Load session
        uploaded_file = st.file_uploader("Load Session", type=['json'])
        if uploaded_file:
            session_data = json.load(uploaded_file)
            if session_mgr.load_session_from_file(uploaded_file.name):
                st.rerun()

        st.divider()

        # Recent sessions
        st.markdown("### 📁 Recent Sessions")
        recent_sessions = session_mgr.get_recent_sessions(limit=3)
        for session in recent_sessions:
            if st.button(f"Load: {session['session_id'][:8]}...", key=session['filename']):
                if session_mgr.load_session_from_file(session['filename']):
                    st.rerun()

        # Clear session
        if st.button("🗑️ Clear Session", type="secondary", use_container_width=True):
            session_mgr.clear_session()
            st.rerun()

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Search Criteria",
    "🔍 Execute Search",
    "✅ Validate Results",
    "📊 Export Results"
])

# Tab 1: Enhanced Search Criteria
with tab1:
    st.header("Define Search Criteria")

    # Use columns for better layout
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("🌍 Geographic Criteria")

        # Location inputs
        countries = st.multiselect(
            "Countries",
            ["United States", "United Kingdom", "Canada", "Australia", "Germany",
             "France", "Japan", "Singapore", "India", "Brazil"],
            default=["Australia"]
        )

        states = st.text_input(
            "States/Regions (comma-separated)",
            placeholder="e.g., New South Wales, Victoria"
        )

        cities = st.text_input(
            "Cities (comma-separated)",
            placeholder="e.g., Sydney, Melbourne, Brisbane"
        )

        # Proximity search
        use_proximity = st.checkbox("Enable proximity search")
        proximity_location = None
        proximity_radius = None

        if use_proximity:
            proximity_location = st.text_input(
                "Central location",
                placeholder="e.g., Sydney CBD"
            )
            proximity_radius = st.number_input(
                "Radius (km)",
                min_value=1,
                max_value=500,
                value=50
            )

        location_exclusions = st.text_input(
            "Exclude locations (comma-separated)",
            placeholder="e.g., Rural areas"
        )

    with col2:
        st.subheader("💰 Financial Criteria")

        # Revenue range
        col2a, col2b = st.columns(2)
        with col2a:
            revenue_min = st.number_input(
                "Min Revenue (millions)",
                min_value=0.0,
                value=5.0,
                step=1.0
            )
        with col2b:
            revenue_max = st.number_input(
                "Max Revenue (millions)",
                min_value=0.0,
                value=100.0,
                step=1.0
            )

        revenue_currency = st.selectbox(
            "Currency",
            ["USD", "AUD", "EUR", "GBP", "CAD"],
            index=1
        )

        # Giving capacity
        giving_capacity = st.number_input(
            "Min Giving Capacity (thousands)",
            min_value=0.0,
            value=20.0,
            step=5.0
        )

        # Financial health
        require_profitable = st.checkbox("Profitable companies only")
        min_growth_rate = st.number_input(
            "Min Growth Rate (%)",
            min_value=0.0,
            value=0.0,
            step=5.0
        )

    with col3:
        st.subheader("🏢 Organizational Criteria")

        # Employee count
        col3a, col3b = st.columns(2)
        with col3a:
            employee_min = st.number_input(
                "Min Employees",
                min_value=0,
                value=50,
                step=10
            )
        with col3b:
            employee_max = st.number_input(
                "Max Employees",
                min_value=0,
                value=1000,
                step=50
            )

        # Office types
        office_types = st.multiselect(
            "Office Types",
            ["Headquarters", "Regional Office", "Branch Office", "Major Office"],
            default=["Headquarters", "Major Office"]
        )

        # Company stage
        company_stage = st.selectbox(
            "Company Stage",
            ["Any", "Startup", "Growth", "Mature", "Enterprise"],
            index=0
        )

    # Second row of criteria
    col4, col5, col6 = st.columns([1, 1, 1])

    with col4:
        st.subheader("🏭 Industry & Business")

        # Business type
        business_types = st.multiselect(
            "Business Types",
            [bt.value for bt in BusinessType],
            default=["B2B", "B2C"]
        )

        # Industries with priority
        st.caption("Add industries in priority order")
        industries = []
        num_industries = st.number_input("Number of industries", min_value=1, max_value=10, value=3)

        for i in range(num_industries):
            col_ind, col_pri = st.columns([3, 1])
            with col_ind:
                industry_name = st.text_input(
                    f"Industry {i + 1}",
                    key=f"ind_{i}",
                    placeholder="e.g., Construction/Trades"
                )
            with col_pri:
                priority = st.number_input(
                    "Priority",
                    min_value=1,
                    max_value=10,
                    value=i + 1,
                    key=f"pri_{i}"
                )
            if industry_name:
                industries.append({"name": industry_name, "priority": priority})

    with col5:
        st.subheader("🌟 CSR & Behavioral Signals")

        # CSR focus areas
        csr_focus_areas = st.multiselect(
            "CSR Focus Areas",
            ["Children", "Education", "Health", "Environment", "Community",
             "Diversity", "Veterans", "Elderly", "Arts", "Sports"],
            default=["Children", "Community"]
        )

        # Certifications
        certifications = st.multiselect(
            "Required Certifications",
            ["B-Corp", "ISO 26000", "ISO 14001", "LEED", "Fair Trade", "Carbon Neutral"],
            default=[]
        )

        # Recent events
        recent_events = st.multiselect(
            "Recent Events/Triggers",
            ["Office Move", "Expansion", "CSR Launch", "New Leadership",
             "Award Won", "IPO", "Acquisition", "Partnership"],
            default=["Office Move", "CSR Launch"]
        )

        # ESG maturity
        esg_maturity = st.select_slider(
            "Min ESG Maturity",
            options=["None", "Basic", "Developing", "Mature", "Leading"],
            value="Basic"
        )

    with col6:
        st.subheader("🚫 Exclusions & Custom")

        # Excluded industries
        excluded_industries = st.text_area(
            "Excluded Industries",
            placeholder="One per line\ne.g., Gambling\nTobacco\nAlcohol",
            height=100
        )

        # Excluded behaviors
        excluded_behaviors = st.multiselect(
            "Excluded Behaviors",
            ["Recent Misconduct", "Bankruptcy", "Major Lawsuits",
             "Environmental Violations", "Labor Issues"],
            default=["Recent Misconduct"]
        )

        # Custom keywords
        custom_keywords = st.text_input(
            "Additional Keywords",
            placeholder="e.g., innovative, technology-driven"
        )

        # Free text criteria
        custom_prompt = st.text_area(
            "Additional Requirements (free text)",
            placeholder="Any other specific requirements...",
            height=100
        )

    # Save criteria button
    if st.button("💾 Save Search Criteria", type="primary", use_container_width=True):
        # Build criteria object
        criteria = SearchCriteria(
            location=LocationCriteria(
                countries=countries,
                states=[s.strip() for s in states.split(',')] if states else [],
                cities=[c.strip() for c in cities.split(',')] if cities else [],
                proximity={"location": proximity_location, "radius_km": proximity_radius} if use_proximity else None,
                exclusions=[e.strip() for e in location_exclusions.split(',')] if location_exclusions else []
            ),
            financial=FinancialCriteria(
                revenue_min=revenue_min * 1_000_000 if revenue_min else None,
                revenue_max=revenue_max * 1_000_000 if revenue_max else None,
                revenue_currency=revenue_currency,
                giving_capacity_min=giving_capacity * 1_000 if giving_capacity else None,
                growth_rate_min=min_growth_rate if min_growth_rate > 0 else None,
                profitable=require_profitable if require_profitable else None
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=employee_min if employee_min > 0 else None,
                employee_count_max=employee_max if employee_max > 0 else None,
                office_types=office_types,
                company_stage=company_stage if company_stage != "Any" else None
            ),
            behavioral=BehavioralSignals(
                csr_focus_areas=csr_focus_areas,
                certifications=certifications,
                recent_events=recent_events,
                esg_maturity=esg_maturity if esg_maturity != "None" else None
            ),
            business_types=business_types,
            industries=industries,
            keywords=[k.strip() for k in custom_keywords.split(',')] if custom_keywords else [],
            custom_prompt=custom_prompt if custom_prompt else None,
            excluded_industries=[i.strip() for i in excluded_industries.split('\n')] if excluded_industries else [],
            excluded_behaviors=excluded_behaviors
        )

        # Save to session
        criteria_dict = {
            'location': {
                'countries': criteria.location.countries,
                'states': criteria.location.states,
                'cities': criteria.location.cities,
                'regions': criteria.location.regions,
                'proximity': criteria.location.proximity,
                'exclusions': criteria.location.exclusions
            },
            'financial': {
                'revenue_min': criteria.financial.revenue_min,
                'revenue_max': criteria.financial.revenue_max,
                'revenue_currency': criteria.financial.revenue_currency,
                'giving_capacity_min': criteria.financial.giving_capacity_min,
                'growth_rate_min': criteria.financial.growth_rate_min,
                'profitable': criteria.financial.profitable
            },
            'organizational': {
                'employee_count_min': criteria.organizational.employee_count_min,
                'employee_count_max': criteria.organizational.employee_count_max,
                'employee_count_by_location': criteria.organizational.employee_count_by_location,
                'office_types': criteria.organizational.office_types,
                'company_stage': criteria.organizational.company_stage
            },
            'behavioral': {
                'csr_programs': criteria.behavioral.csr_programs,
                'csr_focus_areas': criteria.behavioral.csr_focus_areas,
                'certifications': criteria.behavioral.certifications,
                'recent_events': criteria.behavioral.recent_events,
                'technology_stack': criteria.behavioral.technology_stack,
                'esg_maturity': criteria.behavioral.esg_maturity
            },
            'business_types': criteria.business_types,
            'industries': criteria.industries,
            'keywords': criteria.keywords,
            'custom_prompt': criteria.custom_prompt,
            'excluded_industries': criteria.excluded_industries,
            'excluded_companies': criteria.excluded_companies,
            'excluded_behaviors': criteria.excluded_behaviors
        }
        session_mgr.update_search_criteria(criteria_dict)
        st.success("✅ Search criteria saved!")
        st.session_state.current_step = 'search_execution'

# Tab 2: Execute Search
with tab2:
    st.header("Execute Company Search")

    if not st.session_state.get('search_criteria'):
        st.warning("⚠️ Please define search criteria first in the 'Search Criteria' tab.")
    else:
        # Display saved criteria
        with st.expander("📋 Review Search Criteria", expanded=True):
            criteria_dict = st.session_state.search_criteria

            # Format criteria for display
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Geographic:**")
                if criteria_dict['location']['countries']:
                    st.write(f"Countries: {', '.join(criteria_dict['location']['countries'])}")
                if criteria_dict['location']['cities']:
                    st.write(f"Cities: {', '.join(criteria_dict['location']['cities'])}")
                if criteria_dict['location']['proximity']:
                    st.write(
                        f"Proximity: {criteria_dict['location']['proximity']['radius_km']}km from {criteria_dict['location']['proximity']['location']}")

            with col2:
                st.markdown("**Financial:**")
                if criteria_dict['financial']['revenue_min'] or criteria_dict['financial']['revenue_max']:
                    rev_str = f"{criteria_dict['financial']['revenue_currency']} "
                    if criteria_dict['financial']['revenue_min']:
                        rev_str += f"{criteria_dict['financial']['revenue_min'] / 1_000_000:.0f}M"
                    rev_str += " - "
                    if criteria_dict['financial']['revenue_max']:
                        rev_str += f"{criteria_dict['financial']['revenue_max'] / 1_000_000:.0f}M"
                    st.write(f"Revenue: {rev_str}")
                if criteria_dict['financial']['giving_capacity_min']:
                    st.write(
                        f"Min Giving: {criteria_dict['financial']['revenue_currency']} {criteria_dict['financial']['giving_capacity_min'] / 1_000:.0f}K")

            with col3:
                st.markdown("**CSR/Behavioral:**")
                if criteria_dict['behavioral']['csr_focus_areas']:
                    st.write(f"CSR Focus: {', '.join(criteria_dict['behavioral']['csr_focus_areas'])}")
                if criteria_dict['behavioral']['recent_events']:
                    st.write(f"Events: {', '.join(criteria_dict['behavioral']['recent_events'])}")

        # Search configuration
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            # Target count per location
            target_per_location = st.number_input(
                "Target companies per location",
                min_value=10,
                max_value=500,
                value=50,
                step=10
            )

        with col2:
            # GPT deployment selection
            available_deployments = ["gpt-4.1", "gpt-4.1-2", "gpt-4.1-3"]
            selected_deployment = st.selectbox(
                "GPT-4 Deployment",
                available_deployments
            )

        with col3:
            # Process free text
            if criteria_dict.get('custom_prompt'):
                process_free_text = st.checkbox(
                    "Process free text criteria",
                    value=True,
                    help="Use GPT-4 to extract structured criteria from free text"
                )
            else:
                process_free_text = False

        # Calculate estimated cost
        num_locations = len(criteria_dict['location']['countries']) + len(criteria_dict['location']['cities'])
        if num_locations == 0:
            num_locations = 1
        estimated_gpt_calls = num_locations + (1 if process_free_text else 0)
        estimated_cost = estimated_gpt_calls * 0.02

        st.info(f"📊 Estimated: {estimated_gpt_calls} GPT-4 calls, ~${estimated_cost:.2f}")

        # Execute search button
        if st.button("🚀 Start Company Search", type="primary", use_container_width=True):
            with st.spinner("Searching for companies..."):
                # Initialize agent
                agent = EnhancedSearchStrategistAgent(deployment_name=selected_deployment)

                # Process free text if needed
                if process_free_text and criteria_dict.get('custom_prompt'):
                    with st.status("Processing free text criteria..."):
                        extracted = agent.extract_criteria_from_text(criteria_dict['custom_prompt'])
                        st.write("Extracted criteria:", extracted)

                # Reconstruct criteria object
                criteria = SearchCriteria(
                    location=LocationCriteria(**criteria_dict['location']),
                    financial=FinancialCriteria(**criteria_dict['financial']),
                    organizational=OrganizationalCriteria(**criteria_dict['organizational']),
                    behavioral=BehavioralSignals(**criteria_dict['behavioral']),
                    business_types=criteria_dict['business_types'],
                    industries=criteria_dict['industries'],
                    keywords=criteria_dict.get('keywords', []),
                    custom_prompt=criteria_dict.get('custom_prompt'),
                    excluded_industries=criteria_dict.get('excluded_industries', []),
                    excluded_companies=criteria_dict.get('excluded_companies', []),
                    excluded_behaviors=criteria_dict.get('excluded_behaviors', [])
                )

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()

                all_results = []
                start_time = time.time()

                # Search each location
                locations = criteria.location.cities if criteria.location.cities else criteria.location.countries
                if not locations:
                    locations = ["Global"]

                for idx, location in enumerate(locations):
                    status_text.text(f"Searching {location}...")

                    # Modify criteria for this location
                    location_criteria = criteria
                    if criteria.location.cities:
                        location_criteria.location.cities = [location]
                    else:
                        location_criteria.location.countries = [location]


                    # Execute search
                    async def run_search():
                        return await agent.generate_enhanced_strategy(
                            location_criteria,
                            target_count=target_per_location
                        )


                    result = asyncio.run(run_search())

                    # Store results
                    search_result = {
                        "location": location,
                        "timestamp": datetime.now().isoformat(),
                        "companies": [company.dict() for company in result['companies']],
                        "criteria": result['search_criteria']
                    }
                    all_results.append(search_result)

                    # Update progress
                    progress = (idx + 1) / len(locations)
                    progress_bar.progress(progress)

                    # Update session
                    session_mgr.add_search_results([search_result])
                    session_mgr.update_api_usage('gpt4', 1, 0.02)

                    # Show interim results
                    with results_container:
                        st.success(f"✅ {location}: Found {len(result['companies'])} companies")

                # Complete
                elapsed_time = time.time() - start_time
                status_text.text(f"Search completed in {elapsed_time:.1f} seconds!")

                # Summary
                total_companies = sum(len(r['companies']) for r in all_results)
                st.success(f"🎉 Found {total_companies} companies across {len(locations)} locations")

                # Preview results
                st.subheader("Search Results Preview")

                # Group by ICP tier
                all_companies = []
                for result in all_results:
                    for company in result['companies']:
                        company['location'] = result['location']
                        all_companies.append(company)

                # Create tiers
                tier_a = [c for c in all_companies if c.get('icp_tier') == 'A']
                tier_b = [c for c in all_companies if c.get('icp_tier') == 'B']
                tier_c = [c for c in all_companies if c.get('icp_tier') == 'C']
                others = [c for c in all_companies if c.get('icp_tier') not in ['A', 'B', 'C']]

                # Display by tier
                tab_a, tab_b, tab_c, tab_other = st.tabs([
                    f"Tier A ({len(tier_a)})",
                    f"Tier B ({len(tier_b)})",
                    f"Tier C ({len(tier_c)})",
                    f"Others ({len(others)})"
                ])


                def display_companies(companies, tab):
                    with tab:
                        if companies:
                            # Convert to dataframe
                            df_data = []
                            for c in companies:
                                df_data.append({
                                    "Company": c['name'],
                                    "Location": c['location'],
                                    "Industry": c['industry_category'],
                                    "Revenue": c.get('estimated_revenue', 'Unknown'),
                                    "Employees": c.get('estimated_employees', 'Unknown'),
                                    "ICP Score": f"{c.get('icp_score', 0):.1f}",
                                    "CSR Focus": ', '.join(c.get('csr_focus_areas', [])),
                                    "Matched Criteria": len(c.get('matched_criteria', []))
                                })

                            df = pd.DataFrame(df_data)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No companies in this tier")


                display_companies(tier_a, tab_a)
                display_companies(tier_b, tab_b)
                display_companies(tier_c, tab_c)
                display_companies(others, tab_other)

                # Update session state
                st.session_state.current_step = 'validation'

# Tab 3: Validate Results
with tab3:
    st.header("Validate Company Results")

    if not st.session_state.get('search_results'):
        st.warning("⚠️ Please execute a search first in the 'Execute Search' tab.")
    else:
        # Compile all companies
        all_companies = []
        for result in st.session_state.search_results:
            for company_data in result['companies']:
                # Reconstruct company object
                company = EnhancedCompanyEntry(**company_data)
                all_companies.append({
                    'company': company,
                    'location': result['location']
                })

        st.info(f"📊 {len(all_companies)} companies ready for validation")

        # Validation settings
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            # Serper API key
            serper_api_key = st.text_input(
                "Serper API Key",
                value="99c44b79892f5f7499accf2d7c26d93313880937",
                type="password"
            )

        with col2:
            # Validation mode
            validation_mode = st.selectbox(
                "Validation Mode",
                [
                    "Skip Validation",
                    "Places Only (Location)",
                    "Web Only (Company Info)",
                    "News Only (Recent Events)",
                    "Full (Places + Web)",
                    "Comprehensive (All Sources)"
                ],
                index=4
            )

            mode_map = {
                "Skip Validation": EnhancedValidationMode.SKIP,
                "Places Only (Location)": EnhancedValidationMode.PLACES_ONLY,
                "Web Only (Company Info)": EnhancedValidationMode.WEB_ONLY,
                "News Only (Recent Events)": EnhancedValidationMode.NEWS_ONLY,
                "Full (Places + Web)": EnhancedValidationMode.FULL,
                "Comprehensive (All Sources)": EnhancedValidationMode.COMPREHENSIVE
            }

        with col3:
            # Filter by tier
            tier_filter = st.multiselect(
                "Validate Tiers",
                ["A", "B", "C", "D", "Untiered"],
                default=["A", "B"]
            )

        # Additional filters
        col4, col5, col6 = st.columns([1, 1, 1])

        with col4:
            validate_unverified_only = st.checkbox(
                "Skip high-confidence companies",
                value=True
            )

        with col5:
            parallel_limit = st.slider(
                "Parallel Requests",
                min_value=1,
                max_value=10,
                value=3
            )

        with col6:
            # Calculate cost
            companies_to_validate = [
                c for c in all_companies
                if (not tier_filter or c['company'].icp_tier in tier_filter) and
                   (not validate_unverified_only or c['company'].confidence not in ['absolute', 'high'])
            ]

            if validation_mode != "Skip Validation":
                queries_per_company = {
                    "Places Only (Location)": 1,
                    "Web Only (Company Info)": 1,
                    "News Only (Recent Events)": 1,
                    "Full (Places + Web)": 2,
                    "Comprehensive (All Sources)": 4
                }.get(validation_mode, 1)

                estimated_queries = len(companies_to_validate) * queries_per_company
                estimated_cost = estimated_queries * 0.001

                st.metric("Est. Serper Cost", f"${estimated_cost:.2f}")
                st.caption(f"{estimated_queries} queries")

        # Companies to validate preview
        with st.expander(f"📋 Companies to Validate ({len(companies_to_validate)})", expanded=False):
            preview_data = []
            for item in companies_to_validate[:20]:  # Show first 20
                c = item['company']
                preview_data.append({
                    "Company": c.name,
                    "Location": item['location'],
                    "Tier": c.icp_tier,
                    "Current Confidence": c.confidence,
                    "ICP Score": f"{c.icp_score:.1f}"
                })

            if preview_data:
                df = pd.DataFrame(preview_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                if len(companies_to_validate) > 20:
                    st.caption(f"... and {len(companies_to_validate) - 20} more")

        # Execute validation
        if st.button("🔍 Start Validation", type="primary", use_container_width=True):
            if not serper_api_key and validation_mode != "Skip Validation":
                st.error("Please provide a Serper API key")
            else:
                with st.spinner("Validating companies..."):
                    # Initialize validation agent
                    validator = EnhancedValidationAgent(serper_api_key)

                    # Reconstruct criteria
                    criteria = SearchCriteria(**st.session_state.search_criteria)

                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Group by location for efficient validation
                    companies_by_location = {}
                    for item in companies_to_validate:
                        loc = item['location']
                        if loc not in companies_by_location:
                            companies_by_location[loc] = []
                        companies_by_location[loc].append(item['company'])

                    all_validation_results = []
                    processed = 0

                    # Validate each location group
                    for location, companies in companies_by_location.items():
                        status_text.text(f"Validating {len(companies)} companies in {location}...")

                        # Update criteria location
                        validation_criteria = criteria
                        if location in criteria.location.cities:
                            validation_criteria.location.cities = [location]
                        elif location in criteria.location.countries:
                            validation_criteria.location.countries = [location]


                        # Run validation
                        async def run_validation():
                            return await validator.validate_batch_enhanced(
                                companies,
                                validation_criteria,
                                mode=mode_map[validation_mode],
                                parallel_limit=parallel_limit
                            )


                        results = asyncio.run(run_validation())
                        all_validation_results.extend(results)

                        # Update progress
                        processed += len(companies)
                        progress_bar.progress(processed / len(companies_to_validate))

                        # Update session
                        session_mgr.update_api_usage(
                            'serper',
                            sum(r.serper_queries_used for r in results),
                            sum(r.serper_queries_used for r in results) * 0.001
                        )

                    # Save validation results
                    session_mgr.add_validation_results(all_validation_results)

                    # Complete
                    status_text.text("Validation complete!")

                    # Show summary
                    verified = sum(1 for r in all_validation_results if r.validation_status == "verified")
                    partial = sum(1 for r in all_validation_results if r.validation_status == "partial")
                    rejected = sum(1 for r in all_validation_results if r.validation_status == "rejected")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("✅ Verified", verified)
                    with col2:
                        st.metric("⚠️ Partial", partial)
                    with col3:
                        st.metric("❌ Rejected", rejected)
                    with col4:
                        st.metric("Total Validated", len(all_validation_results))

                    # Preview results
                    st.subheader("Validation Results")

                    # Create tabs for different statuses
                    tab_verified, tab_partial, tab_rejected = st.tabs([
                        f"Verified ({verified})",
                        f"Partial ({partial})",
                        f"Rejected ({rejected})"
                    ])


                    def display_validation_results(results, status, tab):
                        with tab:
                            filtered = [r for r in results if r.validation_status == status]
                            if filtered:
                                df_data = []
                                for r in filtered[:50]:  # Limit display
                                    df_data.append({
                                        "Company": r.company_name,
                                        "Location": r.country,
                                        "Original Conf.": r.original_confidence,
                                        "New Conf.": r.confidence_after_validation,
                                        "Val Score": f"{r.validation_score:.1f}",
                                        "Location OK": "✓" if r.location_verified else "✗",
                                        "Revenue OK": "✓" if r.revenue_verified else "✗",
                                        "CSR Found": "✓" if r.csr_evidence else "✗",
                                        "Queries": r.serper_queries_used
                                    })

                                df = pd.DataFrame(df_data)
                                st.dataframe(df, use_container_width=True, hide_index=True)

                                if len(filtered) > 50:
                                    st.caption(f"Showing first 50 of {len(filtered)} results")
                            else:
                                st.info(f"No {status} companies")


                    display_validation_results(all_validation_results, "verified", tab_verified)
                    display_validation_results(all_validation_results, "partial", tab_partial)
                    display_validation_results(all_validation_results, "rejected", tab_rejected)

                    # Update session state
                    st.session_state.current_step = 'export'

# Tab 4: Export Results
with tab4:
    st.header("Export Results")

    if not st.session_state.get('search_results'):
        st.warning("⚠️ No results to export. Please run a search first.")
    else:
        # Export configuration
        st.subheader("📋 Select Fields to Export")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Basic Information**")
            export_name = st.checkbox("Company Name", value=True, disabled=True)
            export_location = st.checkbox("Location", value=True)
            export_industry = st.checkbox("Industry", value=True)
            export_business_type = st.checkbox("Business Type", value=True)
            export_confidence = st.checkbox("Confidence Score", value=True)
            export_icp = st.checkbox("ICP Tier & Score", value=True)

        with col2:
            st.markdown("**Financial & Organizational**")
            export_revenue = st.checkbox("Revenue", value=True)
            export_employees = st.checkbox("Employee Count", value=True)
            export_hq = st.checkbox("Headquarters Info", value=True)
            export_offices = st.checkbox("Office Locations", value=False)
            export_financial_health = st.checkbox("Financial Health", value=False)
            export_giving = st.checkbox("Giving History", value=True)

        with col3:
            st.markdown("**CSR & Validation**")
            export_csr_programs = st.checkbox("CSR Programs", value=True)
            export_csr_focus = st.checkbox("CSR Focus Areas", value=True)
            export_certifications = st.checkbox("Certifications", value=True)
            export_recent_events = st.checkbox("Recent Events", value=False)
            export_validation = st.checkbox("Validation Results", value=True)
            export_sources = st.checkbox("Data Sources", value=False)

        # Filter options
        st.subheader("🔍 Filter Results")

        col4, col5, col6 = st.columns(3)

        with col4:
            # Status filter
            if st.session_state.get('validation_results'):
                status_filter = st.multiselect(
                    "Validation Status",
                    ["verified", "partial", "unverified", "rejected"],
                    default=["verified", "partial"]
                )
            else:
                status_filter = []

            # Tier filter
            tier_filter = st.multiselect(
                "ICP Tiers",
                ["A", "B", "C", "D"],
                default=["A", "B"]
            )

        with col5:
            # Location filter
            all_locations = list(set(r['location'] for r in st.session_state.search_results))
            location_filter = st.multiselect(
                "Locations",
                all_locations,
                default=all_locations
            )

            # Industry filter
            all_industries = list(set(
                c['industry_category']
                for r in st.session_state.search_results
                for c in r['companies']
            ))
            industry_filter = st.multiselect(
                "Industries",
                all_industries,
                default=all_industries[:3] if len(all_industries) > 3 else all_industries
            )

        with col6:
            # Size filter
            size_filter = st.multiselect(
                "Company Size",
                ["small", "medium", "enterprise", "unknown"],
                default=["medium", "enterprise"]
            )

            # CSR match filter
            require_csr_match = st.checkbox("CSR criteria match only", value=False)

        # Compile filtered results
        filtered_companies = []

        # Create validation lookup
        validation_lookup = {}
        if st.session_state.get('validation_results'):
            for val_result in st.session_state.validation_results:
                if hasattr(val_result, 'company_name'):
                    key = f"{val_result.company_name.lower()}_{val_result.country}"
                    validation_lookup[key] = val_result

        # Filter companies
        for result in st.session_state.search_results:
            if result['location'] not in location_filter:
                continue

            for company_data in result['companies']:
                company = EnhancedCompanyEntry(**company_data)

                # Apply filters
                if tier_filter and company.icp_tier not in tier_filter:
                    continue
                if industry_filter and company.industry_category not in industry_filter:
                    continue
                if size_filter and company.company_size not in size_filter:
                    continue

                # Check validation status
                val_key = f"{company.name.lower()}_{result['location']}"
                val_result = validation_lookup.get(val_key)

                if status_filter and val_result and val_result.validation_status not in status_filter:
                    continue

                # Check CSR match
                if require_csr_match:
                    criteria = SearchCriteria(**st.session_state.search_criteria)
                    if criteria.behavioral.csr_focus_areas:
                        if not any(area in company.csr_focus_areas for area in criteria.behavioral.csr_focus_areas):
                            continue

                # Add to filtered list
                filtered_companies.append({
                    'company': company,
                    'location': result['location'],
                    'validation': val_result
                })

        st.info(f"📊 {len(filtered_companies)} companies match your filters")

        # Export formats
        st.subheader("📥 Download Results")

        col7, col8, col9 = st.columns(3)

        with col7:
            # CSV export
            if st.button("📄 Generate CSV", use_container_width=True):
                # Build CSV data
                csv_data = []

                for item in filtered_companies:
                    company = item['company']
                    val = item['validation']

                    row = {}

                    # Basic info
                    row['Company Name'] = company.name
                    if export_location:
                        row['Location'] = item['location']
                    if export_industry:
                        row['Industry'] = company.industry_category
                        row['Sub-Industry'] = company.sub_industry or ''
                    if export_business_type:
                        row['Business Type'] = company.business_type
                    if export_confidence:
                        row['Confidence'] = company.confidence
                    if export_icp:
                        row['ICP Tier'] = company.icp_tier or ''
                        row['ICP Score'] = company.icp_score or 0

                    # Financial
                    if export_revenue:
                        row['Revenue'] = company.estimated_revenue or ''
                        row['Revenue Currency'] = company.revenue_currency or ''
                    if export_employees:
                        row['Employees'] = company.estimated_employees or ''
                    if export_hq and company.headquarters:
                        row['HQ Address'] = company.headquarters.get('address', '')
                        row['HQ City'] = company.headquarters.get('city', '')
                    if export_financial_health:
                        row['Financial Health'] = company.financial_health or ''

                    # CSR
                    if export_csr_programs:
                        row['CSR Programs'] = ', '.join(company.csr_programs)
                    if export_csr_focus:
                        row['CSR Focus Areas'] = ', '.join(company.csr_focus_areas)
                    if export_certifications:
                        row['Certifications'] = ', '.join(company.certifications)

                    # Validation
                    if export_validation and val:
                        row['Validation Status'] = val.validation_status
                        row['Validation Score'] = val.validation_score
                        row['Location Verified'] = 'Yes' if val.location_verified else 'No'
                        row['Revenue Verified'] = 'Yes' if val.revenue_verified else 'No'

                    csv_data.append(row)

                # Convert to DataFrame
                df = pd.DataFrame(csv_data)

                # Download
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    data=csv,
                    file_name=f"company_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col8:
            # JSON export
            if st.button("📋 Generate JSON", use_container_width=True):
                # Build JSON data
                json_data = {
                    'export_time': datetime.now().isoformat(),
                    'search_criteria': st.session_state.search_criteria,
                    'filters_applied': {
                        'status': status_filter,
                        'tiers': tier_filter,
                        'locations': location_filter,
                        'industries': industry_filter,
                        'sizes': size_filter
                    },
                    'total_companies': len(filtered_companies),
                    'companies': []
                }

                for item in filtered_companies:
                    company_dict = item['company'].dict()

                    # Add validation data if available
                    if item['validation']:
                        company_dict['validation_result'] = {
                            'status': item['validation'].validation_status,
                            'score': item['validation'].validation_score,
                            'location_verified': item['validation'].location_verified,
                            'revenue_verified': item['validation'].revenue_verified,
                            'csr_evidence': bool(item['validation'].csr_evidence)
                        }

                    json_data['companies'].append(company_dict)

                # Download
                json_str = json.dumps(json_data, indent=2, default=str)
                st.download_button(
                    "📥 Download JSON",
                    data=json_str,
                    file_name=f"company_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

        with col9:
            # Executive summary
            if st.button("📊 Generate Summary", use_container_width=True):
                # Create summary
                summary = f"""# Company Search Results Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Search Overview
- **Total Companies Found:** {len(filtered_companies)}
- **Locations Searched:** {', '.join(location_filter)}
- **Industries Targeted:** {', '.join(industry_filter[:5])}{'...' if len(industry_filter) > 5 else ''}

## Results by Tier
"""

                # Tier breakdown
                tier_counts = {}
                for item in filtered_companies:
                    tier = item['company'].icp_tier or 'Untiered'
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1

                for tier in ['A', 'B', 'C', 'D']:
                    if tier in tier_counts:
                        summary += f"- **Tier {tier}:** {tier_counts[tier]} companies\n"

                # Validation summary
                if validation_lookup:
                    summary += "\n## Validation Results\n"
                    val_counts = {}
                    for item in filtered_companies:
                        if item['validation']:
                            status = item['validation'].validation_status
                            val_counts[status] = val_counts.get(status, 0) + 1

                    for status, count in val_counts.items():
                        summary += f"- **{status.capitalize()}:** {count} companies\n"

                # Top companies
                summary += "\n## Top 10 Companies by ICP Score\n"
                sorted_companies = sorted(
                    filtered_companies,
                    key=lambda x: x['company'].icp_score or 0,
                    reverse=True
                )[:10]

                for i, item in enumerate(sorted_companies, 1):
                    company = item['company']
                    summary += f"{i}. **{company.name}** ({item['location']}) - Score: {company.icp_score:.1f}\n"
                    summary += f"   - Industry: {company.industry_category}\n"
                    summary += f"   - Revenue: {company.estimated_revenue or 'Unknown'}\n"
                    if company.csr_focus_areas:
                        summary += f"   - CSR Focus: {', '.join(company.csr_focus_areas)}\n"

                # Cost summary
                summary += f"\n## Search Costs\n"
                summary += f"- **Total Cost:** ${st.session_state.total_cost:.2f}\n"
                summary += f"- **GPT-4 Calls:** {st.session_state.api_calls.get('gpt4', 0)}\n"
                summary += f"- **Serper Queries:** {st.session_state.api_calls.get('serper', 0)}\n"

                # Download summary
                st.download_button(
                    "📥 Download Summary",
                    data=summary,
                    file_name=f"search_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

        # Preview export
        if filtered_companies:
            st.subheader("📋 Export Preview")

            # Show first 10 rows
            preview_data = []
            for item in filtered_companies[:10]:
                company = item['company']
                val = item['validation']

                preview_data.append({
                    "Company": company.name,
                    "Location": item['location'],
                    "Tier": company.icp_tier,
                    "Score": f"{company.icp_score:.1f}" if company.icp_score else "N/A",
                    "Revenue": company.estimated_revenue or "Unknown",
                    "CSR Match": "✓" if any(area in company.csr_focus_areas for area in
                                            st.session_state.search_criteria.get('behavioral', {}).get(
                                                'csr_focus_areas', [])) else "✗",
                    "Validated": val.validation_status if val else "No"
                })

            df_preview = pd.DataFrame(preview_data)
            st.dataframe(df_preview, use_container_width=True, hide_index=True)

            if len(filtered_companies) > 10:
                st.caption(f"Showing first 10 of {len(filtered_companies)} companies")

# Footer
st.divider()
st.caption(f"Session: {st.session_state.session_id} | Cost: ${st.session_state.total_cost:.2f}")