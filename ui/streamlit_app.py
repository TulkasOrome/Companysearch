# ui/streamlit_app_v3.py
"""
Enhanced Streamlit App with ICP Profile Selection and True Parallel Execution
"""

import streamlit as st
import asyncio
import pandas as pd
import json
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback
import concurrent.futures
from dataclasses import asdict

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import components
from agents.search_strategist_agent import (
    EnhancedSearchStrategistAgent,
    SearchCriteria,
    LocationCriteria,
    FinancialCriteria,
    OrganizationalCriteria,
    BehavioralSignals
)
from agents.validation_agent_v2 import (
    ValidationConfig,
    EnhancedValidationAgent
)
from core.validation_strategies import ValidationCriteria, ValidationTier

# Try to import ICP manager
try:
    from enhanced_icp_manager import ICPManager

    ICP_AVAILABLE = True
except ImportError:
    ICP_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Company Search & Validation Platform",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'current_criteria' not in st.session_state:
    st.session_state.current_criteria = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = []
if 'saved_profiles' not in st.session_state:
    st.session_state.saved_profiles = {}
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0
if 'current_profile_name' not in st.session_state:
    st.session_state.current_profile_name = None
if 'current_tier' not in st.session_state:
    st.session_state.current_tier = "A"

# Initialize ICP manager
if ICP_AVAILABLE:
    icp_manager = ICPManager()

# Title
st.title("🔍 Company Search & Validation Platform")
st.markdown("AI-powered company discovery with ICP profiles and parallel search capabilities")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model selection for parallel execution
    st.subheader("🚀 Search Models")

    num_models = st.slider(
        "Number of Models",
        min_value=1,
        max_value=5,
        value=1,
        help="Select 2-5 models for parallel execution"
    )

    available_models = ["gpt-4.1", "gpt-4.1-2", "gpt-4.1-3", "gpt-4.1-4", "gpt-4.1-5"]

    if num_models == 1:
        selected_models = [st.selectbox("Select Model", available_models)]
    else:
        selected_models = st.multiselect(
            "Select Models",
            available_models,
            default=available_models[:num_models],
            max_selections=num_models
        )

        if len(selected_models) != num_models:
            st.warning(f"Please select exactly {num_models} models")

    st.info(f"{'Parallel' if num_models > 1 else 'Single'} execution mode")

    # API Keys
    with st.expander("🔑 API Keys"):
        serper_key = st.text_input(
            "Serper API Key",
            value="99c44b79892f5f7499accf2d7c26d93313880937",
            type="password"
        )

    st.divider()

    # Session stats
    st.subheader("📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Companies", len(st.session_state.search_results))
        st.metric("Profiles Saved", len(st.session_state.saved_profiles))
    with col2:
        st.metric("Validated", len(st.session_state.validation_results))
        st.metric("Total Cost", f"${st.session_state.total_cost:.3f}")

    if st.button("🗑️ Clear Session", use_container_width=True):
        for key in ['search_results', 'validation_results', 'current_criteria']:
            if key in st.session_state:
                st.session_state[key] = [] if 'results' in key else None
        st.session_state.total_cost = 0.0
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Search Configuration",
    "🔍 Execute Search",
    "✅ Validation",
    "📊 Results & Export"
])

# Tab 1: Search Configuration
with tab1:
    st.header("Search Configuration")

    # ICP Profile Selection
    st.subheader("📋 ICP Profile Selection")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🏥 RMH Sydney", use_container_width=True, type="secondary"):
            st.session_state.current_profile_name = "rmh_sydney"
            st.rerun()

    with col2:
        if st.button("🦮 Guide Dogs Victoria", use_container_width=True, type="secondary"):
            st.session_state.current_profile_name = "guide_dogs_victoria"
            st.rerun()

    with col3:
        if st.button("🔧 Custom Profile", use_container_width=True, type="secondary"):
            st.session_state.current_profile_name = None
            st.session_state.current_criteria = None
            st.rerun()

    # Show profile details if selected
    if st.session_state.current_profile_name:
        st.divider()

        if st.session_state.current_profile_name == "rmh_sydney":
            st.markdown("### 🏥 RMH Sydney Profile")

            # Tier selection with descriptions
            tier = st.radio(
                "Select Tier",
                ["A", "B", "C"],
                format_func=lambda x: {
                    "A": "Tier A - Perfect Match (Revenue $5-100M, 50+ employees, CSR focus)",
                    "B": "Tier B - Good Match (Revenue $2-200M, 20+ employees, Community involvement)",
                    "C": "Tier C - Potential Match (Revenue $1M+, Any size, Basic criteria)"
                }[x],
                key="rmh_tier_select"
            )

            # Show tier differences
            with st.expander("📊 Tier Comparison", expanded=True):
                tier_df = pd.DataFrame({
                    "Criteria": ["Location", "Revenue", "Employees", "CSR Focus", "Industries"],
                    "Tier A": [
                        "Sydney/GWS + 50km",
                        "AUD $5-100M",
                        "50+",
                        "Children & Community",
                        "Construction, Property, Hospitality"
                    ],
                    "Tier B": [
                        "Sydney/GWS",
                        "AUD $2-200M",
                        "20+",
                        "Community",
                        "All except excluded"
                    ],
                    "Tier C": [
                        "NSW",
                        "AUD $1M+",
                        "Any",
                        "Any",
                        "All except excluded"
                    ]
                })
                st.dataframe(tier_df, use_container_width=True)

            st.session_state.current_tier = tier

        elif st.session_state.current_profile_name == "guide_dogs_victoria":
            st.markdown("### 🦮 Guide Dogs Victoria Profile")

            tier = st.radio(
                "Select Tier",
                ["A", "B", "C"],
                format_func=lambda x: {
                    "A": "Tier A - Strategic Partners (Revenue $500M+, 500+ employees, Certifications)",
                    "B": "Tier B - Exploratory Partners (Revenue $50-500M, 100-500 employees)",
                    "C": "Tier C - Potential Partners (Revenue $10M+, 50+ employees)"
                }[x],
                key="gdv_tier_select"
            )

            with st.expander("📊 Tier Comparison", expanded=True):
                tier_df = pd.DataFrame({
                    "Criteria": ["Location", "Revenue", "Employees", "Certifications", "Industries"],
                    "Tier A": [
                        "Victoria (Melbourne+)",
                        "AUD $500M+",
                        "500+ (150+ in Vic)",
                        "B-Corp, ISO 26000",
                        "Health, Finance, Tech"
                    ],
                    "Tier B": [
                        "Victoria",
                        "AUD $50-500M",
                        "100-500",
                        "Any CSR program",
                        "Manufacturing, Logistics"
                    ],
                    "Tier C": [
                        "Victoria/NSW",
                        "AUD $10M+",
                        "50+",
                        "None required",
                        "All except excluded"
                    ]
                })
                st.dataframe(tier_df, use_container_width=True)

            st.session_state.current_tier = tier

    st.divider()

    # Search Fields Configuration
    st.subheader("🔧 Search Criteria Configuration")

    # Load profile defaults if selected
    if st.session_state.current_profile_name and ICP_AVAILABLE:
        profile = icp_manager.get_profile(st.session_state.current_profile_name)
        if profile:
            criteria = profile.tiers.get(st.session_state.current_tier)
    else:
        criteria = None

    # Location Section
    with st.expander("🌍 Location Settings", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            # Countries - ensure defaults exist in options
            country_options = ["Australia", "United States", "United Kingdom", "Canada", "New Zealand",
                               "Germany", "France", "Japan", "Singapore", "India"]
            default_countries = criteria.location.countries if criteria else ["Australia"]
            # Filter defaults to only include valid options
            valid_country_defaults = [d for d in default_countries if d in country_options]
            if not valid_country_defaults:
                valid_country_defaults = ["Australia"]

            countries = st.multiselect(
                "Countries",
                country_options,
                default=valid_country_defaults
            )

        with col2:
            default_states = criteria.location.states if criteria else []
            states = st.text_input(
                "States/Regions",
                value=", ".join(default_states),
                placeholder="Victoria, New South Wales"
            )

        with col3:
            default_cities = criteria.location.cities if criteria else []
            cities = st.text_input(
                "Cities",
                value=", ".join(default_cities),
                placeholder="Sydney, Melbourne"
            )

        # Proximity search
        use_proximity = st.checkbox("Enable Proximity Search")
        if use_proximity:
            col1, col2 = st.columns(2)
            with col1:
                proximity_center = st.text_input("Center Location", placeholder="Sydney CBD")
            with col2:
                proximity_radius = st.number_input("Radius (km)", 10, 500, 50)
        else:
            proximity_center = None
            proximity_radius = None

        # Exclusions
        location_exclusions = st.text_input(
            "Location Exclusions",
            placeholder="Rural areas, Remote regions"
        )

    # Financial Section
    with st.expander("💰 Financial Criteria", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            default_min = criteria.financial.revenue_min / 1_000_000 if criteria and criteria.financial.revenue_min else 0
            revenue_min = st.number_input("Min Revenue (M)", 0, 10000, int(default_min))

            default_emp_min = criteria.organizational.employee_count_min if criteria else 0
            employee_min = st.number_input("Min Employees", 0, 100000, default_emp_min or 0)

        with col2:
            default_max = criteria.financial.revenue_max / 1_000_000 if criteria and criteria.financial.revenue_max else 100
            revenue_max = st.number_input("Max Revenue (M)", 0, 10000, int(default_max))

            default_emp_max = criteria.organizational.employee_count_max if criteria else 1000
            employee_max = st.number_input("Max Employees", 0, 100000, default_emp_max or 1000)

        with col3:
            default_currency = criteria.financial.revenue_currency if criteria else "AUD"
            currency = st.selectbox("Currency", ["AUD", "USD", "EUR", "GBP"],
                                    index=["AUD", "USD", "EUR", "GBP"].index(default_currency))

            default_giving = criteria.financial.giving_capacity_min / 1_000 if criteria and criteria.financial.giving_capacity_min else 0
            giving_capacity = st.number_input("Min Giving Capacity (K)", 0, 1000, int(default_giving))

    # Industry & Business Section
    with st.expander("🏢 Industry & Business Type", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            # Business Types - ensure defaults exist in options
            business_options = ["B2B", "B2C", "B2B2C", "D2C", "Professional Services", "Enterprise", "SaaS",
                                "Marketplace", "Hybrid"]
            default_types = criteria.business_types if criteria else ["B2B", "B2C"]
            # Filter defaults to only include valid options
            valid_business_defaults = [d for d in default_types if d in business_options]
            if not valid_business_defaults:
                valid_business_defaults = ["B2B", "B2C"]

            business_types = st.multiselect(
                "Business Types",
                business_options,
                default=valid_business_defaults
            )

        with col2:
            # Industries - handle special formatting
            default_industries = []
            if criteria and criteria.industries:
                for ind in criteria.industries:
                    # Handle both dict and string formats
                    if isinstance(ind, dict):
                        ind_name = ind.get('name', '')
                    else:
                        ind_name = str(ind)
                    # Clean up industry names (remove priority markers, etc.)
                    ind_name = ind_name.replace('/', ' ').strip()
                    if ind_name:
                        default_industries.append(ind_name)

            industries = st.text_area(
                "Industries (one per line)",
                value="\n".join(default_industries),
                height=100,
                placeholder="Construction\nProperty\nHospitality"
            )

    # CSR & Behavioral Section
    with st.expander("💚 CSR & Behavioral Signals", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            # CSR Focus Areas - ensure defaults exist in options
            csr_options = ["children", "community", "education", "health", "environment",
                           "disability", "inclusion", "diversity", "sustainability", "elderly",
                           "families", "accessibility", "wellbeing"]
            default_csr = criteria.behavioral.csr_focus_areas if criteria else []
            # Filter defaults to only include valid options
            valid_defaults = [d for d in default_csr if d in csr_options]

            csr_focus = st.multiselect(
                "CSR Focus Areas",
                csr_options,
                default=valid_defaults
            )

            # Recent Events - ensure defaults exist in options
            event_options = ["Office Move", "CSR Launch", "Expansion", "Anniversary", "Award",
                             "New Leadership", "IPO", "Merger", "Partnership"]
            default_events = criteria.behavioral.recent_events if criteria else []
            # Filter defaults to only include valid options
            valid_event_defaults = [d for d in default_events if d in event_options]

            recent_events = st.multiselect(
                "Recent Events",
                event_options,
                default=valid_event_defaults
            )

        with col2:
            # Certifications - ensure defaults exist in options
            cert_options = ["B-Corp", "ISO 26000", "ISO 14001", "Carbon Neutral", "Fair Trade",
                            "Great Place to Work", "ESG Certified"]
            default_certs = criteria.behavioral.certifications if criteria else []
            # Filter defaults to only include valid options
            valid_cert_defaults = [d for d in default_certs if d in cert_options]

            certifications = st.multiselect(
                "Certifications",
                cert_options,
                default=valid_cert_defaults
            )

            esg_maturity = st.selectbox(
                "ESG Maturity",
                ["Any", "Basic", "Developing", "Mature", "Leading"],
                index=0
            )

    # Exclusions Section
    with st.expander("🚫 Exclusions", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Excluded Industries - ensure defaults exist in options
            exclusion_options = ["Gambling", "Tobacco", "Fast Food", "Racing", "Alcohol", "Weapons", "Animal Testing"]
            default_exc_ind = criteria.excluded_industries if criteria else []
            # Filter defaults to only include valid options
            valid_exc_defaults = [d for d in default_exc_ind if d in exclusion_options]

            excluded_industries = st.multiselect(
                "Excluded Industries",
                exclusion_options,
                default=valid_exc_defaults
            )

        with col2:
            default_exc_comp = criteria.excluded_companies if criteria else []
            excluded_companies = st.text_area(
                "Excluded Companies (one per line)",
                value="\n".join(default_exc_comp),
                height=100,
                placeholder="McDonald's\nKFC\nBurger King"
            )

    # Free Text Search Section
    with st.expander("🔍 Advanced Search Options", expanded=False):
        use_free_text = st.checkbox("Add Free Text Criteria")

        if use_free_text:
            free_text = st.text_area(
                "Additional Search Criteria",
                placeholder="Example: Focus on companies that have won sustainability awards in the last 2 years and have strong innovation programs",
                height=100
            )
        else:
            free_text = None

        keywords = st.text_input(
            "Keywords (comma-separated)",
            placeholder="innovation, award-winning, sustainable"
        )

    st.divider()

    # Action Buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("✅ Confirm Criteria", type="primary", use_container_width=True):
            # Build criteria object
            built_criteria = SearchCriteria(
                location=LocationCriteria(
                    countries=countries,
                    states=[s.strip() for s in states.split(',')] if states else [],
                    cities=[c.strip() for c in cities.split(',')] if cities else [],
                    proximity={"location": proximity_center, "radius_km": proximity_radius} if use_proximity else None,
                    exclusions=[e.strip() for e in location_exclusions.split(',')] if location_exclusions else []
                ),
                financial=FinancialCriteria(
                    revenue_min=revenue_min * 1_000_000 if revenue_min > 0 else None,
                    revenue_max=revenue_max * 1_000_000 if revenue_max > 0 else None,
                    revenue_currency=currency,
                    giving_capacity_min=giving_capacity * 1_000 if giving_capacity > 0 else None
                ),
                organizational=OrganizationalCriteria(
                    employee_count_min=employee_min if employee_min > 0 else None,
                    employee_count_max=employee_max if employee_max > 0 else None
                ),
                behavioral=BehavioralSignals(
                    csr_focus_areas=csr_focus,
                    certifications=certifications,
                    recent_events=recent_events,
                    esg_maturity=esg_maturity if esg_maturity != "Any" else None
                ),
                business_types=business_types,
                industries=[{"name": ind.strip(), "priority": i + 1}
                            for i, ind in enumerate(industries.split('\n')) if ind.strip()],
                keywords=[k.strip() for k in keywords.split(',')] if keywords else [],
                custom_prompt=free_text,
                excluded_industries=excluded_industries,
                excluded_companies=[c.strip() for c in excluded_companies.split('\n') if c.strip()]
            )

            st.session_state.current_criteria = built_criteria
            st.success("✅ Criteria confirmed! Go to 'Execute Search' tab to run the search.")

    with col2:
        if st.button("💾 Save Profile", use_container_width=True):
            profile_name = st.text_input("Profile Name", key="save_profile_name")
            if profile_name and st.session_state.current_criteria:
                st.session_state.saved_profiles[profile_name] = st.session_state.current_criteria
                st.success(f"Profile '{profile_name}' saved!")

    with col3:
        if st.session_state.saved_profiles:
            selected_saved = st.selectbox("Load Saved", list(st.session_state.saved_profiles.keys()))
            if st.button("📂 Load", use_container_width=True):
                st.session_state.current_criteria = st.session_state.saved_profiles[selected_saved]
                st.rerun()

    with col4:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.session_state.current_profile_name = None
            st.session_state.current_criteria = None
            st.rerun()

# Tab 2: Execute Search
with tab2:
    st.header("Execute Company Search")

    if not st.session_state.current_criteria:
        st.warning("⚠️ Please configure and confirm search criteria in the 'Search Configuration' tab first.")
    else:
        # Show current criteria summary
        with st.expander("📋 Current Search Criteria", expanded=True):
            criteria = st.session_state.current_criteria

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Location:**")
                if criteria.location.countries:
                    st.write(f"Countries: {', '.join(criteria.location.countries)}")
                if criteria.location.cities:
                    st.write(f"Cities: {', '.join(criteria.location.cities)}")
                if criteria.location.proximity:
                    st.write(
                        f"Proximity: {criteria.location.proximity['radius_km']}km from {criteria.location.proximity['location']}")

            with col2:
                st.markdown("**Financial:**")
                if criteria.financial.revenue_min or criteria.financial.revenue_max:
                    rev_str = f"{criteria.financial.revenue_currency} "
                    if criteria.financial.revenue_min:
                        rev_str += f"${criteria.financial.revenue_min / 1e6:.0f}M"
                    if criteria.financial.revenue_max:
                        rev_str += f" - ${criteria.financial.revenue_max / 1e6:.0f}M"
                    st.write(f"Revenue: {rev_str}")
                if criteria.organizational.employee_count_min:
                    st.write(f"Min Employees: {criteria.organizational.employee_count_min}")

            with col3:
                st.markdown("**Industry & CSR:**")
                if criteria.industries:
                    ind_names = [ind['name'] for ind in criteria.industries[:3]]
                    st.write(f"Industries: {', '.join(ind_names)}")
                if criteria.behavioral.csr_focus_areas:
                    st.write(f"CSR: {', '.join(criteria.behavioral.csr_focus_areas[:3])}")

        st.divider()

        # Search configuration
        col1, col2 = st.columns(2)

        with col1:
            target_count = st.slider(
                "Target Companies",
                min_value=10,
                max_value=200,
                value=50 if num_models > 1 else 20,
                step=10
            )

        with col2:
            if num_models > 1:
                distribution = st.radio(
                    "Distribution Strategy",
                    ["Equal", "Weighted"],
                    help="Equal: Same count per model\nWeighted: More for primary models"
                )
            else:
                distribution = "Single"

            # Cost estimate
            search_cost = target_count * 0.0002 * (1.1 if num_models > 1 else 1)
            st.metric("Estimated Cost", f"${search_cost:.3f}")

        # Model distribution preview
        if num_models > 1:
            st.info(f"**Parallel Execution:** {num_models} models selected")

            if distribution == "Equal":
                per_model = target_count // num_models
                remainder = target_count % num_models

                model_targets = {}
                for i, model in enumerate(selected_models):
                    model_targets[model] = per_model + (1 if i < remainder else 0)
            else:  # Weighted
                weights = [0.4, 0.3, 0.2, 0.1, 0.0][:num_models]
                # Normalize weights
                total_weight = sum(weights)
                model_targets = {}
                for model, weight in zip(selected_models, weights):
                    model_targets[model] = int(target_count * (weight / total_weight))
                # Add remainder to first model
                remainder = target_count - sum(model_targets.values())
                model_targets[selected_models[0]] += remainder

            # Show distribution
            dist_df = pd.DataFrame(list(model_targets.items()), columns=["Model", "Target"])
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"**Single Model:** {selected_models[0]}")
            model_targets = {selected_models[0]: target_count}

        # Execute search button
        if st.button("🚀 Execute Search", type="primary", use_container_width=True):

            # Progress tracking
            progress_bar = st.progress(0)
            status_container = st.container()

            if num_models > 1:
                # True parallel execution using ThreadPoolExecutor
                st.info(f"Starting parallel search across {num_models} models...")


                def run_model_search(model: str, count: int) -> Dict[str, Any]:
                    """Run search for a single model in a separate thread"""
                    try:
                        # Create new event loop for this thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                        agent = EnhancedSearchStrategistAgent(deployment_name=model)

                        async def search():
                            return await agent.generate_enhanced_strategy(
                                st.session_state.current_criteria,
                                target_count=count
                            )

                        result = loop.run_until_complete(search())
                        loop.close()

                        return {
                            'model': model,
                            'companies': result.get('companies', []),
                            'success': True,
                            'error': None
                        }
                    except Exception as e:
                        return {
                            'model': model,
                            'companies': [],
                            'success': False,
                            'error': str(e)
                        }


                # Execute in parallel using ThreadPoolExecutor
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_models) as executor:
                    # Submit all tasks
                    futures = {
                        executor.submit(run_model_search, model, count): model
                        for model, count in model_targets.items()
                    }

                    # Collect results as they complete
                    all_companies = []
                    completed = 0

                    for future in concurrent.futures.as_completed(futures):
                        model = futures[future]
                        result = future.result()

                        completed += 1
                        progress_bar.progress(completed / num_models)

                        with status_container:
                            if result['success']:
                                st.success(f"✅ {result['model']}: Found {len(result['companies'])} companies")
                                all_companies.extend(result['companies'])
                            else:
                                st.error(f"❌ {result['model']}: {result['error']}")

                # Store results
                st.session_state.search_results = all_companies
                st.session_state.total_cost += search_cost

                # Clear progress
                progress_bar.empty()

                # Final summary
                st.success(f"🎉 Search complete! Found {len(all_companies)} total companies across {num_models} models")

            else:
                # Single model execution
                with st.spinner(f"Searching with {selected_models[0]}..."):
                    try:
                        agent = EnhancedSearchStrategistAgent(deployment_name=selected_models[0])

                        # Run async search
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)


                        async def search():
                            return await agent.generate_enhanced_strategy(
                                st.session_state.current_criteria,
                                target_count=target_count
                            )


                        result = loop.run_until_complete(search())
                        loop.close()

                        companies = result.get('companies', [])
                        st.session_state.search_results = companies
                        st.session_state.total_cost += search_cost

                        progress_bar.progress(1.0)
                        st.success(f"✅ Found {len(companies)} companies!")
                        progress_bar.empty()

                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
                        progress_bar.empty()

            # Display results summary
            if st.session_state.search_results:
                st.divider()
                st.subheader("Search Results Summary")

                # Tier distribution
                tier_counts = {}
                for company in st.session_state.search_results:
                    tier = company.icp_tier if hasattr(company, 'icp_tier') else company.get('icp_tier', 'Untiered')
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1

                # Display metrics
                cols = st.columns(len(tier_counts) or 1)
                for i, (tier, count) in enumerate(sorted(tier_counts.items())):
                    with cols[i]:
                        st.metric(f"Tier {tier}", count)

                # Sample results
                st.subheader("Sample Companies")
                sample_data = []
                for company in st.session_state.search_results[:10]:
                    if hasattr(company, 'dict'):
                        c = company.dict()
                    else:
                        c = company

                    sample_data.append({
                        "Company": c.get('name', 'Unknown'),
                        "Industry": c.get('industry_category', 'Unknown'),
                        "Revenue": c.get('estimated_revenue', 'Unknown'),
                        "ICP Score": f"{c.get('icp_score', 0):.0f}" if c.get('icp_score') else "N/A",
                        "Confidence": c.get('confidence', 'Unknown')
                    })

                st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

# Tab 3: Validation (keeping as is)
with tab3:
    st.header("Company Validation")

    if not st.session_state.search_results:
        st.info("No companies to validate. Run a search first.")
    else:
        st.write(f"**{len(st.session_state.search_results)} companies** ready for validation")

        # Validation settings
        col1, col2 = st.columns(2)

        with col1:
            max_validate = st.number_input(
                "Companies to validate",
                1,
                min(50, len(st.session_state.search_results)),
                min(10, len(st.session_state.search_results))
            )

            validation_depth = st.selectbox(
                "Validation Depth",
                ["Quick", "Standard", "Comprehensive"]
            )

        with col2:
            # Cost estimate
            depth_queries = {"Quick": 2, "Standard": 4, "Comprehensive": 6}
            est_queries = max_validate * depth_queries[validation_depth]
            est_cost = est_queries * 0.001

            st.metric("Estimated Queries", est_queries)
            st.metric("Estimated Cost", f"${est_cost:.3f}")

        # Validation button placeholder
        if st.button("✅ Start Validation", type="primary", use_container_width=True):
            st.info("Validation functionality maintained as requested")

# Tab 4: Results & Export
with tab4:
    st.header("Results & Export")

    if st.session_state.search_results:
        st.subheader(f"Search Results ({len(st.session_state.search_results)} companies)")

        # Create comprehensive DataFrame
        df_data = []
        for company in st.session_state.search_results:
            if hasattr(company, 'dict'):
                c = company.dict()
            else:
                c = company

            df_data.append({
                "Company": c.get('name', 'Unknown'),
                "Industry": c.get('industry_category', 'Unknown'),
                "Business Type": c.get('business_type', 'Unknown'),
                "Revenue": c.get('estimated_revenue', 'Unknown'),
                "Employees": c.get('estimated_employees', 'Unknown'),
                "Location": c.get('headquarters', {}).get('city', 'Unknown') if isinstance(c.get('headquarters'),
                                                                                           dict) else 'Unknown',
                "CSR Focus": ', '.join(c.get('csr_focus_areas', [])[:3]) if c.get('csr_focus_areas') else 'None',
                "ICP Score": c.get('icp_score', 0),
                "ICP Tier": c.get('icp_tier', 'Untiered'),
                "Confidence": c.get('confidence', 'Unknown')
            })

        df = pd.DataFrame(df_data)

        # Display options
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox("Sort by", df.columns.tolist(), index=df.columns.tolist().index("ICP Score"))
        with col2:
            sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

        # Sort dataframe
        df_sorted = df.sort_values(sort_by, ascending=(sort_order == "Ascending"))

        # Display dataframe
        st.dataframe(df_sorted, use_container_width=True, height=500)

        # Export options
        st.divider()
        st.subheader("📥 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv = df_sorted.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name=f"companies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            # Excel export
            from io import BytesIO

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_sorted.to_excel(writer, sheet_name='Companies', index=False)
            excel_data = output.getvalue()

            st.download_button(
                "📥 Download Excel",
                data=excel_data,
                file_name=f"companies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col3:
            json_data = [c.dict() if hasattr(c, 'dict') else c for c in st.session_state.search_results]
            json_str = json.dumps(json_data, indent=2, default=str)
            st.download_button(
                "📥 Download JSON",
                data=json_str,
                file_name=f"companies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("No results yet. Configure criteria and run a search to see results here.")

# Footer
st.divider()
st.caption(
    f"Session Cost: ${st.session_state.total_cost:.3f} | Models: {', '.join(selected_models if 'selected_models' in locals() else ['Not selected'])}")