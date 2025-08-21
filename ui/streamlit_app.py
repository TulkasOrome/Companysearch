# streamlit_app_v4.py
"""
Enhanced Streamlit App with Advanced Validation Modes
Fixed version with all syntax and variable errors resolved
"""

import streamlit as st
import asyncio
import pandas as pd
import json
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import traceback
import concurrent.futures
from dataclasses import asdict
from io import BytesIO
import time

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
    EnhancedValidationAgent,
    ValidationOrchestrator
)

from core.validation_strategies import ValidationCriteria, ValidationTier
from core.serper_client import SerperEndpoint

# Import Serper validation
try:
    from serper_validation_integration import validate_company_with_serper

    SERPER_VALIDATION_AVAILABLE = True
except ImportError:
    SERPER_VALIDATION_AVAILABLE = False
    print("Warning: Serper validation not available")

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
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = ["gpt-4.1"]

# Initialize ICP manager
icp_manager = None
if ICP_AVAILABLE:
    icp_manager = ICPManager()

# Title
st.title("🔍 Company Search & Validation Platform")
st.markdown("AI-powered company discovery with advanced validation modes")

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

    # Store selected models in session state
    st.session_state.selected_models = selected_models

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Search Configuration",
    "🔍 Execute Search",
    "✅ Validation",
    "📊 Results & Export",
    "❓ Help"
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
            st.session_state.current_tier = tier

    st.divider()

    # Search Fields Configuration
    st.subheader("🔧 Search Criteria Configuration")

    # Load profile defaults if selected
    criteria = None
    if st.session_state.current_profile_name and ICP_AVAILABLE and icp_manager:
        profile = icp_manager.get_profile(st.session_state.current_profile_name)
        if profile:
            criteria = profile.tiers.get(st.session_state.current_tier)

    # Location Section
    with st.expander("🌍 Location Settings", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            country_options = ["Australia", "United States", "United Kingdom", "Canada", "New Zealand",
                               "Germany", "France", "Japan", "Singapore", "India"]
            default_countries = criteria.location.countries if criteria else ["Australia"]
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
            business_options = ["B2B", "B2C", "B2B2C", "D2C", "Professional Services", "Enterprise", "SaaS",
                                "Marketplace", "Hybrid"]
            default_types = criteria.business_types if criteria else ["B2B", "B2C"]
            valid_business_defaults = [d for d in default_types if d in business_options]
            if not valid_business_defaults:
                valid_business_defaults = ["B2B", "B2C"]

            business_types = st.multiselect(
                "Business Types",
                business_options,
                default=valid_business_defaults
            )

        with col2:
            default_industries = []
            if criteria and criteria.industries:
                for ind in criteria.industries:
                    if isinstance(ind, dict):
                        ind_name = ind.get('name', '')
                    else:
                        ind_name = str(ind)
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
            csr_options = ["children", "community", "education", "health", "environment",
                           "disability", "inclusion", "diversity", "sustainability", "elderly",
                           "families", "accessibility", "wellbeing"]
            default_csr = criteria.behavioral.csr_focus_areas if criteria else []
            valid_defaults = [d for d in default_csr if d in csr_options]

            csr_focus = st.multiselect(
                "CSR Focus Areas",
                csr_options,
                default=valid_defaults
            )

            event_options = ["Office Move", "CSR Launch", "Expansion", "Anniversary", "Award",
                             "New Leadership", "IPO", "Merger", "Partnership"]
            default_events = criteria.behavioral.recent_events if criteria else []
            valid_event_defaults = [d for d in default_events if d in event_options]

            recent_events = st.multiselect(
                "Recent Events",
                event_options,
                default=valid_event_defaults
            )

        with col2:
            cert_options = ["B-Corp", "ISO 26000", "ISO 14001", "Carbon Neutral", "Fair Trade",
                            "Great Place to Work", "ESG Certified"]
            default_certs = criteria.behavioral.certifications if criteria else []
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
            exclusion_options = ["Gambling", "Tobacco", "Fast Food", "Racing", "Alcohol", "Weapons", "Animal Testing"]
            default_exc_ind = criteria.excluded_industries if criteria else []
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
                placeholder="Example: Focus on companies that have won sustainability awards in the last 2 years",
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

            with col2:
                st.markdown("**Financial:**")
                if criteria.financial.revenue_min or criteria.financial.revenue_max:
                    rev_str = f"{criteria.financial.revenue_currency} "
                    if criteria.financial.revenue_min:
                        rev_str += f"${criteria.financial.revenue_min / 1e6:.0f}M"
                    if criteria.financial.revenue_max:
                        rev_str += f" - ${criteria.financial.revenue_max / 1e6:.0f}M"
                    st.write(f"Revenue: {rev_str}")

            with col3:
                st.markdown("**Industry & CSR:**")
                if criteria.industries:
                    ind_names = [ind['name'] for ind in criteria.industries[:3]]
                    st.write(f"Industries: {', '.join(ind_names)}")

        st.divider()

        # Search configuration
        col1, col2 = st.columns(2)

        with col1:
            target_count = st.slider(
                "Target Companies",
                min_value=10,
                max_value=5000,
                value=50 if len(st.session_state.selected_models) > 1 else 20,
                step=10
            )

        with col2:
            if len(st.session_state.selected_models) > 1:
                distribution = st.radio(
                    "Distribution Strategy",
                    ["Equal", "Weighted"],
                    help="Equal: Same count per model\nWeighted: More for primary models"
                )
            else:
                distribution = "Single"

            # Cost estimate
            search_cost = target_count * 0.0002 * (1.1 if len(st.session_state.selected_models) > 1 else 1)
            st.metric("Estimated Cost", f"${search_cost:.3f}")

        # Execute search button
        if st.button("🚀 Execute Search", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_container = st.container()

            # Single model execution (simplified for now)
            if len(st.session_state.selected_models) == 1:
                with st.spinner(f"Searching with {st.session_state.selected_models[0]}..."):
                    try:
                        agent = EnhancedSearchStrategistAgent(deployment_name=st.session_state.selected_models[0])

                        # Run async search
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)


                        async def search():
                            result = await agent.generate_enhanced_strategy(
                                st.session_state.current_criteria,
                                target_count=target_count
                            )
                            # Convert any EnhancedCompanyEntry objects to dicts
                            companies_list = result.get('companies', [])
                            converted_companies = []
                            for company in companies_list:
                                if hasattr(company, 'dict'):
                                    # It's a Pydantic model, convert to dict
                                    converted_companies.append(company.dict())
                                elif hasattr(company, '__dict__'):
                                    # It's some other object, convert to dict
                                    converted_companies.append(company.__dict__)
                                elif isinstance(company, dict):
                                    # Already a dict, use as is
                                    converted_companies.append(company)
                                else:
                                    # Unknown type, try to convert
                                    try:
                                        converted_companies.append(dict(company))
                                    except:
                                        continue
                            result['companies'] = converted_companies
                            return result


                        result = loop.run_until_complete(search())
                        loop.close()

                        # Process and enhance company entries
                        enhanced_companies = []
                        for company_data in result.get("companies", []):
                            try:
                                # Fix operates_in_country if it's a string
                                if 'operates_in_country' in company_data:
                                    oic = company_data['operates_in_country']
                                    if isinstance(oic, str):
                                        # Convert string to boolean - if it's a country name, assume true
                                        company_data['operates_in_country'] = True
                                    elif not isinstance(oic, bool):
                                        company_data['operates_in_country'] = True

                                # Ensure all required fields exist with defaults
                                company_data = agent._ensure_company_fields(company_data)
                                company = EnhancedCompanyEntry(**company_data)
                                # Calculate ICP score
                                company = agent._calculate_icp_score(company, st.session_state.current_criteria)
                                enhanced_companies.append(company)
                            except Exception as e:
                                st.warning(f"Error processing company: {e}")
                                # Try to at least get the company name
                                if 'name' in company_data:
                                    st.write(f"  Company: {company_data['name']}")
                                continue

                        companies = enhanced_companies

                        # Debug: Show what we got back
                        if not companies:
                            st.warning(
                                f"No companies returned from {st.session_state.selected_models[0]}. Check the search criteria or try again.")
                            st.info("Tip: Try simplifying your criteria or selecting different industries.")
                        else:
                            st.session_state.search_results = companies
                            st.session_state.total_cost += search_cost

                            progress_bar.progress(1.0)
                            st.success(f"✅ Found {len(companies)} companies!")

                        progress_bar.empty()

                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
                        progress_bar.empty()

        # Display search results directly in this tab
        if st.session_state.search_results:
            st.divider()
            st.subheader("📊 Search Results")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            # Calculate tier distribution
            tier_counts = {}
            for company in st.session_state.search_results:
                if hasattr(company, 'icp_tier'):
                    tier = company.icp_tier
                elif isinstance(company, dict):
                    tier = company.get('icp_tier', 'Untiered')
                else:
                    tier = 'Untiered'
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

            with col1:
                st.metric("Total Found", len(st.session_state.search_results))
            with col2:
                st.metric("Tier A", tier_counts.get('A', 0))
            with col3:
                st.metric("Tier B", tier_counts.get('B', 0))
            with col4:
                st.metric("Tier C", tier_counts.get('C', 0))

            # Create results dataframe
            results_data = []
            for company in st.session_state.search_results[:100]:  # Show first 100
                if hasattr(company, 'dict'):
                    c = company.dict()
                elif isinstance(company, dict):
                    c = company
                else:
                    c = {'name': str(company)}

                results_data.append({
                    "Company": c.get('name', 'Unknown'),
                    "Industry": c.get('industry_category', 'Unknown'),
                    "Business Type": c.get('business_type', 'Unknown'),
                    "Revenue": c.get('estimated_revenue', 'Unknown'),
                    "Employees": c.get('estimated_employees', 'Unknown'),
                    "Location": c.get('headquarters', {}).get('city', 'Unknown') if isinstance(c.get('headquarters'),
                                                                                               dict) else 'Unknown',
                    "ICP Score": f"{c.get('icp_score', 0):.0f}" if c.get('icp_score') else "N/A",
                    "ICP Tier": c.get('icp_tier', 'Untiered'),
                    "Confidence": c.get('confidence', 'Unknown')
                })

            # Display dataframe with sorting options
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(
                    f"Showing {min(100, len(st.session_state.search_results))} of {len(st.session_state.search_results)} companies")
            with col2:
                if len(st.session_state.search_results) > 100:
                    st.info("View all in Results & Export tab")

            results_df = pd.DataFrame(results_data)
            st.dataframe(
                results_df,
                use_container_width=True,
                height=400,
                hide_index=True
            )

# Tab 3: Validation
with tab3:
    st.header("Company Validation")

    if not st.session_state.search_results:
        st.info("No companies to validate. Run a search first.")
    else:
        st.write(f"**{len(st.session_state.search_results)} companies** ready for validation")

        # Validation mode selection
        st.subheader("🎯 Validation Mode")

        validation_mode = st.selectbox(
            "Select Validation Mode",
            [
                "Simple Check (2-3 credits)",
                "Smart Contact Extraction (3-5 credits)",
                "Smart CSR Verification (3-5 credits)",
                "Smart Financial Check (3-4 credits)",
                "Full Validation (10-15 credits)",
                "Raw Endpoint Access",
                "Custom Configuration"
            ],
            help="Choose validation depth based on your needs"
        )

        # Mode-specific information
        mode_info = {
            "Simple Check (2-3 credits)": {
                "description": "Quick existence and location verification",
                "extracts": ["Company exists", "Location verified", "Basic website"],
                "credits": 2.5
            },
            "Smart Contact Extraction (3-5 credits)": {
                "description": "Extract emails, phones, and contact names",
                "extracts": ["Email addresses", "Phone numbers", "Executive names", "LinkedIn profiles"],
                "credits": 4
            },
            "Smart CSR Verification (3-5 credits)": {
                "description": "Verify CSR programs and community involvement",
                "extracts": ["CSR programs", "Focus areas", "Certifications", "Giving evidence"],
                "credits": 4
            },
            "Smart Financial Check (3-4 credits)": {
                "description": "Verify revenue and employee information",
                "extracts": ["Revenue range", "Employee count", "Growth indicators", "Financial health"],
                "credits": 3.5
            },
            "Full Validation (10-15 credits)": {
                "description": "Comprehensive validation with all checks",
                "extracts": ["All of the above", "Recent news", "Risk signals", "Detailed analysis"],
                "credits": 12
            }
        }

        if validation_mode in mode_info:
            with st.expander("ℹ️ Mode Information", expanded=True):
                info = mode_info[validation_mode]
                st.write(f"**Description:** {info['description']}")
                st.write("**Extracts:**")
                for item in info['extracts']:
                    st.write(f"  • {item}")

        # Validation settings
        col1, col2 = st.columns(2)

        with col1:
            max_validate = st.number_input(
                "Companies to validate",
                1,
                min(50, len(st.session_state.search_results)),
                min(10, len(st.session_state.search_results))
            )

        with col2:
            # Cost estimate
            if validation_mode in mode_info:
                credits_per = mode_info[validation_mode]['credits']
            else:
                credits_per = 5  # Default

            est_credits = max_validate * credits_per
            est_cost = est_credits * 0.001

            st.metric("Estimated Credits", int(est_credits))
            st.metric("Estimated Cost", f"${est_cost:.3f}")

        # Raw endpoint configuration (if selected)
        if validation_mode == "Raw Endpoint Access":
            st.subheader("🔧 Raw Endpoint Configuration")

            col1, col2 = st.columns(2)
            with col1:
                endpoint = st.selectbox(
                    "Serper Endpoint",
                    ["search", "places", "news", "maps", "shopping"]
                )

            with col2:
                custom_query = st.text_input(
                    "Query Template",
                    placeholder="{company_name} {location}",
                    help="Use {company_name} and {location} as placeholders"
                )

        # Custom configuration (if selected)
        if validation_mode == "Custom Configuration":
            st.subheader("🔧 Custom Validation Configuration")

            selected_endpoints = st.multiselect(
                "Select Endpoints",
                ["places", "search", "news", "maps", "shopping"],
                default=["places", "search"]
            )

            st.write("**Query Templates:**")
            custom_queries = {}
            for endpoint in selected_endpoints:
                custom_queries[endpoint] = st.text_input(
                    f"{endpoint.capitalize()} Query",
                    placeholder=f"{{company_name}} {{location}}",
                    key=f"query_{endpoint}"
                )

        # Validation button
        if st.button("✅ Start Validation", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Check if Serper validation is available
            if not SERPER_VALIDATION_AVAILABLE or not serper_key:
                st.error("⚠️ Serper validation not available. Please check API key and modules.")
                progress_bar.empty()
            else:
                # Run ACTUAL Serper validation
                validated_companies = []

                for i, company in enumerate(st.session_state.search_results[:max_validate]):
                    progress = (i + 1) / max_validate
                    progress_bar.progress(progress)

                    # Extract company details
                    if hasattr(company, 'dict'):
                        company_dict = company.dict()
                    elif isinstance(company, dict):
                        company_dict = company
                    else:
                        company_dict = {'name': str(company)}

                    company_name = company_dict.get('name', 'Unknown')
                    status_text.text(f"Validating {i + 1}/{max_validate}: {company_name}")

                    try:
                        # Run ACTUAL Serper validation
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)


                        async def run_validation():
                            return await validate_company_with_serper(
                                company_dict,
                                validation_mode,
                                serper_key
                            )


                        validation_result = loop.run_until_complete(run_validation())
                        loop.close()

                        # Update the company data with validation results
                        company_dict['validation_result'] = validation_result
                        company_dict['validated'] = True
                        company_dict['validation_status'] = validation_result['validation_status']

                        validated_companies.append(validation_result)

                        # Update the company in search results
                        if not hasattr(st.session_state.search_results[i], 'dict'):
                            st.session_state.search_results[i] = company_dict

                        # Rate limiting between companies
                        if i < max_validate - 1:
                            time.sleep(0.5)  # Half second between companies

                    except Exception as e:
                        st.error(f"Error validating {company_name}: {str(e)}")
                        # Add error result
                        validation_result = {
                            'company_name': company_name,
                            'validation_status': 'error',
                            'mode': validation_mode,
                            'credits_used': 0,
                            'validation_timestamp': datetime.now().isoformat(),
                            'error': str(e)
                        }
                        validated_companies.append(validation_result)

                st.session_state.validation_results = validated_companies

                # Calculate actual cost based on credits used
                total_credits = sum(v.get('credits_used', 0) for v in validated_companies)
                actual_cost = total_credits * 0.001
                st.session_state.total_cost += actual_cost

                progress_bar.empty()
                status_text.empty()

                # Show completion message with actual credits
                st.success(
                    f"✅ Validated {len(validated_companies)} companies using {total_credits} credits (${actual_cost:.3f})")

                # Show validation results summary
                st.divider()
                st.subheader("Validation Results Summary")

                # Summary metrics
                verified = len([v for v in validated_companies if v['validation_status'] == 'verified'])
                partial = len([v for v in validated_companies if v['validation_status'] == 'partial'])
                unverified = len([v for v in validated_companies if v['validation_status'] == 'unverified'])
                errors = len([v for v in validated_companies if v['validation_status'] == 'error'])

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Verified", verified)
                with col2:
                    st.metric("Partial", partial)
                with col3:
                    st.metric("Unverified", unverified)
                with col4:
                    st.metric("Errors", errors)

                # Show sample results based on mode
                st.write("**Sample Validation Results:**")

                sample_results = []
                for val in validated_companies[:10]:  # Show up to 10
                    result_row = {
                        "Company": val.get('company_name', 'Unknown'),
                        "Status": val.get('validation_status', 'unknown')
                    }

                    # Add mode-specific columns with REAL data
                    if "Contact" in validation_mode:
                        # Show actual extracted data - handle both list and single values
                        emails = val.get('emails', [])
                        phones = val.get('phones', [])
                        names = val.get('names', [])

                        # Also check for single value fields
                        if not emails and val.get('email'):
                            emails = [val.get('email')]
                        if not phones and val.get('phone'):
                            phones = [val.get('phone')]
                        if not names and val.get('contact_name'):
                            names = [val.get('contact_name')]

                        result_row["Email"] = emails[0] if emails else 'Not found'
                        result_row["Phone"] = phones[0] if phones else 'Not found'
                        result_row["Contact"] = names[0] if names else 'Not found'

                    elif "CSR" in validation_mode:
                        programs = val.get('csr_programs', [])
                        certs = val.get('certifications', [])
                        result_row["CSR Programs"] = ', '.join(programs) if programs else 'None'
                        result_row["Certifications"] = ', '.join(certs) if certs else 'None'

                    elif "Financial" in validation_mode:
                        result_row["Revenue"] = val.get('revenue_range', 'Not verified')
                        result_row["Employees"] = val.get('employee_range', 'Not verified')

                    elif "Full" in validation_mode:
                        # Show key data from full validation
                        emails = val.get('emails', [])
                        email = emails[0] if emails else val.get('email', 'Not found')
                        result_row["Email"] = email
                        result_row["Revenue"] = val.get('revenue_range', 'Not verified')
                        programs = val.get('csr_programs', [])
                        result_row["CSR"] = 'Yes' if programs else 'No'
                        risks = val.get('risk_signals', [])
                        result_row["Risks"] = 'Yes' if risks else 'No'

                    sample_results.append(result_row)

                if sample_results:
                    sample_df = pd.DataFrame(sample_results)
                    st.dataframe(sample_df, use_container_width=True, hide_index=True)

                    # Show detailed validation data in expander
                    with st.expander("📋 Detailed Validation Results", expanded=False):
                        for val in validated_companies[:5]:
                            st.write(f"**{val.get('company_name', 'Unknown')}**")

                            # Show all emails found
                            emails = val.get('emails', [])
                            if emails:
                                st.write(f"  📧 Emails found: {', '.join(emails[:3])}")

                            # Show all phones found
                            phones = val.get('phones', [])
                            if phones:
                                st.write(f"  📞 Phones found: {', '.join(phones[:3])}")

                            # Show all names found
                            names = val.get('names', [])
                            if names:
                                st.write(f"  👤 Names found: {', '.join(names[:3])}")

                            # Show CSR programs
                            programs = val.get('csr_programs', [])
                            if programs:
                                st.write(f"  💚 CSR: {', '.join(programs)}")

                            # Show certifications
                            certs = val.get('certifications', [])
                            if certs:
                                st.write(f"  🏆 Certifications: {', '.join(certs)}")

                            st.write(f"  💳 Credits used: {val.get('credits_used', 0)}")
                            st.write("")

                    # Show credits breakdown
                    st.write("**Credits Usage:**")
                    total_credits_used = sum(v.get('credits_used', 0) for v in validated_companies)
                    st.write(f"Total credits used: {total_credits_used} (${total_credits_used * 0.001:.3f})")
                else:
                    st.warning("No validation results to display")

# Tab 4: Results & Export
with tab4:
    st.header("Results & Export")

    # Combined results view
    if st.session_state.search_results or st.session_state.validation_results:
        st.subheader(f"Combined Results")

        # Create comprehensive DataFrame with validation data
        df_data = []

        # Create a mapping of validation results by company name
        validation_map = {}
        for val in st.session_state.validation_results:
            validation_map[val.get('company_name')] = val

        for company in st.session_state.search_results:
            if hasattr(company, 'dict'):
                c = company.dict()
            elif isinstance(company, dict):
                c = company
            else:
                c = {'name': str(company)}

            company_name = c.get('name', 'Unknown')

            # Get validation result for this company
            validation_info = validation_map.get(company_name, None)

            # Build base row
            row = {
                "Company": company_name,
                "Industry": c.get('industry_category', 'Unknown'),
                "Revenue": c.get('estimated_revenue', 'Unknown'),
                "Employees": c.get('estimated_employees', 'Unknown'),
                "ICP Score": f"{c.get('icp_score', 0):.0f}" if c.get('icp_score') else "N/A",
                "ICP Tier": c.get('icp_tier', 'Untiered'),
                "Validation Status": validation_info.get('validation_status',
                                                         'Not Validated') if validation_info else 'Not Validated',
                "Confidence": c.get('confidence', 'Unknown')
            }

            # Add validation-specific columns if validated
            if validation_info:
                # Add contact info if available
                if 'email' in validation_info:
                    row['Email'] = validation_info.get('email', '')
                if 'phone' in validation_info:
                    row['Phone'] = validation_info.get('phone', '')
                if 'contact_name' in validation_info:
                    row['Contact Name'] = validation_info.get('contact_name', '')

                # Add financial validation if available
                if 'revenue_range' in validation_info:
                    row['Verified Revenue'] = validation_info.get('revenue_range', '')
                if 'employee_range' in validation_info:
                    row['Verified Employees'] = validation_info.get('employee_range', '')

                # Add CSR validation if available
                if 'csr_programs' in validation_info:
                    row['CSR Programs'] = ', '.join(validation_info.get('csr_programs', []))
                if 'certifications' in validation_info:
                    row['Certifications'] = ', '.join(validation_info.get('certifications', []))

                # Add risk signals if available
                if 'risk_signals' in validation_info:
                    row['Risk Signals'] = ', '.join(validation_info.get('risk_signals', []))

            df_data.append(row)

        df = pd.DataFrame(df_data)

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)

        total_companies = len(df_data)
        validated_count = len([d for d in df_data if d['Validation Status'] != 'Not Validated'])
        verified_count = len([d for d in df_data if d['Validation Status'] == 'verified'])

        with col1:
            st.metric("Total Companies", total_companies)
        with col2:
            st.metric("Validated", validated_count)
        with col3:
            st.metric("Verified", verified_count)
        with col4:
            validation_rate = (validated_count / total_companies * 100) if total_companies > 0 else 0
            st.metric("Validation Rate", f"{validation_rate:.1f}%")

        # Filtering options
        st.subheader("🔍 Filter Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            filter_tier = st.selectbox(
                "Filter by ICP Tier",
                ["All"] + sorted(df['ICP Tier'].unique().tolist()),
                key="filter_tier"
            )

        with col2:
            filter_status = st.selectbox(
                "Filter by Validation Status",
                ["All"] + sorted(df['Validation Status'].unique().tolist()),
                key="filter_status"
            )

        with col3:
            search_term = st.text_input(
                "Search Companies",
                placeholder="Enter company name...",
                key="search_companies"
            )

        # Apply filters
        filtered_df = df.copy()

        if filter_tier != "All":
            filtered_df = filtered_df[filtered_df['ICP Tier'] == filter_tier]

        if filter_status != "All":
            filtered_df = filtered_df[filtered_df['Validation Status'] == filter_status]

        if search_term:
            filtered_df = filtered_df[filtered_df['Company'].str.contains(search_term, case=False, na=False)]

        # Display filtered dataframe
        st.write(f"Showing {len(filtered_df)} of {len(df)} companies")
        st.dataframe(filtered_df, use_container_width=True, height=500)

        # Export options
        st.divider()
        st.subheader("📥 Export Options")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # CSV export
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV (Filtered)",
                data=csv,
                file_name=f"companies_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            # Excel export with multiple sheets
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main data sheet
                filtered_df.to_excel(writer, sheet_name='Companies', index=False)

                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Total Companies',
                        'Validated',
                        'Verified',
                        'Partial',
                        'Not Validated',
                        'Tier A',
                        'Tier B',
                        'Tier C',
                        'With Email',
                        'With Phone'
                    ],
                    'Count': [
                        len(df),
                        validated_count,
                        verified_count,
                        len([d for d in df_data if d['Validation Status'] == 'partial']),
                        len([d for d in df_data if d['Validation Status'] == 'Not Validated']),
                        len([d for d in df_data if d['ICP Tier'] == 'A']),
                        len([d for d in df_data if d['ICP Tier'] == 'B']),
                        len([d for d in df_data if d['ICP Tier'] == 'C']),
                        len([d for d in df_data if d.get('Email')]),
                        len([d for d in df_data if d.get('Phone')])
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                # Validation details sheet (if validations exist)
                if st.session_state.validation_results:
                    val_df = pd.DataFrame(st.session_state.validation_results)
                    val_df.to_excel(writer, sheet_name='Validation Details', index=False)

            excel_data = output.getvalue()

            st.download_button(
                "📥 Download Excel (Full)",
                data=excel_data,
                file_name=f"companies_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col3:
            # JSON export with all data
            json_data = []
            for company in st.session_state.search_results:
                if hasattr(company, 'dict'):
                    c = company.dict()
                elif isinstance(company, dict):
                    c = company
                else:
                    c = {'name': str(company)}

                # Add validation data if exists
                company_name = c.get('name', 'Unknown')
                if company_name in validation_map:
                    c['validation'] = validation_map[company_name]

                json_data.append(c)

            export_data = {
                'search_results': json_data,
                'validation_results': st.session_state.validation_results,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_cost': st.session_state.total_cost,
                    'companies_found': len(st.session_state.search_results),
                    'companies_validated': len(st.session_state.validation_results),
                    'validation_rate': f"{validation_rate:.1f}%"
                }
            }

            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                "📥 Download JSON (Complete)",
                data=json_str,
                file_name=f"companies_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        with col4:
            # Download only validated companies
            if validated_count > 0:
                validated_df = filtered_df[filtered_df['Validation Status'] != 'Not Validated']
                validated_csv = validated_df.to_csv(index=False)
                st.download_button(
                    "📥 Validated Only (CSV)",
                    data=validated_csv,
                    file_name=f"validated_companies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No validated companies")
    else:
        st.info("No results yet. Configure criteria and run a search to see results here.")

# Tab 5: Help
with tab5:
    st.header("Help & Documentation")

    st.subheader("📖 Validation Modes Guide")

    with st.expander("Simple Check (2-3 credits)", expanded=False):
        st.write("""
        **Use Cases:**
        - Quick verification that companies exist
        - Basic location confirmation
        - Initial filtering of search results

        **What It Does:**
        - Searches Google Places for company presence
        - Verifies location match
        - Extracts basic contact info if available

        **Best For:**
        - Large batches where you need quick filtering
        - Initial screening before deeper validation
        """)

    with st.expander("Smart Contact Extraction (3-5 credits)", expanded=False):
        st.write("""
        **Use Cases:**
        - Building outreach lists
        - Finding decision makers
        - Email and phone collection

        **What It Does:**
        - Searches for contact pages
        - Extracts emails with pattern matching
        - Finds phone numbers in multiple formats
        - Identifies executive names from LinkedIn

        **Best For:**
        - Sales prospecting
        - Partnership outreach
        - Donor recruitment
        """)

    with st.expander("Smart CSR Verification (3-5 credits)", expanded=False):
        st.write("""
        **Use Cases:**
        - Verifying alignment with social causes
        - Finding corporate giving programs
        - Checking sustainability commitments

        **What It Does:**
        - Searches for CSR/sustainability pages
        - Identifies focus areas (children, environment, etc.)
        - Finds certifications (B-Corp, ISO 26000)
        - Detects foundation or giving programs

        **Best For:**
        - Nonprofit partnership identification
        - ESG-focused campaigns
        - Values-aligned partnerships
        """)

    st.subheader("💡 Credit Optimization Tips")

    st.info("""
    **Tips to minimize credit usage:**
    1. Start with Simple Check for initial filtering
    2. Use targeted modes for specific data needs
    3. Validate high-priority companies first
    4. Use Raw Endpoint Access for custom queries
    5. Batch similar companies together
    """)

    st.subheader("📊 Field Extraction Reference")

    extraction_df = pd.DataFrame({
        "Field": ["Email", "Phone", "Executive Name", "Revenue", "Employees", "CSR Programs", "Certifications"],
        "Simple": ["❌", "✅", "❌", "❌", "❌", "❌", "❌"],
        "Contact": ["✅", "✅", "✅", "❌", "❌", "❌", "❌"],
        "CSR": ["❌", "❌", "❌", "❌", "❌", "✅", "✅"],
        "Financial": ["❌", "❌", "❌", "✅", "✅", "❌", "❌"],
        "Full": ["✅", "✅", "✅", "✅", "✅", "✅", "✅"]
    })

    st.dataframe(extraction_df, use_container_width=True)

# Footer
st.divider()
selected_models_display = st.session_state.selected_models if st.session_state.selected_models else ['Not selected']
st.caption(
    f"Session Cost: ${st.session_state.total_cost:.3f} | "
    f"Models: {', '.join(selected_models_display)} | "
    f"Companies Found: {len(st.session_state.search_results)} | "
    f"Validated: {len(st.session_state.validation_results)}"
)