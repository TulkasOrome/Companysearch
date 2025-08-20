# ui/streamlit_app.py
"""
Enhanced Streamlit App with proper error handling and template loading
"""

import streamlit as st
import asyncio
import pandas as pd
import json
from datetime import datetime
import time
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import from new structure
from agents.search_strategist_agent import EnhancedSearchStrategistAgent
from agents.validation_agent_v2 import (
    EnhancedValidationAgent,
    ValidationConfig,
    ValidationOrchestrator
)
from core.data_models import (
    SearchCriteria,
    LocationCriteria,
    FinancialCriteria,
    OrganizationalCriteria,
    BehavioralSignals,
    BusinessType,
    EnhancedCompanyEntry,
    RMH_SYDNEY_CONFIG,
    GUIDE_DOGS_VICTORIA_CONFIG
)
from core.validation_strategies import ValidationCriteria, ValidationTier
from session_manager import SessionManager  # Import from same directory

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
st.markdown("Comprehensive company discovery with enhanced Serper validation")

# Initialize session state with defaults if not present
if 'search_criteria' not in st.session_state:
    st.session_state.search_criteria = {}
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = []
if 'azure_key' not in st.session_state:
    st.session_state.azure_key = ''
if 'serper_key' not in st.session_state:
    st.session_state.serper_key = '99c44b79892f5f7499accf2d7c26d93313880937'
if 'current_template' not in st.session_state:
    st.session_state.current_template = None

# Sidebar for session management and configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Keys section
    with st.expander("🔑 API Keys", expanded=False):
        serper_key = st.text_input(
            "Serper API Key",
            value=st.session_state.serper_key,
            type="password",
            help="Your Serper.dev API key for validation"
        )
        if serper_key:
            st.session_state.serper_key = serper_key

    # Validation Settings
    with st.expander("🎯 Validation Settings", expanded=False):
        validation_mode = st.selectbox(
            "Default Validation Mode",
            ["Smart (Adaptive)", "Quick", "Standard", "Comprehensive"],
            help="Choose validation depth"
        )

        max_cost_per_company = st.slider(
            "Max Cost per Company ($)",
            min_value=0.001,
            max_value=0.05,
            value=0.01,
            step=0.001,
            format="%.3f"
        )

        parallel_queries = st.slider(
            "Parallel Serper Queries",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of simultaneous Serper API calls"
        )

    st.divider()

    # Session stats
    with st.expander("📊 Session Stats", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Companies Found", len(st.session_state.search_results))
        with col2:
            st.metric("Validated", len(st.session_state.validation_results))

        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.search_criteria = {}
            st.session_state.search_results = []
            st.session_state.validation_results = []
            st.session_state.current_template = None
            st.rerun()

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Search Setup",
    "🔍 Execute Search",
    "✅ Validation",
    "📊 Results & Export"
])

# Tab 1: Search Configuration
with tab1:
    st.header("Configure Search Criteria")

    # Template selection
    st.subheader("📋 Quick Templates")
    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        if st.button("🏥 RMH Sydney", use_container_width=True, type="primary"):
            st.session_state.current_template = "RMH"
            st.rerun()

    with col_t2:
        if st.button("🦮 Guide Dogs Victoria", use_container_width=True, type="primary"):
            st.session_state.current_template = "GDV"
            st.rerun()

    with col_t3:
        if st.button("🔧 Custom Search", use_container_width=True, type="secondary"):
            st.session_state.current_template = None
            st.session_state.search_criteria = {}
            st.rerun()

    # Show current template
    if st.session_state.current_template:
        if st.session_state.current_template == "RMH":
            st.success("✅ Using RMH Sydney template")
            criteria = RMH_SYDNEY_CONFIG.tier_a_criteria
        elif st.session_state.current_template == "GDV":
            st.success("✅ Using Guide Dogs Victoria template")
            criteria = GUIDE_DOGS_VICTORIA_CONFIG.tier_a_criteria

        # Display template details
        with st.expander("Template Details", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Location:**")
                st.write(f"Countries: {', '.join(criteria.location.countries)}")
                if criteria.location.cities:
                    st.write(f"Cities: {', '.join(criteria.location.cities)}")

            with col2:
                st.markdown("**Financial:**")
                if criteria.financial.revenue_min:
                    st.write(
                        f"Min Revenue: ${criteria.financial.revenue_min / 1e6:.0f}M {criteria.financial.revenue_currency}")
                if criteria.financial.revenue_max:
                    st.write(
                        f"Max Revenue: ${criteria.financial.revenue_max / 1e6:.0f}M {criteria.financial.revenue_currency}")

            with col3:
                st.markdown("**CSR Focus:**")
                if criteria.behavioral.csr_focus_areas:
                    st.write(f"Areas: {', '.join(criteria.behavioral.csr_focus_areas)}")
                if criteria.behavioral.certifications:
                    st.write(f"Certs: {', '.join(criteria.behavioral.certifications)}")

    else:
        # Manual configuration
        st.subheader("🔧 Custom Configuration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🌍 Location")
            countries = st.multiselect(
                "Countries",
                ["Australia", "United States", "United Kingdom", "Canada"],
                default=["Australia"]
            )

            cities = st.text_input("Cities (comma-separated)", placeholder="Sydney, Melbourne")

            use_proximity = st.checkbox("Proximity search")
            proximity_location = None
            proximity_radius = None
            if use_proximity:
                proximity_location = st.text_input("Central location", placeholder="Sydney CBD")
                proximity_radius = st.number_input("Radius (km)", min_value=1, max_value=500, value=50)

        with col2:
            st.markdown("### 💰 Financial")
            revenue_min = st.number_input("Min Revenue (M)", min_value=0.0, value=5.0)
            revenue_max = st.number_input("Max Revenue (M)", min_value=0.0, value=100.0)
            revenue_currency = st.selectbox("Currency", ["USD", "AUD", "EUR", "GBP"], index=1)

            employee_min = st.number_input("Min Employees", min_value=0, value=50)
            employee_max = st.number_input("Max Employees", min_value=0, value=1000)

        with col3:
            st.markdown("### 🎯 Industry & CSR")
            business_types = st.multiselect(
                "Business Types",
                ["B2B", "B2C", "B2B2C", "D2C", "Professional Services"],
                default=["B2B", "B2C"]
            )

            industries = st.text_input("Industries (comma-separated)", placeholder="Technology, Retail")

            csr_focus = st.multiselect(
                "CSR Focus Areas",
                ["children", "education", "health", "environment", "community"],
                default=[]
            )

        # Save custom criteria
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            # Build criteria object
            criteria = SearchCriteria(
                location=LocationCriteria(
                    countries=countries,
                    cities=[c.strip() for c in cities.split(',')] if cities else [],
                    proximity={"location": proximity_location, "radius_km": proximity_radius} if use_proximity else None
                ),
                financial=FinancialCriteria(
                    revenue_min=revenue_min * 1_000_000 if revenue_min > 0 else None,
                    revenue_max=revenue_max * 1_000_000 if revenue_max > 0 else None,
                    revenue_currency=revenue_currency
                ),
                organizational=OrganizationalCriteria(
                    employee_count_min=employee_min if employee_min > 0 else None,
                    employee_count_max=employee_max if employee_max > 0 else None
                ),
                behavioral=BehavioralSignals(
                    csr_focus_areas=csr_focus
                ),
                business_types=business_types,
                industries=[{"name": ind.strip(), "priority": i + 1} for i, ind in
                            enumerate(industries.split(','))] if industries else []
            )

            st.session_state.search_criteria = criteria
            st.success("✅ Configuration saved!")

# Tab 2: Execute Search
with tab2:
    st.header("Execute Company Search")

    # Check if we have criteria
    if st.session_state.current_template:
        if st.session_state.current_template == "RMH":
            criteria = RMH_SYDNEY_CONFIG.tier_a_criteria
        elif st.session_state.current_template == "GDV":
            criteria = GUIDE_DOGS_VICTORIA_CONFIG.tier_a_criteria
        ready_to_search = True
    elif st.session_state.search_criteria:
        criteria = st.session_state.search_criteria
        ready_to_search = True
    else:
        ready_to_search = False
        st.warning("⚠️ Please configure search criteria in the 'Search Setup' tab first.")

    if ready_to_search:
        # Search settings
        col1, col2, col3 = st.columns(3)

        with col1:
            target_count = st.number_input(
                "Target Companies",
                min_value=10,
                max_value=200,
                value=50,
                step=10
            )

        with col2:
            deployment = st.selectbox(
                "GPT Deployment",
                ["gpt-4.1", "gpt-4.1-2", "gpt-4.1-3"],
                index=0
            )

        with col3:
            estimated_cost = target_count * 0.0002  # Rough estimate
            st.metric("Estimated Cost", f"${estimated_cost:.3f}")

        # Execute search
        if st.button("🚀 Start Search", type="primary", use_container_width=True):
            with st.spinner(f"Searching for {target_count} companies..."):
                # Initialize agent
                agent = EnhancedSearchStrategistAgent(deployment_name=deployment)


                # Run search
                async def run_search():
                    return await agent.generate_enhanced_strategy(criteria, target_count=target_count)


                try:
                    result = asyncio.run(run_search())
                    companies = result['companies']

                    # Store results
                    st.session_state.search_results = [c.dict() for c in companies]

                    st.success(f"✅ Found {len(companies)} companies!")

                    # Display summary
                    if companies:
                        # Group by tier
                        tier_counts = {}
                        for c in companies:
                            tier = c.icp_tier or "Untiered"
                            tier_counts[tier] = tier_counts.get(tier, 0) + 1

                        # Display tier breakdown
                        cols = st.columns(len(tier_counts))
                        for i, (tier, count) in enumerate(sorted(tier_counts.items())):
                            with cols[i]:
                                st.metric(f"Tier {tier}", count)

                        # Show sample results
                        st.subheader("Sample Results")
                        df_data = []
                        for c in companies[:10]:
                            df_data.append({
                                "Company": c.name,
                                "Industry": c.industry_category,
                                "Revenue": c.estimated_revenue or "Unknown",
                                "Confidence": c.confidence,
                                "ICP Score": f"{c.icp_score:.1f}" if c.icp_score else "N/A"
                            })

                        st.dataframe(pd.DataFrame(df_data), use_container_width=True)

                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    st.exception(e)

# Tab 3: Validation
with tab3:
    st.header("Company Validation with Serper")

    if not st.session_state.search_results:
        st.warning("⚠️ Please run a search first in the 'Execute Search' tab.")
    else:
        all_companies = st.session_state.search_results
        st.info(f"📊 {len(all_companies)} companies ready for validation")

        # Validation settings
        col1, col2, col3 = st.columns(3)

        with col1:
            validation_tier = st.selectbox(
                "Validation Depth",
                ["Quick (1 query)", "Standard (2-3 queries)", "Comprehensive (4-6 queries)"],
                index=0
            )

        with col2:
            tiers_to_validate = st.multiselect(
                "ICP Tiers to Validate",
                ["A", "B", "C", "D", "Untiered"],
                default=["A", "B"]
            )

        with col3:
            # Filter companies by tier
            filtered_companies = [
                c for c in all_companies
                if (c.get('icp_tier') or 'Untiered') in tiers_to_validate
            ]

            if filtered_companies:
                max_companies = st.number_input(
                    "Max Companies",
                    min_value=1,
                    max_value=min(50, len(filtered_companies)),
                    value=min(10, len(filtered_companies))
                )
            else:
                st.info("No companies match tier filter")
                max_companies = 0

        if max_companies > 0:
            # Cost estimate
            queries_map = {
                "Quick (1 query)": 1,
                "Standard (2-3 queries)": 2.5,
                "Comprehensive (4-6 queries)": 5
            }
            est_queries = max_companies * queries_map[validation_tier]
            est_cost = est_queries * 0.001

            st.info(f"💰 Estimated: {est_queries:.0f} Serper queries, ~${est_cost:.3f}")

            # Execute validation
            if st.button("🔍 Start Validation", type="primary", use_container_width=True):
                with st.spinner(f"Validating {max_companies} companies..."):
                    # Create config
                    config = ValidationConfig(
                        serper_api_key=st.session_state.serper_key,
                        max_parallel_queries=parallel_queries,
                        max_cost_per_company=max_cost_per_company
                    )

                    # Get companies to validate
                    companies_to_validate = filtered_companies[:max_companies]

                    # Determine location from criteria
                    if st.session_state.current_template == "RMH":
                        location = "Sydney"
                        country = "Australia"
                    elif st.session_state.current_template == "GDV":
                        location = "Melbourne"
                        country = "Australia"
                    else:
                        # Get from criteria
                        if hasattr(criteria, 'location'):
                            location = criteria.location.cities[0] if criteria.location.cities else \
                            criteria.location.countries[0]
                            country = criteria.location.countries[0] if criteria.location.countries else "Unknown"
                        else:
                            location = "Unknown"
                            country = "Unknown"

                    # Create validation criteria
                    val_criteria = ValidationCriteria(
                        must_be_in_locations=[location]
                    )

                    # Map tier
                    tier_map = {
                        "Quick (1 query)": ValidationTier.QUICK,
                        "Standard (2-3 queries)": ValidationTier.STANDARD,
                        "Comprehensive (4-6 queries)": ValidationTier.COMPREHENSIVE
                    }


                    # Run validation
                    async def validate():
                        async with EnhancedValidationAgent(config) as agent:
                            validations = []
                            progress_bar = st.progress(0)

                            for i, company in enumerate(companies_to_validate):
                                validation = await agent.validate_company(
                                    company,
                                    val_criteria,
                                    location,
                                    country,
                                    force_tier=tier_map[validation_tier]
                                )
                                validations.append(validation)
                                progress_bar.progress((i + 1) / max_companies)

                            return validations


                    try:
                        validations = asyncio.run(validate())
                        st.session_state.validation_results = validations

                        # Show summary
                        verified = sum(1 for v in validations if v.validation_status == "verified")
                        partial = sum(1 for v in validations if v.validation_status == "partial")
                        rejected = sum(1 for v in validations if v.validation_status == "rejected")

                        st.success("✅ Validation complete!")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("✅ Verified", verified)
                        with col2:
                            st.metric("⚠️ Partial", partial)
                        with col3:
                            st.metric("❌ Rejected", rejected)
                        with col4:
                            st.metric("Total Queries", sum(v.serper_queries_used for v in validations))

                    except Exception as e:
                        st.error(f"Validation failed: {str(e)}")

# Tab 4: Results & Export
with tab4:
    st.header("Results & Export")

    # Check for results
    has_search = len(st.session_state.search_results) > 0
    has_validation = len(st.session_state.validation_results) > 0

    if not has_search and not has_validation:
        st.info("No results yet. Run a search and/or validation first.")
    else:
        # Export options
        col1, col2 = st.columns(2)

        with col1:
            export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"])

        with col2:
            include_search = st.checkbox("Include search results", value=has_search)
            include_validation = st.checkbox("Include validation results", value=has_validation)

        # Generate export
        if st.button("📥 Generate Export", type="primary", use_container_width=True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if export_format == "CSV":
                # Prepare data
                export_data = []

                if include_search and has_search:
                    for company in st.session_state.search_results:
                        export_data.append({
                            "Source": "Search",
                            "Company": company.get('name'),
                            "Industry": company.get('industry_category'),
                            "Revenue": company.get('estimated_revenue'),
                            "Employees": company.get('estimated_employees'),
                            "ICP Tier": company.get('icp_tier'),
                            "ICP Score": company.get('icp_score'),
                            "Confidence": company.get('confidence')
                        })

                if include_validation and has_validation:
                    for val in st.session_state.validation_results:
                        export_data.append({
                            "Source": "Validation",
                            "Company": val.company_name,
                            "Validation Status": val.validation_status,
                            "Validation Score": val.overall_score,
                            "Location Verified": val.location.verified,
                            "Revenue Verified": val.financial.revenue_verified,
                            "Queries Used": val.serper_queries_used
                        })

                if export_data:
                    df = pd.DataFrame(export_data)
                    csv = df.to_csv(index=False)

                    st.download_button(
                        "📥 Download CSV",
                        data=csv,
                        file_name=f"company_data_{timestamp}.csv",
                        mime="text/csv"
                    )

            elif export_format == "JSON":
                export_json = {
                    "export_date": datetime.now().isoformat(),
                    "search_results": st.session_state.search_results if include_search else [],
                    "validation_results": [
                        {
                            "company": v.company_name,
                            "status": v.validation_status,
                            "score": v.overall_score
                        } for v in st.session_state.validation_results
                    ] if include_validation else []
                }

                json_str = json.dumps(export_json, indent=2, default=str)

                st.download_button(
                    "📥 Download JSON",
                    data=json_str,
                    file_name=f"company_data_{timestamp}.json",
                    mime="application/json"
                )

# Footer
st.divider()
st.caption(f"Company Search & Validation System v2.0 | Session: {st.session_state.get('session_id', 'New')[:8]}...")