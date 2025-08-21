# ui/streamlit_app_v4.py
"""
Enhanced Streamlit App with New Validation Modes Support
Includes Simple, Raw Endpoint, Smart (Contact/CSR/Financial), and Custom validation options
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
from io import BytesIO

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
    ValidationMode,
    ValidationCriteria,
    EndpointConfig,
    SerperEndpoint
)

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
if 'validation_mode' not in st.session_state:
    st.session_state.validation_mode = ValidationMode.SIMPLE
if 'custom_endpoints' not in st.session_state:
    st.session_state.custom_endpoints = []

# Initialize ICP manager
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
        st.metric("Validated", len(st.session_state.validation_results))
    with col2:
        st.metric("Total Cost", f"${st.session_state.total_cost:.3f}")
        st.metric("Credits Used", f"{int(st.session_state.total_cost * 1000)}")

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

# Tab 1: Search Configuration (keeping existing)
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

    # [Rest of Tab 1 configuration remains the same as previous version]
    # ... (keeping all the existing search configuration code)

# Tab 2: Execute Search (keeping existing)
with tab2:
    st.header("Execute Company Search")
    # [Keeping all existing search execution code]
    # ...

# Tab 3: Enhanced Validation
with tab3:
    st.header("Company Validation")

    if not st.session_state.search_results:
        st.info("No companies to validate. Run a search first.")
    else:
        st.write(f"**{len(st.session_state.search_results)} companies** ready for validation")

        # Validation Mode Selection
        st.subheader("🎯 Validation Mode")

        col1, col2 = st.columns(2)

        with col1:
            validation_mode = st.selectbox(
                "Select Validation Mode",
                options=[
                    "Simple Check",
                    "Raw Endpoint Query",
                    "Smart Contact Extraction",
                    "Smart CSR Verification",
                    "Smart Financial Check",
                    "Full Validation",
                    "Custom Configuration"
                ],
                help="""
                • Simple Check: Verify existence (2-3 credits)
                • Raw Endpoint: Direct API access (1-2 credits)
                • Smart Contact: Extract emails/phones (3-5 credits)
                • Smart CSR: Verify CSR programs (3-5 credits)
                • Smart Financial: Revenue/employee data (3-4 credits)
                • Full Validation: All checks (10-15 credits)
                • Custom: Define your own queries
                """
            )

            # Map selection to enum
            mode_map = {
                "Simple Check": ValidationMode.SIMPLE,
                "Raw Endpoint Query": ValidationMode.RAW_ENDPOINT,
                "Smart Contact Extraction": ValidationMode.SMART_CONTACT,
                "Smart CSR Verification": ValidationMode.SMART_CSR,
                "Smart Financial Check": ValidationMode.SMART_FINANCIAL,
                "Full Validation": ValidationMode.FULL,
                "Custom Configuration": ValidationMode.CUSTOM
            }
            st.session_state.validation_mode = mode_map[validation_mode]

        with col2:
            # Show mode-specific information
            if st.session_state.validation_mode == ValidationMode.SIMPLE:
                st.info("""
                **Simple Validation**
                - Verifies company exists
                - Confirms location match
                - Basic contact info
                - ~2-3 credits per company
                """)
            elif st.session_state.validation_mode == ValidationMode.SMART_CONTACT:
                st.info("""
                **Contact Extraction**
                - Email addresses
                - Phone numbers
                - Executive names
                - LinkedIn profiles
                - ~3-5 credits per company
                """)
            elif st.session_state.validation_mode == ValidationMode.SMART_CSR:
                st.info("""
                **CSR Verification**
                - CSR programs
                - Foundation check
                - Certifications
                - Recent donations
                - ~3-5 credits per company
                """)
            elif st.session_state.validation_mode == ValidationMode.SMART_FINANCIAL:
                st.info("""
                **Financial Check**
                - Annual revenue
                - Employee count
                - Stock listing (ASX)
                - Growth indicators
                - ~3-4 credits per company
                """)
            elif st.session_state.validation_mode == ValidationMode.FULL:
                st.info("""
                **Full Validation**
                - All validation checks
                - Comprehensive data
                - High confidence
                - ~10-15 credits per company
                """)

        # Raw Endpoint Configuration
        if st.session_state.validation_mode == ValidationMode.RAW_ENDPOINT:
            st.subheader("🔧 Raw Endpoint Configuration")

            col1, col2, col3 = st.columns(3)

            with col1:
                endpoint = st.selectbox(
                    "Select Endpoint",
                    ["search", "places", "maps", "news", "shopping"],
                    help="Choose Serper API endpoint"
                )

            with col2:
                custom_query = st.text_input(
                    "Query Template",
                    value="{company} Australia",
                    help="Use {company} as placeholder"
                )

            with col3:
                num_results = st.number_input(
                    "Results per Query",
                    min_value=1,
                    max_value=20,
                    value=10
                )

        # Custom Configuration
        elif st.session_state.validation_mode == ValidationMode.CUSTOM:
            st.subheader("🔧 Custom Validation Configuration")

            # Endpoint selection
            st.write("**Select Endpoints to Use:**")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                use_places = st.checkbox("Places (2cr)", value=True)
            with col2:
                use_search = st.checkbox("Search (1cr)", value=True)
            with col3:
                use_news = st.checkbox("News (1cr)", value=False)
            with col4:
                use_maps = st.checkbox("Maps (2cr)", value=False)
            with col5:
                use_shopping = st.checkbox("Shopping (1cr)", value=False)

            # Query templates
            st.write("**Define Query Templates:**")

            custom_endpoints = []

            if use_places:
                places_query = st.text_input(
                    "Places Query",
                    value="{company} {location}",
                    key="places_query"
                )
                custom_endpoints.append(EndpointConfig(
                    endpoint=SerperEndpoint.PLACES,
                    enabled=True,
                    query_templates=[places_query]
                ))

            if use_search:
                search_queries = st.text_area(
                    "Search Queries (one per line)",
                    value='"{company}" contact email\n"{company}" about us',
                    key="search_queries",
                    height=100
                )
                custom_endpoints.append(EndpointConfig(
                    endpoint=SerperEndpoint.SEARCH,
                    enabled=True,
                    query_templates=search_queries.split('\n')
                ))

            if use_news:
                news_query = st.text_input(
                    "News Query",
                    value="{company}",
                    key="news_query"
                )
                custom_endpoints.append(EndpointConfig(
                    endpoint=SerperEndpoint.NEWS,
                    enabled=True,
                    query_templates=[news_query]
                ))

            st.session_state.custom_endpoints = custom_endpoints

        # Company Selection
        st.divider()
        st.subheader("📋 Company Selection")

        col1, col2 = st.columns(2)

        with col1:
            max_validate = st.number_input(
                "Companies to validate",
                min_value=1,
                max_value=min(50, len(st.session_state.search_results)),
                value=min(10, len(st.session_state.search_results))
            )

        with col2:
            # Cost estimate based on mode
            credits_per_company = {
                ValidationMode.SIMPLE: 3,
                ValidationMode.RAW_ENDPOINT: 1,
                ValidationMode.SMART_CONTACT: 4,
                ValidationMode.SMART_CSR: 4,
                ValidationMode.SMART_FINANCIAL: 4,
                ValidationMode.FULL: 12,
                ValidationMode.CUSTOM: len(
                    st.session_state.custom_endpoints) * 2 if st.session_state.custom_endpoints else 5
            }

            est_credits = max_validate * credits_per_company.get(st.session_state.validation_mode, 5)
            est_cost = est_credits * 0.001

            st.metric("Estimated Credits", est_credits)
            st.metric("Estimated Cost", f"${est_cost:.3f}")

        # Company preview
        with st.expander("👀 Preview Companies to Validate", expanded=False):
            preview_data = []
            for company in st.session_state.search_results[:max_validate]:
                if hasattr(company, 'dict'):
                    c = company.dict()
                else:
                    c = company

                preview_data.append({
                    "Company": c.get('name', 'Unknown'),
                    "Industry": c.get('industry_category', 'Unknown'),
                    "Location": c.get('headquarters', {}).get('city', 'Unknown') if isinstance(c.get('headquarters'),
                                                                                               dict) else 'Unknown',
                    "ICP Score": c.get('icp_score', 0)
                })

            st.dataframe(pd.DataFrame(preview_data), use_container_width=True)

        # Validation Execution
        st.divider()

        if st.button("🚀 Start Validation", type="primary", use_container_width=True):
            with st.spinner(f"Validating {max_validate} companies using {validation_mode} mode..."):

                # Create validation config
                config = ValidationConfig(
                    serper_api_key=serper_key,
                    max_parallel_queries=5,
                    max_cost_per_company=0.015
                )

                # Initialize progress tracking
                progress_bar = st.progress(0)
                status_container = st.container()


                async def run_validation():
                    """Run validation asynchronously"""
                    async with EnhancedValidationAgent(config) as agent:
                        results = []

                        # Get location from search criteria
                        location = "Australia"  # Default
                        if st.session_state.current_criteria:
                            if st.session_state.current_criteria.location.cities:
                                location = st.session_state.current_criteria.location.cities[0]
                            elif st.session_state.current_criteria.location.countries:
                                location = st.session_state.current_criteria.location.countries[0]

                        # Validate each company
                        for i, company in enumerate(st.session_state.search_results[:max_validate]):
                            if hasattr(company, 'dict'):
                                company_dict = company.dict()
                            else:
                                company_dict = company

                            company_name = company_dict.get('name', 'Unknown')

                            # Run validation based on mode
                            if st.session_state.validation_mode == ValidationMode.SIMPLE:
                                result = await agent.validate_simple(company_name, location)
                            elif st.session_state.validation_mode == ValidationMode.RAW_ENDPOINT:
                                result = await agent.validate_raw_endpoint(
                                    SerperEndpoint[endpoint.upper()],
                                    custom_query.replace("{company}", company_name),
                                    {"num": num_results}
                                )
                            elif st.session_state.validation_mode == ValidationMode.SMART_CONTACT:
                                result = await agent.validate_smart_contact(company_name, location)
                            elif st.session_state.validation_mode == ValidationMode.SMART_CSR:
                                result = await agent.validate_smart_csr(company_name)
                            elif st.session_state.validation_mode == ValidationMode.SMART_FINANCIAL:
                                result = await agent.validate_smart_financial(company_name)
                            elif st.session_state.validation_mode == ValidationMode.FULL:
                                result = await agent.validate_full(company_name, location)
                            elif st.session_state.validation_mode == ValidationMode.CUSTOM:
                                result = await agent.validate_custom(company_name, st.session_state.custom_endpoints)
                            else:
                                continue

                            results.append(result)

                            # Update progress
                            progress = (i + 1) / max_validate
                            progress_bar.progress(progress)

                            # Show status
                            with status_container:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"✅ {company_name}")
                                with col2:
                                    st.write(f"Status: {result.validation_status}")
                                with col3:
                                    st.write(f"Credits: {result.credits_used}")

                        # Get summary
                        summary = agent.get_validation_summary()
                        return results, summary


                # Run validation
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                validation_results, summary = loop.run_until_complete(run_validation())
                loop.close()

                # Store results
                st.session_state.validation_results = validation_results
                st.session_state.total_cost += summary['estimated_cost']

                # Clear progress
                progress_bar.empty()

                # Show summary
                st.success(f"✅ Validation complete! {len(validation_results)} companies validated")

                col1, col2, col3 = st.columns(3)
                with col1:
                    verified = sum(1 for r in validation_results if r.validation_status == "verified")
                    st.metric("Verified", f"{verified}/{len(validation_results)}")
                with col2:
                    st.metric("Credits Used", summary['total_credits_used'])
                with col3:
                    st.metric("Cost", f"${summary['estimated_cost']:.3f}")

        # Display Validation Results
        if st.session_state.validation_results:
            st.divider()
            st.subheader("📊 Validation Results")

            # Create results dataframe based on mode
            results_data = []

            for result in st.session_state.validation_results:
                row = {
                    "Company": result.company_name,
                    "Status": result.validation_status,
                    "Confidence": f"{result.confidence_score:.0f}%",
                    "Credits": result.credits_used
                }

                # Add mode-specific columns
                if result.validation_mode == ValidationMode.SMART_CONTACT:
                    contact_info = result.contact_info
                    row["Emails"] = len(contact_info.get('emails', []))
                    row["Phones"] = len(contact_info.get('phones', []))
                    row["Names"] = len(contact_info.get('names', []))

                elif result.validation_mode == ValidationMode.SMART_CSR:
                    csr_info = result.csr_info
                    row["Has CSR"] = "✓" if csr_info.get('has_csr') else "✗"
                    row["Focus Areas"] = ", ".join(csr_info.get('focus_areas', []))
                    row["Certifications"] = ", ".join(csr_info.get('certifications', []))

                elif result.validation_mode == ValidationMode.SMART_FINANCIAL:
                    fin_info = result.financial_info
                    row["Revenue"] = fin_info.get('revenue', 'N/A')
                    row["Employees"] = fin_info.get('employees', 'N/A')
                    row["Listed"] = "✓" if fin_info.get('stock_listed') else "✗"

                results_data.append(row)

            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True, height=400)

            # Export validation results
            with st.expander("💾 Export Validation Results"):
                col1, col2 = st.columns(2)

                with col1:
                    # CSV export
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        "📥 Download Excel Report",
                        data=excel_data,
                        file_name=f"companies_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        with col3:
            # Full JSON export
            export_data = {
                "export_date": datetime.now().isoformat(),
                "search_criteria": asdict(
                    st.session_state.current_criteria) if st.session_state.current_criteria and hasattr(
                    st.session_state.current_criteria, '__dict__') else {},
                "companies": combined_data,
                "validation_details": [asdict(r) for r in
                                       st.session_state.validation_results] if st.session_state.validation_results else [],
                "statistics": {
                    "total_companies": len(df_combined),
                    "validated_count": len(
                        st.session_state.validation_results) if st.session_state.validation_results else 0,
                    "total_cost": st.session_state.total_cost,
                    "total_credits": int(st.session_state.total_cost * 1000)
                }
            }

            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                "📥 Download Full JSON",
                data=json_str,
                file_name=f"companies_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    elif st.session_state.search_results:
    # Only search results available
    st.subheader(f"Search Results ({len(st.session_state.search_results)} companies)")

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
        sort_by = st.selectbox("Sort by", df.columns.tolist(),
                               index=df.columns.tolist().index("ICP Score") if "ICP Score" in df.columns else 0)
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

# Footer with enhanced stats
st.divider()

# Session statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.caption(f"💰 Session Cost: ${st.session_state.total_cost:.3f}")

with col2:
    st.caption(f"🎯 Credits Used: {int(st.session_state.total_cost * 1000)}")

with col3:
    if st.session_state.validation_results:
        verified = sum(1 for r in st.session_state.validation_results if r.validation_status == "verified")
        st.caption(f"✅ Verified: {verified}/{len(st.session_state.validation_results)}")
    else:
        st.caption("✅ No validations yet")

with col4:
    st.caption(f"🤖 Models: {', '.join(selected_models if 'selected_models' in locals() else ['Not selected'])}")

# Help section in sidebar
with st.sidebar:
    st.divider()
    with st.expander("❓ Help & Tips"):
        st.markdown("""
        ### Validation Modes Guide

        **🎯 Use Cases:**
        - **Simple Check**: Quick verification before outreach
        - **Contact Extraction**: Building contact lists
        - **CSR Verification**: RMH/Guide Dogs requirements
        - **Financial Check**: Verify revenue claims
        - **Full Validation**: Complete due diligence

        **💡 Tips:**
        - Start with Simple Check to filter companies
        - Use Contact Extraction for top prospects
        - CSR mode essential for charity partnerships
        - Custom mode for specific requirements

        **💰 Credit Optimization:**
        - Simple: 2-3 credits (most efficient)
        - Smart modes: 3-5 credits (targeted)
        - Full: 10-15 credits (comprehensive)
        - Batch similar companies together

        **🔍 Raw Endpoint Access:**
        - Direct Serper API queries
        - Full control over search
        - See raw JSON responses
        - Good for testing queries
        """)

    with st.expander("📊 Validation Fields"):
        st.markdown("""
        ### Data Extracted by Mode

        **Simple Check:**
        - Company exists
        - Location match
        - Basic contact (phone, address)
        - Website

        **Contact Extraction:**
        - Email addresses
        - Phone numbers
        - Executive names
        - Job titles
        - LinkedIn profiles

        **CSR Verification:**
        - CSR programs
        - Company foundation
        - Focus areas (children, community, etc.)
        - Certifications (B-Corp, ISO)
        - Recent donations
        - Sustainability reports

        **Financial Check:**
        - Annual revenue
        - Employee count
        - Growth stage
        - Stock listing (ASX)
        - Recent funding
        - Financial events

        **Full Validation:**
        - All of the above
        - Weighted confidence score
        - Comprehensive coverage
        """)

# Add version info
st.sidebar.divider()
st.sidebar.caption("Version 4.0 - Enhanced Validation Modes")
st.sidebar.caption("© 2024 Company Search & Validation Platform")
d_button(
    "📥 Download Validation CSV",
    data=csv,
    file_name=f"validation_{st.session_state.validation_mode.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

with col2:
    # JSON export with full details
    json_data = [asdict(r) for r in st.session_state.validation_results]
    json_str = json.dumps(json_data, indent=2, default=str)
    st.download_button(
        "📥 Download Full JSON",
        data=json_str,
        file_name=f"validation_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# Tab 4: Results & Export (enhanced with validation data)
with tab4:
    st.header("Results & Export")

    # Combined results if both search and validation done
    if st.session_state.search_results and st.session_state.validation_results:
        st.subheader("🎯 Combined Results")

        # Merge search and validation data
        combined_data = []

        # Create validation lookup
        validation_lookup = {}
        for val_result in st.session_state.validation_results:
            validation_lookup[val_result.company_name] = val_result

        for company in st.session_state.search_results:
            if hasattr(company, 'dict'):
                c = company.dict()
            else:
                c = company

            company_name = c.get('name', 'Unknown')

            # Get validation data if exists
            validation = validation_lookup.get(company_name)

            row = {
                "Company": company_name,
                "Industry": c.get('industry_category', 'Unknown'),
                "Business Type": c.get('business_type', 'Unknown'),
                "Revenue": c.get('estimated_revenue', 'Unknown'),
                "Employees": c.get('estimated_employees', 'Unknown'),
                "Location": c.get('headquarters', {}).get('city', 'Unknown') if isinstance(c.get('headquarters'),
                                                                                           dict) else 'Unknown',
                "ICP Score": c.get('icp_score', 0),
                "ICP Tier": c.get('icp_tier', 'Untiered'),
                "Search Confidence": c.get('confidence', 'Unknown')
            }

            # Add validation data if available
            if validation:
                row["Validation Status"] = validation.validation_status
                row["Validation Confidence"] = f"{validation.confidence_score:.0f}%"

                # Add contact info if available
                if validation.contact_info:
                    emails = validation.contact_info.get('emails', [])
                    phones = validation.contact_info.get('phones', [])
                    row["Email"] = emails[0] if emails else ""
                    row["Phone"] = phones[0] if phones else ""

                # Add CSR info if available
                if validation.csr_info:
                    row["CSR Programs"] = "✓" if validation.csr_info.get('has_csr') else "✗"

                # Add financial info if available
                if validation.financial_info:
                    if validation.financial_info.get('revenue'):
                        row["Verified Revenue"] = validation.financial_info['revenue']
                    if validation.financial_info.get('employees'):
                        row["Verified Employees"] = validation.financial_info['employees']
            else:
                row["Validation Status"] = "Not Validated"

            combined_data.append(row)

        df_combined = pd.DataFrame(combined_data)

        # Display options
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox("Sort by", df_combined.columns.tolist())
        with col2:
            sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
        with col3:
            filter_validated = st.checkbox("Show Validated Only", value=False)

        # Apply filters
        if filter_validated and "Validation Status" in df_combined.columns:
            df_display = df_combined[df_combined["Validation Status"] != "Not Validated"]
        else:
            df_display = df_combined

        # Sort
        df_display = df_display.sort_values(sort_by, ascending=(sort_order == "Ascending"))

        # Display
        st.dataframe(df_display, use_container_width=True, height=500)

        # Export options
        st.divider()
        st.subheader("📥 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv = df_display.to_csv(index=False)
            st.download_button(
                "📥 Download Complete CSV",
                data=csv,
                file_name=f"companies_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            # Excel export
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_display.to_excel(writer, sheet_name='Companies', index=False)

                # Add summary sheet
                summary_data = {
                    "Metric": [
                        "Total Companies",
                        "Validated Companies",
                        "Verified Companies",
                        "Average ICP Score",
                        "Total Credits Used",
                        "Total Cost"
                    ],
                    "Value": [
                        len(df_combined),
                        sum(1 for r in
                            st.session_state.validation_results) if st.session_state.validation_results else 0,
                        sum(1 for r in st.session_state.validation_results if
                            r.validation_status == "verified") if st.session_state.validation_results else 0,
                        df_combined["ICP Score"].mean() if "ICP Score" in df_combined.columns else 0,
                        int(st.session_state.total_cost * 1000),
                        f"${st.session_state.total_cost:.3f}"
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

            excel_data = output.getvalue()

            st.downloa