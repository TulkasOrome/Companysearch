# tabs/validation_tab.py
"""
Validation Tab - Simplified with validation approach in closed accordion
"""

import streamlit as st
import asyncio
import pandas as pd
import time
from datetime import datetime
from shared.validation_utils import validate_company_with_serper
from shared.session_state import update_cost


def render_validation_tab(serper_key):
    """Render the Validation tab with simplified approach selection"""

    st.header("Company Validation")

    if not st.session_state.search_results:
        st.info("No companies to validate. Run a search first.")
    else:
        # Track validated company names
        if 'validated_company_names' not in st.session_state:
            st.session_state.validated_company_names = set()

        # Update the set from existing validation results
        if st.session_state.validation_results:
            for v in st.session_state.validation_results:
                st.session_state.validated_company_names.add(v.get('company_name', '').lower())

        # Count companies
        unvalidated_companies = []
        for company in st.session_state.search_results:
            if hasattr(company, 'dict'):
                company_dict = company.dict()
            elif isinstance(company, dict):
                company_dict = company
            else:
                company_dict = {'name': str(company)}

            company_name = company_dict.get('name', 'Unknown').lower()
            if company_name not in st.session_state.validated_company_names:
                unvalidated_companies.append(company)

        total_companies = len(st.session_state.search_results)
        validated_count = len(st.session_state.validated_company_names)
        unvalidated_count = len(unvalidated_companies)

        # Display status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Companies", total_companies)
        with col2:
            st.metric("Already Validated", validated_count)
        with col3:
            st.metric("Not Yet Validated", unvalidated_count)

        if unvalidated_count == 0:
            st.success("✅ All companies have been validated!")
        else:
            st.info(f"📋 {unvalidated_count} companies available for validation")

        # Validation approach in accordion (CLOSED by default)
        with st.expander("⚙️ Validation Approach", expanded=False):
            validation_approach = st.radio(
                "Choose validation approach",
                [
                    "Continue from last position (skip already validated)",
                    "Re-validate from beginning (overwrite existing)",
                    "Select specific companies to validate"
                ],
                index=0,  # Default to "Continue from last position"
                help="Choose how to handle previously validated companies"
            )

        # Get the selected approach (use default if expander not opened)
        if 'validation_approach' not in locals():
            validation_approach = "Continue from last position (skip already validated)"

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

        # Mode information in expander
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
            with st.expander("ℹ️ Mode Information", expanded=False):
                info = mode_info[validation_mode]
                st.write(f"**Description:** {info['description']}")
                st.write("**Extracts:**")
                for item in info['extracts']:
                    st.write(f"  • {item}")

        # Handle different validation approaches
        companies_to_validate = []

        if validation_approach == "Continue from last position (skip already validated)":
            companies_to_validate = unvalidated_companies

        elif validation_approach == "Re-validate from beginning (overwrite existing)":
            companies_to_validate = st.session_state.search_results
            if st.checkbox("⚠️ Confirm: This will overwrite existing validation data"):
                st.warning("Existing validation results will be replaced!")
            else:
                companies_to_validate = []

        else:  # Select specific companies
            # Create a selectable list
            company_options = []
            for i, company in enumerate(st.session_state.search_results):
                if hasattr(company, 'dict'):
                    company_dict = company.dict()
                elif isinstance(company, dict):
                    company_dict = company
                else:
                    company_dict = {'name': str(company)}

                name = company_dict.get('name', f'Company {i + 1}')
                validated_status = "✅" if name.lower() in st.session_state.validated_company_names else "❌"
                company_options.append(f"{validated_status} {name}")

            selected_indices = st.multiselect(
                "Select companies to validate",
                options=range(len(company_options)),
                format_func=lambda x: company_options[x],
                help="✅ = Already validated, ❌ = Not validated"
            )

            companies_to_validate = [st.session_state.search_results[i] for i in selected_indices]

        # Validation settings
        if companies_to_validate:
            col1, col2 = st.columns(2)

            with col1:
                max_available = len(companies_to_validate)
                max_validate = st.number_input(
                    f"Companies to validate (max {max_available})",
                    1,
                    min(500, max_available),
                    min(10, max_available)
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

            # Show what will be validated
            if validation_approach == "Continue from last position (skip already validated)":
                st.success(
                    f"✅ Will validate {max_validate} new companies (skipping {validated_count} already validated)")
            elif validation_approach == "Select specific companies to validate":
                st.info(f"📋 Will validate {max_validate} selected companies")
            else:
                st.warning(f"⚠️ Will re-validate {max_validate} companies from the beginning")

            # Validation button
            if st.button("✅ Start Validation", type="primary", use_container_width=True):
                execute_validation(
                    companies_to_validate[:max_validate],
                    validation_mode,
                    serper_key,
                    validation_approach
                )
        else:
            if validation_approach == "Re-validate from beginning (overwrite existing)":
                st.warning("Please confirm overwriting existing validation data above.")
            else:
                st.info("No companies available to validate with current settings.")

    # Display existing validation results
    if st.session_state.validation_results and len(st.session_state.validation_results) > 0:
        display_validation_results()


def execute_validation(companies_to_validate, validation_mode, serper_key, validation_approach):
    """Execute validation with progress tracking"""

    progress_bar = st.progress(0)
    status_text = st.empty()

    validated_companies = []

    # Keep existing validation results if continuing
    if validation_approach == "Continue from last position (skip already validated)":
        validated_companies = st.session_state.validation_results.copy() if st.session_state.validation_results else []

    for i, company in enumerate(companies_to_validate):
        progress = (i + 1) / len(companies_to_validate)
        progress_bar.progress(progress)

        # Extract company details
        if hasattr(company, 'dict'):
            company_dict = company.dict()
        elif isinstance(company, dict):
            company_dict = company
        else:
            company_dict = {'name': str(company)}

        company_name = company_dict.get('name', 'Unknown')
        status_text.text(f"Validating {i + 1}/{len(companies_to_validate)}: {company_name}")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            validation_result = loop.run_until_complete(
                validate_company_with_serper(
                    company_dict,
                    validation_mode,
                    serper_key
                )
            )
            loop.close()

            validated_companies.append(validation_result)

            # Add to tracked set
            st.session_state.validated_company_names.add(company_name.lower())

            # Rate limiting
            if i < len(companies_to_validate) - 1:
                time.sleep(0.5)

        except Exception as e:
            st.error(f"Error validating {company_name}: {str(e)}")
            validation_result = {
                'company_name': company_name,
                'validation_status': 'error',
                'mode': validation_mode,
                'credits_used': 0,
                'validation_timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            validated_companies.append(validation_result)

    # Update session state
    if validation_approach == "Re-validate from beginning (overwrite existing)":
        st.session_state.validation_results = validated_companies
    else:
        # For continuing or select specific, merge appropriately
        if validation_approach == "Continue from last position (skip already validated)":
            st.session_state.validation_results = validated_companies
        else:
            # Merge for selected companies
            existing_names = {v.get('company_name', '').lower() for v in
                              st.session_state.validation_results} if st.session_state.validation_results else set()
            new_validations = [v for v in validated_companies if
                               v.get('company_name', '').lower() not in existing_names]

            if st.session_state.validation_results:
                st.session_state.validation_results.extend(new_validations)
            else:
                st.session_state.validation_results = new_validations

    # Calculate cost
    total_credits = sum(v.get('credits_used', 0) for v in validated_companies)
    actual_cost = total_credits * 0.001
    update_cost(actual_cost)

    progress_bar.empty()
    status_text.empty()

    # Show completion
    st.success(f"✅ Validated {len(companies_to_validate)} companies using {total_credits} credits (${actual_cost:.3f})")

    # Show summary
    display_validation_summary(validated_companies)


def display_validation_summary(validated_companies):
    """Display summary of validation results"""
    st.divider()
    st.subheader("Validation Results Summary")

    verified = len([v for v in validated_companies if
                    v.get('validation_status', '').lower() in ['verified', 'verify', 'valid']])
    partial = len([v for v in validated_companies if
                   v.get('validation_status', '').lower() in ['partial', 'partially']])
    unverified = len([v for v in validated_companies if
                      v.get('validation_status', '').lower() in ['unverified', 'not verified']])
    errors = len([v for v in validated_companies if
                  v.get('validation_status', '').lower() in ['error', 'failed']])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Verified", verified)
    with col2:
        st.metric("Partial", partial)
    with col3:
        st.metric("Unverified", unverified)
    with col4:
        st.metric("Errors", errors)


def display_validation_results():
    """Display all validation results"""
    st.divider()
    st.subheader("📊 All Validation Results")

    # Summary metrics
    all_verified = len([v for v in st.session_state.validation_results if
                        v.get('validation_status', '').lower() in ['verified', 'verify', 'valid']])
    all_partial = len([v for v in st.session_state.validation_results if
                       v.get('validation_status', '').lower() in ['partial', 'partially']])
    all_unverified = len([v for v in st.session_state.validation_results if
                          v.get('validation_status', '').lower() in ['unverified', 'not verified']])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Validated", len(st.session_state.validation_results))
    with col2:
        st.metric("Verified", all_verified)
    with col3:
        st.metric("Partial", all_partial)
    with col4:
        st.metric("Unverified", all_unverified)

    # Create display DataFrame
    display_data = []
    for val_result in st.session_state.validation_results:
        display_row = {
            "Company": val_result.get('company_name', 'Unknown'),
            "Status": val_result.get('validation_status', 'unknown'),
            "Mode": val_result.get('mode', 'Unknown'),
            "Credits": val_result.get('credits_used', 0)
        }

        # Add key fields
        if val_result.get('emails'):
            display_row["Email"] = val_result['emails'][0]
        if val_result.get('revenue_range'):
            display_row["Revenue"] = val_result['revenue_range']

        display_data.append(display_row)

    if display_data:
        df = pd.DataFrame(display_data)

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Filter by Status", ["All", "Verified", "Partial", "Unverified"])
        with col2:
            search_term = st.text_input("Search companies", placeholder="Company name...")
        with col3:
            if st.button("🗑️ Clear All Validations"):
                st.session_state.validation_results = []
                st.session_state.validated_company_names = set()
                st.rerun()

        # Apply filters
        filtered_df = df.copy()

        if status_filter != "All":
            status_map = {
                "Verified": ['verified', 'verify', 'valid'],
                "Partial": ['partial', 'partially'],
                "Unverified": ['unverified', 'not verified']
            }
            if status_filter in status_map:
                filtered_df = filtered_df[filtered_df['Status'].str.lower().isin(status_map[status_filter])]

        if search_term:
            filtered_df = filtered_df[filtered_df['Company'].str.contains(search_term, case=False, na=False)]

        st.write(f"Showing {len(filtered_df)} of {len(df)} companies")

        # Display with color coding
        def color_status(val):
            if isinstance(val, str):
                val_lower = val.lower()
                if val_lower in ['verified', 'verify', 'valid']:
                    return 'background-color: #90EE90'
                elif val_lower in ['partial', 'partially']:
                    return 'background-color: #FFE4B5'
                elif val_lower in ['error', 'failed']:
                    return 'background-color: #FFB6C1'
            return 'background-color: #D3D3D3'

        if 'Status' in filtered_df.columns:
            styled_df = filtered_df.style.applymap(color_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True, height=400)
        else:
            st.dataframe(filtered_df, use_container_width=True, height=400)