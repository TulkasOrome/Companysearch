# tabs/results_export_tab.py
"""
Results & Export Tab
TODO: This tab was missing in the original implementation
This is a placeholder with the expected functionality
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from io import BytesIO
import xlsxwriter


def render_results_export_tab():
    """Render the Results & Export tab"""

    st.header("📊 Results & Export")

    # Check if there are results to display
    has_search = len(st.session_state.search_results) > 0
    has_validation = len(st.session_state.validation_results) > 0

    if not has_search and not has_validation:
        st.info("No results to display. Please run a search or validation first.")
        return

    # Results summary
    st.subheader("📈 Results Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Companies Found", len(st.session_state.search_results))
    with col2:
        st.metric("Companies Validated", len(st.session_state.validation_results))
    with col3:
        if has_validation:
            verified = len([v for v in st.session_state.validation_results
                            if v.get('validation_status', '').lower() in ['verified', 'verify', 'valid']])
            st.metric("Verified Companies", verified)
        else:
            st.metric("Verified Companies", 0)
    with col4:
        st.metric("Total Cost", f"${st.session_state.total_cost:.3f}")

    st.divider()

    # Combined results view
    st.subheader("🔍 Combined Results View")

    # Prepare combined data
    combined_data = []

    for company in st.session_state.search_results:
        if hasattr(company, 'dict'):
            company_dict = company.dict()
        elif isinstance(company, dict):
            company_dict = company
        else:
            continue

        # Base company data
        row = {
            "Company Name": company_dict.get('name', 'Unknown'),
            "Industry": company_dict.get('industry_category', 'Unknown'),
            "Business Type": company_dict.get('business_type', 'Unknown'),
            "Revenue": company_dict.get('estimated_revenue', 'Unknown'),
            "Employees": company_dict.get('estimated_employees', 'Unknown'),
            "ICP Score": company_dict.get('icp_score', 0),
            "ICP Tier": company_dict.get('icp_tier', 'D'),
            "Confidence": company_dict.get('confidence', 'Unknown')
        }

        # Add location data if available
        if company_dict.get('headquarters'):
            hq = company_dict['headquarters']
            if isinstance(hq, dict):
                row["HQ City"] = hq.get('city', 'Unknown')
            else:
                row["HQ City"] = str(hq)
        else:
            row["HQ City"] = 'Unknown'

        # Add CSR data if available
        csr_areas = company_dict.get('csr_focus_areas', [])
        row["CSR Focus Areas"] = ', '.join(csr_areas) if csr_areas else 'None'

        # Check if company has been validated
        validation_data = next((v for v in st.session_state.validation_results
                                if v.get('company_name', '').lower() == company_dict.get('name', '').lower()),
                               None)

        if validation_data:
            row["Validation Status"] = validation_data.get('validation_status', 'Not Validated')
            row["Validation Mode"] = validation_data.get('mode', 'N/A')

            # Add contact info if available
            emails = validation_data.get('emails', [])
            row["Email"] = emails[0] if emails else ''

            phones = validation_data.get('phones', [])
            row["Phone"] = phones[0] if phones else validation_data.get('phone', '')

            # Add risk signals if available
            risk_signals = validation_data.get('risk_signals', [])
            row["Risk Signals"] = len(risk_signals)
        else:
            row["Validation Status"] = "Not Validated"
            row["Validation Mode"] = "N/A"
            row["Email"] = ""
            row["Phone"] = ""
            row["Risk Signals"] = 0

        combined_data.append(row)

    # Create DataFrame
    if combined_data:
        df = pd.DataFrame(combined_data)

        # Filter options
        st.subheader("🔧 Filter Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            icp_filter = st.multiselect(
                "Filter by ICP Tier",
                ["A", "B", "C", "D"],
                default=["A", "B", "C", "D"]
            )

        with col2:
            validation_filter = st.selectbox(
                "Filter by Validation Status",
                ["All", "Validated Only", "Not Validated", "Verified Only"]
            )

        with col3:
            search_term = st.text_input(
                "Search Companies",
                placeholder="Enter company name..."
            )

        # Apply filters
        filtered_df = df.copy()

        # ICP filter
        filtered_df = filtered_df[filtered_df['ICP Tier'].isin(icp_filter)]

        # Validation filter
        if validation_filter == "Validated Only":
            filtered_df = filtered_df[filtered_df['Validation Status'] != "Not Validated"]
        elif validation_filter == "Not Validated":
            filtered_df = filtered_df[filtered_df['Validation Status'] == "Not Validated"]
        elif validation_filter == "Verified Only":
            filtered_df = filtered_df[
                filtered_df['Validation Status'].str.lower().isin(['verified', 'verify', 'valid'])]

        # Search filter
        if search_term:
            filtered_df = filtered_df[
                filtered_df['Company Name'].str.contains(search_term, case=False, na=False)
            ]

        # Display results
        st.write(f"Showing {len(filtered_df)} of {len(df)} companies")

        # Sort options
        sort_col = st.selectbox(
            "Sort by",
            filtered_df.columns.tolist(),
            index=filtered_df.columns.tolist().index("ICP Score") if "ICP Score" in filtered_df.columns else 0
        )

        sort_order = st.radio("Sort order", ["Descending", "Ascending"], horizontal=True)

        filtered_df = filtered_df.sort_values(
            by=sort_col,
            ascending=(sort_order == "Ascending")
        )

        # Display the filtered DataFrame
        st.dataframe(filtered_df, use_container_width=True, height=400)

        st.divider()

        # Export options
        st.subheader("💾 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Export to CSV
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name=f"companies_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Export to Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                filtered_df.to_excel(writer, sheet_name='Companies', index=False)

                # Get workbook and worksheet objects
                workbook = writer.book
                worksheet = writer.sheets['Companies']

                # Add formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D7E4BD',
                    'border': 1
                })

                # Write the header row with formatting
                for col_num, value in enumerate(filtered_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                # Auto-fit columns
                for i, col in enumerate(filtered_df.columns):
                    column_width = max(filtered_df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, min(column_width, 50))

            excel_data = output.getvalue()

            st.download_button(
                label="📊 Download as Excel",
                data=excel_data,
                file_name=f"companies_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with col3:
            # Export to JSON
            json_data = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                label="🔧 Download as JSON",
                data=json_data,
                file_name=f"companies_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        # Session save/load
        st.divider()
        st.subheader("💼 Session Management")

        col1, col2 = st.columns(2)

        with col1:
            # Save session
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "search_criteria": st.session_state.current_criteria.__dict__ if st.session_state.current_criteria else None,
                "search_results": [c.dict() if hasattr(c, 'dict') else c for c in st.session_state.search_results],
                "validation_results": st.session_state.validation_results,
                "total_cost": st.session_state.total_cost,
                "model_performance": st.session_state.model_success_status
            }

            session_json = json.dumps(session_data, indent=2, default=str)

            st.download_button(
                label="💾 Save Session",
                data=session_json,
                file_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                help="Save your current session to continue later"
            )

        with col2:
            # Load session
            uploaded_file = st.file_uploader(
                "Load Session",
                type=['json'],
                help="Load a previously saved session"
            )

            if uploaded_file is not None:
                try:
                    session_data = json.load(uploaded_file)

                    # Restore session data
                    if st.button("🔄 Restore Session", use_container_width=True):
                        st.session_state.search_results = session_data.get('search_results', [])
                        st.session_state.validation_results = session_data.get('validation_results', [])
                        st.session_state.total_cost = session_data.get('total_cost', 0)

                        st.success("✅ Session restored successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error loading session: {str(e)}")

    else:
        st.warning("No data available for display")