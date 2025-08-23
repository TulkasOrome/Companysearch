# tabs/search_config_tab.py
"""
Search Configuration Tab - Exact replica from original streamlit_app.py
"""

import streamlit as st
from shared.data_models import (
    SearchCriteria,
    LocationCriteria,
    FinancialCriteria,
    OrganizationalCriteria,
    BehavioralSignals
)


def render_search_config_tab(icp_manager):
    """Render the Search Configuration tab - exact replica from original"""

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
    if st.session_state.current_profile_name and icp_manager:
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
            countries = st.multiselect(
                "Countries",
                country_options,
                default=default_countries
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

        # Office types field
        office_types = st.multiselect(
            "Office Types",
            ["Headquarters", "Major Office", "Regional Office", "Branch Office"],
            default=criteria.organizational.office_types if criteria else []
        )

    # Industry & Business Section
    with st.expander("🏢 Industry & Business Type", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            business_options = ["B2B", "B2C", "B2B2C", "D2C", "Professional Services", "Enterprise", "SaaS"]
            default_types = criteria.business_types if criteria else ["B2B", "B2C"]
            business_types = st.multiselect(
                "Business Types",
                business_options,
                default=default_types
            )

        with col2:
            default_industries = []
            if criteria and criteria.industries:
                for ind in criteria.industries:
                    default_industries.append(ind.get('name', ''))

            industries = st.text_area(
                "Industries (one per line)",
                value="\n".join(default_industries),
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
            # Ensure defaults exist in options
            valid_csr_defaults = [d for d in default_csr if d in csr_options]

            csr_focus = st.multiselect(
                "CSR Focus Areas",
                csr_options,
                default=valid_csr_defaults
            )

            event_options = ["Office Move", "CSR Launch", "Expansion", "Anniversary", "Award",
                             "New Leadership", "IPO", "Merger", "Partnership"]
            default_events = criteria.behavioral.recent_events if criteria else []
            # Ensure defaults exist in options
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
            # Ensure defaults exist in options
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

    # Exclusions Section (RESTORED)
    with st.expander("🚫 Exclusions", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            exclusion_options = ["Gambling", "Tobacco", "Fast Food", "Racing", "Alcohol",
                                 "Weapons", "Animal Testing"]
            default_exc_ind = criteria.excluded_industries if criteria else []
            # Ensure defaults exist in options
            valid_exc_defaults = [d for d in default_exc_ind if d in exclusion_options]

            excluded_industries = st.multiselect(
                "Excluded Industries",
                exclusion_options,
                default=valid_exc_defaults
            )

            behavior_exclusion_options = ["Recent Misconduct", "Bankruptcy", "Litigation",
                                          "Environmental Violations", "Labor Issues"]
            default_exc_behaviors = criteria.excluded_behaviors if criteria else []
            valid_behavior_defaults = [d for d in default_exc_behaviors if d in behavior_exclusion_options]

            excluded_behaviors = st.multiselect(
                "Excluded Behaviors",
                behavior_exclusion_options,
                default=valid_behavior_defaults
            )

        with col2:
            default_exc_comp = criteria.excluded_companies if criteria else []
            excluded_companies = st.text_area(
                "Excluded Companies (one per line)",
                value="\n".join(default_exc_comp),
                height=100,
                placeholder="McDonald's\nKFC\nBurger King"
            )

            location_exclusions = st.text_input(
                "Location Exclusions",
                placeholder="Rural areas, Remote regions"
            )

    # Advanced Search Options Section
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
                employee_count_max=employee_max if employee_max > 0 else None,
                office_types=office_types if office_types else []
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
            excluded_companies=[c.strip() for c in excluded_companies.split('\n') if c.strip()],
            excluded_behaviors=excluded_behaviors
        )

        st.session_state.current_criteria = built_criteria
        st.success("✅ Criteria confirmed! Go to 'Execute Search' tab.")