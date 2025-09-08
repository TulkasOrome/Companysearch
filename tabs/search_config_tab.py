# tabs/search_config_tab.py
"""
Search Configuration Tab - Updated with revenue categories
"""

import streamlit as st
from shared.data_models import (
    SearchCriteria,
    LocationCriteria,
    FinancialCriteria,
    OrganizationalCriteria,
    BehavioralSignals,
    determine_revenue_categories_from_range
)


def render_search_config_tab(icp_manager):
    """Render the Search Configuration tab with revenue categories"""

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
                    "A": "Tier A - Perfect Match (Revenue $5M-$100M, 50+ employees, CSR focus)",
                    "B": "Tier B - Good Match (Revenue $2M-$200M, 20+ employees, Community involvement)",
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
                    "B": "Tier B - Exploratory Partners (Revenue $50M-$500M, 100-500 employees)",
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

    # Financial Section - UPDATED WITH REVENUE CATEGORIES
    with st.expander("💰 Financial Criteria", expanded=True):
        st.markdown("### Revenue Range")

        # Revenue category selector
        revenue_categories = {
            "Very Low": {"min": 0, "max": 1, "desc": "< $1M", "cat": "very_low"},
            "Low": {"min": 1, "max": 10, "desc": "$1M - $10M", "cat": "low"},
            "Medium": {"min": 10, "max": 100, "desc": "$10M - $100M", "cat": "medium"},
            "High": {"min": 100, "max": 1000, "desc": "$100M - $1B", "cat": "high"},
            "Very High": {"min": 1000, "max": float('inf'), "desc": "$1B+", "cat": "very_high"}
        }

        # Create a double-ended slider for revenue range
        col1, col2 = st.columns(2)

        with col1:
            # Minimum revenue selector
            min_revenue_options = ["No minimum", "< $1M", "$1M+", "$10M+", "$100M+", "$1B+"]
            min_revenue_values = [0, 0, 1, 10, 100, 1000]  # in millions

            default_min_idx = 0
            if criteria and criteria.financial.revenue_min:
                min_val = criteria.financial.revenue_min / 1_000_000
                # Find closest option
                for i, val in enumerate(min_revenue_values):
                    if val <= min_val:
                        default_min_idx = i

            min_revenue_selection = st.selectbox(
                "Minimum Revenue",
                min_revenue_options,
                index=default_min_idx,
                help="Select minimum revenue threshold"
            )
            min_revenue_millions = min_revenue_values[min_revenue_options.index(min_revenue_selection)]

        with col2:
            # Maximum revenue selector
            max_revenue_options = ["No maximum", "< $1M", "< $10M", "< $100M", "< $1B"]
            max_revenue_values = [float('inf'), 1, 10, 100, 1000]  # in millions

            default_max_idx = 0
            if criteria and criteria.financial.revenue_max:
                max_val = criteria.financial.revenue_max / 1_000_000
                # Find closest option
                for i, val in enumerate(max_revenue_values):
                    if val >= max_val:
                        default_max_idx = i
                        break

            max_revenue_selection = st.selectbox(
                "Maximum Revenue",
                max_revenue_options,
                index=default_max_idx,
                help="Select maximum revenue threshold"
            )
            max_revenue_millions = max_revenue_values[max_revenue_options.index(max_revenue_selection)]

        # Show which categories will be included
        if min_revenue_millions > 0 or max_revenue_millions < float('inf'):
            included_categories = []
            category_list = []

            for cat_name, cat_info in revenue_categories.items():
                # Check if this category overlaps with selected range
                cat_overlaps = False

                if max_revenue_millions == float('inf'):
                    # No max limit
                    if cat_info["max"] == float('inf') or cat_info["max"] > min_revenue_millions:
                        cat_overlaps = True
                elif min_revenue_millions == 0:
                    # No min limit
                    if cat_info["min"] < max_revenue_millions:
                        cat_overlaps = True
                else:
                    # Both limits
                    if (cat_info["min"] < max_revenue_millions and
                            (cat_info["max"] == float('inf') or cat_info["max"] > min_revenue_millions)):
                        cat_overlaps = True

                if cat_overlaps:
                    included_categories.append(cat_info["cat"])
                    category_list.append(f"**{cat_name}** ({cat_info['desc']})")

            st.info(f"📊 This will search for companies in these revenue categories:\n" +
                    ", ".join(category_list))

        st.divider()

        # Employee counts
        col1, col2, col3 = st.columns(3)

        with col1:
            default_emp_min = criteria.organizational.employee_count_min if criteria else 0
            employee_min = st.number_input("Min Employees", 0, 100000, default_emp_min or 0)

        with col2:
            default_emp_max = criteria.organizational.employee_count_max if criteria else 0
            employee_max = st.number_input("Max Employees", 0, 100000, default_emp_max or 0,
                                           help="0 = no maximum")

        with col3:
            default_currency = criteria.financial.revenue_currency if criteria else "AUD"
            currency = st.selectbox("Currency", ["AUD", "USD", "EUR", "GBP"],
                                    index=["AUD", "USD", "EUR", "GBP"].index(default_currency))

        # Giving capacity (optional)
        default_giving = criteria.financial.giving_capacity_min / 1_000 if criteria and criteria.financial.giving_capacity_min else 0
        giving_capacity = st.number_input("Min Giving Capacity ($K)", 0, 10000, int(default_giving),
                                          help="Estimated philanthropic capacity in thousands")

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
                placeholder="Construction\nProperty\nHospitality",
                help="Leave empty to search all industries"
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
                default=valid_csr_defaults,
                help="Optional: Filter for specific CSR focus areas"
            )

            event_options = ["Office Move", "CSR Launch", "Expansion", "Anniversary", "Award",
                             "New Leadership", "IPO", "Merger", "Partnership"]
            default_events = criteria.behavioral.recent_events if criteria else []
            # Ensure defaults exist in options
            valid_event_defaults = [d for d in default_events if d in event_options]

            recent_events = st.multiselect(
                "Recent Events",
                event_options,
                default=valid_event_defaults,
                help="Optional: Look for companies with these recent events"
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
                default=valid_cert_defaults,
                help="Optional: Filter for specific certifications"
            )

            esg_maturity = st.selectbox(
                "ESG Maturity",
                ["Any", "Basic", "Developing", "Mature", "Leading"],
                index=0,
                help="Optional: Minimum ESG maturity level"
            )

    # Exclusions Section
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
        # Convert revenue to actual values
        revenue_min = min_revenue_millions * 1_000_000 if min_revenue_millions > 0 else None
        revenue_max = max_revenue_millions * 1_000_000 if max_revenue_millions < float('inf') else None

        # Calculate revenue categories
        revenue_categories_list = determine_revenue_categories_from_range(revenue_min, revenue_max)

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
                revenue_min=revenue_min,
                revenue_max=revenue_max,
                revenue_currency=currency,
                revenue_categories=revenue_categories_list,  # Add categories
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