# streamlit_app.py

import streamlit as st
import asyncio
import pandas as pd
import json
from datetime import datetime
import time
from search_strategist_agent import (
    SearchStrategistAgent,
    ParallelSearchCoordinator,
    BusinessType,
    TARGET_COUNTRIES,
    INDUSTRY_TAXONOMIES,
    DistributionMode,
    CompanyEntry
)
from validation_agent import (
    ValidationAgent,
    ValidationMode,
    ConfidenceFilter,
    ValidationResult
)

# Page config
st.set_page_config(
    page_title="Company Search & Validation Platform",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = []
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'companies_for_validation' not in st.session_state:
    st.session_state.companies_for_validation = []

# Title and description
st.title("🔍 AI-Powered Company Search & Validation Platform")
st.markdown("Search for companies using GPT-4.1, then validate with Serper API")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Search Companies", "✅ Validate Companies", "📊 Results & Export"])

# Tab 1: Search Companies
with tab1:
    col1_search, col2_search = st.columns([1, 5])  # Further reduced from [1, 4] to [1, 5]

    with col1_search:
        # Use smaller headers and compact layout
        st.markdown("#### Search Config")  # Changed from st.header

        # Deployment selection - now multi-select
        st.markdown("##### GPT-4.1")  # Changed from st.subheader
        available_deployments = ["gpt-4.1", "gpt-4.1-2", "gpt-4.1-3", "gpt-4.1-4", "gpt-4.1-5"]
        selected_deployments = st.multiselect(
            "Deployments",  # Shortened label
            available_deployments,
            default=["gpt-4.1"],
            help="Select multiple deployments to run searches in parallel"
        )

        # Show deployment info
        if len(selected_deployments) > 1:
            st.info(f"Using {len(selected_deployments)} deployments for {len(selected_deployments)}x speed")

        # Business type selection
        business_type = st.selectbox(
            "Business Type",
            [bt for bt in BusinessType],
            format_func=lambda x: x.value
        )

        # Include financial data option
        include_financials = st.checkbox(
            "Include Revenue & Employee Estimates",
            value=True,
            help="Search for estimated revenue and employee counts"
        )

        # Company size filter (for search)
        st.markdown("##### Size Filter")  # Changed from st.subheader
        size_preference = st.multiselect(
            "Target Sizes",  # Shortened label
            ["Small", "Medium", "Enterprise", "All Sizes"],
            default=["All Sizes"],
            help="Filter search results by company size (requires financial data)"
        )

        # If specific sizes selected, show definitions
        if size_preference and "All Sizes" not in size_preference:
            with st.expander("Size Definitions"):
                st.markdown("""
                - **Small**: 1-50 employees or <$10M revenue
                - **Medium**: 51-500 employees or $10M-$100M revenue
                - **Enterprise**: 500+ employees or $100M+ revenue
                """)

        # Region filter (optional)
        regions = sorted(set(info["region"] for info in TARGET_COUNTRIES.values()))
        selected_regions = st.multiselect(
            "Filter by Region (optional)",
            regions,
            default=[]
        )

        # Country selection
        if selected_regions:
            available_countries = [
                country for country, info in TARGET_COUNTRIES.items()
                if info["region"] in selected_regions
            ]
        else:
            available_countries = list(TARGET_COUNTRIES.keys())

        selected_countries = st.multiselect(
            "Select Countries",
            available_countries,
            default=["United States"] if "United States" in available_countries else [available_countries[0]]
        )

        # Industry selection
        industries = INDUSTRY_TAXONOMIES.get(business_type, {})
        selected_industry = st.selectbox(
            "Industry Category",
            list(industries.keys()) if industries else ["General"]
        )

        # Sub-industry selection
        if selected_industry and selected_industry in industries:
            sub_industries = industries[selected_industry]
            selected_sub_industry = st.selectbox(
                "Sub-Industry (optional)",
                ["All"] + sub_industries
            )
            if selected_sub_industry == "All":
                selected_sub_industry = None
        else:
            selected_sub_industry = None

        # Distribution mode
        st.markdown("##### Distribution")  # Changed from st.subheader
        distribution_mode = st.radio(
            "Mode",  # Shortened label
            ["Equal per Country", "Population-Based", "Custom"],
            help="How to distribute the total company target across countries"
        )

        # Total target
        if distribution_mode != "Custom":
            total_target = st.number_input(
                "Total Companies Target",
                min_value=10,
                max_value=30000,
                value=500,  # Reduced from 1000
                step=100,
                help="Total companies to find across all selected countries"
            )

        # Per-country targets (either calculated or custom)
        if distribution_mode == "Custom":
            st.markdown("#### Set targets per country:")
            custom_targets = {}
            total_custom = 0
            for country in selected_countries:
                count = st.number_input(
                    f"{country}",
                    min_value=10,
                    max_value=5000,
                    value=100,
                    step=10,
                    key=f"target_{country}"
                )
                custom_targets[country] = count
                total_custom += count
            st.info(f"Total: {total_custom:,} companies")
            total_target = total_custom
        else:
            # Calculate distribution
            if selected_countries:
                if distribution_mode == "Equal per Country":
                    per_country = total_target // len(selected_countries)
                    st.info(f"~{per_country} companies per country")
                else:  # Population-based
                    st.info("Companies distributed proportionally to population")

        # Cost estimate
        estimated_calls = len(selected_countries)
        if len(selected_deployments) > 1:
            st.success(f"Parallel execution with {len(selected_deployments)} deployments")
        estimated_cost = estimated_calls * 0.02
        st.info(f"Estimated API calls: {estimated_calls}")
        st.info(f"Estimated cost: ${estimated_cost:.2f}")

    with col2_search:
        st.header("Search Execution")

        # Display selected parameters
        params_data = {
            "Parameter": ["Business Type", "Countries", "Industry", "Sub-Industry", "Total Target", "Distribution",
                          "Deployments"],
            "Value": [
                str(business_type.value),
                f"{len(selected_countries)} selected",
                str(selected_industry),
                str(selected_sub_industry or "All"),
                f"{total_target:,}",
                distribution_mode,
                f"{len(selected_deployments)} selected"
            ]
        }

        if include_financials:
            params_data["Parameter"].append("Financial Data")
            params_data["Value"].append("Included")

        params_df = pd.DataFrame(params_data)
        st.dataframe(params_df, hide_index=True)

        # Search button
        if st.button("🚀 Start Search", type="primary", disabled=not selected_countries):
            with st.spinner("Searching for companies..."):
                # Create progress container
                progress_container = st.container()

                # Determine distribution mode
                if distribution_mode == "Equal per Country":
                    dist_mode = DistributionMode.EQUAL
                elif distribution_mode == "Population-Based":
                    dist_mode = DistributionMode.POPULATION
                else:
                    dist_mode = DistributionMode.CUSTOM

                # Store results
                all_results = []

                # Progress tracking
                progress_bar = progress_container.progress(0)
                status_text = progress_container.empty()
                detail_text = progress_container.empty()
                eta_text = progress_container.empty()

                # Time tracking for ETA
                start_time = time.time()


                def update_eta(countries_done, total_countries):
                    """Update ETA display"""
                    elapsed = time.time() - start_time
                    if countries_done > 0:
                        avg_time_per_country = elapsed / countries_done
                        remaining_countries = total_countries - countries_done
                        eta_seconds = avg_time_per_country * remaining_countries

                        # Format time nicely
                        if eta_seconds < 60:
                            eta_str = f"{int(eta_seconds)} seconds"
                        elif eta_seconds < 3600:
                            minutes = int(eta_seconds / 60)
                            seconds = int(eta_seconds % 60)
                            eta_str = f"{minutes}m {seconds}s"
                        else:
                            hours = int(eta_seconds / 3600)
                            minutes = int((eta_seconds % 3600) / 60)
                            eta_str = f"{hours}h {minutes}m"

                        # Speed metrics
                        speed = countries_done / (elapsed / 60)  # countries per minute

                        eta_text.markdown(f"""
                        **Progress Metrics:**
                        - ⏱️ Elapsed: {int(elapsed)}s
                        - 🏁 ETA: {eta_str}
                        - 🚀 Speed: {speed:.1f} countries/min
                        - 📊 Progress: {countries_done}/{total_countries} countries
                        """)


                # Use parallel coordinator if multiple deployments selected
                if len(selected_deployments) > 1:
                    # Parallel execution
                    coordinator = ParallelSearchCoordinator(selected_deployments)

                    # Progress callback using list to track count
                    countries_completed = []


                    def progress_callback(country, deployment, companies_found):
                        countries_completed.append(country)
                        progress = len(countries_completed) / len(selected_countries)
                        progress_bar.progress(progress)
                        detail_text.text(f"Completed {country} on {deployment}: {companies_found} companies")
                        update_eta(len(countries_completed), len(selected_countries))


                    # Run parallel search
                    async def run_parallel_search():
                        return await coordinator.search_parallel(
                            countries=selected_countries,
                            business_type=business_type,
                            industry=selected_industry,
                            sub_industry=selected_sub_industry,
                            total_target=total_target,
                            distribution_mode=dist_mode,
                            custom_distribution=custom_targets if distribution_mode == "Custom" else None,
                            include_financials=include_financials,
                            size_preferences=size_preference if "All Sizes" not in size_preference else None,
                            progress_callback=progress_callback
                        )


                    status_text.text(f"Running parallel search with {len(selected_deployments)} deployments...")
                    update_eta(0, len(selected_countries))
                    search_results = asyncio.run(run_parallel_search())

                    # Process results
                    for result_item in search_results["results"]:
                        if result_item["strategy"]:
                            strategy = result_item["strategy"]
                            result = {
                                "country": result_item["country"],
                                "business_type": business_type.value,
                                "industry": selected_industry,
                                "sub_industry": selected_sub_industry,
                                "companies": [company.dict() for company in strategy.known_companies],
                                "search_queries": strategy.search_queries,
                                "deployment": result_item["deployment"],
                                "timestamp": datetime.now().isoformat()
                            }
                            all_results.append(result)

                    # Update cost from coordinator
                    st.session_state.total_cost += search_results["summary"]["total_cost"]

                else:
                    # Single deployment execution (original logic)
                    agent = SearchStrategistAgent(deployment_name=selected_deployments[0])

                    # Calculate targets per country
                    if distribution_mode == "Custom":
                        country_targets = custom_targets
                    else:
                        per_country = total_target // len(selected_countries)
                        country_targets = {country: per_country for country in selected_countries}
                        # Distribute remainder
                        remainder = total_target % len(selected_countries)
                        for i, country in enumerate(selected_countries[:remainder]):
                            country_targets[country] += 1

                    # Initial ETA
                    update_eta(0, len(selected_countries))

                    # Search each country
                    for idx, country in enumerate(selected_countries):
                        target_count = country_targets[country]
                        status_text.text(f"Searching {country} (target: {target_count})...")


                        # Run async search
                        async def run_search():
                            return await agent.generate_strategy(
                                country=country,
                                business_type=business_type,
                                industry=selected_industry,
                                sub_industry=selected_sub_industry,
                                target_count=target_count,
                                include_financials=include_financials,
                                size_preferences=size_preference if "All Sizes" not in size_preference else None
                            )


                        # Execute search
                        strategy = asyncio.run(run_search())

                        # Store results
                        result = {
                            "country": country,
                            "business_type": business_type.value,
                            "industry": selected_industry,
                            "sub_industry": selected_sub_industry,
                            "companies": [company.dict() for company in strategy.known_companies],
                            "search_queries": strategy.search_queries,
                            "deployment": selected_deployments[0],
                            "timestamp": datetime.now().isoformat()
                        }
                        all_results.append(result)

                        # Update progress
                        countries_processed = idx + 1
                        progress = countries_processed / len(selected_countries)
                        progress_bar.progress(progress)
                        detail_text.text(f"Found {len(strategy.known_companies)} companies")
                        update_eta(countries_processed, len(selected_countries))

                    # Update cost
                    st.session_state.total_cost += len(selected_countries) * 0.02

                # Clear ETA text
                eta_text.empty()

                # Store in session state
                st.session_state.search_results.extend(all_results)
                st.session_state.search_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "countries": selected_countries,
                    "business_type": business_type.value,
                    "results_count": sum(len(r['companies']) for r in all_results),
                    "deployments": selected_deployments,
                    "total_target": total_target,
                    "duration": time.time() - start_time
                })

                # Prepare companies for validation
                companies_for_validation = []
                for result in all_results:
                    for company_data in result["companies"]:
                        companies_for_validation.append({
                            "company": CompanyEntry(**company_data),
                            "country": result["country"],
                            "business_type": result["business_type"],
                            "industry": result["industry"]
                        })

                st.session_state.companies_for_validation = companies_for_validation

                status_text.text("Search complete!")
                detail_text.empty()

                # Show completion summary
                total_time = time.time() - start_time
                time_str = f"{int(total_time / 60)}m {int(total_time % 60)}s" if total_time > 60 else f"{int(total_time)}s"
                st.success(
                    f"✅ Found {sum(len(r['companies']) for r in all_results)} companies across {len(selected_countries)} countries in {time_str}")
                st.info(
                    f"💡 {len(companies_for_validation)} companies ready for validation. Go to the 'Validate Companies' tab to continue.")

        # Results section in Tab 1
        if st.session_state.search_results:
            st.header("Search Results")

            # Filter options
            filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(5)

            with filter_col1:
                filter_country = st.selectbox(
                    "Filter by Country",
                    ["All"] + list(set(r['country'] for r in st.session_state.search_results)),
                    key="tab1_country"
                )

            with filter_col2:
                confidence_levels = ["All", "absolute", "high", "medium", "low"]
                filter_confidence = st.selectbox("Filter by Confidence", confidence_levels, key="tab1_confidence")

            with filter_col3:
                filter_operates = st.selectbox("Operating Status", ["All", "Operating", "Not Operating"],
                                               key="tab1_operating")

            with filter_col4:
                filter_has_financials = st.selectbox("Financial Data", ["All", "Has Data", "No Data"],
                                                     key="tab1_financial")

            with filter_col5:
                size_options = ["All", "Small", "Medium", "Enterprise", "Unknown"]
                filter_size = st.selectbox("Company Size", size_options, key="tab1_size")

            # Compile all companies
            all_companies = []
            for result in st.session_state.search_results:
                if filter_country != "All" and result['country'] != filter_country:
                    continue

                for company in result['companies']:
                    if filter_confidence != "All" and company.get('confidence') != filter_confidence:
                        continue
                    if filter_operates == "Operating" and not company.get('operates_in_country', True):
                        continue
                    if filter_operates == "Not Operating" and company.get('operates_in_country', True):
                        continue

                    has_financial = bool(company.get('estimated_revenue') or company.get('estimated_employees'))
                    if filter_has_financials == "Has Data" and not has_financial:
                        continue
                    if filter_has_financials == "No Data" and has_financial:
                        continue

                    company_size = company.get('company_size', 'unknown')
                    if filter_size != "All" and company_size.lower() != filter_size.lower():
                        continue

                    all_companies.append({
                        "Company Name": company['name'],
                        "Country": result['country'],
                        "Business Type": result['business_type'],
                        "Industry": result['industry'],
                        "Size": company_size.capitalize(),
                        "Confidence": company['confidence'],
                        "Operating": "✓" if company.get('operates_in_country', True) else "✗",
                        "Revenue": company.get('estimated_revenue', 'Unknown'),
                        "Employees": company.get('estimated_employees', 'Unknown'),
                        "Reasoning": company.get('reasoning', ''),
                        "Deployment": result.get('deployment', 'Unknown')
                    })

            # Display results
            if all_companies:
                st.subheader(f"Showing {len(all_companies)} companies")

                # Summary statistics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    with_revenue = sum(1 for c in all_companies if c['Revenue'] != 'Unknown')
                    st.metric("With Revenue Data", f"{with_revenue} ({with_revenue / len(all_companies) * 100:.1f}%)")
                with col2:
                    with_employees = sum(1 for c in all_companies if c['Employees'] != 'Unknown')
                    st.metric("With Employee Data",
                              f"{with_employees} ({with_employees / len(all_companies) * 100:.1f}%)")
                with col3:
                    operating = sum(1 for c in all_companies if c['Operating'] == "✓")
                    st.metric("Operating", f"{operating} ({operating / len(all_companies) * 100:.1f}%)")
                with col4:
                    # Size breakdown
                    size_counts = pd.DataFrame(all_companies)['Size'].value_counts()
                    if 'Enterprise' in size_counts:
                        st.metric("Enterprise",
                                  f"{size_counts['Enterprise']} ({size_counts['Enterprise'] / len(all_companies) * 100:.1f}%)")
                    else:
                        st.metric("Enterprise", "0 (0%)")
                with col5:
                    unique_deployments = len(set(c['Deployment'] for c in all_companies))
                    st.metric("Deployments Used", unique_deployments)

                # Convert to DataFrame for display
                df = pd.DataFrame(all_companies)

                # Display with styling
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Size": st.column_config.TextColumn("Size", width="small"),
                        "Operating": st.column_config.TextColumn("Operating", width="small"),
                        "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                        "Revenue": st.column_config.TextColumn("Revenue", width="medium"),
                        "Employees": st.column_config.TextColumn("Employees", width="medium"),
                        "Reasoning": st.column_config.TextColumn("Reasoning", width="large")
                    }
                )

                # Export options
                st.subheader("Export Options")
                col1, col2, col3 = st.columns(3)

                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"company_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with col2:
                    json_data = json.dumps(all_companies, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"company_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                with col3:
                    # Summary statistics
                    st.metric("Total Companies", len(all_companies))

                    # Show size breakdown
                    if 'Size' in df.columns:
                        size_counts = df['Size'].value_counts()
                        st.text("By Size:")
                        for size, count in size_counts.items():
                            st.text(f"  {size}: {count}")

                    # Show confidence breakdown
                    confidence_counts = df['Confidence'].value_counts()
                    st.text("By Confidence:")
                    for conf, count in confidence_counts.items():
                        st.text(f"  {conf}: {count}")
            else:
                st.info("No companies match the selected filters")

            # Progress tracking section
            with st.expander("Search History & Progress"):
                if st.session_state.search_history:
                    history_df = pd.DataFrame(st.session_state.search_history)
                    st.dataframe(history_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No searches performed yet")

# Tab 2: Validate Companies
with tab2:
    st.header("Company Validation with Serper API")

    if not st.session_state.companies_for_validation:
        st.warning("No companies to validate. Please run a search first in the 'Search Companies' tab.")
    else:
        col1_validate, col2_validate = st.columns([1, 4])  # Changed from [1, 3] to [1, 4]

        with col1_validate:
            st.subheader("Validation Settings")

            # Serper API key
            serper_api_key = st.text_input(
                "Serper API Key",
                value="99c44b79892f5f7499accf2d7c26d93313880937",
                type="password",
                help="Enter your Serper API key for validation"
            )

            # Validation mode
            st.subheader("Validation Mode")
            validation_mode = st.radio(
                "Select Validation Type",
                [
                    "Skip Validation",
                    "Places Only (Verify Existence)",
                    "Web Search Only (Find Company Info)",
                    "Full Validation (Places + Web)"
                ],
                index=1,
                help="Choose how to validate companies"
            )

            # Map to enum
            mode_map = {
                "Skip Validation": ValidationMode.SKIP,
                "Places Only (Verify Existence)": ValidationMode.PLACES_ONLY,
                "Web Search Only (Find Company Info)": ValidationMode.WEB_ONLY,
                "Full Validation (Places + Web)": ValidationMode.FULL
            }

            # Confidence filter
            st.subheader("Which Companies to Validate")
            confidence_filter = st.radio(
                "Filter by Confidence",
                [
                    "All Companies",
                    "Low Confidence Only",
                    "Medium and Low",
                    "High, Medium and Low (skip Absolute)"
                ],
                help="Choose which confidence levels to validate"
            )

            # Map to enum
            filter_map = {
                "All Companies": ConfidenceFilter.ALL,
                "Low Confidence Only": ConfidenceFilter.LOW_ONLY,
                "Medium and Low": ConfidenceFilter.MEDIUM_AND_LOW,
                "High, Medium and Low (skip Absolute)": ConfidenceFilter.HIGH_AND_BELOW
            }

            # Parallel limit
            parallel_limit = st.slider(
                "Parallel Requests",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of concurrent Serper API requests"
            )

            # Show statistics
            st.subheader("Companies to Validate")
            total_companies = len(st.session_state.companies_for_validation)
            st.metric("Total Companies", total_companies)

            # Count by confidence
            confidence_counts = {}
            for item in st.session_state.companies_for_validation:
                conf = item["company"].confidence
                confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

            for conf, count in sorted(confidence_counts.items()):
                st.text(f"{conf}: {count}")

            # Cost estimate for Serper
            if validation_mode != "Skip Validation":
                queries_per_company = 1 if "Only" in validation_mode else 2
                estimated_queries = total_companies * queries_per_company
                estimated_serper_cost = estimated_queries * 0.001  # $0.001 per query
                st.info(f"Estimated Serper queries: {estimated_queries}")
                st.info(f"Estimated Serper cost: ${estimated_serper_cost:.2f}")

        with col2_validate:
            st.subheader("Validation Execution")

            # Validation button
            if st.button("🔍 Start Validation", type="primary",
                         disabled=not serper_api_key and validation_mode != "Skip Validation"):
                with st.spinner("Validating companies..."):
                    # Initialize validation agent
                    validation_agent = ValidationAgent(serper_api_key)

                    # Group companies by country for efficient processing
                    companies_by_country = {}
                    for item in st.session_state.companies_for_validation:
                        country = item["country"]
                        if country not in companies_by_country:
                            companies_by_country[country] = []
                        companies_by_country[country].append(item)

                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    detail_text = st.empty()

                    all_validation_results = []
                    countries_processed = 0

                    # Process each country
                    for country, company_items in companies_by_country.items():
                        status_text.text(f"Validating {len(company_items)} companies in {country}...")

                        # Extract companies
                        companies = [item["company"] for item in company_items]
                        business_type = company_items[0]["business_type"]


                        # Run validation
                        async def run_validation():
                            return await validation_agent.validate_batch(
                                companies=companies,
                                country=country,
                                business_type=business_type,
                                mode=mode_map[validation_mode],
                                confidence_filter=filter_map[confidence_filter],
                                parallel_limit=parallel_limit
                            )


                        # Execute validation
                        validation_results = asyncio.run(run_validation())
                        all_validation_results.extend(validation_results)

                        # Update progress
                        countries_processed += 1
                        progress = countries_processed / len(companies_by_country)
                        progress_bar.progress(progress)

                        # Show results summary
                        verified = sum(1 for r in validation_results if r.validation_status == "verified")
                        rejected = sum(1 for r in validation_results if r.validation_status == "rejected")
                        detail_text.text(f"{country}: {verified} verified, {rejected} rejected")

                    # Store validation results
                    st.session_state.validation_results = all_validation_results

                    # Update costs
                    total_serper_calls = validation_agent.progress.serper_calls
                    serper_cost = validation_agent.progress.estimated_cost
                    st.session_state.total_cost += serper_cost

                    # Show summary
                    status_text.text("Validation complete!")
                    detail_text.empty()

                    # Display results summary
                    st.success(f"✅ Validated {len(all_validation_results)} companies")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        verified_count = sum(1 for r in all_validation_results if r.validation_status == "verified")
                        st.metric("Verified", verified_count)
                    with col2:
                        rejected_count = sum(1 for r in all_validation_results if r.validation_status == "rejected")
                        st.metric("Rejected", rejected_count)
                    with col3:
                        unverified_count = sum(1 for r in all_validation_results if r.validation_status == "unverified")
                        st.metric("Unverified", unverified_count)
                    with col4:
                        st.metric("Serper Calls", total_serper_calls)

                    st.info(f"💡 Validation results ready. Go to the 'Results & Export' tab to view and export.")

# Tab 3: Results & Export
with tab3:
    st.header("Combined Results")

    # Check if we have results
    has_search_results = bool(st.session_state.search_results)
    has_validation_results = bool(st.session_state.validation_results)

    if not has_search_results and not has_validation_results:
        st.warning("No results yet. Please run a search and/or validation first.")
    else:
        # Filter options
        st.subheader("Filter Results")
        filter_cols = st.columns(6)

        with filter_cols[0]:
            filter_country = st.selectbox(
                "Country",
                ["All"] + list(set(r['country'] for r in st.session_state.search_results))
            )

        with filter_cols[1]:
            confidence_levels = ["All", "absolute", "high", "medium", "low"]
            filter_confidence = st.selectbox("Confidence", confidence_levels)

        with filter_cols[2]:
            filter_operates = st.selectbox("Operating", ["All", "Operating", "Not Operating"])

        with filter_cols[3]:
            filter_has_financials = st.selectbox("Financial Data", ["All", "Has Data", "No Data"])

        with filter_cols[4]:
            size_options = ["All", "Small", "Medium", "Enterprise", "Unknown"]
            filter_size = st.selectbox("Company Size", size_options)

        with filter_cols[5]:
            if has_validation_results:
                filter_validation = st.selectbox("Validation Status", ["All", "Verified", "Rejected", "Unverified"])
            else:
                filter_validation = "All"

        # Compile all companies with validation results
        all_companies = []

        # Create validation lookup
        validation_lookup = {}
        if has_validation_results:
            for val_result in st.session_state.validation_results:
                key = f"{val_result.company_name.lower()}_{val_result.country}"
                validation_lookup[key] = val_result

        # Process search results
        for result in st.session_state.search_results:
            if filter_country != "All" and result['country'] != filter_country:
                continue

            for company in result['companies']:
                if filter_confidence != "All" and company.get('confidence') != filter_confidence:
                    continue
                if filter_operates == "Operating" and not company.get('operates_in_country', True):
                    continue
                if filter_operates == "Not Operating" and company.get('operates_in_country', True):
                    continue

                has_financial = bool(company.get('estimated_revenue') or company.get('estimated_employees'))
                if filter_has_financials == "Has Data" and not has_financial:
                    continue
                if filter_has_financials == "No Data" and has_financial:
                    continue

                company_size = company.get('company_size', 'unknown')
                if filter_size != "All" and company_size.lower() != filter_size.lower():
                    continue

                # Check validation status
                val_key = f"{company['name'].lower()}_{result['country']}"
                val_result = validation_lookup.get(val_key)

                if filter_validation != "All":
                    if not val_result:
                        continue
                    if filter_validation == "Verified" and val_result.validation_status != "verified":
                        continue
                    if filter_validation == "Rejected" and val_result.validation_status != "rejected":
                        continue
                    if filter_validation == "Unverified" and val_result.validation_status != "unverified":
                        continue

                all_companies.append({
                    "Company Name": company['name'],
                    "Country": result['country'],
                    "Business Type": result['business_type'],
                    "Industry": result['industry'],
                    "Size": company_size.capitalize(),
                    "Confidence": company['confidence'],
                    "Operating": "✓" if company.get('operates_in_country', True) else "✗",
                    "Revenue": company.get('estimated_revenue', 'Unknown'),
                    "Employees": company.get('estimated_employees', 'Unknown'),
                    "Validation Status": val_result.validation_status if val_result else "Not Validated",
                    "Validation Method": val_result.validation_mode if val_result else "-",
                    "Final Confidence": val_result.confidence_after_validation if val_result else company['confidence'],
                    "Reasoning": company.get('reasoning', ''),
                    "Deployment": result.get('deployment', 'Unknown')
                })

        # Display results
        if all_companies:
            st.subheader(f"Showing {len(all_companies)} companies")

            # Summary statistics
            summary_cols = st.columns(6)
            with summary_cols[0]:
                st.metric("Total Companies", len(all_companies))
            with summary_cols[1]:
                validated = sum(1 for c in all_companies if c['Validation Status'] != "Not Validated")
                st.metric("Validated", f"{validated} ({validated / len(all_companies) * 100:.1f}%)")
            with summary_cols[2]:
                verified = sum(1 for c in all_companies if c['Validation Status'] == "verified")
                st.metric("Verified", f"{verified} ({verified / len(all_companies) * 100:.1f}%)")
            with summary_cols[3]:
                with_revenue = sum(1 for c in all_companies if c['Revenue'] != 'Unknown')
                st.metric("With Revenue", f"{with_revenue} ({with_revenue / len(all_companies) * 100:.1f}%)")
            with summary_cols[4]:
                enterprise = sum(1 for c in all_companies if c['Size'] == 'Enterprise')
                st.metric("Enterprise", f"{enterprise} ({enterprise / len(all_companies) * 100:.1f}%)")
            with summary_cols[5]:
                st.metric("Total Cost", f"${st.session_state.total_cost:.2f}")

            # Convert to DataFrame for display
            df = pd.DataFrame(all_companies)

            # Display with styling
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Size": st.column_config.TextColumn("Size", width="small"),
                    "Operating": st.column_config.TextColumn("Operating", width="small"),
                    "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "Validation Status": st.column_config.TextColumn("Val Status", width="small"),
                    "Revenue": st.column_config.TextColumn("Revenue", width="medium"),
                    "Employees": st.column_config.TextColumn("Employees", width="medium"),
                    "Reasoning": st.column_config.TextColumn("Reasoning", width="large")
                }
            )

            # Export options
            st.subheader("Export Options")
            export_cols = st.columns(3)

            with export_cols[0]:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full CSV",
                    data=csv,
                    file_name=f"company_search_validated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with export_cols[1]:
                # Verified only CSV
                verified_df = df[df['Validation Status'] == 'verified']
                if not verified_df.empty:
                    verified_csv = verified_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Verified Only CSV",
                        data=verified_csv,
                        file_name=f"verified_companies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.button("📥 Download Verified Only CSV", disabled=True, help="No verified companies")

            with export_cols[2]:
                json_data = json.dumps(all_companies, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"company_search_validated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            # Show breakdowns
            with st.expander("Detailed Statistics"):
                stat_cols = st.columns(3)

                with stat_cols[0]:
                    st.markdown("**By Size:**")
                    size_counts = df['Size'].value_counts()
                    for size, count in size_counts.items():
                        st.text(f"{size}: {count}")

                with stat_cols[1]:
                    st.markdown("**By Confidence:**")
                    conf_counts = df['Final Confidence'].value_counts()
                    for conf, count in conf_counts.items():
                        st.text(f"{conf}: {count}")

                with stat_cols[2]:
                    st.markdown("**By Validation:**")
                    val_counts = df['Validation Status'].value_counts()
                    for status, count in val_counts.items():
                        st.text(f"{status}: {count}")
        else:
            st.info("No companies match the selected filters")