# example_integration.py
"""
Example integration showing how to use the enhanced validation system
with the company search functionality
"""

import asyncio
import json
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

from search_strategist_agent import EnhancedSearchStrategistAgent
from validation_agent_v2 import (
    EnhancedValidationAgent,
    ValidationConfig,
    ValidationOrchestrator
)
from validation_strategies import ValidationCriteria
from data_models import (
    SearchCriteria,
    LocationCriteria,
    FinancialCriteria,
    OrganizationalCriteria,
    BehavioralSignals,
    RMH_SYDNEY_CONFIG,
    GUIDE_DOGS_VICTORIA_CONFIG
)


async def example_rmh_sydney_search_and_validate():
    """
    Example: Complete workflow for RMH Sydney prospect search and validation
    """
    print("=" * 80)
    print("RMH SYDNEY PROSPECT SEARCH AND VALIDATION")
    print("=" * 80)

    # Step 1: Initialize search agent
    search_agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")

    # Step 2: Use predefined RMH Sydney criteria
    search_criteria = RMH_SYDNEY_CONFIG.tier_a_criteria

    # Step 3: Execute search
    print("\n🔍 Searching for companies...")
    search_result = await search_agent.generate_enhanced_strategy(
        search_criteria,
        target_count=50
    )

    companies = search_result['companies']
    print(f"✅ Found {len(companies)} companies")

    # Display ICP tier breakdown
    tier_counts = {}
    for company in companies:
        tier = company.icp_tier or "Untiered"
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print("\nICP Tier Distribution:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  Tier {tier}: {count} companies")

    # Step 4: Initialize validation agent
    validation_config = ValidationConfig(
        serper_api_key="99c44b79892f5f7499accf2d7c26d93313880937",  # Your Serper API key
        max_parallel_queries=5,
        max_cost_per_company=0.005,  # $0.005 per company
        max_total_cost=1.0  # $1.00 total limit
    )

    # Step 5: Validate companies
    print("\n✅ Starting validation...")

    async with ValidationOrchestrator(validation_config) as orchestrator:
        # Group companies by tier
        companies_by_tier = {}
        for company in companies:
            tier = company.icp_tier or "C"
            if tier not in companies_by_tier:
                companies_by_tier[tier] = []
            companies_by_tier[tier].append(company.dict())

        # Convert validation criteria
        val_criteria = ValidationCriteria(
            must_be_in_locations=["Sydney", "Greater Western Sydney"],
            must_be_within_radius=("Sydney CBD", 50),
            min_revenue=5_000_000,
            max_revenue=100_000_000,
            revenue_currency="AUD",
            min_employees=50,
            required_csr_areas=["children", "community"],
            excluded_keywords=["McDonald's competitor", "fast food"]
        )

        # Validate by tier
        validation_results = await orchestrator.validate_tiered_companies(
            companies_by_tier,
            val_criteria,
            "Sydney",
            "Australia"
        )

        # Step 6: Analyze results
        print("\n📊 Validation Results:")

        for tier, validations in validation_results.items():
            verified = sum(1 for v in validations if v.validation_status == "verified")
            partial = sum(1 for v in validations if v.validation_status == "partial")
            rejected = sum(1 for v in validations if v.validation_status == "rejected")

            print(f"\nTier {tier}:")
            print(f"  Verified: {verified}")
            print(f"  Partial: {partial}")
            print(f"  Rejected: {rejected}")

            # Show top verified companies
            if verified > 0:
                print(f"  Top verified companies:")
                for v in validations:
                    if v.validation_status == "verified":
                        print(f"    - {v.company_name} (score: {v.overall_score:.1f})")
                        if len([v for v in validations if v.validation_status == "verified"]) >= 3:
                            break

        # Step 7: Export results
        summary = orchestrator.agent.export_validation_report(
            [v for tier_vals in validation_results.values() for v in tier_vals],
            format="summary"
        )

        print("\n📈 Overall Summary:")
        print(f"  Total companies validated: {summary['total_companies']}")
        print(f"  Verified: {summary['verified']}")
        print(f"  Average score: {summary['average_score']:.1f}")
        print(f"  Total Serper queries: {summary['total_queries']}")
        print(f"  Estimated cost: ${summary['total_queries'] * 0.001:.2f}")

        # Save results
        with open("rmh_sydney_results.json", "w") as f:
            json.dump({
                "search_criteria": str(search_criteria),
                "companies_found": len(companies),
                "validation_summary": summary,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        print("\n✅ Results saved to rmh_sydney_results.json")


async def example_progressive_validation():
    """
    Example: Progressive validation to find verified companies efficiently
    """
    print("=" * 80)
    print("PROGRESSIVE VALIDATION EXAMPLE")
    print("=" * 80)

    # Search for companies with relaxed criteria
    search_agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")

    # Broader search criteria
    broad_criteria = SearchCriteria(
        location=LocationCriteria(countries=["United States"]),
        financial=FinancialCriteria(revenue_min=10_000_000),
        organizational=OrganizationalCriteria(),
        behavioral=BehavioralSignals(),
        business_types=["B2B", "B2C"],
        industries=[{"name": "Technology", "priority": 1}]
    )

    print("\n🔍 Broad search for tech companies...")
    result = await search_agent.generate_enhanced_strategy(broad_criteria, target_count=100)
    companies = result['companies']
    print(f"✅ Found {len(companies)} companies")

    # Progressive validation
    validation_config = ValidationConfig(
        serper_api_key="99c44b79892f5f7499accf2d7c26d93313880937",
        max_parallel_queries=3
    )

    val_criteria = ValidationCriteria(
        must_be_in_locations=["United States"],
        min_revenue=10_000_000
    )

    async with ValidationOrchestrator(validation_config) as orchestrator:
        print("\n🎯 Progressively validating until 10 verified...")

        validations = await orchestrator.progressive_validation(
            [c.dict() for c in companies],
            val_criteria,
            "United States",
            "United States",
            target_verified=10
        )

        verified = [v for v in validations if v.validation_status == "verified"]
        print(f"\n✅ Found {len(verified)} verified companies out of {len(validations)} validated")

        for v in verified:
            print(f"  - {v.company_name} (score: {v.overall_score:.1f})")


async def example_accountant_search():
    """
    Example: Search and validate accounting firms
    """
    print("=" * 80)
    print("PROFESSIONAL SERVICES (ACCOUNTANTS) SEARCH")
    print("=" * 80)

    search_agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")

    # Accounting firm criteria
    criteria = SearchCriteria(
        location=LocationCriteria(
            countries=["Australia"],
            cities=["Sydney", "Melbourne"]
        ),
        financial=FinancialCriteria(
            revenue_min=50_000_000,
            revenue_max=500_000_000
        ),
        organizational=OrganizationalCriteria(
            employee_count_min=100,
            employee_count_max=1000
        ),
        behavioral=BehavioralSignals(
            certifications=["CPA", "CA"],
            technology_stack=["Xero", "MYOB", "QuickBooks"]
        ),
        business_types=["Professional Services"],
        industries=[{"name": "Accounting", "priority": 1}]
    )

    print("\n🔍 Searching for mid-tier accounting firms...")
    result = await search_agent.generate_enhanced_strategy(criteria, target_count=20)

    companies = result['companies']
    print(f"✅ Found {len(companies)} accounting firms")

    # Quick validation for professional services
    validation_config = ValidationConfig(
        serper_api_key="99c44b79892f5f7499accf2d7c26d93313880937",
        max_parallel_queries=5
    )

    val_criteria = ValidationCriteria(
        must_be_in_locations=["Sydney", "Melbourne"],
        required_certifications=["CPA", "CA"]
    )

    async with EnhancedValidationAgent(validation_config) as agent:
        print("\n✅ Validating accounting firms...")

        validations = []
        for company in companies[:10]:  # Validate top 10
            validation = await agent.validate_company(
                company.dict(),
                val_criteria,
                "Australia",
                "Australia"
            )
            validations.append(validation)

            if validation.location.verified:
                print(f"  ✓ {validation.company_name} - Location verified")
                if validation.location.headquarters:
                    print(f"    Address: {validation.location.headquarters.get('address')}")


async def example_enrichment_workflow():
    """
    Example: Search, validate, and enrich company data
    """
    print("=" * 80)
    print("COMPANY ENRICHMENT WORKFLOW")
    print("=" * 80)

    # Simple search
    search_agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")

    criteria = SearchCriteria(
        location=LocationCriteria(countries=["United Kingdom"], cities=["London"]),
        financial=FinancialCriteria(revenue_min=50_000_000),
        organizational=OrganizationalCriteria(),
        behavioral=BehavioralSignals(csr_focus_areas=["environment"]),
        business_types=["B2C"],
        industries=[{"name": "Retail", "priority": 1}]
    )

    print("\n🔍 Searching for UK retail companies...")
    result = await search_agent.generate_enhanced_strategy(criteria, target_count=10)
    companies = result['companies']

    # Validate and enrich
    validation_config = ValidationConfig(
        serper_api_key="99c44b79892f5f7499accf2d7c26d93313880937"
    )

    val_criteria = ValidationCriteria(
        must_be_in_locations=["London"],
        required_csr_areas=["environment"]
    )

    async with ValidationOrchestrator(validation_config) as orchestrator:
        print("\n🔄 Enriching company data...")

        enriched = await orchestrator.validate_with_enrichment(
            [c.dict() for c in companies[:5]],
            val_criteria,
            "London",
            "United Kingdom"
        )

        # Display enriched data
        for company in enriched:
            print(f"\n📊 {company['name']}:")
            print(f"  Original confidence: {company['confidence']}")
            print(f"  Validation status: {company['validation']['status']}")
            print(f"  Validation score: {company['validation']['score']:.1f}")

            if 'verified_headquarters' in company:
                print(f"  Verified HQ: {company['verified_headquarters'].get('address')}")
            if 'verified_revenue' in company:
                print(f"  Verified revenue: {company['verified_revenue']}")
            if 'verified_csr_areas' in company:
                print(f"  CSR areas: {', '.join(company['verified_csr_areas'])}")
            if 'risk_signals' in company:
                print(f"  ⚠️ Risk signals: {', '.join(company['risk_signals'])}")

        # Save enriched data
        df = pd.DataFrame(enriched)
        df.to_csv("enriched_companies.csv", index=False)
        print("\n✅ Enriched data saved to enriched_companies.csv")


async def main():
    """
    Run all examples
    """
    print("ENHANCED COMPANY SEARCH AND VALIDATION SYSTEM")
    print("=" * 80)
    print("\nSelect an example to run:")
    print("1. RMH Sydney - Full search and validation workflow")
    print("2. Progressive validation - Find verified companies efficiently")
    print("3. Accountant search - Professional services example")
    print("4. Enrichment workflow - Search, validate, and enrich data")
    print("5. Run all examples")

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice == "1":
        await example_rmh_sydney_search_and_validate()
    elif choice == "2":
        await example_progressive_validation()
    elif choice == "3":
        await example_accountant_search()
    elif choice == "4":
        await example_enrichment_workflow()
    elif choice == "5":
        await example_rmh_sydney_search_and_validate()
        print("\n" + "=" * 80 + "\n")
        await example_progressive_validation()
        print("\n" + "=" * 80 + "\n")
        await example_accountant_search()
        print("\n" + "=" * 80 + "\n")
        await example_enrichment_workflow()
    else:
        print("Invalid choice. Please run again and select 1-5.")


if __name__ == "__main__":
    asyncio.run(main())