#!/usr/bin/env python3
"""
test_streamlit_search.py - Test why Streamlit isn't working
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent if '__file__' in globals() else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents.search_strategist_agent import (
    EnhancedSearchStrategistAgent,
    SearchCriteria,
    LocationCriteria,
    FinancialCriteria,
    OrganizationalCriteria,
    BehavioralSignals
)


async def test_exact_streamlit_criteria():
    """Test with the exact criteria structure Streamlit creates"""
    print("=" * 60)
    print("TEST: Exact Streamlit Criteria Structure")
    print("=" * 60)

    # This is exactly how Streamlit builds the criteria
    criteria = SearchCriteria(
        location=LocationCriteria(
            countries=["Australia"],
            cities=[],
            states=[],
            regions=[],
            proximity=None,
            exclusions=[]
        ),
        financial=FinancialCriteria(
            revenue_min=None,
            revenue_max=None,
            revenue_currency="USD",
            giving_capacity_min=None,
            growth_rate_min=None,
            profitable=None
        ),
        organizational=OrganizationalCriteria(
            employee_count_min=None,
            employee_count_max=None,
            employee_count_by_location=None,
            office_types=[],
            company_stage=None
        ),
        behavioral=BehavioralSignals(
            csr_programs=[],
            csr_focus_areas=[],
            certifications=[],
            recent_events=[],
            technology_stack=[],
            esg_maturity=None
        ),
        business_types=["B2C"],
        industries=[{"name": "Retail", "priority": 1}],
        keywords=[],
        custom_prompt=None,
        excluded_industries=[],
        excluded_companies=[],
        excluded_behaviors=[]
    )

    print("\nCriteria object created successfully")
    print(f"Type: {type(criteria)}")
    print(f"Has location attr: {hasattr(criteria, 'location')}")
    print(f"Location countries: {criteria.location.countries}")
    print(f"Business types: {criteria.business_types}")
    print(f"Industries: {criteria.industries}")

    # Initialize agent
    agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")

    # Test prompt generation
    try:
        prompt = agent._build_enhanced_prompt(criteria, 5)
        print("\n✅ Prompt generated successfully")
        print("Prompt preview:")
        print("-" * 40)
        print(prompt[:500])
        print("-" * 40)
    except Exception as e:
        print(f"\n❌ Prompt generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test search
    try:
        print("\nRunning search...")
        result = await agent.generate_enhanced_strategy(criteria, target_count=5)
        companies = result.get('companies', [])

        print(f"\n✅ Search successful! Found {len(companies)} companies")

        if companies:
            print("\nFirst company structure:")
            first = companies[0]
            print(f"  Type: {type(first)}")
            print(f"  Name: {first.name if hasattr(first, 'name') else 'No name attr'}")

            # Test dict conversion
            if hasattr(first, 'dict'):
                company_dict = first.dict()
                print(f"  Dict conversion: ✅")
                print(f"  Keys: {list(company_dict.keys())[:5]}...")
            else:
                print(f"  Dict conversion: ❌ No dict method")

    except Exception as e:
        print(f"\n❌ Search failed: {e}")
        import traceback
        traceback.print_exc()


async def test_parallel_execution():
    """Test parallel model execution"""
    print("\n" + "=" * 60)
    print("TEST: Parallel Model Execution")
    print("=" * 60)

    # Simple criteria
    criteria = SearchCriteria(
        location=LocationCriteria(countries=["Australia"]),
        financial=FinancialCriteria(),
        organizational=OrganizationalCriteria(),
        behavioral=BehavioralSignals(),
        business_types=["B2C"],
        industries=[{"name": "Retail", "priority": 1}],
        keywords=[],
        custom_prompt=None,
        excluded_industries=[],
        excluded_companies=[],
        excluded_behaviors=[]
    )

    # Models to test
    models = ["gpt-4.1", "gpt-4.1-2", "gpt-4.1-3"]

    print(f"Testing {len(models)} models in parallel")

    # Create tasks
    tasks = []
    for model in models:
        agent = EnhancedSearchStrategistAgent(deployment_name=model)
        task = agent.generate_enhanced_strategy(criteria, target_count=5)
        tasks.append((model, task))

    # Execute in parallel
    results = {}
    for model, task in tasks:
        try:
            print(f"\nExecuting {model}...")
            result = await task
            companies = result.get('companies', [])
            results[model] = len(companies)
            print(f"  ✅ {model}: {len(companies)} companies")
        except Exception as e:
            results[model] = f"Error: {str(e)[:50]}"
            print(f"  ❌ {model}: {str(e)[:50]}")

    # Summary
    print(f"\nParallel execution summary:")
    total = sum(count for count in results.values() if isinstance(count, int))
    print(f"Total companies found: {total}")


def test_criteria_serialization():
    """Test if criteria can be serialized/deserialized properly"""
    print("\n" + "=" * 60)
    print("TEST: Criteria Serialization")
    print("=" * 60)

    # Create criteria
    criteria = SearchCriteria(
        location=LocationCriteria(countries=["Australia"]),
        financial=FinancialCriteria(revenue_min=5_000_000),
        organizational=OrganizationalCriteria(),
        behavioral=BehavioralSignals(),
        business_types=["B2C"],
        industries=[{"name": "Retail", "priority": 1}]
    )

    print("Original criteria created")

    # Test different access methods
    print("\nTesting access methods:")

    # Direct attribute access
    try:
        countries = criteria.location.countries
        print(f"  Direct access: ✅ Countries = {countries}")
    except Exception as e:
        print(f"  Direct access: ❌ {e}")

    # getattr access
    try:
        location = getattr(criteria, 'location')
        countries = getattr(location, 'countries')
        print(f"  getattr access: ✅ Countries = {countries}")
    except Exception as e:
        print(f"  getattr access: ❌ {e}")

    # Check if it's a dataclass
    from dataclasses import is_dataclass, asdict

    if is_dataclass(criteria):
        print(f"  Is dataclass: ✅")
        try:
            criteria_dict = asdict(criteria)
            print(f"  asdict conversion: ✅")
            print(f"  Dict keys: {list(criteria_dict.keys())}")
        except Exception as e:
            print(f"  asdict conversion: ❌ {e}")
    else:
        print(f"  Is dataclass: ❌")


async def main():
    """Run all tests"""
    print("STREAMLIT SEARCH DIAGNOSTIC")
    print("=" * 60)

    # Test 1: Exact Streamlit criteria
    await test_exact_streamlit_criteria()

    # Test 2: Parallel execution
    await test_parallel_execution()

    # Test 3: Criteria serialization
    test_criteria_serialization()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())