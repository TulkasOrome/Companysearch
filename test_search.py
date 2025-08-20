# test_search.py

import asyncio
import json
from datetime import datetime
from search_strategist_agent import (
    EnhancedSearchStrategistAgent,
    SearchCriteria,
    LocationCriteria,
    FinancialCriteria,
    OrganizationalCriteria,
    BehavioralSignals,
    BusinessType
)


async def test_basic_search():
    """Test 1: Most basic search - UK retail companies"""
    print("=" * 60)
    print("TEST 1: Basic UK Retail Search")
    print("=" * 60)

    # Initialize agent
    agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")

    # Create simple criteria
    criteria = SearchCriteria(
        location=LocationCriteria(
            countries=["United Kingdom"],
            states=[],
            cities=[],
            regions=[],
            proximity=None,
            exclusions=[]
        ),
        financial=FinancialCriteria(
            revenue_min=None,
            revenue_max=100_000_000,  # 100M max
            revenue_currency="GBP",
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

    try:
        # Run search
        print("Running search...")
        result = await agent.generate_enhanced_strategy(criteria, target_count=10)

        print(f"\nSearch completed!")
        print(f"Companies found: {len(result['companies'])}")

        # Display results
        if result['companies']:
            print("\nCompanies found:")
            for i, company in enumerate(result['companies'][:5], 1):
                print(f"\n{i}. {company.name}")
                print(f"   Industry: {company.industry_category}")
                print(f"   Revenue: {company.estimated_revenue}")
                print(f"   Confidence: {company.confidence}")
                print(f"   ICP Score: {company.icp_score}")
        else:
            print("\nNo companies found!")

        # Save raw response for debugging
        with open('test_search_result.json', 'w') as f:
            json.dump({
                'criteria': str(criteria),
                'result': {
                    'companies_count': len(result['companies']),
                    'companies': [c.dict() for c in result['companies']],
                    'metadata': result.get('metadata', {})
                }
            }, f, indent=2)

        print("\nRaw result saved to test_search_result.json")

    except Exception as e:
        print(f"\nError during search: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_minimal_search():
    """Test 2: Absolute minimal search - just country"""
    print("\n" + "=" * 60)
    print("TEST 2: Minimal Search - Just UK Companies")
    print("=" * 60)

    agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")

    # Even simpler criteria
    criteria = SearchCriteria(
        location=LocationCriteria(countries=["United Kingdom"]),
        financial=FinancialCriteria(),
        organizational=OrganizationalCriteria(),
        behavioral=BehavioralSignals(),
        business_types=[],  # No restriction
        industries=[],  # No restriction
        keywords=[],
        custom_prompt=None,
        excluded_industries=[],
        excluded_companies=[],
        excluded_behaviors=[]
    )

    try:
        print("Running minimal search...")
        result = await agent.generate_enhanced_strategy(criteria, target_count=5)

        print(f"\nCompanies found: {len(result['companies'])}")

        if result['companies']:
            for company in result['companies']:
                print(f"- {company.name} ({company.confidence})")

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {str(e)}")


async def test_direct_llm():
    """Test 3: Direct LLM call to check if Azure OpenAI is working"""
    print("\n" + "=" * 60)
    print("TEST 3: Direct Azure OpenAI Test")
    print("=" * 60)

    from openai import AzureOpenAI

    try:
        client = AzureOpenAI(
            api_key="CUxPxhxqutsvRVHmGQcmH59oMim6mu55PjHTjSpM6y9UwIxwVZIuJQQJ99BFACL93NaXJ3w3AAABACOG3kI1",
            api_version="2024-02-01",
            azure_endpoint="https://amex-openai-2025.openai.azure.com/"
        )

        print("Testing direct API call...")

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "List 3 UK retail companies in JSON format"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        print("Response received:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"Error with direct API call: {type(e).__name__}: {str(e)}")


async def test_with_debug_prompt():
    """Test 4: Search with debug output in prompt"""
    print("\n" + "=" * 60)
    print("TEST 4: Search with Debug Prompt")
    print("=" * 60)

    agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")

    # Simple criteria with debug prompt
    criteria = SearchCriteria(
        location=LocationCriteria(countries=["United Kingdom"]),
        financial=FinancialCriteria(revenue_max=100_000_000),
        organizational=OrganizationalCriteria(),
        behavioral=BehavioralSignals(),
        business_types=["B2C"],
        industries=[{"name": "Retail", "priority": 1}],
        keywords=[],
        custom_prompt="DEBUG: Please list exactly 5 companies with all required fields",
        excluded_industries=[],
        excluded_companies=[],
        excluded_behaviors=[]
    )

    try:
        # Manually build and print the prompt
        prompt = agent._build_enhanced_prompt(criteria, 5)
        print("Generated prompt:")
        print("-" * 40)
        print(prompt)
        print("-" * 40)

        result = await agent.generate_enhanced_strategy(criteria, target_count=5)
        print(f"\nCompanies found: {len(result['companies'])}")

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {str(e)}")


async def main():
    """Run all tests"""
    print("Starting Company Search Tests")
    print("=" * 60)

    # Test 1: Basic search
    await test_basic_search()

    # Test 2: Minimal search
    await test_minimal_search()

    # Test 3: Direct API test
    await test_direct_llm()

    # Test 4: Debug prompt
    await test_with_debug_prompt()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("Check test_search_result.json for detailed output")


if __name__ == "__main__":
    asyncio.run(main())