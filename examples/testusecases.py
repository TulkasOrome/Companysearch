# test_use_cases.py

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from agents.search_strategist_agent import (
    EnhancedSearchStrategistAgent,
    SearchCriteria,
    LocationCriteria,
    FinancialCriteria,
    OrganizationalCriteria,
    BehavioralSignals
)


class UseCaseValidator:
    """Validates search results against specific use case requirements"""

    def __init__(self, use_case_name: str, requirements: Dict[str, Any]):
        self.use_case_name = use_case_name
        self.requirements = requirements

    def validate_company(self, company: Any) -> Dict[str, Any]:
        """Validate a single company against requirements"""
        validation_results = {
            'company_name': company.name,
            'passes_all': True,
            'must_have_pass': [],
            'must_have_fail': [],
            'nice_to_have_pass': [],
            'nice_to_have_fail': [],
            'disqualifiers': []
        }

        # Check must-have requirements
        for req_name, req_check in self.requirements.get('must_have', {}).items():
            if req_check(company):
                validation_results['must_have_pass'].append(req_name)
            else:
                validation_results['must_have_fail'].append(req_name)
                validation_results['passes_all'] = False

        # Check nice-to-have requirements
        for req_name, req_check in self.requirements.get('nice_to_have', {}).items():
            if req_check(company):
                validation_results['nice_to_have_pass'].append(req_name)
            else:
                validation_results['nice_to_have_fail'].append(req_name)

        # Check disqualifiers
        for req_name, req_check in self.requirements.get('disqualifiers', {}).items():
            if req_check(company):
                validation_results['disqualifiers'].append(req_name)
                validation_results['passes_all'] = False

        return validation_results


async def test_rmh_sydney_use_case():
    """Test RMH Sydney specific requirements"""
    print("\n" + "=" * 80)
    print("RMH SYDNEY USE CASE TEST")
    print("=" * 80)

    # Define validation requirements
    validator = UseCaseValidator(
        "RMH Sydney",
        {
            'must_have': {
                'Location': lambda c: c.operates_in_country and any(
                    loc in str(getattr(c, 'headquarters', {})).lower() + ' '.join(c.office_locations).lower()
                    for loc in ['sydney', 'western sydney', 'parramatta', 'penrith']
                ),
                'Revenue Range': lambda c: c.estimated_revenue and (
                        '5' in c.estimated_revenue or '10' in c.estimated_revenue or
                        '20' in c.estimated_revenue or '50' in c.estimated_revenue or
                        '100' in c.estimated_revenue
                ),
                'CSR Focus': lambda c: any(
                    area in c.csr_focus_areas
                    for area in ['children', 'community', 'health', 'families']
                ),
                'Giving Capacity': lambda c: c.estimated_revenue is not None  # Proxy for capacity
            },
            'nice_to_have': {
                'Employee Count 50+': lambda c: c.estimated_employees and (
                        '50' in c.estimated_employees or '100' in c.estimated_employees or
                        '200' in c.estimated_employees or '500' in c.estimated_employees
                ),
                'Recent Events': lambda c: len(c.recent_events) > 0,
                'Industry Match': lambda c: any(
                    ind in c.industry_category.lower()
                    for ind in ['construction', 'trades', 'property', 'real estate', 'hospitality']
                )
            },
            'disqualifiers': {
                'Fast Food Competitor': lambda c: any(
                    term in c.name.lower() or c.industry_category.lower()
                    for term in ['burger', 'kfc', 'subway', 'pizza', 'fast food']
                ),
                'Misconduct': lambda c: any(
                    term in str(c.recent_events).lower()
                    for term in ['scandal', 'lawsuit', 'investigation', 'fraud']
                )
            }
        }
    )

    # Create search criteria
    criteria = SearchCriteria(
        location=LocationCriteria(
            countries=["Australia"],
            cities=["Sydney"],
            proximity={"location": "Greater Western Sydney", "radius_km": 50}
        ),
        financial=FinancialCriteria(
            revenue_min=5_000_000,
            revenue_max=100_000_000,
            revenue_currency="AUD",
            giving_capacity_min=20_000
        ),
        organizational=OrganizationalCriteria(
            employee_count_min=50,
            office_types=["Headquarters", "Major Office"]
        ),
        behavioral=BehavioralSignals(
            csr_focus_areas=["children", "community"],
            recent_events=["Office Move", "CSR Launch", "Expansion"]
        ),
        business_types=["B2B", "B2C"],
        industries=[
            {"name": "Construction/Trades", "priority": 1},
            {"name": "Property/Real Estate", "priority": 2},
            {"name": "Hospitality/Food Services", "priority": 3},
            {"name": "Professional Services", "priority": 4}
        ],
        excluded_industries=["Fast Food", "Gambling", "Tobacco"]
    )

    # Run search
    agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")
    result = await agent.generate_enhanced_strategy(criteria, target_count=20)

    print(f"Found {len(result['companies'])} companies")

    # Validate results
    validation_results = []
    for company in result['companies']:
        validation = validator.validate_company(company)
        validation_results.append(validation)

    # Summary
    total_pass = sum(1 for v in validation_results if v['passes_all'])
    print(f"\nValidation Summary:")
    print(f"- Total companies: {len(validation_results)}")
    print(f"- Pass all requirements: {total_pass}")
    if len(validation_results) > 0:
        print(f"- Pass rate: {total_pass / len(validation_results) * 100:.1f}%")
    else:
        print("- Pass rate: N/A (no companies found)")
        print("\n⚠️ No companies found. Debugging information:")
        print(f"- Search criteria location: {criteria.location.countries}, {criteria.location.cities}")
        print(f"- Industries: {[ind['name'] for ind in criteria.industries]}")
        print(f"- Business types: {criteria.business_types}")
        print(
            f"- Revenue range: {criteria.financial.revenue_currency} {criteria.financial.revenue_min / 1_000_000:.0f}M - {criteria.financial.revenue_max / 1_000_000:.0f}M")

    # Show top companies
    print("\nTop Qualifying Companies:")
    for i, (company, validation) in enumerate(zip(result['companies'][:10], validation_results[:10])):
        if validation['passes_all']:
            print(f"\n{i + 1}. {company.name}")
            print(f"   Industry: {company.industry_category}")
            print(f"   Revenue: {company.estimated_revenue}")
            print(f"   CSR Focus: {', '.join(company.csr_focus_areas)}")
            print(f"   ICP Score: {company.icp_score:.1f}")

    return {
        'use_case': 'RMH Sydney',
        'companies_found': len(result['companies']),
        'qualifying_companies': total_pass,
        'validation_results': validation_results
    }


async def test_guide_dogs_victoria_use_case():
    """Test Guide Dogs Victoria requirements"""
    print("\n" + "=" * 80)
    print("GUIDE DOGS VICTORIA USE CASE TEST")
    print("=" * 80)

    # Tier A validator
    tier_a_validator = UseCaseValidator(
        "Guide Dogs Victoria - Tier A",
        {
            'must_have': {
                'Victoria HQ': lambda c: c.operates_in_country and any(
                    loc in str(getattr(c, 'headquarters', {})).lower() + ' '.join(c.office_locations).lower()
                    for loc in ['melbourne', 'geelong', 'ballarat', 'bendigo', 'victoria']
                ),
                'Revenue 500M+': lambda c: c.estimated_revenue and (
                        '500' in c.estimated_revenue or 'billion' in c.estimated_revenue.lower()
                ),
                'Employees 500+': lambda c: c.estimated_employees and any(
                    num in c.estimated_employees
                    for num in ['500', '1000', '5000', '10000']
                ),
                'CSR Maturity': lambda c: (
                        len(c.certifications) > 0 or
                        c.esg_maturity in ['Mature', 'Leading'] or
                        len(c.csr_programs) > 2
                )
            },
            'nice_to_have': {
                'Industry Match': lambda c: any(
                    ind in c.industry_category.lower()
                    for ind in ['health', 'life sciences', 'finance', 'legal', 'tech', 'fmcg', 'property']
                ),
                'Event Hosting': lambda c: 'Major Office' in c.office_locations or c.company_size == 'enterprise',
                'Giving History': lambda c: len(c.giving_history) > 0
            },
            'disqualifiers': {
                'Gambling': lambda c: 'gambling' in c.industry_category.lower() or 'casino' in c.name.lower(),
                'Tobacco': lambda c: 'tobacco' in c.industry_category.lower(),
                'Racing': lambda c: 'racing' in c.industry_category.lower() or 'racing' in c.name.lower(),
                'Animal Welfare Concerns': lambda c: any(
                    term in c.industry_category.lower()
                    for term in ['fur', 'testing', 'meat processing']
                )
            }
        }
    )

    # Tier A criteria
    tier_a_criteria = SearchCriteria(
        location=LocationCriteria(
            countries=["Australia"],
            states=["Victoria"],
            cities=["Melbourne", "Geelong", "Ballarat", "Bendigo"]
        ),
        financial=FinancialCriteria(
            revenue_min=500_000_000,
            revenue_currency="AUD"
        ),
        organizational=OrganizationalCriteria(
            employee_count_min=500,
            employee_count_by_location={"Victoria": 500}
        ),
        behavioral=BehavioralSignals(
            certifications=["B-Corp", "ISO 26000"],
            csr_focus_areas=["disability", "inclusion", "health", "community"],
            esg_maturity="Mature"
        ),
        business_types=["B2B", "B2C"],
        industries=[
            {"name": "Health & Life Sciences", "priority": 1},
            {"name": "Financial Services", "priority": 2},
            {"name": "Legal Services", "priority": 3},
            {"name": "Technology", "priority": 4},
            {"name": "FMCG", "priority": 5}
        ],
        excluded_industries=["Gambling", "Tobacco", "Racing"]
    )

    # Run Tier A search
    agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")
    tier_a_result = await agent.generate_enhanced_strategy(tier_a_criteria, target_count=15)

    print(f"\nTier A Results: Found {len(tier_a_result['companies'])} companies")

    # Validate Tier A
    tier_a_validations = []
    for company in tier_a_result['companies']:
        validation = tier_a_validator.validate_company(company)
        tier_a_validations.append(validation)

    tier_a_pass = sum(1 for v in tier_a_validations if v['passes_all'])

    # Tier B search (lower requirements)
    tier_b_criteria = SearchCriteria(
        location=LocationCriteria(
            countries=["Australia"],
            states=["Victoria"]
        ),
        financial=FinancialCriteria(
            revenue_min=50_000_000,
            revenue_max=500_000_000,
            revenue_currency="AUD"
        ),
        organizational=OrganizationalCriteria(
            employee_count_min=100,
            employee_count_max=500
        ),
        behavioral=BehavioralSignals(
            csr_focus_areas=["community", "health", "education"]
        ),
        business_types=["B2B", "B2C"],
        industries=[
            {"name": "Manufacturing", "priority": 1},
            {"name": "Logistics", "priority": 2},
            {"name": "Universities", "priority": 3}
        ],
        excluded_industries=["Gambling", "Tobacco", "Racing"]
    )

    tier_b_result = await agent.generate_enhanced_strategy(tier_b_criteria, target_count=15)
    print(f"Tier B Results: Found {len(tier_b_result['companies'])} companies")

    # Summary
    print(f"\nGuide Dogs Victoria Summary:")
    print(f"- Tier A: {tier_a_pass}/{len(tier_a_result['companies'])} qualifying")
    print(f"- Tier B: {len(tier_b_result['companies'])} found")

    return {
        'use_case': 'Guide Dogs Victoria',
        'tier_a_found': len(tier_a_result['companies']),
        'tier_a_qualifying': tier_a_pass,
        'tier_b_found': len(tier_b_result['companies'])
    }


async def test_professional_services_use_case():
    """Test searching for professional services (accountants)"""
    print("\n" + "=" * 80)
    print("PROFESSIONAL SERVICES (ACCOUNTANTS) USE CASE TEST")
    print("=" * 80)

    # Test different size categories
    size_categories = [
        {
            'name': 'Big 4 Firms',
            'criteria': SearchCriteria(
                location=LocationCriteria(countries=["Australia", "United States", "United Kingdom"]),
                financial=FinancialCriteria(revenue_min=1_000_000_000),
                organizational=OrganizationalCriteria(employee_count_min=10000),
                behavioral=BehavioralSignals(),
                business_types=["Professional Services"],
                industries=[{"name": "Accounting & Audit", "priority": 1}]
            ),
            'target': 5
        },
        {
            'name': 'Mid-Tier Firms',
            'criteria': SearchCriteria(
                location=LocationCriteria(countries=["Australia"], cities=["Sydney", "Melbourne"]),
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
            ),
            'target': 10
        },
        {
            'name': 'Boutique Firms',
            'criteria': SearchCriteria(
                location=LocationCriteria(
                    countries=["Australia"],
                    cities=["Sydney"],
                    proximity={"location": "Sydney CBD", "radius_km": 20}
                ),
                financial=FinancialCriteria(revenue_max=50_000_000),
                organizational=OrganizationalCriteria(
                    employee_count_min=10,
                    employee_count_max=100
                ),
                behavioral=BehavioralSignals(),
                business_types=["Professional Services"],
                industries=[
                    {"name": "Tax Advisory", "priority": 1},
                    {"name": "Business Advisory", "priority": 2}
                ],
                keywords=["SME", "small business", "tax", "advisory"]
            ),
            'target': 10
        }
    ]

    agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")
    results = {}

    for category in size_categories:
        print(f"\nSearching for {category['name']}...")
        result = await agent.generate_enhanced_strategy(
            category['criteria'],
            target_count=category['target']
        )

        results[category['name']] = {
            'found': len(result['companies']),
            'companies': [c.name for c in result['companies'][:5]]
        }

        print(f"Found {len(result['companies'])} {category['name']}")
        if result['companies']:
            print(f"Sample: {', '.join(results[category['name']]['companies'])}")

    return {
        'use_case': 'Professional Services',
        'categories': results
    }


async def test_real_estate_use_case():
    """Test searching for real estate companies"""
    print("\n" + "=" * 80)
    print("REAL ESTATE USE CASE TEST")
    print("=" * 80)

    # Different real estate segments
    segments = [
        {
            'name': 'Commercial Real Estate',
            'criteria': SearchCriteria(
                location=LocationCriteria(countries=["United States"], cities=["New York", "Los Angeles"]),
                financial=FinancialCriteria(revenue_min=100_000_000),
                organizational=OrganizationalCriteria(employee_count_min=100),
                behavioral=BehavioralSignals(
                    certifications=["LEED", "Energy Star"],
                    csr_focus_areas=["environment", "sustainability"]
                ),
                business_types=["Real Estate", "B2B"],
                industries=[{"name": "Commercial Real Estate", "priority": 1}],
                keywords=["office", "retail", "industrial", "property management"]
            )
        },
        {
            'name': 'Residential Developers',
            'criteria': SearchCriteria(
                location=LocationCriteria(countries=["Australia"], cities=["Sydney", "Melbourne"]),
                financial=FinancialCriteria(
                    revenue_min=50_000_000,
                    revenue_max=500_000_000
                ),
                organizational=OrganizationalCriteria(),
                behavioral=BehavioralSignals(
                    csr_focus_areas=["community", "affordable housing"],
                    recent_events=["New Development", "Project Launch"]
                ),
                business_types=["Real Estate", "B2C"],
                industries=[{"name": "Residential Development", "priority": 1}]
            )
        }
    ]

    agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")
    results = {}

    for segment in segments:
        print(f"\nSearching for {segment['name']}...")
        result = await agent.generate_enhanced_strategy(segment['criteria'], target_count=10)

        results[segment['name']] = {
            'found': len(result['companies']),
            'avg_score': sum(c.icp_score or 0 for c in result['companies']) / max(1, len(result['companies']))
        }

        print(f"Found {results[segment['name']]['found']} companies")
        print(f"Average ICP Score: {results[segment['name']]['avg_score']:.1f}")

    return {
        'use_case': 'Real Estate',
        'segments': results
    }


async def test_custom_prompt_scenarios():
    """Test various custom prompt scenarios"""
    print("\n" + "=" * 80)
    print("CUSTOM PROMPT SCENARIOS TEST")
    print("=" * 80)

    scenarios = [
        {
            'name': 'Award Winners',
            'prompt': "Find companies that have won sustainability or CSR awards in the last 2 years and are known for innovative employee programs",
            'location': ["United States"],
            'min_revenue': 50_000_000
        },
        {
            'name': 'Tech Innovators',
            'prompt': "Identify B2B SaaS companies that use AI/ML, have raised Series B+ funding, and have strong diversity initiatives",
            'location': ["United States", "United Kingdom"],
            'min_revenue': 10_000_000
        },
        {
            'name': 'Local Heroes',
            'prompt': "Find family-owned businesses that have been operating for 20+ years and are actively involved in local community support",
            'location': ["Australia"],
            'min_revenue': 5_000_000
        }
    ]

    agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")
    results = []

    for scenario in scenarios:
        print(f"\nTesting: {scenario['name']}")
        print(f"Prompt: {scenario['prompt']}")

        criteria = SearchCriteria(
            location=LocationCriteria(countries=scenario['location']),
            financial=FinancialCriteria(revenue_min=scenario['min_revenue']),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=[],
            industries=[],
            custom_prompt=scenario['prompt']
        )

        result = await agent.generate_enhanced_strategy(criteria, target_count=5)

        results.append({
            'scenario': scenario['name'],
            'companies_found': len(result['companies']),
            'sample': [c.name for c in result['companies'][:3]]
        })

        print(f"Found {len(result['companies'])} companies")
        if result['companies']:
            print(f"Sample: {', '.join(results[-1]['sample'])}")

    return {
        'use_case': 'Custom Prompts',
        'scenarios': results
    }


async def main():
    """Run all use case tests"""
    print("=" * 80)
    print("REAL-WORLD USE CASE TEST SUITE")
    print(f"Started at: {datetime.now()}")
    print("=" * 80)

    all_results = []

    # Run each use case test
    all_results.append(await test_rmh_sydney_use_case())
    all_results.append(await test_guide_dogs_victoria_use_case())
    all_results.append(await test_professional_services_use_case())
    all_results.append(await test_real_estate_use_case())
    all_results.append(await test_custom_prompt_scenarios())

    # Save results
    with open('use_case_test_results.json', 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL USE CASE TESTS COMPLETED")
    print("Results saved to use_case_test_results.json")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())