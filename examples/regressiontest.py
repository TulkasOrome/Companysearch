#!/usr/bin/env python3
"""
test_icp_regression.py - Regression testing for ICP profiles
Tests different variations to identify why ICP profiles aren't returning results
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

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

# Try to import ICP manager
try:
    from enhanced_icp_manager import ICPManager

    ICP_AVAILABLE = True
except ImportError:
    ICP_AVAILABLE = False
    print("Warning: ICP Manager not available")


class ICPRegressionTester:
    """Regression tester for ICP profiles"""

    def __init__(self):
        self.agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")
        self.test_results = []
        if ICP_AVAILABLE:
            self.icp_manager = ICPManager()

    async def test_criteria(self, test_name: str, criteria: SearchCriteria, target: int = 5) -> Dict[str, Any]:
        """Test a single criteria configuration"""
        print(f"\nTesting: {test_name}")
        print("-" * 40)

        try:
            # Show what we're searching for
            print(f"Countries: {criteria.location.countries}")
            print(f"Cities: {criteria.location.cities}")
            print(
                f"Revenue: {criteria.financial.revenue_min}-{criteria.financial.revenue_max} {criteria.financial.revenue_currency}")
            print(
                f"Industries: {[ind.get('name') if isinstance(ind, dict) else ind for ind in criteria.industries[:3]]}")

            # Generate and show prompt
            prompt = self.agent._build_enhanced_prompt(criteria, target)
            print(f"\nPrompt preview (first 300 chars):")
            print(prompt[:300])

            # Run search
            result = await self.agent.generate_enhanced_strategy(criteria, target_count=target)
            companies = result.get('companies', [])

            print(f"✅ Found {len(companies)} companies")

            if companies:
                print("Sample companies:")
                for c in companies[:3]:
                    print(f"  - {c.name}: {c.industry_category}")

            return {
                'test_name': test_name,
                'success': True,
                'companies_found': len(companies),
                'sample_companies': [c.name for c in companies[:3]] if companies else [],
                'criteria_summary': {
                    'countries': criteria.location.countries,
                    'cities': criteria.location.cities,
                    'industries': len(criteria.industries),
                    'revenue_range': f"{criteria.financial.revenue_min}-{criteria.financial.revenue_max}"
                }
            }

        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            return {
                'test_name': test_name,
                'success': False,
                'error': str(e),
                'companies_found': 0
            }

    async def test_rmh_progressive(self):
        """Progressively test RMH criteria from most restrictive to least"""
        print("\n" + "=" * 60)
        print("RMH SYDNEY PROGRESSIVE TESTING")
        print("=" * 60)

        test_cases = []

        # Test 1: Full Tier A criteria (most restrictive)
        if ICP_AVAILABLE:
            profile = self.icp_manager.get_profile("rmh_sydney")
            criteria_a = profile.tiers.get("A")
        else:
            criteria_a = SearchCriteria(
                location=LocationCriteria(
                    countries=["Australia"],
                    cities=["Sydney"],
                    regions=["Greater Western Sydney"]
                ),
                financial=FinancialCriteria(
                    revenue_min=5_000_000,
                    revenue_max=100_000_000,
                    revenue_currency="AUD"
                ),
                organizational=OrganizationalCriteria(
                    employee_count_min=50
                ),
                behavioral=BehavioralSignals(
                    csr_focus_areas=["children", "community"]
                ),
                business_types=["B2B", "B2C"],
                industries=[
                    {"name": "Construction", "priority": 1},
                    {"name": "Property", "priority": 2}
                ],
                excluded_industries=["Fast Food", "Gambling"]
            )

        result = await self.test_criteria("RMH Tier A - Full", criteria_a)
        test_cases.append(result)

        # Test 2: Remove CSR requirement
        criteria_no_csr = SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                cities=["Sydney"]
            ),
            financial=FinancialCriteria(
                revenue_min=5_000_000,
                revenue_max=100_000_000,
                revenue_currency="AUD"
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=50
            ),
            behavioral=BehavioralSignals(),  # No CSR
            business_types=["B2B", "B2C"],
            industries=[
                {"name": "Construction", "priority": 1},
                {"name": "Property", "priority": 2}
            ],
            excluded_industries=[]
        )

        result = await self.test_criteria("RMH - No CSR", criteria_no_csr)
        test_cases.append(result)

        # Test 3: Remove financial constraints
        criteria_no_financial = SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                cities=["Sydney"]
            ),
            financial=FinancialCriteria(),  # No financial constraints
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=["B2B", "B2C"],
            industries=[
                {"name": "Construction", "priority": 1},
                {"name": "Property", "priority": 2}
            ],
            excluded_industries=[]
        )

        result = await self.test_criteria("RMH - No Financial", criteria_no_financial)
        test_cases.append(result)

        # Test 4: Just location and industry
        criteria_minimal = SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                cities=["Sydney"]
            ),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=[],  # No business type restriction
            industries=[{"name": "Construction", "priority": 1}],
            excluded_industries=[]
        )

        result = await self.test_criteria("RMH - Minimal (Sydney + Construction)", criteria_minimal)
        test_cases.append(result)

        # Test 5: Just Australia
        criteria_country_only = SearchCriteria(
            location=LocationCriteria(countries=["Australia"]),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=["B2C"],
            industries=[],
            excluded_industries=[]
        )

        result = await self.test_criteria("RMH - Australia Only", criteria_country_only)
        test_cases.append(result)

        return test_cases

    async def test_guide_dogs_progressive(self):
        """Progressively test Guide Dogs criteria"""
        print("\n" + "=" * 60)
        print("GUIDE DOGS VICTORIA PROGRESSIVE TESTING")
        print("=" * 60)

        test_cases = []

        # Test 1: Full Tier A (most restrictive)
        criteria_tier_a = SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                states=["Victoria"],
                cities=["Melbourne"]
            ),
            financial=FinancialCriteria(
                revenue_min=500_000_000,
                revenue_currency="AUD"
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=500
            ),
            behavioral=BehavioralSignals(
                certifications=["B-Corp", "ISO 26000"],
                csr_focus_areas=["disability", "inclusion"]
            ),
            business_types=["B2B", "B2C"],
            industries=[
                {"name": "Health", "priority": 1},
                {"name": "Finance", "priority": 2}
            ],
            excluded_industries=["Gambling", "Tobacco"]
        )

        result = await self.test_criteria("GDV Tier A - Full", criteria_tier_a)
        test_cases.append(result)

        # Test 2: Tier B (less restrictive)
        criteria_tier_b = SearchCriteria(
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
            behavioral=BehavioralSignals(),
            business_types=["B2B", "B2C"],
            industries=[
                {"name": "Manufacturing", "priority": 1},
                {"name": "Logistics", "priority": 2}
            ],
            excluded_industries=["Gambling", "Tobacco"]
        )

        result = await self.test_criteria("GDV Tier B", criteria_tier_b)
        test_cases.append(result)

        # Test 3: Remove certifications requirement
        criteria_no_cert = SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                states=["Victoria"]
            ),
            financial=FinancialCriteria(
                revenue_min=100_000_000,
                revenue_currency="AUD"
            ),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),  # No certifications
            business_types=["B2B", "B2C"],
            industries=[],
            excluded_industries=[]
        )

        result = await self.test_criteria("GDV - No Certifications", criteria_no_cert)
        test_cases.append(result)

        # Test 4: Just Victoria location
        criteria_victoria_only = SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                states=["Victoria"]
            ),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=[],
            industries=[],
            excluded_industries=[]
        )

        result = await self.test_criteria("GDV - Victoria Only", criteria_victoria_only)
        test_cases.append(result)

        return test_cases

    async def test_free_text_variations(self):
        """Test free text search variations"""
        print("\n" + "=" * 60)
        print("FREE TEXT SEARCH TESTING")
        print("=" * 60)

        test_cases = []

        free_text_queries = [
            {
                "name": "Simple Construction",
                "text": "Construction companies in Sydney",
                "custom_prompt": "Find construction companies in Sydney, Australia"
            },
            {
                "name": "Revenue Specific",
                "text": "Australian companies with revenue between $10M and $100M",
                "custom_prompt": "Find Australian companies with annual revenue between 10 million and 100 million dollars"
            },
            {
                "name": "CSR Focus",
                "text": "Companies in Melbourne with strong community programs",
                "custom_prompt": "Find companies in Melbourne, Australia that have strong community involvement and CSR programs"
            },
            {
                "name": "Industry Mix",
                "text": "B2B technology and professional services companies in Australia",
                "custom_prompt": "Find B2B technology and professional services companies operating in Australia"
            }
        ]

        for query in free_text_queries:
            # Extract criteria from text
            extracted = self.agent.extract_criteria_from_text(query["text"])

            # Build criteria with custom prompt
            criteria = SearchCriteria(
                location=LocationCriteria(
                    countries=extracted.get('locations', {}).get('countries', ['Australia']),
                    cities=extracted.get('locations', {}).get('cities', [])
                ),
                financial=FinancialCriteria(
                    revenue_min=extracted.get('financial', {}).get('revenue_min'),
                    revenue_max=extracted.get('financial', {}).get('revenue_max')
                ),
                organizational=OrganizationalCriteria(),
                behavioral=BehavioralSignals(
                    csr_focus_areas=extracted.get('behavioral', {}).get('csr_focus_areas', [])
                ),
                business_types=extracted.get('business_types', []),
                industries=extracted.get('industries', []),
                keywords=extracted.get('keywords', []),
                custom_prompt=query["custom_prompt"],
                excluded_industries=[],
                excluded_companies=[]
            )

            result = await self.test_criteria(f"Free Text: {query['name']}", criteria)
            test_cases.append(result)

        return test_cases

    async def test_field_combinations(self):
        """Test different field combinations to isolate issues"""
        print("\n" + "=" * 60)
        print("FIELD COMBINATION TESTING")
        print("=" * 60)

        test_cases = []

        # Base criteria
        base = SearchCriteria(
            location=LocationCriteria(countries=["Australia"]),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=[],
            industries=[],
            excluded_industries=[],
            excluded_companies=[]
        )

        # Test 1: Location only
        result = await self.test_criteria("Field Test: Location Only", base)
        test_cases.append(result)

        # Test 2: Location + Business Type
        criteria_bus = SearchCriteria(
            location=LocationCriteria(countries=["Australia"]),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=["B2C"],
            industries=[],
            excluded_industries=[],
            excluded_companies=[]
        )
        result = await self.test_criteria("Field Test: Location + Business Type", criteria_bus)
        test_cases.append(result)

        # Test 3: Location + Industry
        criteria_ind = SearchCriteria(
            location=LocationCriteria(countries=["Australia"]),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=[],
            industries=[{"name": "Retail", "priority": 1}],
            excluded_industries=[],
            excluded_companies=[]
        )
        result = await self.test_criteria("Field Test: Location + Industry", criteria_ind)
        test_cases.append(result)

        # Test 4: Location + Revenue
        criteria_rev = SearchCriteria(
            location=LocationCriteria(countries=["Australia"]),
            financial=FinancialCriteria(
                revenue_min=10_000_000,
                revenue_max=100_000_000,
                revenue_currency="AUD"
            ),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=[],
            industries=[],
            excluded_industries=[],
            excluded_companies=[]
        )
        result = await self.test_criteria("Field Test: Location + Revenue", criteria_rev)
        test_cases.append(result)

        # Test 5: Complex regions/cities parsing
        criteria_complex_loc = SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                regions=["Greater Western Sydney", "Western Sydney"],  # Test regions
                cities=["Sydney", "Parramatta"]  # Multiple cities
            ),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=["B2B"],
            industries=[],
            excluded_industries=[],
            excluded_companies=[]
        )
        result = await self.test_criteria("Field Test: Complex Location", criteria_complex_loc)
        test_cases.append(result)

        return test_cases

    async def run_all_tests(self):
        """Run all regression tests"""
        print("=" * 60)
        print("ICP REGRESSION TESTING SUITE")
        print(f"Started: {datetime.now()}")
        print("=" * 60)

        all_results = {}

        # Test RMH progressively
        print("\n[1/4] Testing RMH Sydney variations...")
        rmh_results = await self.test_rmh_progressive()
        all_results['rmh'] = rmh_results

        # Test Guide Dogs progressively
        print("\n[2/4] Testing Guide Dogs Victoria variations...")
        gdv_results = await self.test_guide_dogs_progressive()
        all_results['guide_dogs'] = gdv_results

        # Test free text
        print("\n[3/4] Testing free text variations...")
        free_text_results = await self.test_free_text_variations()
        all_results['free_text'] = free_text_results

        # Test field combinations
        print("\n[4/4] Testing field combinations...")
        field_results = await self.test_field_combinations()
        all_results['field_combinations'] = field_results

        # Generate report
        self.generate_report(all_results)

        return all_results

    def generate_report(self, all_results: Dict[str, List[Dict]]):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("REGRESSION TEST REPORT")
        print("=" * 60)

        # Summary statistics
        total_tests = sum(len(results) for results in all_results.values())
        successful_tests = sum(
            sum(1 for r in results if r.get('success', False))
            for results in all_results.values()
        )

        print(f"\n📊 Overall Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Successful: {successful_tests}/{total_tests} ({successful_tests / total_tests * 100:.1f}%)")

        # Results by category
        for category, results in all_results.items():
            print(f"\n📋 {category.upper().replace('_', ' ')}:")

            for result in results:
                status = "✅" if result.get('success') else "❌"
                count = result.get('companies_found', 0)
                name = result.get('test_name', 'Unknown')

                print(f"  {status} {name}: {count} companies")

                if not result.get('success'):
                    error = result.get('error', 'Unknown error')
                    print(f"      Error: {error[:100]}")

        # Identify patterns
        print(f"\n🔍 Pattern Analysis:")

        # Check which fields cause issues
        working_tests = []
        failing_tests = []

        for category, results in all_results.items():
            for result in results:
                if result.get('success') and result.get('companies_found', 0) > 0:
                    working_tests.append(result)
                else:
                    failing_tests.append(result)

        if working_tests:
            print(f"\n✅ Working configurations ({len(working_tests)}):")
            for test in working_tests[:5]:  # Show first 5
                print(f"  - {test['test_name']}: {test['companies_found']} companies")

        if failing_tests:
            print(f"\n❌ Failing configurations ({len(failing_tests)}):")
            for test in failing_tests[:5]:  # Show first 5
                print(f"  - {test['test_name']}")

        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'icp_regression_results_{timestamp}.json'

        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: {filename}")

        # Recommendations
        print(f"\n💡 Diagnostic Findings:")

        # Check if basic searches work
        basic_working = any(
            r.get('test_name', '').endswith('Only') and r.get('success')
            for results in all_results.values()
            for r in results
        )

        if basic_working:
            print("  ✅ Basic searches (country only) are working")
            print("  ⚠️  Complex criteria may be too restrictive")
        else:
            print("  ❌ Even basic searches are failing - check API connection")

        # Check if any ICP profiles work
        icp_working = any(
            'Tier' in r.get('test_name', '') and r.get('success') and r.get('companies_found', 0) > 0
            for results in all_results.values()
            for r in results
        )

        if not icp_working:
            print("  ⚠️  ICP profiles may be too restrictive or have conflicting criteria")
            print("  💡 Try relaxing revenue ranges or removing CSR requirements")

        # Check regions handling
        region_test = next(
            (r for results in all_results.values()
             for r in results
             if 'Complex Location' in r.get('test_name', '')),
            None
        )

        if region_test and not region_test.get('success'):
            print("  ⚠️  'regions' field may not be properly handled in prompt generation")
            print("  💡 Consider moving regions to cities or using custom_prompt")


async def main():
    """Main test runner"""
    tester = ICPRegressionTester()
    results = await tester.run_all_tests()

    print("\n" + "=" * 60)
    print("REGRESSION TESTING COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    asyncio.run(main())