#!/usr/bin/env python3
"""
test_icp_profiles.py - Comprehensive test for RMH Sydney and Guide Dogs Victoria ICPs
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
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


class ICPProfileTester:
    """Test harness for ICP profiles"""

    def __init__(self):
        self.agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")
        self.results = []

    def create_rmh_criteria(self, tier: str = "standard") -> SearchCriteria:
        """Create RMH Sydney search criteria"""

        if tier == "strict":
            # Exact ICP match
            return SearchCriteria(
                location=LocationCriteria(
                    countries=["Australia"],
                    cities=["Sydney"],
                    regions=["Greater Western Sydney"],
                    proximity={"location": "Western Sydney", "radius_km": 50}
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
                    recent_events=["Office Move", "CSR Launch"]
                ),
                business_types=["B2B", "B2C"],
                industries=[
                    {"name": "Construction/Trades", "priority": 1},
                    {"name": "Property/Real Estate", "priority": 2},
                    {"name": "Hospitality", "priority": 3},
                    {"name": "Professional Services", "priority": 4}
                ],
                excluded_industries=["Fast Food", "Gambling"],
                excluded_companies=["McDonald's", "KFC", "Burger King"],
                custom_prompt="Focus on companies with strong community ties in Western Sydney"
            )
        else:
            # Standard search
            return SearchCriteria(
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
                behavioral=BehavioralSignals(
                    csr_focus_areas=["children", "community"]
                ),
                business_types=["B2B", "B2C"],
                industries=[
                    {"name": "Construction", "priority": 1},
                    {"name": "Property", "priority": 2},
                    {"name": "Hospitality", "priority": 3}
                ],
                excluded_industries=["Fast Food", "Gambling"]
            )

    def create_guide_dogs_criteria(self, tier: str = "A") -> SearchCriteria:
        """Create Guide Dogs Victoria search criteria"""

        if tier == "A":
            # Tier A - Strategic partners
            return SearchCriteria(
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
                    employee_count_by_location={"Victoria": 150}
                ),
                behavioral=BehavioralSignals(
                    certifications=["B-Corp", "ISO 26000"],
                    csr_focus_areas=["disability", "inclusion", "health"],
                    esg_maturity="Mature"
                ),
                business_types=["B2B", "B2C"],
                industries=[
                    {"name": "Health/Life Sciences", "priority": 1},
                    {"name": "Financial Services", "priority": 2},
                    {"name": "Legal Services", "priority": 3},
                    {"name": "Technology", "priority": 4},
                    {"name": "FMCG", "priority": 5},
                    {"name": "Property", "priority": 6}
                ],
                excluded_industries=["Gambling", "Tobacco", "Racing"],
                custom_prompt="Find companies with strong CSR programs and disability inclusion initiatives"
            )
        else:
            # Tier B - Exploratory partners
            return SearchCriteria(
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
                    csr_focus_areas=["community", "health", "wellbeing"]
                ),
                business_types=["B2B", "B2C"],
                industries=[
                    {"name": "Manufacturing", "priority": 1},
                    {"name": "Logistics", "priority": 2},
                    {"name": "Universities", "priority": 3}
                ],
                excluded_industries=["Gambling", "Tobacco", "Racing"]
            )

    async def test_rmh_sydney(self):
        """Test RMH Sydney ICP"""
        print("\n" + "=" * 70)
        print("TESTING: RMH SYDNEY ICP")
        print("=" * 70)

        # Test standard criteria
        print("\n1. Testing RMH Standard Criteria...")
        criteria = self.create_rmh_criteria("standard")

        try:
            result = await self.agent.generate_enhanced_strategy(criteria, target_count=10)
            companies = result.get('companies', [])

            print(f"✅ Found {len(companies)} companies")

            # Analyze results
            if companies:
                print("\nTop 5 Companies:")
                for i, company in enumerate(companies[:5], 1):
                    print(f"{i}. {company.name}")
                    print(f"   Industry: {company.industry_category}")
                    print(f"   Revenue: {company.estimated_revenue or 'Unknown'}")
                    print(f"   CSR: {', '.join(company.csr_focus_areas) if company.csr_focus_areas else 'None'}")
                    print(f"   ICP Score: {company.icp_score:.1f}" if company.icp_score else "   ICP Score: N/A")

                # Check ICP alignment
                self.analyze_icp_match(companies, "RMH Sydney")

            self.results.append(("RMH Sydney Standard", len(companies), True))

        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("RMH Sydney Standard", 0, False))

        # Test strict criteria
        print("\n2. Testing RMH Strict Criteria (with proximity)...")
        criteria = self.create_rmh_criteria("strict")

        try:
            result = await self.agent.generate_enhanced_strategy(criteria, target_count=5)
            companies = result.get('companies', [])

            print(f"✅ Found {len(companies)} companies with strict criteria")

            self.results.append(("RMH Sydney Strict", len(companies), True))

        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("RMH Sydney Strict", 0, False))

    async def test_guide_dogs_victoria(self):
        """Test Guide Dogs Victoria ICP"""
        print("\n" + "=" * 70)
        print("TESTING: GUIDE DOGS VICTORIA ICP")
        print("=" * 70)

        # Test Tier A
        print("\n1. Testing Guide Dogs Tier A (Strategic)...")
        criteria = self.create_guide_dogs_criteria("A")

        try:
            result = await self.agent.generate_enhanced_strategy(criteria, target_count=10)
            companies = result.get('companies', [])

            print(f"✅ Found {len(companies)} Tier A companies")

            if companies:
                print("\nTop 5 Tier A Companies:")
                for i, company in enumerate(companies[:5], 1):
                    print(f"{i}. {company.name}")
                    print(f"   Industry: {company.industry_category}")
                    print(f"   Revenue: {company.estimated_revenue or 'Unknown'}")
                    print(f"   Employees: {company.estimated_employees or 'Unknown'}")
                    print(
                        f"   Certifications: {', '.join(company.certifications) if company.certifications else 'None'}")

                self.analyze_icp_match(companies, "Guide Dogs Tier A")

            self.results.append(("Guide Dogs Tier A", len(companies), True))

        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("Guide Dogs Tier A", 0, False))

        # Test Tier B
        print("\n2. Testing Guide Dogs Tier B (Exploratory)...")
        criteria = self.create_guide_dogs_criteria("B")

        try:
            result = await self.agent.generate_enhanced_strategy(criteria, target_count=10)
            companies = result.get('companies', [])

            print(f"✅ Found {len(companies)} Tier B companies")

            if companies:
                print("\nTop 3 Tier B Companies:")
                for i, company in enumerate(companies[:3], 1):
                    print(f"{i}. {company.name} - {company.industry_category}")

            self.results.append(("Guide Dogs Tier B", len(companies), True))

        except Exception as e:
            print(f"❌ Failed: {e}")
            self.results.append(("Guide Dogs Tier B", 0, False))

    async def test_free_text_search(self):
        """Test free text search capability"""
        print("\n" + "=" * 70)
        print("TESTING: FREE TEXT SEARCH")
        print("=" * 70)

        test_cases = [
            {
                "name": "Sustainability Leaders",
                "text": "Find Australian companies that have won sustainability awards in the last 2 years, have strong environmental programs, and are actively reducing their carbon footprint. Focus on companies with 100-1000 employees.",
                "country": "Australia"
            },
            {
                "name": "Tech Innovators",
                "text": "Identify B2B SaaS companies in Melbourne that use AI/ML, have raised Series B+ funding, and have strong diversity and inclusion programs. Revenue between $10M-$100M AUD.",
                "country": "Australia",
                "city": "Melbourne"
            },
            {
                "name": "Community Champions",
                "text": "Find family-owned businesses in Sydney that have been operating for 20+ years, actively support local communities, and have programs for youth development or education.",
                "country": "Australia",
                "city": "Sydney"
            }
        ]

        for test in test_cases:
            print(f"\nTesting: {test['name']}")
            print(f"Query: {test['text'][:100]}...")

            # Extract criteria from free text
            extracted = self.agent.extract_criteria_from_text(test['text'])

            # Create criteria combining extracted and defaults
            criteria = SearchCriteria(
                location=LocationCriteria(
                    countries=[test.get('country', 'Australia')],
                    cities=[test.get('city')] if test.get('city') else []
                ),
                financial=FinancialCriteria(
                    revenue_min=extracted.get('financial', {}).get('revenue_min'),
                    revenue_max=extracted.get('financial', {}).get('revenue_max')
                ),
                organizational=OrganizationalCriteria(
                    employee_count_min=extracted.get('organizational', {}).get('employee_count_min'),
                    employee_count_max=extracted.get('organizational', {}).get('employee_count_max')
                ),
                behavioral=BehavioralSignals(
                    csr_focus_areas=extracted.get('behavioral', {}).get('csr_focus_areas', [])
                ),
                business_types=[],
                industries=extracted.get('industries', []),
                keywords=extracted.get('keywords', []),
                custom_prompt=test['text']
            )

            try:
                result = await self.agent.generate_enhanced_strategy(criteria, target_count=5)
                companies = result.get('companies', [])

                print(f"✅ Found {len(companies)} companies")

                if companies:
                    for company in companies[:2]:
                        print(f"  - {company.name} ({company.industry_category})")

                self.results.append((test['name'], len(companies), True))

            except Exception as e:
                print(f"❌ Failed: {e}")
                self.results.append((test['name'], 0, False))

    def analyze_icp_match(self, companies: List[Any], profile_name: str):
        """Analyze how well companies match the ICP"""

        if not companies:
            return

        # Calculate statistics
        scores = [c.icp_score for c in companies if c.icp_score]
        tiers = {}
        for c in companies:
            tier = c.icp_tier or "Untiered"
            tiers[tier] = tiers.get(tier, 0) + 1

        print(f"\n📊 ICP Analysis for {profile_name}:")

        if scores:
            print(f"  Average ICP Score: {sum(scores) / len(scores):.1f}")
            print(f"  Score Range: {min(scores):.1f} - {max(scores):.1f}")

        print("  Tier Distribution:")
        for tier in ['A', 'B', 'C', 'D', 'Untiered']:
            if tier in tiers:
                pct = (tiers[tier] / len(companies)) * 100
                print(f"    Tier {tier}: {tiers[tier]} ({pct:.0f}%)")

    async def run_all_tests(self):
        """Run all ICP tests"""
        print("=" * 70)
        print("ICP PROFILE TESTING SUITE")
        print(f"Started: {datetime.now()}")
        print("=" * 70)

        # Test RMH Sydney
        await self.test_rmh_sydney()

        # Test Guide Dogs Victoria
        await self.test_guide_dogs_victoria()

        # Test free text search
        await self.test_free_text_search()

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate test report"""
        print("\n" + "=" * 70)
        print("TEST REPORT SUMMARY")
        print("=" * 70)

        # Overall stats
        total_tests = len(self.results)
        passed_tests = sum(1 for _, _, passed in self.results if passed)
        total_companies = sum(count for _, count, _ in self.results)

        print(f"\n📊 Overall Statistics:")
        print(f"  Tests Run: {total_tests}")
        print(f"  Tests Passed: {passed_tests}/{total_tests} ({passed_tests / total_tests * 100:.0f}%)")
        print(f"  Total Companies Found: {total_companies}")

        print(f"\n📋 Individual Test Results:")
        for test_name, count, passed in self.results:
            status = "✅" if passed else "❌"
            print(f"  {status} {test_name}: {count} companies")

        # Save detailed results
        results_df = pd.DataFrame(self.results, columns=['Test', 'Companies Found', 'Passed'])
        results_df.to_csv(f'icp_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
        print(f"\n💾 Detailed results saved to CSV")

        # Recommendations
        print(f"\n💡 Recommendations:")
        if passed_tests < total_tests:
            print("  - Some tests failed. Review error messages above.")
        if any(count < 5 for _, count, passed in self.results if passed):
            print("  - Some searches returned few results. Consider relaxing criteria.")
        print("  - Consider implementing validation with Serper API for found companies")
        print("  - Add more specific industry mappings for better matches")


async def main():
    """Main test runner"""
    tester = ICPProfileTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())