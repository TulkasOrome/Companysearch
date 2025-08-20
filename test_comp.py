# test_comprehensive.py

import asyncio
import json
import itertools
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

from search_strategist_agent import (
    EnhancedSearchStrategistAgent,
    SearchCriteria,
    LocationCriteria,
    FinancialCriteria,
    OrganizationalCriteria,
    BehavioralSignals,
    BusinessType
)


class ComprehensiveSearchTester:
    """Comprehensive test suite for company search functionality"""

    def __init__(self):
        self.test_results = []
        self.agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")

    async def run_test(self, test_name: str, criteria: SearchCriteria, expected_attributes: Dict[str, Any]) -> Dict[
        str, Any]:
        """Run a single test and record results"""
        print(f"\n{'=' * 60}")
        print(f"TEST: {test_name}")
        print(f"{'=' * 60}")

        start_time = datetime.now()
        try:
            result = await self.agent.generate_enhanced_strategy(criteria, target_count=5)
            companies_found = len(result['companies'])

            # Analyze results
            analysis = self.analyze_results(result['companies'], expected_attributes)

            test_result = {
                'test_name': test_name,
                'status': 'SUCCESS',
                'companies_found': companies_found,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'analysis': analysis,
                'sample_companies': [c.name for c in result['companies'][:3]] if result['companies'] else []
            }

            print(f"✓ Found {companies_found} companies")
            if result['companies']:
                print(f"  Sample: {', '.join(test_result['sample_companies'])}")

        except Exception as e:
            test_result = {
                'test_name': test_name,
                'status': 'FAILED',
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
            print(f"✗ Test failed: {str(e)[:100]}")

        self.test_results.append(test_result)
        return test_result

    def analyze_results(self, companies: List[Any], expected: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if results match expected attributes"""
        if not companies:
            return {'matches_expected': False, 'reason': 'No companies found'}

        analysis = {
            'total_companies': len(companies),
            'confidence_distribution': {},
            'has_financial_data': 0,
            'has_csr_data': 0,
            'avg_icp_score': 0,
            'tier_distribution': {}
        }

        for company in companies:
            # Confidence distribution
            conf = company.confidence
            analysis['confidence_distribution'][conf] = analysis['confidence_distribution'].get(conf, 0) + 1

            # Financial data
            if company.estimated_revenue or company.estimated_employees:
                analysis['has_financial_data'] += 1

            # CSR data
            if company.csr_programs or company.csr_focus_areas:
                analysis['has_csr_data'] += 1

            # ICP scores
            if company.icp_score:
                analysis['avg_icp_score'] += company.icp_score

            # Tier distribution
            tier = company.icp_tier or 'Untiered'
            analysis['tier_distribution'][tier] = analysis['tier_distribution'].get(tier, 0) + 1

        if companies and analysis['avg_icp_score'] > 0:
            analysis['avg_icp_score'] /= len(companies)

        return analysis

    async def test_location_variations(self):
        """Test different location criteria"""
        print("\n" + "=" * 80)
        print("LOCATION CRITERIA TESTS")
        print("=" * 80)

        # Test 1: Single country
        criteria = SearchCriteria(
            location=LocationCriteria(countries=["United States"]),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=["B2C"],
            industries=[{"name": "Technology", "priority": 1}]
        )
        await self.run_test("Single Country (US)", criteria, {'location': 'US'})

        # Test 2: Multiple countries
        criteria.location.countries = ["United Kingdom", "Germany", "France"]
        await self.run_test("Multiple Countries (EU)", criteria, {'location': 'EU'})

        # Test 3: City-specific
        criteria.location = LocationCriteria(
            countries=["Australia"],
            cities=["Sydney", "Melbourne"]
        )
        await self.run_test("City-Specific (Australia)", criteria, {'location': 'AU Cities'})

        # Test 4: Proximity search
        criteria.location = LocationCriteria(
            countries=["United Kingdom"],
            cities=["London"],
            proximity={"location": "London CBD", "radius_km": 50}
        )
        await self.run_test("Proximity Search (London 50km)", criteria, {'location': 'Proximity'})

        # Test 5: With exclusions
        criteria.location = LocationCriteria(
            countries=["United States"],
            states=["California", "New York"],
            exclusions=["Rural areas"]
        )
        await self.run_test("Location with Exclusions", criteria, {'location': 'Exclusions'})

    async def test_financial_variations(self):
        """Test different financial criteria"""
        print("\n" + "=" * 80)
        print("FINANCIAL CRITERIA TESTS")
        print("=" * 80)

        base_criteria = SearchCriteria(
            location=LocationCriteria(countries=["United States"]),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=["B2B"],
            industries=[{"name": "Software", "priority": 1}]
        )

        # Test 1: Revenue range
        base_criteria.financial = FinancialCriteria(
            revenue_min=10_000_000,
            revenue_max=50_000_000,
            revenue_currency="USD"
        )
        await self.run_test("Revenue $10M-$50M", base_criteria, {'revenue': '10-50M'})

        # Test 2: High revenue only
        base_criteria.financial.revenue_min = 100_000_000
        base_criteria.financial.revenue_max = None
        await self.run_test("Revenue $100M+", base_criteria, {'revenue': '100M+'})

        # Test 3: With giving capacity
        base_criteria.financial = FinancialCriteria(
            revenue_min=50_000_000,
            revenue_max=500_000_000,
            giving_capacity_min=50_000
        )
        await self.run_test("Revenue + Giving Capacity", base_criteria, {'giving': '50K+'})

        # Test 4: Growth companies
        base_criteria.financial = FinancialCriteria(
            growth_rate_min=20,
            profitable=True
        )
        await self.run_test("High Growth + Profitable", base_criteria, {'growth': '20%+'})

        # Test 5: Different currency
        base_criteria.financial = FinancialCriteria(
            revenue_min=5_000_000,
            revenue_max=100_000_000,
            revenue_currency="AUD"
        )
        base_criteria.location.countries = ["Australia"]
        await self.run_test("Australian Revenue (AUD)", base_criteria, {'currency': 'AUD'})

    async def test_organizational_variations(self):
        """Test different organizational criteria"""
        print("\n" + "=" * 80)
        print("ORGANIZATIONAL CRITERIA TESTS")
        print("=" * 80)

        base_criteria = SearchCriteria(
            location=LocationCriteria(countries=["United Kingdom"]),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=["B2C"],
            industries=[{"name": "Retail", "priority": 1}]
        )

        # Test 1: Small companies
        base_criteria.organizational = OrganizationalCriteria(
            employee_count_min=10,
            employee_count_max=50
        )
        await self.run_test("Small Companies (10-50 emp)", base_criteria, {'size': 'small'})

        # Test 2: Medium companies
        base_criteria.organizational.employee_count_min = 51
        base_criteria.organizational.employee_count_max = 500
        await self.run_test("Medium Companies (51-500 emp)", base_criteria, {'size': 'medium'})

        # Test 3: Large companies
        base_criteria.organizational = OrganizationalCriteria(
            employee_count_min=500,
            office_types=["Headquarters", "Regional Office"]
        )
        await self.run_test("Large Companies with HQ", base_criteria, {'size': 'large'})

        # Test 4: By location employee count
        base_criteria.organizational = OrganizationalCriteria(
            employee_count_by_location={"London": 100, "Manchester": 50}
        )
        await self.run_test("Location-specific headcount", base_criteria, {'location_emp': True})

        # Test 5: Company stage
        base_criteria.organizational = OrganizationalCriteria(
            company_stage="Growth",
            employee_count_min=100
        )
        await self.run_test("Growth Stage Companies", base_criteria, {'stage': 'growth'})

    async def test_behavioral_signals(self):
        """Test CSR and behavioral criteria"""
        print("\n" + "=" * 80)
        print("BEHAVIORAL/CSR CRITERIA TESTS")
        print("=" * 80)

        base_criteria = SearchCriteria(
            location=LocationCriteria(countries=["United States"]),
            financial=FinancialCriteria(revenue_min=50_000_000),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=["B2B", "B2C"],
            industries=[{"name": "Technology", "priority": 1}]
        )

        # Test 1: CSR focus areas
        base_criteria.behavioral = BehavioralSignals(
            csr_focus_areas=["children", "education"]
        )
        await self.run_test("CSR: Children & Education", base_criteria, {'csr': 'children'})

        # Test 2: Environmental focus
        base_criteria.behavioral = BehavioralSignals(
            csr_focus_areas=["environment", "sustainability"],
            certifications=["Carbon Neutral", "ISO 14001"]
        )
        await self.run_test("Environmental CSR + Certs", base_criteria, {'csr': 'environment'})

        # Test 3: B-Corp companies
        base_criteria.behavioral = BehavioralSignals(
            certifications=["B-Corp"],
            esg_maturity="Leading"
        )
        await self.run_test("B-Corp Certified", base_criteria, {'cert': 'B-Corp'})

        # Test 4: Recent events
        base_criteria.behavioral = BehavioralSignals(
            recent_events=["Office Move", "Expansion", "CSR Launch"]
        )
        await self.run_test("Recent Events Triggers", base_criteria, {'events': True})

        # Test 5: Tech stack
        base_criteria.behavioral = BehavioralSignals(
            technology_stack=["Salesforce", "AWS", "Microsoft"]
        )
        await self.run_test("Specific Tech Stack", base_criteria, {'tech': True})

    async def test_industry_combinations(self):
        """Test different industry and business type combinations"""
        print("\n" + "=" * 80)
        print("INDUSTRY & BUSINESS TYPE TESTS")
        print("=" * 80)

        base_criteria = SearchCriteria(
            location=LocationCriteria(countries=["United States"]),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=[],
            industries=[]
        )

        # Test combinations
        test_cases = [
            (["B2C"], [{"name": "Retail", "priority": 1}, {"name": "Food & Beverage", "priority": 2}],
             "B2C Retail/Food"),
            (["B2B"], [{"name": "Software", "priority": 1}, {"name": "IT Services", "priority": 2}], "B2B Tech"),
            (["D2C"], [{"name": "Consumer Brands", "priority": 1}], "D2C Brands"),
            (["Professional Services"], [{"name": "Accounting", "priority": 1}], "Professional Services"),
            (["Real Estate"], [{"name": "Commercial Real Estate", "priority": 1}], "Real Estate"),
        ]

        for business_types, industries, test_name in test_cases:
            base_criteria.business_types = business_types
            base_criteria.industries = industries
            await self.run_test(test_name, base_criteria, {'industry': test_name})

    async def test_complex_combinations(self):
        """Test complex real-world scenarios"""
        print("\n" + "=" * 80)
        print("COMPLEX REAL-WORLD SCENARIOS")
        print("=" * 80)

        # Scenario 1: RMH Sydney ICP
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
                recent_events=["Office Move", "CSR Launch"]
            ),
            business_types=["B2B", "B2C"],
            industries=[
                {"name": "Construction/Trades", "priority": 1},
                {"name": "Property/Real Estate", "priority": 2},
                {"name": "Hospitality", "priority": 3}
            ],
            excluded_industries=["Fast Food", "Gambling"]
        )
        await self.run_test("RMH Sydney Full ICP", criteria, {'scenario': 'RMH'})

        # Scenario 2: Guide Dogs Victoria Tier A
        criteria = SearchCriteria(
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
                csr_focus_areas=["disability", "inclusion"],
                esg_maturity="Mature"
            ),
            business_types=["B2B", "B2C"],
            industries=[
                {"name": "Health", "priority": 1},
                {"name": "Finance", "priority": 2},
                {"name": "Technology", "priority": 3}
            ],
            excluded_industries=["Gambling", "Tobacco", "Racing"]
        )
        await self.run_test("Guide Dogs Victoria Tier A", criteria, {'scenario': 'GDV'})

        # Scenario 3: Custom prompt test
        criteria = SearchCriteria(
            location=LocationCriteria(countries=["United Kingdom"]),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=["B2B"],
            industries=[],
            custom_prompt="Find innovative tech companies that have won awards in the last 2 years and have strong diversity programs"
        )
        await self.run_test("Custom Prompt Search", criteria, {'custom': True})

    async def test_edge_cases(self):
        """Test edge cases and error conditions"""
        print("\n" + "=" * 80)
        print("EDGE CASES & ERROR CONDITIONS")
        print("=" * 80)

        # Test 1: No criteria (everything empty)
        criteria = SearchCriteria(
            location=LocationCriteria(),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=[],
            industries=[]
        )
        await self.run_test("Empty Criteria", criteria, {'edge': 'empty'})

        # Test 2: Conflicting criteria
        criteria = SearchCriteria(
            location=LocationCriteria(countries=["Japan"]),
            financial=FinancialCriteria(revenue_currency="USD"),
            organizational=OrganizationalCriteria(employee_count_max=10),
            behavioral=BehavioralSignals(certifications=["Fortune 500"]),
            business_types=["B2C"],
            industries=[{"name": "Automotive", "priority": 1}]
        )
        await self.run_test("Conflicting Criteria", criteria, {'edge': 'conflict'})

        # Test 3: Very narrow criteria
        criteria = SearchCriteria(
            location=LocationCriteria(cities=["Reykjavik"]),
            financial=FinancialCriteria(
                revenue_min=100_000_000,
                revenue_max=150_000_000
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=1000,
                employee_count_max=1500
            ),
            behavioral=BehavioralSignals(
                certifications=["B-Corp", "ISO 26000", "Carbon Neutral"],
                csr_focus_areas=["arctic", "fishing"]
            ),
            business_types=["B2B"],
            industries=[{"name": "Fishing", "priority": 1}]
        )
        await self.run_test("Ultra-Narrow Criteria", criteria, {'edge': 'narrow'})

    async def test_performance_scaling(self):
        """Test performance with different target counts"""
        print("\n" + "=" * 80)
        print("PERFORMANCE & SCALING TESTS")
        print("=" * 80)

        base_criteria = SearchCriteria(
            location=LocationCriteria(countries=["United States"]),
            financial=FinancialCriteria(),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=["B2C"],
            industries=[{"name": "Retail", "priority": 1}]
        )

        # Test different target counts
        for target_count in [1, 10, 25, 50]:
            start_time = datetime.now()
            try:
                result = await self.agent.generate_enhanced_strategy(base_criteria, target_count=target_count)
                execution_time = (datetime.now() - start_time).total_seconds()

                test_result = {
                    'test_name': f"Performance Test - {target_count} companies",
                    'status': 'SUCCESS',
                    'target_count': target_count,
                    'actual_count': len(result['companies']),
                    'execution_time': execution_time,
                    'time_per_company': execution_time / max(1, len(result['companies']))
                }

                print(f"✓ Target: {target_count}, Found: {len(result['companies'])}, Time: {execution_time:.2f}s")

            except Exception as e:
                test_result = {
                    'test_name': f"Performance Test - {target_count} companies",
                    'status': 'FAILED',
                    'error': str(e)
                }
                print(f"✗ Failed for {target_count} companies: {str(e)[:50]}")

            self.test_results.append(test_result)

    async def run_all_tests(self):
        """Run all test suites"""
        print("=" * 80)
        print("COMPREHENSIVE COMPANY SEARCH TEST SUITE")
        print(f"Started at: {datetime.now()}")
        print("=" * 80)

        # Run all test categories
        await self.test_location_variations()
        await self.test_financial_variations()
        await self.test_organizational_variations()
        await self.test_behavioral_signals()
        await self.test_industry_combinations()
        await self.test_complex_combinations()
        await self.test_edge_cases()
        await self.test_performance_scaling()

        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY REPORT")
        print("=" * 80)

        # Overall statistics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for t in self.test_results if t['status'] == 'SUCCESS')
        failed_tests = total_tests - successful_tests

        print(f"\nTotal Tests: {total_tests}")
        print(f"Successful: {successful_tests} ({successful_tests / total_tests * 100:.1f}%)")
        print(f"Failed: {failed_tests} ({failed_tests / total_tests * 100:.1f}%)")

        # Performance statistics
        success_results = [t for t in self.test_results if t['status'] == 'SUCCESS']
        if success_results:
            avg_execution_time = sum(t['execution_time'] for t in success_results) / len(success_results)
            avg_companies_found = sum(t.get('companies_found', 0) for t in success_results) / len(success_results)

            print(f"\nAverage Execution Time: {avg_execution_time:.2f}s")
            print(f"Average Companies Found: {avg_companies_found:.1f}")

        # Failed tests details
        if failed_tests > 0:
            print("\nFAILED TESTS:")
            for test in self.test_results:
                if test['status'] == 'FAILED':
                    print(f"- {test['test_name']}: {test.get('error', 'Unknown error')[:100]}")

        # Category breakdown
        categories = {
            'Location': 0,
            'Financial': 0,
            'Organizational': 0,
            'CSR': 0,
            'Industry': 0,
            'Complex': 0,
            'Edge': 0,
            'Performance': 0
        }

        for test in self.test_results:
            if test['status'] == 'SUCCESS':
                if 'Location' in test['test_name'] or 'Country' in test['test_name']:
                    categories['Location'] += 1
                elif 'Revenue' in test['test_name'] or 'Financial' in test['test_name']:
                    categories['Financial'] += 1
                elif 'Companies' in test['test_name'] or 'Employee' in test['test_name']:
                    categories['Organizational'] += 1
                elif 'CSR' in test['test_name'] or 'Environmental' in test['test_name']:
                    categories['CSR'] += 1
                elif 'B2B' in test['test_name'] or 'B2C' in test['test_name']:
                    categories['Industry'] += 1
                elif 'Full ICP' in test['test_name'] or 'Tier' in test['test_name']:
                    categories['Complex'] += 1
                elif 'Edge' in test['test_name'] or 'Empty' in test['test_name']:
                    categories['Edge'] += 1
                elif 'Performance' in test['test_name']:
                    categories['Performance'] += 1

        print("\nSUCCESS BY CATEGORY:")
        for category, count in categories.items():
            if count > 0:
                print(f"- {category}: {count} tests passed")

        # Save detailed results
        with open('comprehensive_test_results.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'successful': successful_tests,
                    'failed': failed_tests,
                    'execution_date': datetime.now().isoformat()
                },
                'test_results': self.test_results
            }, f, indent=2)

        print("\nDetailed results saved to comprehensive_test_results.json")

        # Create DataFrame for analysis
        df_results = pd.DataFrame(self.test_results)
        df_results.to_csv('comprehensive_test_results.csv', index=False)
        print("Results also saved to comprehensive_test_results.csv")


async def main():
    """Run comprehensive test suite"""
    tester = ComprehensiveSearchTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())