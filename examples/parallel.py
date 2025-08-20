# test_parallel_gpt.py

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from dataclasses import dataclass, asdict
import statistics

from agents.search_strategist_agent import (
    EnhancedSearchStrategistAgent,
    SearchCriteria,
    LocationCriteria,
    FinancialCriteria,
    OrganizationalCriteria,
    BehavioralSignals,
    EnhancedCompanyEntry
)


@dataclass
class PerformanceMetrics:
    """Performance metrics for each test run"""
    deployment: str
    target_count: int
    actual_count: int
    execution_time: float
    time_per_company: float
    success: bool
    error: str = None
    response_quality_score: float = 0.0
    companies_with_all_fields: int = 0
    avg_icp_score: float = 0.0
    timestamp: str = ""


class ParallelGPTTester:
    """Test harness for parallel GPT-4.1 deployment testing"""

    def __init__(self):
        # All available deployments
        self.deployments = [
            "gpt-4.1",
            "gpt-4.1-2",
            "gpt-4.1-3",
            "gpt-4.1-4",
            "gpt-4.1-5"
        ]

        self.test_results = []
        self.deployment_agents = {}

        # Initialize agents for each deployment
        for deployment in self.deployments:
            self.deployment_agents[deployment] = EnhancedSearchStrategistAgent(deployment_name=deployment)

    def create_test_criteria(self, complexity: str = "medium") -> SearchCriteria:
        """Create test criteria with varying complexity"""

        if complexity == "simple":
            return SearchCriteria(
                location=LocationCriteria(countries=["United States"]),
                financial=FinancialCriteria(),
                organizational=OrganizationalCriteria(),
                behavioral=BehavioralSignals(),
                business_types=["B2C"],
                industries=[{"name": "Retail", "priority": 1}]
            )

        elif complexity == "medium":
            return SearchCriteria(
                location=LocationCriteria(
                    countries=["Australia"],
                    cities=["Sydney", "Melbourne"]
                ),
                financial=FinancialCriteria(
                    revenue_min=10_000_000,
                    revenue_max=100_000_000,
                    revenue_currency="AUD"
                ),
                organizational=OrganizationalCriteria(
                    employee_count_min=50,
                    employee_count_max=500
                ),
                behavioral=BehavioralSignals(
                    csr_focus_areas=["community", "environment"]
                ),
                business_types=["B2B", "B2C"],
                industries=[
                    {"name": "Technology", "priority": 1},
                    {"name": "Professional Services", "priority": 2}
                ]
            )

        else:  # complex
            return SearchCriteria(
                location=LocationCriteria(
                    countries=["United States", "United Kingdom"],
                    cities=["New York", "London"],
                    proximity={"location": "Manhattan", "radius_km": 50}
                ),
                financial=FinancialCriteria(
                    revenue_min=100_000_000,
                    revenue_max=1_000_000_000,
                    revenue_currency="USD",
                    giving_capacity_min=100_000,
                    growth_rate_min=10
                ),
                organizational=OrganizationalCriteria(
                    employee_count_min=500,
                    employee_count_max=5000,
                    office_types=["Headquarters", "Regional Office"]
                ),
                behavioral=BehavioralSignals(
                    csr_focus_areas=["diversity", "education", "health"],
                    certifications=["B-Corp", "ISO 26000"],
                    recent_events=["Expansion", "CSR Launch"],
                    esg_maturity="Mature"
                ),
                business_types=["B2B", "B2C", "B2B2C"],
                industries=[
                    {"name": "Financial Services", "priority": 1},
                    {"name": "Healthcare", "priority": 2},
                    {"name": "Technology", "priority": 3}
                ],
                excluded_industries=["Gambling", "Tobacco"]
            )

    def evaluate_response_quality(self, companies: List[EnhancedCompanyEntry]) -> Dict[str, Any]:
        """Evaluate the quality of the response"""
        if not companies:
            return {
                "quality_score": 0,
                "companies_with_all_fields": 0,
                "avg_icp_score": 0
            }

        required_fields = [
            'name', 'confidence', 'operates_in_country',
            'business_type', 'industry_category', 'reasoning'
        ]

        enhanced_fields = [
            'estimated_revenue', 'estimated_employees', 'company_size',
            'headquarters', 'office_locations', 'csr_programs',
            'csr_focus_areas', 'icp_tier', 'icp_score'
        ]

        companies_with_required = 0
        companies_with_enhanced = 0
        total_icp_score = 0
        field_completeness = []

        for company in companies:
            company_dict = company.dict() if hasattr(company, 'dict') else company

            # Check required fields
            has_required = all(
                company_dict.get(field) is not None
                for field in required_fields
            )
            if has_required:
                companies_with_required += 1

            # Check enhanced fields
            enhanced_count = sum(
                1 for field in enhanced_fields
                if company_dict.get(field) is not None
            )
            companies_with_enhanced += enhanced_count / len(enhanced_fields)

            # Track ICP score
            if company_dict.get('icp_score'):
                total_icp_score += company_dict['icp_score']

            # Calculate field completeness
            total_fields = len(required_fields) + len(enhanced_fields)
            filled_fields = sum(
                1 for field in required_fields + enhanced_fields
                if company_dict.get(field) is not None
            )
            field_completeness.append(filled_fields / total_fields)

        quality_score = (
                (companies_with_required / len(companies)) * 50 +  # 50% for required fields
                (companies_with_enhanced / len(companies)) * 30 +  # 30% for enhanced fields
                (statistics.mean(field_completeness) * 20)  # 20% for overall completeness
        )

        return {
            "quality_score": quality_score,
            "companies_with_all_fields": companies_with_required,
            "avg_icp_score": total_icp_score / len(companies) if companies else 0
        }

    async def test_single_deployment(
            self,
            deployment: str,
            criteria: SearchCriteria,
            target_count: int
    ) -> PerformanceMetrics:
        """Test a single deployment"""
        start_time = time.time()

        try:
            agent = self.deployment_agents[deployment]
            result = await agent.generate_enhanced_strategy(criteria, target_count=target_count)

            execution_time = time.time() - start_time
            companies = result.get('companies', [])

            # Evaluate quality
            quality_metrics = self.evaluate_response_quality(companies)

            return PerformanceMetrics(
                deployment=deployment,
                target_count=target_count,
                actual_count=len(companies),
                execution_time=execution_time,
                time_per_company=execution_time / max(1, len(companies)),
                success=True,
                response_quality_score=quality_metrics['quality_score'],
                companies_with_all_fields=quality_metrics['companies_with_all_fields'],
                avg_icp_score=quality_metrics['avg_icp_score'],
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return PerformanceMetrics(
                deployment=deployment,
                target_count=target_count,
                actual_count=0,
                execution_time=time.time() - start_time,
                time_per_company=0,
                success=False,
                error=str(e)[:200],
                timestamp=datetime.now().isoformat()
            )

    async def test_parallel_deployments(
            self,
            criteria: SearchCriteria,
            target_count: int,
            deployments_to_test: List[str] = None
    ) -> List[PerformanceMetrics]:
        """Test multiple deployments in parallel"""
        if deployments_to_test is None:
            deployments_to_test = self.deployments[:3]  # Default to first 3

        print(f"\n{'=' * 60}")
        print(f"Parallel Test: {len(deployments_to_test)} deployments, {target_count} companies each")
        print(f"Deployments: {', '.join(deployments_to_test)}")
        print(f"{'=' * 60}")

        tasks = []
        for deployment in deployments_to_test:
            task = self.test_single_deployment(deployment, criteria, target_count)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Print summary
        for result in results:
            status = "✓" if result.success else "✗"
            print(
                f"{status} {result.deployment}: {result.actual_count}/{result.target_count} in {result.execution_time:.2f}s")
            if result.success:
                print(f"  Quality: {result.response_quality_score:.1f}%, Avg ICP: {result.avg_icp_score:.1f}")

        return results

    async def test_scaling_performance(self):
        """Test performance at different scales"""
        print("\n" + "=" * 80)
        print("SCALING PERFORMANCE TEST")
        print("=" * 80)

        # Test configurations
        test_configs = [
            (1, ["gpt-4.1"], "simple"),  # Single small request
            (10, ["gpt-4.1"], "simple"),  # Small batch
            (25, ["gpt-4.1", "gpt-4.1-2"], "medium"),  # Medium batch, 2 deployments
            (50, self.deployments[:3], "medium"),  # Large batch, 3 deployments
            (100, self.deployments[:5], "medium"),  # Very large batch, 5 deployments
            (200, self.deployments[:5], "simple"),  # Extreme batch, all deployments
        ]

        all_results = []

        for target_count, deployments, complexity in test_configs:
            print(f"\n--- Testing {target_count} companies with {len(deployments)} deployment(s) ---")

            criteria = self.create_test_criteria(complexity)
            results = await self.test_parallel_deployments(
                criteria,
                target_count,
                deployments
            )

            all_results.extend(results)

            # Calculate aggregate metrics
            successful = [r for r in results if r.success]
            if successful:
                avg_time = statistics.mean(r.execution_time for r in successful)
                avg_quality = statistics.mean(r.response_quality_score for r in successful)
                total_companies = sum(r.actual_count for r in successful)

                print(f"\nAggregate Metrics:")
                print(f"  Avg Time: {avg_time:.2f}s")
                print(f"  Avg Quality: {avg_quality:.1f}%")
                print(f"  Total Companies: {total_companies}")
                print(f"  Success Rate: {len(successful)}/{len(results)}")

            # Add delay between tests
            await asyncio.sleep(2)

        self.test_results.extend(all_results)
        return all_results

    async def test_to_1000_companies(self):
        """Progressive test up to 1000 companies total"""
        print("\n" + "=" * 80)
        print("PROGRESSIVE SCALING TO 1000 COMPANIES")
        print("=" * 80)

        criteria = self.create_test_criteria("medium")
        total_companies_target = 1000
        companies_collected = 0
        batch_results = []

        # Strategy: Use all 5 deployments in parallel with increasing batch sizes
        batch_configs = [
            (50, 5),  # 50 per deployment × 5 = 250 companies
            (75, 5),  # 75 per deployment × 5 = 375 companies
            (75, 5),  # 75 per deployment × 5 = 375 companies
        ]

        print(f"Target: {total_companies_target} total companies")
        print(f"Strategy: {len(batch_configs)} batches using all {len(self.deployments)} deployments")

        for batch_num, (companies_per_deployment, num_deployments) in enumerate(batch_configs, 1):
            deployments_to_use = self.deployments[:num_deployments]
            expected_companies = companies_per_deployment * num_deployments

            print(f"\n--- Batch {batch_num}: {expected_companies} companies ---")
            print(f"Deployments: {', '.join(deployments_to_use)}")

            batch_start = time.time()

            # Run parallel requests
            tasks = []
            for deployment in deployments_to_use:
                task = self.test_single_deployment(
                    deployment,
                    criteria,
                    companies_per_deployment
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            batch_time = time.time() - batch_start

            # Collect metrics
            batch_total = sum(r.actual_count for r in results if r.success)
            companies_collected += batch_total

            batch_summary = {
                'batch_num': batch_num,
                'target_per_deployment': companies_per_deployment,
                'deployments_used': num_deployments,
                'expected_total': expected_companies,
                'actual_total': batch_total,
                'batch_time': batch_time,
                'avg_time_per_deployment': batch_time / num_deployments,
                'companies_collected_so_far': companies_collected,
                'results': results
            }
            batch_results.append(batch_summary)

            # Print batch summary
            print(f"Batch {batch_num} Complete:")
            print(f"  Expected: {expected_companies}, Actual: {batch_total}")
            print(f"  Time: {batch_time:.2f}s")
            print(f"  Total so far: {companies_collected}/{total_companies_target}")

            for result in results:
                status = "✓" if result.success else "✗"
                print(f"  {status} {result.deployment}: {result.actual_count} companies")

            if companies_collected >= total_companies_target:
                print(f"\n✓ Target reached! Collected {companies_collected} companies")
                break

            # Rate limiting between batches
            await asyncio.sleep(3)

        return batch_results

    async def test_deployment_consistency(self):
        """Test consistency across deployments with same input"""
        print("\n" + "=" * 80)
        print("DEPLOYMENT CONSISTENCY TEST")
        print("=" * 80)

        # Use same criteria for all deployments
        criteria = self.create_test_criteria("medium")
        target_count = 20

        print(f"Testing all {len(self.deployments)} deployments with identical input")
        print(f"Target: {target_count} companies each")

        # Run all deployments with same input
        results = await self.test_parallel_deployments(
            criteria,
            target_count,
            self.deployments
        )

        # Analyze consistency
        successful_results = [r for r in results if r.success]

        if len(successful_results) >= 2:
            # Calculate variance metrics
            counts = [r.actual_count for r in successful_results]
            times = [r.execution_time for r in successful_results]
            qualities = [r.response_quality_score for r in successful_results]

            consistency_metrics = {
                'count_variance': statistics.stdev(counts) if len(counts) > 1 else 0,
                'count_range': max(counts) - min(counts),
                'time_variance': statistics.stdev(times) if len(times) > 1 else 0,
                'time_range': max(times) - min(times),
                'quality_variance': statistics.stdev(qualities) if len(qualities) > 1 else 0,
                'quality_range': max(qualities) - min(qualities),
                'avg_count': statistics.mean(counts),
                'avg_time': statistics.mean(times),
                'avg_quality': statistics.mean(qualities)
            }

            print("\nConsistency Analysis:")
            print(f"  Count: {consistency_metrics['avg_count']:.1f} ± {consistency_metrics['count_variance']:.1f}")
            print(f"  Time: {consistency_metrics['avg_time']:.2f}s ± {consistency_metrics['time_variance']:.2f}s")
            print(
                f"  Quality: {consistency_metrics['avg_quality']:.1f}% ± {consistency_metrics['quality_variance']:.1f}%")

            # Flag inconsistencies
            if consistency_metrics['count_range'] > 5:
                print("  ⚠️ High variance in company counts across deployments")
            if consistency_metrics['time_range'] > 10:
                print("  ⚠️ High variance in execution times across deployments")
            if consistency_metrics['quality_range'] > 20:
                print("  ⚠️ High variance in response quality across deployments")

            return consistency_metrics

        return None

    async def test_failure_recovery(self):
        """Test how deployments handle failures and retries"""
        print("\n" + "=" * 80)
        print("FAILURE RECOVERY TEST")
        print("=" * 80)

        # Test with potentially problematic criteria
        problematic_criteria = [
            {
                'name': 'Empty Criteria',
                'criteria': SearchCriteria(
                    location=LocationCriteria(),
                    financial=FinancialCriteria(),
                    organizational=OrganizationalCriteria(),
                    behavioral=BehavioralSignals(),
                    business_types=[],
                    industries=[]
                )
            },
            {
                'name': 'Impossible Criteria',
                'criteria': SearchCriteria(
                    location=LocationCriteria(cities=["Atlantis"]),
                    financial=FinancialCriteria(
                        revenue_min=1_000_000_000_000,  # 1 trillion
                        revenue_max=1_000_000_000_001  # Very narrow range
                    ),
                    organizational=OrganizationalCriteria(
                        employee_count_min=1_000_000
                    ),
                    behavioral=BehavioralSignals(),
                    business_types=["Impossible Type"],
                    industries=[{"name": "Non-existent Industry", "priority": 1}]
                )
            },
            {
                'name': 'Very Large Request',
                'criteria': self.create_test_criteria("simple"),
                'target_count': 500  # Very large single request
            }
        ]

        recovery_results = []

        for test_case in problematic_criteria:
            print(f"\nTesting: {test_case['name']}")

            target = test_case.get('target_count', 10)

            # Test first 3 deployments
            tasks = []
            for deployment in self.deployments[:3]:
                task = self.test_single_deployment(
                    deployment,
                    test_case['criteria'],
                    target
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            # Analyze recovery
            success_count = sum(1 for r in results if r.success)
            partial_success = sum(1 for r in results if r.actual_count > 0)

            recovery_summary = {
                'test_case': test_case['name'],
                'deployments_tested': 3,
                'successful': success_count,
                'partial_success': partial_success,
                'results': [asdict(r) for r in results]
            }

            recovery_results.append(recovery_summary)

            print(f"  Success rate: {success_count}/3")
            print(f"  Partial success: {partial_success}/3")

            for result in results:
                if not result.success:
                    print(f"  ✗ {result.deployment}: {result.error[:50]}...")
                else:
                    print(f"  ✓ {result.deployment}: {result.actual_count} companies")

        return recovery_results

    async def run_comprehensive_parallel_tests(self):
        """Run all parallel tests"""
        print("=" * 80)
        print("COMPREHENSIVE PARALLEL GPT-4.1 TESTING SUITE")
        print(f"Started at: {datetime.now()}")
        print(f"Testing {len(self.deployments)} deployments")
        print("=" * 80)

        all_test_results = {}

        # 1. Scaling Performance Test
        print("\n[1/5] Running Scaling Performance Test...")
        scaling_results = await self.test_scaling_performance()
        all_test_results['scaling'] = scaling_results

        # 2. Consistency Test
        print("\n[2/5] Running Deployment Consistency Test...")
        consistency_results = await self.test_deployment_consistency()
        all_test_results['consistency'] = consistency_results

        # 3. Failure Recovery Test
        print("\n[3/5] Running Failure Recovery Test...")
        recovery_results = await self.test_failure_recovery()
        all_test_results['recovery'] = recovery_results

        # 4. Progressive to 1000 Test
        print("\n[4/5] Running Progressive Scaling to 1000...")
        thousand_results = await self.test_to_1000_companies()
        all_test_results['thousand_companies'] = thousand_results

        # 5. Generate comprehensive report
        print("\n[5/5] Generating Comprehensive Report...")
        self.generate_comprehensive_report(all_test_results)

        return all_test_results

    def generate_comprehensive_report(self, all_results: Dict[str, Any]):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST REPORT")
        print("=" * 80)

        # Overall statistics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        total_companies = sum(r.actual_count for r in self.test_results if r.success)

        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Test Runs: {total_tests}")
        print(f"  Successful Runs: {successful_tests} ({successful_tests / total_tests * 100:.1f}%)")
        print(f"  Total Companies Generated: {total_companies}")

        # Deployment performance summary
        deployment_stats = {}
        for deployment in self.deployments:
            deployment_results = [r for r in self.test_results if r.deployment == deployment]
            if deployment_results:
                successful = [r for r in deployment_results if r.success]
                deployment_stats[deployment] = {
                    'total_runs': len(deployment_results),
                    'success_rate': len(successful) / len(deployment_results) * 100,
                    'avg_time': statistics.mean(r.execution_time for r in successful) if successful else 0,
                    'avg_quality': statistics.mean(r.response_quality_score for r in successful) if successful else 0,
                    'total_companies': sum(r.actual_count for r in successful)
                }

        print(f"\nDEPLOYMENT PERFORMANCE:")
        for deployment, stats in deployment_stats.items():
            print(f"\n  {deployment}:")
            print(f"    Success Rate: {stats['success_rate']:.1f}%")
            print(f"    Avg Time: {stats['avg_time']:.2f}s")
            print(f"    Avg Quality: {stats['avg_quality']:.1f}%")
            print(f"    Total Companies: {stats['total_companies']}")

        # Best performing deployment
        if deployment_stats:
            best_by_speed = min(deployment_stats.items(),
                                key=lambda x: x[1]['avg_time'] if x[1]['avg_time'] > 0 else float('inf'))
            best_by_quality = max(deployment_stats.items(),
                                  key=lambda x: x[1]['avg_quality'])
            best_by_reliability = max(deployment_stats.items(),
                                      key=lambda x: x[1]['success_rate'])

            print(f"\nBEST PERFORMERS:")
            print(f"  Fastest: {best_by_speed[0]} ({best_by_speed[1]['avg_time']:.2f}s avg)")
            print(f"  Highest Quality: {best_by_quality[0]} ({best_by_quality[1]['avg_quality']:.1f}% avg)")
            print(f"  Most Reliable: {best_by_reliability[0]} ({best_by_reliability[1]['success_rate']:.1f}% success)")

        # Scaling insights
        if 'thousand_companies' in all_results:
            thousand_data = all_results['thousand_companies']
            total_collected = thousand_data[-1]['companies_collected_so_far'] if thousand_data else 0
            total_time = sum(batch['batch_time'] for batch in thousand_data)

            print(f"\nSCALING TO 1000 COMPANIES:")
            print(f"  Total Collected: {total_collected}")
            print(f"  Total Time: {total_time:.2f}s")
            print(f"  Avg Rate: {total_collected / total_time:.1f} companies/second")
            print(f"  Batches Required: {len(thousand_data)}")

        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save raw test results
        with open(f'parallel_gpt_test_results_{timestamp}.json', 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'successful_tests': successful_tests,
                    'total_companies': total_companies,
                    'timestamp': datetime.now().isoformat()
                },
                'deployment_stats': deployment_stats,
                'all_results': all_results,
                'individual_results': [asdict(r) for r in self.test_results]
            }, f, indent=2, default=str)

        # Create DataFrame for analysis
        df_results = pd.DataFrame([asdict(r) for r in self.test_results])
        df_results.to_csv(f'parallel_gpt_test_results_{timestamp}.csv', index=False)

        print(f"\nResults saved to:")
        print(f"  - parallel_gpt_test_results_{timestamp}.json")
        print(f"  - parallel_gpt_test_results_{timestamp}.csv")


async def main():
    """Run the parallel GPT testing suite"""
    tester = ParallelGPTTester()

    # Run comprehensive tests
    results = await tester.run_comprehensive_parallel_tests()

    print("\n" + "=" * 80)
    print("ALL PARALLEL TESTS COMPLETED")
    print("=" * 80)

    return results


if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())