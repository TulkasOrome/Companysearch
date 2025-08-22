# test_parallel_gpt.py
"""
Comprehensive test script for parallel GPT-4.1 deployment execution
Tests multiple Azure OpenAI deployments running simultaneously
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass, asdict
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp

# Try both import methods for compatibility
try:
    from openai import AsyncAzureOpenAI, AzureOpenAI

    ASYNC_AVAILABLE = True
except ImportError:
    import openai

    ASYNC_AVAILABLE = False
    print("Warning: AsyncAzureOpenAI not available, using sync methods")


@dataclass
class TestResult:
    """Results from a single test execution"""
    deployment: str
    test_name: str
    success: bool
    response_time: float
    tokens_used: int
    error: Optional[str] = None
    response_content: Optional[str] = None
    timestamp: str = ""


@dataclass
class ParallelTestSummary:
    """Summary of parallel test results"""
    total_tests: int
    successful_tests: int
    failed_tests: int
    average_response_time: float
    total_execution_time: float
    deployments_tested: List[str]
    tokens_used: int
    timestamp: str


class ParallelGPTTester:
    """Test harness for parallel GPT-4.1 deployment testing"""

    def __init__(self):
        # Load configuration from environment or defaults
        self.api_key = os.getenv("AZURE_OPENAI_KEY",
                                 "CUxPxhxqutsvRVHmGQcmH59oMim6mu55PjHTjSpM6y9UwIxwVZIuJQQJ99BFACL93NaXJ3w3AAABACOG3kI1")
        self.api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://amex-openai-2025.openai.azure.com/")

        # Available deployments
        self.deployments = [
            "gpt-4.1",
            "gpt-4.1-2",
            "gpt-4.1-3",
            "gpt-4.1-4",
            "gpt-4.1-5"
        ]

        # Initialize clients for each deployment
        self.sync_clients = {}
        self.async_clients = {}
        self._initialize_clients()

        # Test results storage
        self.test_results: List[TestResult] = []

    def _initialize_clients(self):
        """Initialize both sync and async clients for each deployment"""
        print(f"Initializing clients for {len(self.deployments)} deployments...")

        for deployment in self.deployments:
            try:
                if ASYNC_AVAILABLE:
                    # Create async client
                    self.async_clients[deployment] = AsyncAzureOpenAI(
                        api_key=self.api_key,
                        api_version=self.api_version,
                        azure_endpoint=self.azure_endpoint
                    )

                    # Create sync client
                    self.sync_clients[deployment] = AzureOpenAI(
                        api_key=self.api_key,
                        api_version=self.api_version,
                        azure_endpoint=self.azure_endpoint
                    )
                else:
                    # Fallback to old OpenAI library syntax
                    import openai
                    openai.api_type = "azure"
                    openai.api_key = self.api_key
                    openai.api_version = self.api_version
                    openai.api_base = self.azure_endpoint
                    self.sync_clients[deployment] = openai

                print(f"  ✓ Initialized {deployment}")

            except Exception as e:
                print(f"  ✗ Failed to initialize {deployment}: {str(e)}")

    # ============================================================================
    # TEST 1: Basic Connectivity Test
    # ============================================================================

    def test_single_deployment_sync(self, deployment: str) -> TestResult:
        """Test a single deployment synchronously"""
        start_time = time.time()

        try:
            if ASYNC_AVAILABLE:
                client = self.sync_clients.get(deployment)
                if not client:
                    raise Exception(f"No client for {deployment}")

                response = client.chat.completions.create(
                    model=deployment,
                    messages=[
                        {"role": "system", "content": "You are a test assistant. Respond with exactly 'OK'."},
                        {"role": "user", "content": "Respond with OK"}
                    ],
                    temperature=0,
                    max_tokens=10
                )

                content = response.choices[0].message.content
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0

            else:
                # Old API syntax
                response = openai.ChatCompletion.create(
                    engine=deployment,
                    messages=[
                        {"role": "system", "content": "You are a test assistant. Respond with exactly 'OK'."},
                        {"role": "user", "content": "Respond with OK"}
                    ],
                    temperature=0,
                    max_tokens=10
                )
                content = response['choices'][0]['message']['content']
                tokens = response.get('usage', {}).get('total_tokens', 0)

            response_time = time.time() - start_time

            return TestResult(
                deployment=deployment,
                test_name="connectivity",
                success=True,
                response_time=response_time,
                tokens_used=tokens,
                response_content=content,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return TestResult(
                deployment=deployment,
                test_name="connectivity",
                success=False,
                response_time=time.time() - start_time,
                tokens_used=0,
                error=str(e),
                timestamp=datetime.now().isoformat()
            )

    # ============================================================================
    # TEST 2: Async Parallel Execution
    # ============================================================================

    async def test_single_deployment_async(self, deployment: str, prompt: str) -> TestResult:
        """Test a single deployment asynchronously"""
        if not ASYNC_AVAILABLE:
            return TestResult(
                deployment=deployment,
                test_name="async",
                success=False,
                response_time=0,
                tokens_used=0,
                error="Async not available",
                timestamp=datetime.now().isoformat()
            )

        start_time = time.time()

        try:
            client = self.async_clients.get(deployment)
            if not client:
                raise Exception(f"No async client for {deployment}")

            response = await client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )

            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
            response_time = time.time() - start_time

            return TestResult(
                deployment=deployment,
                test_name="async_parallel",
                success=True,
                response_time=response_time,
                tokens_used=tokens,
                response_content=content[:100],  # Truncate for display
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            return TestResult(
                deployment=deployment,
                test_name="async_parallel",
                success=False,
                response_time=time.time() - start_time,
                tokens_used=0,
                error=str(e),
                timestamp=datetime.now().isoformat()
            )

    async def test_async_parallel(self, deployments: List[str], prompt: str) -> List[TestResult]:
        """Test multiple deployments in parallel using asyncio"""
        print(f"\n🚀 Testing {len(deployments)} deployments in parallel (async)...")

        tasks = []
        for deployment in deployments:
            task = self.test_single_deployment_async(deployment, prompt)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    # ============================================================================
    # TEST 3: Thread-based Parallel Execution
    # ============================================================================

    def test_thread_parallel(self, deployments: List[str], prompt: str) -> List[TestResult]:
        """Test multiple deployments in parallel using ThreadPoolExecutor"""
        print(f"\n🧵 Testing {len(deployments)} deployments in parallel (threads)...")

        results = []

        def make_request(deployment):
            start_time = time.time()
            try:
                if ASYNC_AVAILABLE:
                    client = self.sync_clients.get(deployment)
                    response = client.chat.completions.create(
                        model=deployment,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=100
                    )
                    content = response.choices[0].message.content
                    tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
                else:
                    response = openai.ChatCompletion.create(
                        engine=deployment,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=100
                    )
                    content = response['choices'][0]['message']['content']
                    tokens = response.get('usage', {}).get('total_tokens', 0)

                return TestResult(
                    deployment=deployment,
                    test_name="thread_parallel",
                    success=True,
                    response_time=time.time() - start_time,
                    tokens_used=tokens,
                    response_content=content[:100],
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                return TestResult(
                    deployment=deployment,
                    test_name="thread_parallel",
                    success=False,
                    response_time=time.time() - start_time,
                    tokens_used=0,
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                )

        with ThreadPoolExecutor(max_workers=len(deployments)) as executor:
            futures = {executor.submit(make_request, dep): dep for dep in deployments}

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        return results

    # ============================================================================
    # TEST 4: Load Distribution Test
    # ============================================================================

    async def test_load_distribution(self, num_requests: int = 20) -> Dict[str, Any]:
        """Test load distribution across deployments"""
        print(f"\n⚖️ Testing load distribution with {num_requests} requests...")

        deployment_usage = {dep: 0 for dep in self.deployments}
        deployment_times = {dep: [] for dep in self.deployments}

        async def make_request_with_deployment(deployment: str, request_id: int):
            start_time = time.time()
            try:
                if ASYNC_AVAILABLE:
                    client = self.async_clients.get(deployment)
                    await client.chat.completions.create(
                        model=deployment,
                        messages=[
                            {"role": "user", "content": f"Request {request_id}: Say hello"}
                        ],
                        max_tokens=10
                    )
                response_time = time.time() - start_time
                deployment_usage[deployment] += 1
                deployment_times[deployment].append(response_time)
                return True
            except:
                return False

        # Create tasks distributed across deployments
        tasks = []
        for i in range(num_requests):
            deployment = self.deployments[i % len(self.deployments)]
            task = make_request_with_deployment(deployment, i)
            tasks.append(task)

        # Execute all tasks
        results = await asyncio.gather(*tasks)

        # Calculate statistics
        stats = {}
        for deployment in self.deployments:
            if deployment_times[deployment]:
                stats[deployment] = {
                    'requests': deployment_usage[deployment],
                    'avg_time': statistics.mean(deployment_times[deployment]),
                    'min_time': min(deployment_times[deployment]),
                    'max_time': max(deployment_times[deployment])
                }
            else:
                stats[deployment] = {
                    'requests': 0,
                    'avg_time': 0,
                    'min_time': 0,
                    'max_time': 0
                }

        return {
            'total_requests': num_requests,
            'successful_requests': sum(results),
            'deployment_stats': stats
        }

    # ============================================================================
    # TEST 5: Stress Test
    # ============================================================================

    async def stress_test(self, duration_seconds: int = 10) -> Dict[str, Any]:
        """Run continuous requests for a specified duration"""
        print(f"\n💪 Running stress test for {duration_seconds} seconds...")

        start_time = time.time()
        end_time = start_time + duration_seconds
        request_count = 0
        success_count = 0
        deployment_counts = {dep: 0 for dep in self.deployments}

        async def continuous_requests(deployment: str):
            nonlocal request_count, success_count

            while time.time() < end_time:
                try:
                    if ASYNC_AVAILABLE:
                        client = self.async_clients.get(deployment)
                        await client.chat.completions.create(
                            model=deployment,
                            messages=[{"role": "user", "content": "Hello"}],
                            max_tokens=5
                        )
                    success_count += 1
                    deployment_counts[deployment] += 1
                except:
                    pass

                request_count += 1
                await asyncio.sleep(0.1)  # Small delay to prevent overwhelming

        # Start continuous requests on all deployments
        tasks = [continuous_requests(dep) for dep in self.deployments]
        await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        return {
            'duration': total_time,
            'total_requests': request_count,
            'successful_requests': success_count,
            'success_rate': (success_count / request_count * 100) if request_count > 0 else 0,
            'requests_per_second': request_count / total_time,
            'deployment_distribution': deployment_counts
        }

    # ============================================================================
    # Main Test Runner
    # ============================================================================

    def run_connectivity_tests(self):
        """Run basic connectivity tests for all deployments"""
        print("\n" + "=" * 60)
        print("🔌 CONNECTIVITY TEST")
        print("=" * 60)

        for deployment in self.deployments:
            print(f"\nTesting {deployment}...")
            result = self.test_single_deployment_sync(deployment)
            self.test_results.append(result)

            if result.success:
                print(f"  ✓ Success - Response time: {result.response_time:.2f}s")
            else:
                print(f"  ✗ Failed - Error: {result.error}")

        # Summary
        successful = sum(1 for r in self.test_results if r.success)
        print(f"\n📊 Connectivity Summary: {successful}/{len(self.deployments)} deployments connected")

    async def run_parallel_tests(self):
        """Run various parallel execution tests"""
        print("\n" + "=" * 60)
        print("🚀 PARALLEL EXECUTION TESTS")
        print("=" * 60)

        test_prompt = "Write a haiku about parallel processing"

        # Test 1: Async parallel (all deployments)
        print("\n--- Async Parallel Test ---")
        start_time = time.time()
        async_results = await self.test_async_parallel(self.deployments, test_prompt)
        async_time = time.time() - start_time

        successful_async = sum(1 for r in async_results if r.success)
        print(f"Completed in {async_time:.2f}s - {successful_async}/{len(self.deployments)} successful")

        for result in async_results:
            self.test_results.append(result)
            if result.success:
                print(f"  ✓ {result.deployment}: {result.response_time:.2f}s")
            else:
                print(f"  ✗ {result.deployment}: {result.error}")

        # Test 2: Thread parallel (all deployments)
        print("\n--- Thread Parallel Test ---")
        start_time = time.time()
        thread_results = self.test_thread_parallel(self.deployments, test_prompt)
        thread_time = time.time() - start_time

        successful_thread = sum(1 for r in thread_results if r.success)
        print(f"Completed in {thread_time:.2f}s - {successful_thread}/{len(self.deployments)} successful")

        for result in thread_results:
            self.test_results.append(result)
            if result.success:
                print(f"  ✓ {result.deployment}: {result.response_time:.2f}s")
            else:
                print(f"  ✗ {result.deployment}: {result.error}")

        # Comparison
        print("\n📊 Parallel Execution Comparison:")
        print(f"  Async method: {async_time:.2f}s")
        print(f"  Thread method: {thread_time:.2f}s")
        print(f"  Speed improvement vs sequential: ~{len(self.deployments)}x")

    async def run_load_tests(self):
        """Run load distribution and stress tests"""
        print("\n" + "=" * 60)
        print("⚖️ LOAD DISTRIBUTION & STRESS TESTS")
        print("=" * 60)

        # Load distribution test
        load_results = await self.test_load_distribution(num_requests=20)

        print("\n📊 Load Distribution Results:")
        print(f"  Total requests: {load_results['total_requests']}")
        print(f"  Successful: {load_results['successful_requests']}")

        print("\n  Per-deployment statistics:")
        for deployment, stats in load_results['deployment_stats'].items():
            print(f"    {deployment}:")
            print(f"      Requests: {stats['requests']}")
            print(f"      Avg time: {stats['avg_time']:.2f}s")

        # Stress test (shorter duration for demo)
        stress_results = await self.stress_test(duration_seconds=5)

        print("\n💪 Stress Test Results:")
        print(f"  Duration: {stress_results['duration']:.2f}s")
        print(f"  Total requests: {stress_results['total_requests']}")
        print(f"  Success rate: {stress_results['success_rate']:.1f}%")
        print(f"  Requests/second: {stress_results['requests_per_second']:.1f}")

    def generate_report(self) -> ParallelTestSummary:
        """Generate comprehensive test report"""
        successful_tests = [r for r in self.test_results if r.success]
        failed_tests = [r for r in self.test_results if not r.success]

        if successful_tests:
            avg_response_time = statistics.mean(r.response_time for r in successful_tests)
            total_tokens = sum(r.tokens_used for r in successful_tests)
        else:
            avg_response_time = 0
            total_tokens = 0

        deployments_tested = list(set(r.deployment for r in self.test_results))

        summary = ParallelTestSummary(
            total_tests=len(self.test_results),
            successful_tests=len(successful_tests),
            failed_tests=len(failed_tests),
            average_response_time=avg_response_time,
            total_execution_time=sum(r.response_time for r in self.test_results),
            deployments_tested=deployments_tested,
            tokens_used=total_tokens,
            timestamp=datetime.now().isoformat()
        )

        return summary

    async def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 60)
        print("🧪 AZURE OPENAI PARALLEL DEPLOYMENT TEST SUITE")
        print(f"Testing {len(self.deployments)} deployments")
        print(f"Timestamp: {datetime.now()}")
        print("=" * 60)

        # Run connectivity tests
        self.run_connectivity_tests()

        # Run parallel tests
        await self.run_parallel_tests()

        # Run load tests
        await self.run_load_tests()

        # Generate final report
        print("\n" + "=" * 60)
        print("📊 FINAL TEST REPORT")
        print("=" * 60)

        summary = self.generate_report()

        print(f"\nTest Summary:")
        print(f"  Total tests run: {summary.total_tests}")
        print(f"  Successful: {summary.successful_tests}")
        print(f"  Failed: {summary.failed_tests}")
        print(f"  Success rate: {(summary.successful_tests / summary.total_tests * 100):.1f}%")
        print(f"  Average response time: {summary.average_response_time:.2f}s")
        print(f"  Total tokens used: {summary.tokens_used}")
        print(f"  Deployments tested: {', '.join(summary.deployments_tested)}")

        # Save results to file
        results_file = f"parallel_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': asdict(summary),
                'detailed_results': [asdict(r) for r in self.test_results]
            }, f, indent=2)

        print(f"\n💾 Detailed results saved to: {results_file}")

        return summary


# ============================================================================
# Quick Test Functions
# ============================================================================

def quick_connectivity_test():
    """Quick test to verify basic connectivity"""
    print("Running quick connectivity test...")
    tester = ParallelGPTTester()

    # Test just the first deployment
    result = tester.test_single_deployment_sync(tester.deployments[0])

    if result.success:
        print(f"✓ Connection successful to {result.deployment}")
        print(f"  Response time: {result.response_time:.2f}s")
        print(f"  Response: {result.response_content}")
    else:
        print(f"✗ Connection failed: {result.error}")

    return result.success


async def quick_parallel_test():
    """Quick test of parallel execution"""
    print("Running quick parallel test...")
    tester = ParallelGPTTester()

    # Test first 3 deployments in parallel
    deployments_to_test = tester.deployments[:3]
    prompt = "Say 'Hello from parallel testing' in exactly 5 words"

    print(f"Testing {len(deployments_to_test)} deployments in parallel...")
    start_time = time.time()

    results = await tester.test_async_parallel(deployments_to_test, prompt)

    total_time = time.time() - start_time
    successful = sum(1 for r in results if r.success)

    print(f"\nResults:")
    print(f"  Time taken: {total_time:.2f}s")
    print(f"  Successful: {successful}/{len(deployments_to_test)}")

    for result in results:
        if result.success:
            print(f"  ✓ {result.deployment}: {result.response_content}")
        else:
            print(f"  ✗ {result.deployment}: {result.error}")

    return results


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point for the test suite"""
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            # Run quick connectivity test
            success = quick_connectivity_test()
            if success:
                # Run quick parallel test
                await quick_parallel_test()
        elif sys.argv[1] == "full":
            # Run full test suite
            tester = ParallelGPTTester()
            await tester.run_all_tests()
        else:
            print("Usage: python test_parallel_gpt.py [quick|full]")
    else:
        # Default: run full test suite
        tester = ParallelGPTTester()
        await tester.run_all_tests()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())