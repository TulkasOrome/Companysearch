#!/usr/bin/env python3
"""
emergency_fix_and_test.py - Emergency diagnostic and fix for search issues
This will identify why searches are returning 0 results and provide fixes
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import traceback
from openai import AzureOpenAI

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


class EmergencyDiagnostics:
    """Emergency diagnostics to find and fix the issue"""

    def __init__(self):
        self.agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")
        # Initialize the client directly for testing
        self.client = AzureOpenAI(
            api_key="CUxPxhxqutsvRVHmGQcmH59oMim6mu55PjHTjSpM6y9UwIxwVZIuJQQJ99BFACL93NaXJ3w3AAABACOG3kI1",
            api_version="2024-02-01",
            azure_endpoint="https://amex-openai-2025.openai.azure.com/"
        )

    async def test_raw_gpt_call(self):
        """Test if GPT is responding at all"""
        print("\n" + "=" * 80)
        print("TEST 1: RAW GPT CALL TEST")
        print("=" * 80)

        try:
            # Simple test prompt
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "List 3 Australian companies. Reply with just the names."}
                ],
                temperature=0.1,
                max_tokens=100
            )

            content = response.choices[0].message.content
            print(f"✅ GPT Response: {content}")
            return True

        except Exception as e:
            print(f"❌ GPT Call Failed: {e}")
            return False

    async def test_json_format(self):
        """Test if JSON format is working"""
        print("\n" + "=" * 80)
        print("TEST 2: JSON FORMAT TEST")
        print("=" * 80)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system",
                     "content": "You are an expert at finding companies. Always respond with valid JSON."},
                    {"role": "user", "content": """Find 3 Australian companies. Return as JSON:
                    {"companies": [{"name": "Company Name", "industry": "Industry"}]}"""}
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            print(f"Raw response: {content[:200]}...")

            # Try to parse
            parsed = json.loads(content)
            companies = parsed.get("companies", [])
            print(f"✅ Parsed {len(companies)} companies")
            for c in companies:
                print(f"  - {c.get('name', 'Unknown')}")
            return True

        except Exception as e:
            print(f"❌ JSON Format Failed: {e}")
            traceback.print_exc()
            return False

    async def test_simple_criteria_prompt(self):
        """Test with a very simple criteria prompt"""
        print("\n" + "=" * 80)
        print("TEST 3: SIMPLE CRITERIA PROMPT")
        print("=" * 80)

        prompt = """Find 5 companies matching these criteria:
Countries: Australia

Return a JSON object with a "companies" array. Each company MUST have these fields:
{
    "companies": [
        {
            "name": "Company Name",
            "confidence": "high",
            "operates_in_country": true,
            "business_type": "B2C",
            "industry_category": "Retail",
            "reasoning": "Why this company matches"
        }
    ]
}

Return ONLY valid JSON."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system",
                     "content": "You are an expert at finding companies. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)
            companies = parsed.get("companies", [])

            print(f"✅ Found {len(companies)} companies")
            for c in companies[:3]:
                print(f"  - {c.get('name', 'Unknown')}: {c.get('industry_category', 'Unknown')}")

            return len(companies) > 0

        except Exception as e:
            print(f"❌ Simple Criteria Failed: {e}")
            return False

    async def test_with_revenue(self):
        """Test adding revenue criteria"""
        print("\n" + "=" * 80)
        print("TEST 4: WITH REVENUE CRITERIA")
        print("=" * 80)

        prompt = """Find 5 companies matching these criteria:
Countries: Australia
Revenue: AUD 10,000,000 - 100,000,000

Return a JSON object with a "companies" array. Each company MUST have these fields:
{
    "companies": [
        {
            "name": "Company Name",
            "confidence": "high",
            "operates_in_country": true,
            "business_type": "B2C",
            "industry_category": "Retail",
            "reasoning": "Why this company matches",
            "estimated_revenue": "$10M-$100M"
        }
    ]
}

Return ONLY valid JSON."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system",
                     "content": "You are an expert at finding companies. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)
            companies = parsed.get("companies", [])

            print(f"Result: Found {len(companies)} companies")
            if companies:
                print("✅ Revenue criteria working")
                for c in companies[:3]:
                    print(f"  - {c.get('name', 'Unknown')}: {c.get('estimated_revenue', 'Unknown')}")
            else:
                print("❌ No companies returned with revenue criteria")

            return len(companies) > 0

        except Exception as e:
            print(f"❌ Revenue Test Failed: {e}")
            return False

    async def test_actual_failing_scenario(self):
        """Test your exact failing scenario"""
        print("\n" + "=" * 80)
        print("TEST 5: YOUR EXACT FAILING SCENARIO")
        print("=" * 80)

        # Your exact criteria that's failing
        criteria = SearchCriteria(
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

        # Get the actual prompt your app generates
        prompt = self.agent._build_enhanced_prompt(criteria, 5)

        print("Generated Prompt Length:", len(prompt))
        print("\nFirst 500 chars of prompt:")
        print("-" * 40)
        print(prompt[:500])
        print("-" * 40)

        try:
            # Try with the actual prompt
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system",
                     "content": "You are an expert at finding companies. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            print(f"\nRaw response length: {len(content)} chars")
            print(f"First 200 chars: {content[:200]}")

            # Try to parse
            parsed = json.loads(content)
            companies = parsed.get("companies", [])

            print(f"\n{'✅' if companies else '❌'} Found {len(companies)} companies")

            if not companies:
                # Check if there's an error message in the response
                if "error" in parsed:
                    print(f"Error in response: {parsed['error']}")

                # Save the full prompt and response for analysis
                with open("failing_prompt.txt", "w") as f:
                    f.write(prompt)
                with open("failing_response.json", "w") as f:
                    json.dump(parsed, f, indent=2)
                print("\n💾 Saved failing prompt and response for analysis")

            return len(companies) > 0

        except Exception as e:
            print(f"❌ Actual Scenario Failed: {e}")
            traceback.print_exc()
            return False

    async def test_modified_prompt_format(self):
        """Test with a modified, simpler prompt format"""
        print("\n" + "=" * 80)
        print("TEST 6: MODIFIED PROMPT FORMAT")
        print("=" * 80)

        # Simplified prompt format
        prompt = """You are a company finder. Find 5 Australian companies.

Requirements:
- Location: Australia, Victoria state
- Revenue: Between 50 million and 500 million AUD
- Employees: 100-500
- Industries: Manufacturing, Logistics, or Universities

Format your response as JSON:
{
  "companies": [
    {
      "name": "Example Company Pty Ltd",
      "confidence": "high",
      "operates_in_country": true,
      "business_type": "B2B",
      "industry_category": "Manufacturing",
      "reasoning": "Matches criteria because...",
      "estimated_revenue": "$50M-$100M",
      "estimated_employees": "100-500"
    }
  ]
}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)
            companies = parsed.get("companies", [])

            print(f"{'✅' if companies else '❌'} Found {len(companies)} companies with modified format")

            if companies:
                for c in companies[:3]:
                    print(f"  - {c.get('name', 'Unknown')}: {c.get('industry_category', 'Unknown')}")

            return len(companies) > 0

        except Exception as e:
            print(f"❌ Modified Format Failed: {e}")
            return False

    async def test_without_json_format_constraint(self):
        """Test without the JSON format constraint"""
        print("\n" + "=" * 80)
        print("TEST 7: WITHOUT JSON FORMAT CONSTRAINT")
        print("=" * 80)

        prompt = """Find 5 Australian manufacturing companies with revenue between 50-500 million AUD.

For each company, provide:
- Company name
- Industry
- Estimated revenue
- Why it matches

Format as a simple list."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
                # Note: NO response_format constraint
            )

            content = response.choices[0].message.content
            print(f"Response without JSON constraint:")
            print(content[:500])

            # Check if we got companies
            has_companies = any(word in content.lower() for word in ["pty", "ltd", "limited", "company"])
            print(f"\n{'✅' if has_companies else '❌'} Response contains company names")

            return has_companies

        except Exception as e:
            print(f"❌ Non-JSON Test Failed: {e}")
            return False

    async def run_emergency_diagnostics(self):
        """Run all emergency diagnostic tests"""
        print("=" * 80)
        print("EMERGENCY DIAGNOSTICS")
        print(f"Started: {datetime.now()}")
        print("=" * 80)

        results = {}

        # Test 1: Basic GPT connectivity
        print("\n[1/7] Testing GPT connectivity...")
        results['gpt_connectivity'] = await self.test_raw_gpt_call()

        # Test 2: JSON format
        print("\n[2/7] Testing JSON format...")
        results['json_format'] = await self.test_json_format()

        # Test 3: Simple criteria
        print("\n[3/7] Testing simple criteria...")
        results['simple_criteria'] = await self.test_simple_criteria_prompt()

        # Test 4: With revenue
        print("\n[4/7] Testing with revenue...")
        results['with_revenue'] = await self.test_with_revenue()

        # Test 5: Actual failing scenario
        print("\n[5/7] Testing your exact failing scenario...")
        results['actual_scenario'] = await self.test_actual_failing_scenario()

        # Test 6: Modified format
        print("\n[6/7] Testing modified prompt format...")
        results['modified_format'] = await self.test_modified_prompt_format()

        # Test 7: Without JSON constraint
        print("\n[7/7] Testing without JSON constraint...")
        results['without_json'] = await self.test_without_json_format_constraint()

        # Generate diagnosis
        print("\n" + "=" * 80)
        print("DIAGNOSIS RESULTS")
        print("=" * 80)

        working = sum(1 for v in results.values() if v)
        total = len(results)

        print(f"\n📊 Summary: {working}/{total} tests passed")

        for test, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {test}")

        # Provide diagnosis
        print("\n🔍 DIAGNOSIS:")

        if not results['gpt_connectivity']:
            print("  ❌ CRITICAL: GPT API not responding")
            print("  Fix: Check API key and endpoint")
        elif not results['json_format']:
            print("  ❌ CRITICAL: JSON format not working")
            print("  Fix: Issue with response_format parameter")
        elif results['simple_criteria'] and not results['with_revenue']:
            print("  ⚠️ Revenue criteria breaking searches")
            print("  Fix: Simplify revenue format in prompt")
        elif results['modified_format'] and not results['actual_scenario']:
            print("  ⚠️ Current prompt format is too complex")
            print("  Fix: Simplify the prompt structure")
        elif results['without_json'] and not results['json_format']:
            print("  ⚠️ JSON constraint causing issues")
            print("  Fix: Remove or modify response_format parameter")
        else:
            print("  ✅ API is working, issue is in prompt construction")

        # Provide immediate fix
        print("\n💊 IMMEDIATE FIX TO TRY:")
        print("-" * 40)
        self.print_immediate_fix()

        return results

    def print_immediate_fix(self):
        """Print the immediate fix to apply"""
        print("""
1. In search_strategist_agent.py, replace the _build_enhanced_prompt method's revenue section (around line 470) with:

```python
# Financial criteria - SIMPLIFIED FORMAT
if criteria.financial.revenue_min or criteria.financial.revenue_max:
    if criteria.financial.revenue_min and criteria.financial.revenue_max:
        min_m = int(criteria.financial.revenue_min / 1_000_000)
        max_m = int(criteria.financial.revenue_max / 1_000_000)
        prompt_parts.append(f"Revenue: {min_m} to {max_m} million {criteria.financial.revenue_currency}")
    elif criteria.financial.revenue_min:
        min_m = int(criteria.financial.revenue_min / 1_000_000)
        prompt_parts.append(f"Revenue: Above {min_m} million {criteria.financial.revenue_currency}")
    else:
        max_m = int(criteria.financial.revenue_max / 1_000_000)
        prompt_parts.append(f"Revenue: Up to {max_m} million {criteria.financial.revenue_currency}")
```

2. Simplify the JSON output format instruction (around line 540):

```python
# Output format - SIMPLIFIED
prompt_parts.append('''

Return exactly 5 companies as JSON:
{
  "companies": [
    {
      "name": "Company Name",
      "confidence": "high",
      "operates_in_country": true,
      "business_type": "B2B",
      "industry_category": "Manufacturing",
      "reasoning": "Matches criteria",
      "estimated_revenue": "50-100M",
      "estimated_employees": "100-500",
      "company_size": "medium",
      "headquarters": {"city": "Melbourne"},
      "office_locations": ["Melbourne"],
      "csr_programs": [],
      "csr_focus_areas": [],
      "certifications": [],
      "recent_events": [],
      "data_sources": []
    }
  ]
}

Each company must have all fields shown above.
''')
```

3. Test with a single search first before running batch searches.
        """)


async def main():
    """Run emergency diagnostics"""
    diagnostics = EmergencyDiagnostics()
    results = await diagnostics.run_emergency_diagnostics()

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("Check failing_prompt.txt and failing_response.json if test 5 failed")
    print("=" * 80)

    return results


if __name__ == "__main__":
    asyncio.run(main())