#!/usr/bin/env python3
"""
test_icp_system.py - Test the complete ICP system with both profiles
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent if '__file__' in globals() else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the new ICP manager
from enhanced_icp_manager import ICPManager, get_rmh_criteria, get_guide_dogs_criteria

# Import search agent
from agents.search_strategist_agent import EnhancedSearchStrategistAgent


async def test_icp_system():
    """Test the complete ICP system"""
    print("=" * 70)
    print("ICP SYSTEM TEST")
    print(f"Started: {datetime.now()}")
    print("=" * 70)

    # Initialize components
    manager = ICPManager()
    agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")

    # List available profiles
    print("\n📋 Available ICP Profiles:")
    for profile_name in manager.list_profiles():
        profile = manager.get_profile(profile_name)
        print(f"  - {profile.name}: {profile.description}")

    # Test RMH Sydney
    print("\n" + "=" * 70)
    print("TESTING RMH SYDNEY ICP")
    print("=" * 70)

    # Get criteria using convenience function
    rmh_criteria = get_rmh_criteria("A")

    print("\n🔍 Searching for RMH Sydney Tier A companies...")
    try:
        result = await agent.generate_enhanced_strategy(rmh_criteria, target_count=5)
        companies = result.get('companies', [])

        print(f"✅ Found {len(companies)} companies")

        # Validate each company
        print("\n📊 Validating companies against RMH ICP:")
        for company in companies[:3]:
            passes, details = manager.validate_company_against_icp(
                company.dict() if hasattr(company, 'dict') else company,
                "rmh_sydney",
                "A"
            )

            status = "✅ PASS" if passes else "❌ FAIL"
            print(f"\n{company.name}: {status}")
            print(f"  Score: {details['score']}/100")
            print(f"  Passed: {', '.join(details['passed_checks'][:3])}")
            if details['failed_checks']:
                print(f"  Failed: {', '.join(details['failed_checks'][:2])}")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Test Guide Dogs Victoria
    print("\n" + "=" * 70)
    print("TESTING GUIDE DOGS VICTORIA ICP")
    print("=" * 70)

    # Test both tiers
    for tier in ["A", "B"]:
        print(f"\n🔍 Searching for Guide Dogs Tier {tier} companies...")

        gdv_criteria = get_guide_dogs_criteria(tier)

        try:
            result = await agent.generate_enhanced_strategy(gdv_criteria, target_count=5)
            companies = result.get('companies', [])

            print(f"✅ Found {len(companies)} Tier {tier} companies")

            # Show first few
            for company in companies[:2]:
                passes, details = manager.validate_company_against_icp(
                    company.dict() if hasattr(company, 'dict') else company,
                    "guide_dogs_victoria",
                    tier
                )

                status = "✅" if passes else "⚠️"
                print(f"  {status} {company.name} (Score: {details['score']})")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Test free text with fixed extraction
    print("\n" + "=" * 70)
    print("TESTING FREE TEXT SEARCH (FIXED)")
    print("=" * 70)

    test_query = """
    Find B2B SaaS companies in Melbourne that use AI/ML, 
    have raised Series B+ funding, and have strong diversity 
    and inclusion programs. Revenue between $10M-$100M AUD.
    """

    print(f"Query: {test_query[:80]}...")

    # Extract criteria
    extracted = agent.extract_criteria_from_text(test_query)

    print("\nExtracted criteria:")
    print(f"  Cities: {extracted.get('locations', {}).get('cities', [])}")
    print(
        f"  Revenue: ${extracted.get('financial', {}).get('revenue_min', 0) / 1e6:.0f}M - ${extracted.get('financial', {}).get('revenue_max', 0) / 1e6:.0f}M")
    print(f"  Keywords: {extracted.get('keywords', [])[:3]}")

    # Export profile configurations
    print("\n" + "=" * 70)
    print("PROFILE CONFIGURATIONS")
    print("=" * 70)

    for profile_name in ["rmh_sydney", "guide_dogs_victoria"]:
        config = manager.export_profile(profile_name, "json")
        if config:
            print(f"\n{profile_name.upper()} Configuration:")
            # Just show first few lines
            lines = config.split('\n')
            for line in lines[:10]:
                print(line)
            print("  ...")


if __name__ == "__main__":
    asyncio.run(test_icp_system())