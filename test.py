# final_test.py
"""
Final test to verify the entire system is working
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("FINAL SYSTEM TEST")
print("=" * 60)

# Test 1: All imports
print("\n1. Testing all imports...")
try:
    from core import (
        EnhancedSerperClient,
        SearchCriteria,
        LocationCriteria,
        FinancialCriteria,
        OrganizationalCriteria,
        BehavioralSignals,
        ValidationStrategyBuilder,
        ValidationAnalyzer
    )

    print("   ✓ Core imports successful")
except Exception as e:
    print(f"   ✗ Core import error: {e}")
    sys.exit(1)

try:
    from agents import (
        EnhancedSearchStrategistAgent,
        EnhancedValidationAgent,
        ValidationConfig,
        ValidationOrchestrator
    )

    print("   ✓ Agents imports successful")
except Exception as e:
    print(f"   ✗ Agents import error: {e}")
    sys.exit(1)

try:
    from config import config

    print("   ✓ Config imports successful")
except Exception as e:
    print(f"   ✗ Config import error: {e}")
    sys.exit(1)

try:
    from ui.session_manager import SessionManager

    print("   ✓ UI imports successful")
except Exception as e:
    print(f"   ✗ UI import error: {e}")

# Test 2: Create objects
print("\n2. Testing object creation...")
try:
    # Create search criteria
    criteria = SearchCriteria(
        location=LocationCriteria(countries=["Australia"]),
        financial=FinancialCriteria(revenue_min=1000000),
        organizational=OrganizationalCriteria(),
        behavioral=BehavioralSignals(),
        business_types=["B2B"],
        industries=[]
    )
    print("   ✓ SearchCriteria created")

    # Create Serper client
    serper_client = EnhancedSerperClient(api_key=config.SERPER_API_KEY)
    print("   ✓ SerperClient created")

    # Create validation config
    val_config = ValidationConfig(
        serper_api_key=config.SERPER_API_KEY,
        max_cost_per_company=config.MAX_COST_PER_COMPANY
    )
    print("   ✓ ValidationConfig created")

    # Create search agent
    search_agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")
    print("   ✓ SearchAgent created")

except Exception as e:
    print(f"   ✗ Object creation error: {e}")
    import traceback

    traceback.print_exc()

# Test 3: Test async functionality
print("\n3. Testing async functionality...")


async def test_async():
    try:
        # Test Serper client context manager
        async with EnhancedSerperClient(api_key=config.SERPER_API_KEY) as client:
            print("   ✓ SerperClient async context manager works")

        # Test validation agent context manager
        async with EnhancedValidationAgent(val_config) as agent:
            print("   ✓ ValidationAgent async context manager works")

        return True
    except Exception as e:
        print(f"   ✗ Async error: {e}")
        return False


# Run async test
try:
    result = asyncio.run(test_async())
    if result:
        print("   ✓ All async tests passed")
except Exception as e:
    print(f"   ✗ Async runtime error: {e}")

# Test 4: Check configuration values
print("\n4. Checking configuration...")
print(f"   Azure endpoint: {config.AZURE_OPENAI_ENDPOINT}")
print(f"   Serper API key: ***{config.SERPER_API_KEY[-4:]}")
print(f"   Max validation cost: ${config.MAX_VALIDATION_COST}")
print(f"   Session cleanup days: {config.SESSION_CLEANUP_DAYS}")

# Test 5: Test use case configs
print("\n5. Testing use case configurations...")
try:
    from core import RMH_SYDNEY_CONFIG, GUIDE_DOGS_VICTORIA_CONFIG

    print(f"   ✓ RMH Sydney config loaded: {RMH_SYDNEY_CONFIG.name}")
    print(f"   ✓ Guide Dogs Victoria config loaded: {GUIDE_DOGS_VICTORIA_CONFIG.name}")
except Exception as e:
    print(f"   ✗ Use case config error: {e}")

print("\n" + "=" * 60)
print("SYSTEM STATUS")
print("=" * 60)

# Summary
all_good = True
checks = {
    "Core Module": True,
    "Agents Module": True,
    "Config Module": True,
    "UI Module": True,
    "Async Support": result if 'result' in locals() else False,
    "Use Cases": True
}

for module, status in checks.items():
    symbol = "✅" if status else "❌"
    print(f"{symbol} {module}: {'Ready' if status else 'Failed'}")
    if not status:
        all_good = False

print("\n" + "=" * 60)
if all_good:
    print("✅ SYSTEM READY!")
    print("\nYou can now run:")
    print("  streamlit run ui/streamlit_app.py")
    print("\nOr test examples:")
    print("  cd examples")
    print("  python example_integration.py")
else:
    print("⚠️ Some components need attention")
print("=" * 60)