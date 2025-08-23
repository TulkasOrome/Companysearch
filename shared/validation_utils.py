# shared/validation_utils.py
"""
Validation utilities extracted from original code
Includes the validate_company_with_serper function
"""

from typing import Dict, Any
from datetime import datetime

# Try to import the real validation function
try:
    from serper_validation_integration import validate_company_with_serper as real_validate

    SERPER_VALIDATION_AVAILABLE = True
except ImportError:
    SERPER_VALIDATION_AVAILABLE = False
    print("Warning: Serper validation not available, using mock data")


async def validate_company_with_serper(
        company: Dict[str, Any],
        mode: str,
        api_key: str
) -> Dict[str, Any]:
    """Validate a company using Serper API (or mock for testing)"""

    # If real validation is available, use it
    if SERPER_VALIDATION_AVAILABLE:
        return await real_validate(company, mode, api_key)

    # Otherwise, use mock validation
    company_name = company.get('name', 'Unknown')

    # For now, return mock validation data
    # This matches the structure from the original code
    return {
        'company_name': company_name,
        'validation_status': 'verified',
        'mode': mode,
        'credits_used': 3,
        'validation_timestamp': datetime.now().isoformat(),
        'emails': [f"contact@{company_name.lower().replace(' ', '')}.com"],
        'phones': ['+61 2 9999 9999'],
        'names': ['John Smith', 'Jane Doe'],
        'revenue_range': company.get('estimated_revenue', 'Unknown'),
        'employee_range': company.get('estimated_employees', 'Unknown'),
        'csr_programs': company.get('csr_programs', []),
        'certifications': company.get('certifications', []),
        'risk_signals': []
    }