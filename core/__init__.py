# core/__init__.py
"""
Core modules for company search and validation system
"""

from .data_models import (
    # Enums
    BusinessType,
    CompanySize,
    ConfidenceLevel,
    ICPTier,
    ValidationStatus,
    ESGMaturity,

    # Location Models
    LocationCriteria,
    Address,

    # Financial Models
    FinancialCriteria,
    FinancialData,

    # Organizational Models
    OrganizationalCriteria,
    OrganizationalData,

    # CSR/ESG Models
    BehavioralSignals,
    CSRData,

    # Search Criteria
    SearchCriteria,

    # Company Models
    CompanyBase,
    CompanyLocation,
    CompanyFinancials,
    CompanyCSR,
    CompanySignals,
    CompanyICP,
    CompanyMetadata,
    EnhancedCompanyEntry,

    # Validation Models
    ValidationResult,

    # Use Case Models
    UseCaseConfig,
    RMH_SYDNEY_CONFIG,
    GUIDE_DOGS_VICTORIA_CONFIG,

    # Helper Functions
    company_to_dict,
    dict_to_company,
    calculate_company_size
)

from .serper_client import (
    SerperEndpoint,
    SerperQuery,
    SerperResponse,
    SerperRateLimiter,
    EnhancedSerperClient
)

from .validation_strategies import (
    ValidationTier,
    ValidationPriority,
    ValidationStrategy,
    ValidationCriteria,
    ValidationStrategyBuilder,
    ValidationQueryOptimizer
)

from .validation_analyzer import (
    ValidationEvidence,
    LocationValidation,
    FinancialValidation,
    CSRValidation,
    ReputationValidation,
    ComprehensiveValidation,
    ValidationAnalyzer
)

__all__ = [
    # Data Models
    'BusinessType',
    'CompanySize',
    'ConfidenceLevel',
    'ICPTier',
    'ValidationStatus',
    'ESGMaturity',
    'SearchCriteria',
    'LocationCriteria',
    'FinancialCriteria',
    'OrganizationalCriteria',
    'BehavioralSignals',
    'EnhancedCompanyEntry',
    'RMH_SYDNEY_CONFIG',
    'GUIDE_DOGS_VICTORIA_CONFIG',

    # Serper Client
    'EnhancedSerperClient',
    'SerperQuery',
    'SerperResponse',
    'SerperEndpoint',

    # Validation
    'ValidationTier',
    'ValidationStrategy',
    'ValidationCriteria',
    'ValidationStrategyBuilder',
    'ValidationAnalyzer',
    'ComprehensiveValidation',
    'LocationValidation',
    'FinancialValidation',
    'CSRValidation',
    'ReputationValidation',

    # Helper functions
    'company_to_dict',
    'dict_to_company',
    'calculate_company_size'
]