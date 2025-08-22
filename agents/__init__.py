# agents/__init__.py
"""
Agent modules for company search and validation
Fixed version with proper imports
"""

# Initialize empty __all__ list
__all__ = []

# Import search strategist agent with all its classes
try:
    from .search_strategist_agent import (
        EnhancedSearchStrategistAgent,
        SearchStrategistAgent,
        EnhancedCompanyEntry,
        SearchCriteria,
        LocationCriteria,
        FinancialCriteria,
        OrganizationalCriteria,
        BehavioralSignals,
        BusinessType,
        CompanySize
    )
    __all__.extend([
        'EnhancedSearchStrategistAgent',
        'SearchStrategistAgent',
        'EnhancedCompanyEntry',
        'SearchCriteria',
        'LocationCriteria',
        'FinancialCriteria',
        'OrganizationalCriteria',
        'BehavioralSignals',
        'BusinessType',
        'CompanySize'
    ])
except ImportError as e:
    print(f"Warning: Could not import search_strategist_agent: {e}")
    EnhancedSearchStrategistAgent = None
    SearchStrategistAgent = None
    EnhancedCompanyEntry = None
    SearchCriteria = None
    LocationCriteria = None
    FinancialCriteria = None
    OrganizationalCriteria = None
    BehavioralSignals = None
    BusinessType = None
    CompanySize = None

# Import validation agent
try:
    from .validation_agent_v2 import (
        ValidationConfig,
        EnhancedValidationAgent,
        ValidationOrchestrator,
        ValidationMode,
        ValidationCriteria,
        ComprehensiveValidation
    )
    __all__.extend([
        'ValidationConfig',
        'EnhancedValidationAgent',
        'ValidationOrchestrator',
        'ValidationMode',
        'ValidationCriteria',
        'ComprehensiveValidation'
    ])
except ImportError as e:
    print(f"Warning: Could not import validation_agent_v2: {e}")
    ValidationConfig = None
    EnhancedValidationAgent = None
    ValidationOrchestrator = None
    ValidationMode = None
    ValidationCriteria = None
    ComprehensiveValidation = None

# Try importing the original validation agent as fallback
try:
    from .validation_agent import (
        EnhancedValidationMode,
        EnhancedValidationResult,
        EnhancedSerperClient,
        EnhancedValidationAgent as ValidationAgent
    )
    __all__.extend([
        'EnhancedValidationMode',
        'EnhancedValidationResult',
        'EnhancedSerperClient',
        'ValidationAgent'
    ])
except ImportError as e:
    print(f"Warning: Could not import validation_agent: {e}")
    EnhancedValidationMode = None
    EnhancedValidationResult = None
    EnhancedSerperClient = None
    ValidationAgent = None