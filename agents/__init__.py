# agents/__init__.py
"""
Agent modules for company search and validation
"""

# Initialize empty __all__ list
__all__ = []

# Try importing search strategist agent
try:
    from .search_strategist_agent import (
        EnhancedSearchStrategistAgent,
        SearchStrategistAgent  # Backward compatibility
    )
    __all__.extend(['EnhancedSearchStrategistAgent', 'SearchStrategistAgent'])
except ImportError as e:
    print(f"Warning: Could not import search_strategist_agent: {e}")
    EnhancedSearchStrategistAgent = None
    SearchStrategistAgent = None

# Try importing validation agent
try:
    from .validation_agent_v2 import (
        ValidationConfig,
        EnhancedValidationAgent,
        ValidationOrchestrator
    )
    __all__.extend(['ValidationConfig', 'EnhancedValidationAgent', 'ValidationOrchestrator'])
except ImportError as e:
    print(f"Warning: Could not import validation_agent_v2: {e}")
    ValidationConfig = None
    EnhancedValidationAgent = None
    ValidationOrchestrator = None