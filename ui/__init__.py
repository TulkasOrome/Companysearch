# ui/__init__.py
"""
UI components for the application
"""

__all__ = []

try:
    from .session_manager import SessionManager
    __all__.append('SessionManager')
except ImportError as e:
    print(f"Warning: Could not import session_manager: {e}")
    SessionManager = None