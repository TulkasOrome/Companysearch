# shared/session_state.py
"""
Centralized session state management
"""

import streamlit as st
from datetime import datetime


def initialize_session_state():
    """Initialize all session state variables with defaults"""

    defaults = {
        # Search criteria
        'current_criteria': None,
        'current_profile_name': None,
        'current_tier': "A",

        # Results storage
        'search_results': [],
        'validation_results': [],
        'selected_companies': [],

        # Profile management
        'saved_profiles': {},

        # Cost tracking
        'total_cost': 0.0,

        # Model configuration
        'selected_models': ["gpt-4.1"],
        'parallel_execution_enabled': False,
        'model_results': {},
        'model_execution_times': {},
        'model_success_status': {},

        # Export configuration
        'export_config': {},

        # UI state
        'current_tab': 0,
        'show_advanced_options': False,

        # Session metadata
        'session_start_time': datetime.now(),
        'last_search_time': None,
        'last_validation_time': None
    }

    # Initialize only missing keys
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_search_results():
    """Clear all search-related results"""
    st.session_state.search_results = []
    st.session_state.model_results = {}
    st.session_state.model_execution_times = {}
    st.session_state.model_success_status = {}
    st.session_state.last_search_time = None


def clear_validation_results():
    """Clear all validation results"""
    st.session_state.validation_results = []
    st.session_state.last_validation_time = None


def clear_session():
    """Clear entire session"""
    for key in ['search_results', 'validation_results', 'current_criteria',
                'model_results', 'model_execution_times', 'model_success_status']:
        if key in st.session_state:
            st.session_state[key] = [] if 'results' in key else {} if 'model_' in key else None
    st.session_state.total_cost = 0.0
    st.session_state.last_search_time = None
    st.session_state.last_validation_time = None


def update_cost(amount: float):
    """Update total cost"""
    st.session_state.total_cost += amount


def get_session_summary():
    """Get summary of current session"""
    return {
        'companies_found': len(st.session_state.search_results),
        'companies_validated': len(st.session_state.validation_results),
        'total_cost': st.session_state.total_cost,
        'session_duration': datetime.now() - st.session_state.session_start_time,
        'parallel_execution': st.session_state.parallel_execution_enabled,
        'models_used': st.session_state.selected_models
    }