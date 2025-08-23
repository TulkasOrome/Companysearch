# main_streamlit_app.py
"""
Main Streamlit Application Entry Point
Coordinates all tabs and manages session state
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import traceback

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import tab modules
from tabs.search_config_tab import render_search_config_tab
from tabs.execute_search_tab import render_execute_search_tab
from tabs.validation_tab import render_validation_tab
from tabs.results_export_tab import render_results_export_tab
from tabs.help_tab import render_help_tab

# Import shared utilities
from shared.session_state import initialize_session_state
from shared.sidebar import render_sidebar
from shared.data_models import ICPManager

# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================

# Page config
st.set_page_config(
    page_title="Company Search & Validation Platform",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
initialize_session_state()

# Initialize ICP manager
icp_manager = ICPManager()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Title
st.title("🔍 Company Search & Validation Platform")
st.markdown("AI-powered company discovery with advanced validation modes and enhanced token handling")

# Render sidebar
serper_key = render_sidebar()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Search Configuration",
    "🔍 Execute Search",
    "✅ Validation",
    "📊 Results & Export",
    "❓ Help"
])

# Tab 1: Search Configuration
with tab1:
    render_search_config_tab(icp_manager)

# Tab 2: Execute Search
with tab2:
    render_execute_search_tab()

# Tab 3: Validation
with tab3:
    render_validation_tab(serper_key)

# Tab 4: Results & Export
with tab4:
    # TODO: This tab was missing implementation - needs to be restored
    # Should include:
    # - Combined results view (search + validation)
    # - Filtering and sorting options
    # - Export to Excel/CSV functionality
    # - Save/load session functionality
    render_results_export_tab()

# Tab 5: Help
with tab5:
    # TODO: This tab was missing implementation - needs to be restored
    # Should include:
    # - User guide
    # - API documentation
    # - Validation mode explanations
    # - ICP profile descriptions
    # - Troubleshooting tips
    render_help_tab()