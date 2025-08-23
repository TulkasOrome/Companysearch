# shared/sidebar.py
"""
Sidebar component with configuration and stats
"""

import streamlit as st
from shared.session_state import clear_session, get_session_summary


def render_sidebar():
    """Render the sidebar with configuration and stats"""

    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection for parallel execution
        st.subheader("🚀 Search Models")

        num_models = st.slider(
            "Number of Models",
            min_value=1,
            max_value=5,
            value=1,
            help="Select 2-5 models for parallel execution"
        )

        available_models = ["gpt-4.1", "gpt-4.1-2", "gpt-4.1-3", "gpt-4.1-4", "gpt-4.1-5"]

        if num_models == 1:
            selected_models = [st.selectbox("Select Model", available_models)]
            st.session_state.parallel_execution_enabled = False
        else:
            selected_models = st.multiselect(
                "Select Models",
                available_models,
                default=available_models[:num_models],
                max_selections=num_models
            )
            st.session_state.parallel_execution_enabled = True

        st.session_state.selected_models = selected_models

        # Show parallel execution status
        if st.session_state.parallel_execution_enabled:
            st.success(f"⚡ Parallel execution mode with {len(selected_models)} models")
        else:
            st.info(f"Single model execution mode")

        # API Keys
        with st.expander("🔑 API Keys"):
            serper_key = st.text_input(
                "Serper API Key",
                value="99c44b79892f5f7499accf2d7c26d93313880937",
                type="password"
            )

        st.divider()

        # Session stats
        st.subheader("📊 Session Stats")
        summary = get_session_summary()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Companies", summary['companies_found'])
            st.metric("Profiles Saved", len(st.session_state.saved_profiles))
        with col2:
            st.metric("Validated", summary['companies_validated'])
            st.metric("Total Cost", f"${summary['total_cost']:.3f}")

        # Show model performance if available
        if st.session_state.model_success_status:
            st.subheader("🎯 Model Performance")
            for model, status in st.session_state.model_success_status.items():
                if status.get('status') == 'success':
                    st.success(f"{model}: {status.get('companies_found', 0)} companies")
                elif status.get('status') == 'success (retry)':
                    st.warning(f"{model}: {status.get('companies_found', 0)} companies (retry)")
                else:
                    st.error(f"{model}: Failed")

        # Clear session button
        if st.button("🗑️ Clear Session", use_container_width=True):
            clear_session()
            st.rerun()

        return serper_key