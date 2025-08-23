# tabs/help_tab.py
"""
Help Tab - Cleaned version without development team contact and known issues
"""

import streamlit as st


def render_help_tab():
    """Render the Help tab"""

    st.header("❓ Help & Documentation")

    # Quick Start Guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        ### Getting Started with Company Search & Validation

        1. **Configure Search Criteria** (Search Configuration tab)
           - Select an ICP Profile (RMH Sydney or Guide Dogs Victoria) or create custom criteria
           - Choose your tier (A, B, or C) for predefined profiles
           - Adjust location, financial, and behavioral criteria as needed

        2. **Execute Search** (Execute Search tab)
           - Choose single or parallel model execution
           - Set target number of companies
           - Click "Execute Search" to find matching companies

        3. **Validate Companies** (Validation tab)
           - Select validation mode based on your needs
           - Choose number of companies to validate
           - Review validation results

        4. **Export Results** (Results & Export tab)
           - Filter and sort results
           - Export to CSV, Excel, or JSON
           - Save session for later continuation
        """)

    # ICP Profiles
    with st.expander("🎯 ICP Profile Descriptions"):
        st.markdown("""
        ### RMH Sydney (Ronald McDonald House Sydney)

        **Tier A - Perfect Match**
        - Revenue: $5-100M AUD
        - Employees: 50+
        - Location: Sydney area (50km radius)
        - CSR Focus: Children, Community
        - Industries: Construction, Property, Hospitality

        **Tier B - Good Match**
        - Revenue: $2-200M AUD
        - Employees: 20+
        - Location: New South Wales
        - Community involvement required

        **Tier C - Potential Match**
        - Revenue: $1M+ AUD
        - Location: Australia
        - Basic criteria only

        ---

        ### Guide Dogs Victoria

        **Tier A - Strategic Partners**
        - Revenue: $500M+ AUD
        - Employees: 500+ (150+ in Victoria)
        - Certifications: B-Corp, ISO 26000
        - CSR Focus: Disability, Inclusion, Health
        - Industries: Health, Financial Services, Technology

        **Tier B - Exploratory Partners**
        - Revenue: $50-500M AUD
        - Employees: 100-500
        - Location: Victoria
        - CSR Focus: Community, Health

        **Tier C - Potential Partners**
        - Revenue: $10M+ AUD
        - Employees: 50+
        - Location: Victoria
        """)

    # Validation Modes
    with st.expander("✅ Validation Mode Explanations"):
        st.markdown("""
        ### Validation Modes

        **Simple Check (2-3 credits)**
        - Quick existence verification
        - Basic location confirmation
        - Phone and address extraction
        - Best for: High-confidence companies needing basic verification

        **Smart Contact Extraction (3-5 credits)**
        - Email addresses extraction
        - Phone numbers
        - Executive names from LinkedIn
        - Contact information aggregation
        - Best for: Building contact lists for outreach

        **Smart CSR Verification (3-5 credits)**
        - CSR program detection
        - Focus area identification
        - Certification verification
        - Foundation existence check
        - Best for: Validating CSR alignment

        **Smart Financial Check (3-4 credits)**
        - Revenue verification
        - Employee count confirmation
        - Stock listing check
        - Growth indicators
        - Best for: Financial qualification

        **Full Validation (10-15 credits)**
        - Comprehensive all-mode validation
        - Risk signal detection
        - Recent news analysis
        - Complete data extraction
        - Best for: High-priority prospects
        """)

    # Parallel Execution
    with st.expander("⚡ Parallel Execution Guide"):
        st.markdown("""
        ### Parallel Model Execution

        **Benefits:**
        - Faster results for large searches
        - Redundancy if one model fails
        - Diverse results from different models
        - Automatic deduplication

        **When to Use:**
        - Searching for 50+ companies
        - Time-sensitive searches
        - When maximum coverage is needed

        **How it Works:**
        1. Select 2-5 models in the sidebar
        2. Target count is distributed across models
        3. Models execute simultaneously
        4. Results are merged and deduplicated
        5. Failed models are automatically retried with smaller batches

        **Token Optimization:**
        - Large searches (>20 companies) automatically use batch execution
        - Each model can handle up to 50 companies per batch
        - Token limits increased to 16,384 for better handling
        """)

    # API Information
    with st.expander("🔑 API Configuration"):
        st.markdown("""
        ### Required APIs

        **Azure OpenAI**
        - Deployment: gpt-4.1 (and variants)
        - Used for company discovery
        - Cost: ~$0.02 per search

        **Serper API**
        - Used for validation
        - Cost: $0.001 per credit
        - Validation modes use 2-15 credits

        ### API Keys

        Default keys are provided for testing. For production use:
        1. Obtain your own API keys
        2. Enter them in the sidebar under "API Keys"
        3. Keys are stored only for the session
        """)

    # Troubleshooting
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        ### Common Issues and Solutions

        **No companies found:**
        - Relax search criteria (reduce minimum requirements)
        - Expand location range
        - Remove strict filters
        - Try different industries

        **Search fails with token error:**
        - Reduce target company count
        - Use single model instead of parallel
        - Simplify search criteria

        **Validation fails:**
        - Check Serper API key is valid
        - Reduce number of companies to validate
        - Try simpler validation mode

        **Export not working:**
        - Ensure results exist before exporting
        - Check browser download settings
        - Try different export format

        **Session restore fails:**
        - Ensure JSON file is not corrupted
        - Check file was saved from this application
        - Try saving a new session
        """)

    # Best Practices
    with st.expander("💡 Best Practices"):
        st.markdown("""
        ### Search Strategy

        1. **Start Broad, Then Narrow**
           - Begin with Tier C criteria
           - Review results
           - Apply stricter filters

        2. **Use ICP Profiles**
           - Leverage predefined profiles for consistency
           - Modify tiers as needed
           - Save successful criteria combinations

        3. **Validation Strategy**
           - Validate Tier A companies with Full mode
           - Use Smart modes for specific data needs
           - Simple check for large batches

        4. **Cost Optimization**
           - Use parallel execution for large searches
           - Choose appropriate validation depth
           - Export and save sessions to avoid re-running

        5. **Data Quality**
           - Review ICP scores for relevance
           - Check validation status before outreach
           - Monitor risk signals
        """)

    # Version and Updates
    st.divider()
    st.markdown("""
    ### 📌 Version Information

    **Current Version:** 2.1.0
    - Enhanced token handling (16,384 tokens)
    - Parallel model execution
    - Batch processing for large searches (up to 10,000 companies)
    - Comprehensive validation modes
    - Session management
    - Search history tracking
    - Add to existing results functionality

    **Recent Updates:**
    - Added number input for precise company count
    - Simplified UI with compact metrics
    - Validation approach in collapsible accordion
    - Support for appending search results with deduplication
    - Search history tracking
    """)