# run_app.py
"""
Launch script for the Company Search & Validation System
Run this from the root directory
"""

import sys
import os
from pathlib import Path


def main():
    """Main function to run the application"""
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import and run streamlit
    import streamlit.web.cli as stcli

    # Run streamlit app with proper path
    sys.argv = [
        "streamlit",
        "run",
        str(project_root / "ui" / "streamlit_app.py"),
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=false"
    ]

    sys.exit(stcli.main())


if __name__ == "__main__":
    main()