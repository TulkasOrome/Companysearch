# session_manager.py

import streamlit as st
import json
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid
import os


class SessionManager:
    """Comprehensive session management for Streamlit app"""

    def __init__(self, session_dir: str = "sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self._initialize_session()

    def _initialize_session(self):
        """Initialize session state with default values"""
        defaults = {
            'session_id': self._generate_session_id(),
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'current_step': 'search_config',
            'search_criteria': {},
            'search_results': [],
            'validation_results': [],
            'selected_companies': [],
            'export_config': {},
            'steps_completed': {
                'search_config': False,
                'search_execution': False,
                'validation': False,
                'export': False
            },
            'search_history': [],
            'total_cost': 0.0,
            'api_calls': {
                'gpt4': 0,
                'serper': 0
            }
        }

        # Initialize only missing keys
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        # Check for resume parameter in URL
        self._check_resume_link()

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    def _check_resume_link(self):
        """Check if there's a resume link in URL parameters"""
        params = st.query_params
        if 'resume' in params:
            try:
                encoded_data = params['resume']
                resume_data = json.loads(base64.urlsafe_b64decode(encoded_data).decode())

                # Load session data
                if 'session_file' in resume_data:
                    self.load_session_from_file(resume_data['session_file'])
                elif 'session_data' in resume_data:
                    self._restore_session_data(resume_data['session_data'])

                # Clear the URL parameter
                st.query_params.clear()
            except Exception as e:
                st.error(f"Failed to resume session: {str(e)}")

    def save_progress(self, step: str, data: Any):
        """Save progress for a specific step"""
        st.session_state[step] = data
        st.session_state.steps_completed[step] = True
        st.session_state.last_activity = datetime.now()

        # Auto-save to file
        self._autosave()

    def get_progress(self, step: str) -> Any:
        """Get saved progress for a step"""
        return st.session_state.get(step, None)

    def update_search_criteria(self, criteria: Dict[str, Any]):
        """Update and save search criteria"""
        st.session_state.search_criteria = criteria
        st.session_state.last_activity = datetime.now()
        self._autosave()

    def add_search_results(self, results: List[Dict[str, Any]]):
        """Add search results to session"""
        st.session_state.search_results.extend(results)
        st.session_state.steps_completed['search_execution'] = True
        st.session_state.last_activity = datetime.now()
        self._autosave()

    def add_validation_results(self, results: List[Any]):
        """Add validation results to session"""
        st.session_state.validation_results = results
        st.session_state.steps_completed['validation'] = True
        st.session_state.last_activity = datetime.now()
        self._autosave()

    def update_api_usage(self, api_type: str, count: int = 1, cost: float = 0.0):
        """Update API usage tracking"""
        if api_type in st.session_state.api_calls:
            st.session_state.api_calls[api_type] += count
        st.session_state.total_cost += cost

    def _autosave(self):
        """Automatically save session to file"""
        try:
            session_data = self.export_session_data()
            file_path = self.session_dir / f"{st.session_state.session_id}.json"

            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)

            # Clean up old sessions
            self._cleanup_old_sessions()
        except Exception as e:
            print(f"Autosave failed: {e}")

    def _cleanup_old_sessions(self, days: int = 7):
        """Remove session files older than specified days"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            for file_path in self.session_dir.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_time.timestamp():
                    file_path.unlink()
        except Exception as e:
            print(f"Cleanup failed: {e}")

    def export_session_data(self) -> Dict[str, Any]:
        """Export current session data"""
        return {
            'session_id': st.session_state.session_id,
            'created_at': st.session_state.created_at.isoformat(),
            'last_activity': st.session_state.last_activity.isoformat(),
            'current_step': st.session_state.current_step,
            'search_criteria': st.session_state.search_criteria,
            'search_results_count': len(st.session_state.search_results),
            'search_results': st.session_state.search_results[:100],  # Limit for file size
            'validation_results_count': len(st.session_state.validation_results),
            'steps_completed': st.session_state.steps_completed,
            'total_cost': st.session_state.total_cost,
            'api_calls': st.session_state.api_calls
        }

    def generate_resume_link(self) -> str:
        """Generate a shareable resume link"""
        # Save current session
        self._autosave()

        # Create resume data
        resume_data = {
            'session_file': f"{st.session_state.session_id}.json",
            'step': st.session_state.current_step
        }

        # Encode data
        encoded = base64.urlsafe_b64encode(
            json.dumps(resume_data).encode()
        ).decode()

        # Get current URL
        return f"?resume={encoded}"

    def generate_compact_resume_link(self) -> str:
        """Generate a compact resume link with minimal data"""
        resume_data = {
            'session_data': {
                'search_criteria': st.session_state.search_criteria,
                'current_step': st.session_state.current_step,
                'steps_completed': st.session_state.steps_completed
            }
        }

        encoded = base64.urlsafe_b64encode(
            json.dumps(resume_data).encode()
        ).decode()

        return f"?resume={encoded}"

    def save_session_to_file(self, filename: str = None) -> str:
        """Save session to a specific file"""
        if not filename:
            filename = f"{st.session_state.session_id}_export.json"

        session_data = {
            'session_id': st.session_state.session_id,
            'export_time': datetime.now().isoformat(),
            'search_criteria': st.session_state.search_criteria,
            'search_results': st.session_state.search_results,
            'validation_results': [
                result.dict() if hasattr(result, 'dict') else result
                for result in st.session_state.validation_results
            ],
            'selected_companies': st.session_state.selected_companies,
            'export_config': st.session_state.export_config,
            'steps_completed': st.session_state.steps_completed,
            'total_cost': st.session_state.total_cost,
            'api_calls': st.session_state.api_calls
        }

        return json.dumps(session_data, indent=2, default=str)

    def load_session_from_file(self, filename: str):
        """Load session from file"""
        try:
            file_path = self.session_dir / filename
            if not file_path.exists():
                # Try without directory
                file_path = Path(filename)

            with open(file_path, 'r') as f:
                session_data = json.load(f)

            self._restore_session_data(session_data)
            st.success(f"Session loaded successfully!")
            return True
        except Exception as e:
            st.error(f"Failed to load session: {str(e)}")
            return False

    def _restore_session_data(self, session_data: Dict[str, Any]):
        """Restore session data to session state"""
        # Restore key session data
        if 'search_criteria' in session_data:
            st.session_state.search_criteria = session_data['search_criteria']
        if 'search_results' in session_data:
            st.session_state.search_results = session_data['search_results']
        if 'validation_results' in session_data:
            st.session_state.validation_results = session_data['validation_results']
        if 'steps_completed' in session_data:
            st.session_state.steps_completed = session_data['steps_completed']
        if 'current_step' in session_data:
            st.session_state.current_step = session_data['current_step']
        if 'total_cost' in session_data:
            st.session_state.total_cost = session_data['total_cost']
        if 'api_calls' in session_data:
            st.session_state.api_calls = session_data['api_calls']

        # Update activity timestamp
        st.session_state.last_activity = datetime.now()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        return {
            'session_id': st.session_state.session_id,
            'duration': str(datetime.now() - st.session_state.created_at),
            'current_step': st.session_state.current_step,
            'progress': sum(st.session_state.steps_completed.values()) / len(st.session_state.steps_completed),
            'companies_found': len(st.session_state.search_results),
            'companies_validated': len(st.session_state.validation_results),
            'total_cost': st.session_state.total_cost,
            'api_calls': st.session_state.api_calls
        }

    def clear_session(self):
        """Clear current session data"""
        # Keep session ID but clear data
        session_id = st.session_state.session_id
        for key in list(st.session_state.keys()):
            if key != 'session_id':
                del st.session_state[key]

        # Re-initialize
        self._initialize_session()
        st.session_state.session_id = session_id

    def get_recent_sessions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get list of recent sessions"""
        sessions = []

        try:
            # Get all session files
            session_files = sorted(
                self.session_dir.glob("*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            for file_path in session_files[:limit]:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        sessions.append({
                            'filename': file_path.name,
                            'session_id': data.get('session_id', 'Unknown'),
                            'created_at': data.get('created_at', 'Unknown'),
                            'last_activity': data.get('last_activity', 'Unknown'),
                            'companies_found': data.get('search_results_count', 0)
                        })
                except:
                    continue

        except Exception as e:
            print(f"Error getting recent sessions: {e}")

        return sessions