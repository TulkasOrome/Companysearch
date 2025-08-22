# streamlit_app.py
"""
Enhanced Streamlit App with Parallel Model Execution and Improved Token Handling
Updated version with batch execution and token limit management
"""

import streamlit as st
import asyncio
import pandas as pd
import json
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import traceback
import concurrent.futures
from dataclasses import dataclass, asdict, field
from enum import Enum
from io import BytesIO
import time
import re
import os
from pydantic import BaseModel, Field

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))


# ============================================================================
# DEFINE CORE CLASSES LOCALLY TO AVOID IMPORT ISSUES
# ============================================================================

class BusinessType(Enum):
    """Business model types"""
    B2C = "B2C"
    B2B = "B2B"
    B2B2C = "B2B2C"
    D2C = "D2C"
    MARKETPLACE = "Marketplace"
    HYBRID = "Hybrid"
    PROFESSIONAL_SERVICES = "Professional Services"
    REAL_ESTATE = "Real Estate"
    SAAS = "SaaS"
    ENTERPRISE = "Enterprise"


class CompanySize(Enum):
    """Company size classifications"""
    SMALL = "small"
    MEDIUM = "medium"
    ENTERPRISE = "enterprise"
    UNKNOWN = "unknown"


@dataclass
class LocationCriteria:
    """Location search criteria"""
    countries: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    cities: List[str] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    proximity: Optional[Dict[str, Any]] = None
    exclusions: List[str] = field(default_factory=list)


@dataclass
class FinancialCriteria:
    """Financial search criteria"""
    revenue_min: Optional[float] = None
    revenue_max: Optional[float] = None
    revenue_currency: str = "USD"
    giving_capacity_min: Optional[float] = None
    growth_rate_min: Optional[float] = None
    profitable: Optional[bool] = None
    funding_stage: Optional[str] = None


@dataclass
class OrganizationalCriteria:
    """Organizational search criteria"""
    employee_count_min: Optional[int] = None
    employee_count_max: Optional[int] = None
    employee_count_by_location: Optional[Dict[str, int]] = None
    office_types: List[str] = field(default_factory=list)
    company_stage: Optional[str] = None
    years_in_business_min: Optional[int] = None


@dataclass
class BehavioralSignals:
    """Behavioral and CSR signals"""
    csr_programs: List[str] = field(default_factory=list)
    csr_focus_areas: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    recent_events: List[str] = field(default_factory=list)
    technology_stack: List[str] = field(default_factory=list)
    esg_maturity: Optional[str] = None
    partnerships: List[str] = field(default_factory=list)


@dataclass
class SearchCriteria:
    """Complete search criteria"""
    location: LocationCriteria
    financial: FinancialCriteria
    organizational: OrganizationalCriteria
    behavioral: BehavioralSignals
    business_types: List[str] = field(default_factory=list)
    industries: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    custom_prompt: Optional[str] = None
    excluded_industries: List[str] = field(default_factory=list)
    excluded_companies: List[str] = field(default_factory=list)
    excluded_behaviors: List[str] = field(default_factory=list)


class EnhancedCompanyEntry(BaseModel):
    """Complete enhanced company entry with all fields"""
    # Basic info
    name: str = Field(description="Company name")
    confidence: str = Field(description="Confidence level")
    operates_in_country: bool = Field(description="Whether company operates in the specified country")
    business_type: str = Field(description="Type of business")
    industry_category: str = Field(description="Industry category")
    sub_industry: Optional[str] = Field(default=None)
    reasoning: str = Field(description="Brief reasoning for confidence level")

    # Geographic footprint
    headquarters: Optional[Dict[str, Any]] = Field(default=None)
    office_locations: List[Any] = Field(default_factory=list)
    service_areas: List[str] = Field(default_factory=list)

    # Financial profile
    estimated_revenue: Optional[str] = Field(default=None)
    revenue_currency: Optional[str] = Field(default="USD")
    estimated_employees: Optional[str] = Field(default=None)
    employees_by_location: Optional[Dict[str, str]] = Field(default=None)
    company_size: Optional[str] = Field(default="unknown")
    giving_history: List[Dict[str, Any]] = Field(default_factory=list)
    financial_health: Optional[str] = Field(default=None)

    # CSR/ESG Profile
    csr_programs: List[str] = Field(default_factory=list)
    csr_focus_areas: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    esg_score: Optional[float] = Field(default=None)
    esg_maturity: Optional[str] = Field(default=None)
    community_involvement: List[str] = Field(default_factory=list)

    # Signals and triggers
    recent_events: List[Any] = Field(default_factory=list)
    leadership_changes: List[Dict[str, Any]] = Field(default_factory=list)
    growth_signals: List[str] = Field(default_factory=list)

    # ICP Matching
    icp_tier: Optional[str] = Field(default=None)
    icp_score: Optional[float] = Field(default=None)
    matched_criteria: List[str] = Field(default_factory=list)
    missing_criteria: List[str] = Field(default_factory=list)

    # Data quality
    data_freshness: Optional[str] = Field(default=None)
    data_sources: List[str] = Field(default_factory=list)
    validation_notes: Optional[str] = Field(default=None)

    # Parallel execution tracking
    source_model: Optional[str] = Field(default=None)  # Track which model generated this company


# ============================================================================
# ICP PROFILES MANAGER
# ============================================================================

class ICPProfile:
    """ICP Profile definition"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tiers = {}

    def add_tier(self, tier: str, criteria: SearchCriteria):
        self.tiers[tier] = criteria


class ICPManager:
    """Manages ICP profiles"""

    def __init__(self):
        self.profiles = {}
        self._initialize_profiles()

    def _initialize_profiles(self):
        """Initialize RMH Sydney and Guide Dogs Victoria profiles"""

        # RMH Sydney Profile
        rmh = ICPProfile("rmh_sydney", "Ronald McDonald House Sydney")

        # Tier A
        rmh.add_tier("A", SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                cities=["Sydney"],
                proximity={"location": "Greater Western Sydney", "radius_km": 50}
            ),
            financial=FinancialCriteria(
                revenue_min=5_000_000,
                revenue_max=100_000_000,
                revenue_currency="AUD",
                giving_capacity_min=20_000
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=50,
                office_types=["Headquarters", "Major Office"]
            ),
            behavioral=BehavioralSignals(
                csr_focus_areas=["children", "community"],
                recent_events=["Office Move", "CSR Launch"]
            ),
            business_types=["B2B", "B2C"],
            industries=[
                {"name": "Construction", "priority": 1},
                {"name": "Property", "priority": 2},
                {"name": "Hospitality", "priority": 3}
            ],
            excluded_industries=["Fast Food", "Gambling"]
        ))

        # Tier B
        rmh.add_tier("B", SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                states=["New South Wales"]
            ),
            financial=FinancialCriteria(
                revenue_min=2_000_000,
                revenue_max=200_000_000,
                revenue_currency="AUD"
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=20
            ),
            behavioral=BehavioralSignals(
                csr_focus_areas=["community"]
            ),
            business_types=["B2B", "B2C"],
            industries=[{"name": "Any", "priority": 1}],
            excluded_industries=["Fast Food", "Gambling"]
        ))

        # Tier C
        rmh.add_tier("C", SearchCriteria(
            location=LocationCriteria(countries=["Australia"]),
            financial=FinancialCriteria(revenue_min=1_000_000, revenue_currency="AUD"),
            organizational=OrganizationalCriteria(),
            behavioral=BehavioralSignals(),
            business_types=["B2B", "B2C"],
            industries=[],
            excluded_industries=["Fast Food", "Gambling"]
        ))

        self.profiles["rmh_sydney"] = rmh

        # Guide Dogs Victoria Profile
        gdv = ICPProfile("guide_dogs_victoria", "Guide Dogs Victoria")

        # Tier A
        gdv.add_tier("A", SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                states=["Victoria"],
                cities=["Melbourne", "Geelong", "Ballarat", "Bendigo"]
            ),
            financial=FinancialCriteria(
                revenue_min=500_000_000,
                revenue_currency="AUD"
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=500,
                employee_count_by_location={"Victoria": 150}
            ),
            behavioral=BehavioralSignals(
                certifications=["B-Corp", "ISO 26000"],
                csr_focus_areas=["disability", "inclusion", "health"],
                esg_maturity="Mature"
            ),
            business_types=["B2B", "B2C"],
            industries=[
                {"name": "Health", "priority": 1},
                {"name": "Financial Services", "priority": 2},
                {"name": "Technology", "priority": 3}
            ],
            excluded_industries=["Gambling", "Tobacco", "Racing"]
        ))

        # Tier B
        gdv.add_tier("B", SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                states=["Victoria"]
            ),
            financial=FinancialCriteria(
                revenue_min=50_000_000,
                revenue_max=500_000_000,
                revenue_currency="AUD"
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=100,
                employee_count_max=500
            ),
            behavioral=BehavioralSignals(
                csr_focus_areas=["community", "health"]
            ),
            business_types=["B2B", "B2C"],
            industries=[{"name": "Any", "priority": 1}],
            excluded_industries=["Gambling", "Tobacco"]
        ))

        # Tier C
        gdv.add_tier("C", SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                states=["Victoria"]
            ),
            financial=FinancialCriteria(
                revenue_min=10_000_000,
                revenue_currency="AUD"
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=50
            ),
            behavioral=BehavioralSignals(),
            business_types=["B2B", "B2C"],
            industries=[],
            excluded_industries=["Gambling", "Tobacco"]
        ))

        self.profiles["guide_dogs_victoria"] = gdv

    def get_profile(self, name: str) -> Optional[ICPProfile]:
        return self.profiles.get(name)


# ============================================================================
# IMPORT OR DEFINE THE ENHANCED SEARCH AGENT
# ============================================================================

# Try to import the enhanced agent
try:
    from agents.search_strategist_agent import EnhancedSearchStrategistAgent
except ImportError:
    # If import fails, use the local definition
    class EnhancedSearchStrategistAgent:
        """Enhanced agent with comprehensive search capabilities"""

        def __init__(self, deployment_name: str = "gpt-4.1"):
            self.deployment_name = deployment_name
            self.client = None
            self.initialized = False

        def _init_llm(self):
            """Initialize the LLM with Azure OpenAI"""
            if self.initialized:
                return

            try:
                from openai import AzureOpenAI

                api_key = os.getenv("AZURE_OPENAI_KEY",
                                    "CUxPxhxqutsvRVHmGQcmH59oMim6mu55PjHTjSpM6y9UwIxwVZIuJQQJ99BFACL93NaXJ3w3AAABACOG3kI1")
                api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://amex-openai-2025.openai.azure.com/")

                self.client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint
                )

                self.initialized = True
                print(f"Successfully initialized Azure OpenAI client with deployment: {self.deployment_name}")

            except Exception as e:
                print(f"Error initializing Azure OpenAI: {str(e)}")
                raise

        async def generate_enhanced_strategy(
                self,
                criteria: SearchCriteria,
                target_count: int = 100
        ) -> Dict[str, Any]:
            """Generate search strategy with enhanced criteria"""
            if not self.client:
                self._init_llm()

            prompt = self._build_enhanced_prompt(criteria, target_count)

            try:
                # ENHANCED: Increased token limit
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a company finder. Respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=16384,  # Increased from 4000
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                result = json.loads(content)

                enhanced_companies = []
                for company_data in result.get("companies", []):
                    try:
                        company_data = self._ensure_company_fields(company_data)
                        # Add source model tracking
                        company_data['source_model'] = self.deployment_name
                        company = EnhancedCompanyEntry(**company_data)
                        company = self._calculate_icp_score(company, criteria)
                        enhanced_companies.append(company)
                    except Exception as e:
                        print(f"Error processing company: {e}")
                        continue

                return {
                    "companies": enhanced_companies,
                    "search_criteria": asdict(criteria) if hasattr(criteria, '__dict__') else criteria,
                    "metadata": {
                        "total_found": len(enhanced_companies),
                        "timestamp": datetime.now().isoformat(),
                        "deployment": self.deployment_name,
                        "batch_execution": target_count > 20  # Track if batch would be used
                    }
                }

            except Exception as e:
                print(f"Search error in {self.deployment_name}: {e}")
                return {"companies": [], "error": str(e), "deployment": self.deployment_name}

        def _ensure_company_fields(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
            """Ensure all required fields exist in company data"""
            defaults = {
                "name": "Unknown Company",
                "confidence": "low",
                "operates_in_country": True,
                "business_type": "Unknown",
                "industry_category": "Unknown",
                "reasoning": "No reasoning provided",
                "sub_industry": None,
                "headquarters": None,
                "office_locations": [],
                "service_areas": [],
                "estimated_revenue": None,
                "revenue_currency": "USD",
                "estimated_employees": None,
                "employees_by_location": None,
                "company_size": "unknown",
                "giving_history": [],
                "financial_health": None,
                "csr_programs": [],
                "csr_focus_areas": [],
                "certifications": [],
                "esg_score": None,
                "esg_maturity": None,
                "community_involvement": [],
                "recent_events": [],
                "leadership_changes": [],
                "growth_signals": [],
                "icp_tier": None,
                "icp_score": None,
                "matched_criteria": [],
                "missing_criteria": [],
                "data_freshness": None,
                "data_sources": [],
                "validation_notes": None,
                "source_model": None
            }

            for key, default_value in defaults.items():
                if key not in company_data:
                    company_data[key] = default_value

            return company_data

        def _build_enhanced_prompt(self, criteria: SearchCriteria, target_count: int) -> str:
            """Build comprehensive prompt from criteria - OPTIMIZED FOR TOKENS"""
            prompt_parts = []

            # More concise prompt to reduce token usage
            prompt_parts.append(f"Find {target_count} companies matching:")

            # Location - concise
            if criteria.location.countries or criteria.location.cities:
                locations = []
                if criteria.location.countries:
                    locations.append(f"Countries: {', '.join(criteria.location.countries[:3])}")
                if criteria.location.cities:
                    locations.append(f"Cities: {', '.join(criteria.location.cities[:5])}")
                prompt_parts.append("LOCATION: " + "; ".join(locations))

            # Financial - concise
            if criteria.financial.revenue_min or criteria.financial.revenue_max:
                if criteria.financial.revenue_min and criteria.financial.revenue_max:
                    prompt_parts.append(
                        f"REVENUE: ${int(criteria.financial.revenue_min / 1e6)}-{int(criteria.financial.revenue_max / 1e6)}M {criteria.financial.revenue_currency}")
                elif criteria.financial.revenue_min:
                    prompt_parts.append(
                        f"REVENUE: >${int(criteria.financial.revenue_min / 1e6)}M {criteria.financial.revenue_currency}")

            # Employees - concise
            if criteria.organizational.employee_count_min:
                prompt_parts.append(f"EMPLOYEES: {criteria.organizational.employee_count_min}+")

            # Industries - limit to top 3
            if criteria.industries:
                ind_names = [ind['name'] for ind in criteria.industries[:3]]
                prompt_parts.append(f"INDUSTRIES: {', '.join(ind_names)}")

            # Business types
            if criteria.business_types:
                prompt_parts.append(f"TYPES: {', '.join(criteria.business_types[:3])}")

            # CSR - concise
            if criteria.behavioral.csr_focus_areas:
                prompt_parts.append(f"CSR: {', '.join(criteria.behavioral.csr_focus_areas[:3])}")

            # JSON format - minimal fields to reduce response size
            prompt_parts.append("\nReturn JSON:")
            prompt_parts.append("""{"companies":[{
"name":"Company Name",
"confidence":"high",
"operates_in_country":true,
"business_type":"B2B",
"industry_category":"Industry",
"reasoning":"Brief reason",
"estimated_revenue":"50-100M",
"estimated_employees":"100-500",
"headquarters":{"city":"City"},
"csr_focus_areas":[]
}]}""")

            return "\n".join(prompt_parts)

        def _calculate_icp_score(self, company: EnhancedCompanyEntry, criteria: SearchCriteria) -> EnhancedCompanyEntry:
            """Calculate ICP score and tier for a company"""
            score = 0
            max_score = 100

            # Simple scoring logic
            if company.operates_in_country:
                score += 20
            if company.estimated_revenue:
                score += 20
            if company.estimated_employees:
                score += 15
            if company.business_type in criteria.business_types:
                score += 15
            if company.csr_programs or company.csr_focus_areas:
                score += 15
            if company.confidence in ["high", "absolute"]:
                score += 15

            # Determine tier
            if score >= 80:
                tier = "A"
            elif score >= 60:
                tier = "B"
            elif score >= 40:
                tier = "C"
            else:
                tier = "D"

            company.icp_score = score
            company.icp_tier = tier

            return company


# ============================================================================
# ENHANCED PARALLEL EXECUTION HELPER FUNCTIONS
# ============================================================================

async def execute_parallel_search(
        models: List[str],
        criteria: SearchCriteria,
        target_count: int
) -> Dict[str, Any]:
    """Execute search across multiple models in parallel - ENHANCED WITH BATCH HANDLING"""

    # Calculate per-model target with better distribution
    # For large counts, each model gets a reasonable batch
    if target_count > 100:
        # For large searches, limit each model to reasonable batch sizes
        per_model_count = min(50, target_count // len(models))
        remainder = target_count - (per_model_count * len(models))
    else:
        per_model_count = max(10, target_count // len(models))
        remainder = target_count % len(models)

    # Create tasks for each model
    tasks = []
    model_targets = {}

    for i, model in enumerate(models):
        # Distribute remainder across first models
        model_target = per_model_count + (remainder // len(models)) + (1 if i < (remainder % len(models)) else 0)
        model_targets[model] = model_target

        # Create agent and task
        agent = EnhancedSearchStrategistAgent(deployment_name=model)
        task = agent.generate_enhanced_strategy(criteria, target_count=model_target)
        tasks.append(task)

    # Execute all tasks in parallel with better error handling
    start_time = time.time()

    # Use gather with return_exceptions to handle failures gracefully
    results = await asyncio.gather(*tasks, return_exceptions=True)

    execution_time = time.time() - start_time

    # Process results with enhanced error handling
    all_companies = []
    model_stats = {}
    successful_models = []
    failed_models = []
    total_attempts = len(models)
    retry_models = []

    for model, result in zip(models, results):
        if isinstance(result, Exception):
            # Handle failed model
            failed_models.append(model)
            model_stats[model] = {
                'status': 'failed',
                'error': str(result),
                'companies_found': 0
            }
            retry_models.append(model)

        elif isinstance(result, dict):
            if 'error' in result:
                # Model returned error
                failed_models.append(model)
                model_stats[model] = {
                    'status': 'error',
                    'error': result.get('error'),
                    'companies_found': 0
                }
                retry_models.append(model)
            else:
                # Success
                companies = result.get('companies', [])
                all_companies.extend(companies)
                successful_models.append(model)
                model_stats[model] = {
                    'status': 'success',
                    'companies_found': len(companies),
                    'target': model_targets[model],
                    'batch_execution': result.get('metadata', {}).get('batch_execution', False)
                }

    # Retry failed models with smaller batch sizes if needed
    if retry_models and len(all_companies) < target_count * 0.5:
        print(f"Retrying {len(retry_models)} failed models with smaller batches...")

        retry_tasks = []
        for model in retry_models[:2]:  # Limit retries to avoid long delays
            agent = EnhancedSearchStrategistAgent(deployment_name=model)
            # Try with much smaller batch
            retry_task = agent.generate_enhanced_strategy(criteria, target_count=10)
            retry_tasks.append((model, retry_task))

        if retry_tasks:
            retry_results = await asyncio.gather(*[task for _, task in retry_tasks], return_exceptions=True)

            for (model, _), retry_result in zip(retry_tasks, retry_results):
                if not isinstance(retry_result, Exception) and isinstance(retry_result, dict):
                    if 'companies' in retry_result and retry_result['companies']:
                        companies = retry_result['companies']
                        all_companies.extend(companies)

                        # Update model stats
                        if model in failed_models:
                            failed_models.remove(model)
                            successful_models.append(model)

                        model_stats[model] = {
                            'status': 'success (retry)',
                            'companies_found': len(companies),
                            'target': 10,
                            'batch_execution': True
                        }

    # Deduplicate companies by name with better logic
    seen_names = set()
    seen_name_variations = {}  # Track variations of company names
    unique_companies = []
    duplicates_removed = 0

    for company in all_companies:
        company_name = company.name.lower().strip()

        # Remove common suffixes for deduplication
        name_core = company_name
        for suffix in ['pty ltd', 'limited', 'ltd', 'inc', 'corporation', 'corp', 'llc', 'plc']:
            name_core = name_core.replace(suffix, '').strip()

        if name_core not in seen_names and company_name not in seen_name_variations.values():
            seen_names.add(name_core)
            seen_name_variations[name_core] = company_name
            unique_companies.append(company)
        else:
            duplicates_removed += 1

    # Calculate success metrics
    success_rate = (len(successful_models) / total_attempts * 100) if total_attempts > 0 else 0
    companies_per_second = len(all_companies) / execution_time if execution_time > 0 else 0

    return {
        'companies': unique_companies,
        'metadata': {
            'parallel_execution': True,
            'models_used': models,
            'successful_models': successful_models,
            'failed_models': failed_models,
            'model_stats': model_stats,
            'total_companies_before_dedup': len(all_companies),
            'total_companies_after_dedup': len(unique_companies),
            'duplicates_removed': duplicates_removed,
            'execution_time': execution_time,
            'success_rate': success_rate,
            'companies_per_second': companies_per_second,
            'batch_execution_used': any(stats.get('batch_execution', False) for stats in model_stats.values()),
            'timestamp': datetime.now().isoformat()
        }
    }


# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

async def validate_company_with_serper(
        company: Dict[str, Any],
        mode: str,
        api_key: str
) -> Dict[str, Any]:
    """Validate a company using Serper API (or mock for testing)"""

    company_name = company.get('name', 'Unknown')

    # For now, return mock validation data
    # Replace this with actual Serper validation when ready
    return {
        'company_name': company_name,
        'validation_status': 'verified',
        'mode': mode,
        'credits_used': 3,
        'validation_timestamp': datetime.now().isoformat(),
        'emails': [f"contact@{company_name.lower().replace(' ', '')}.com"],
        'phones': ['+61 2 9999 9999'],
        'names': ['John Smith', 'Jane Doe'],
        'revenue_range': company.get('estimated_revenue', 'Unknown'),
        'employee_range': company.get('estimated_employees', 'Unknown'),
        'csr_programs': company.get('csr_programs', []),
        'certifications': company.get('certifications', []),
        'risk_signals': []
    }


# Try to import the real validation function
try:
    from serper_validation_integration import validate_company_with_serper as real_validate

    validate_company_with_serper = real_validate
    SERPER_VALIDATION_AVAILABLE = True
except ImportError:
    SERPER_VALIDATION_AVAILABLE = False
    print("Warning: Serper validation not available, using mock data")

# ============================================================================
# STREAMLIT APP - WITH PARALLEL EXECUTION AND ENHANCED TOKEN HANDLING
# ============================================================================

# Page config
st.set_page_config(
    page_title="Company Search & Validation Platform",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'current_criteria' not in st.session_state:
    st.session_state.current_criteria = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = []
if 'saved_profiles' not in st.session_state:
    st.session_state.saved_profiles = {}
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0
if 'current_profile_name' not in st.session_state:
    st.session_state.current_profile_name = None
if 'current_tier' not in st.session_state:
    st.session_state.current_tier = "A"
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = ["gpt-4.1"]
if 'parallel_execution_enabled' not in st.session_state:
    st.session_state.parallel_execution_enabled = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'model_execution_times' not in st.session_state:
    st.session_state.model_execution_times = {}
if 'model_success_status' not in st.session_state:
    st.session_state.model_success_status = {}

# Initialize ICP manager
icp_manager = ICPManager()

# Title
st.title("🔍 Company Search & Validation Platform")
st.markdown("AI-powered company discovery with advanced validation modes and enhanced token handling")

# Sidebar configuration
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
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Companies", len(st.session_state.search_results))
        st.metric("Profiles Saved", len(st.session_state.saved_profiles))
    with col2:
        st.metric("Validated", len(st.session_state.validation_results))
        st.metric("Total Cost", f"${st.session_state.total_cost:.3f}")

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

    if st.button("🗑️ Clear Session", use_container_width=True):
        for key in ['search_results', 'validation_results', 'current_criteria', 'model_results',
                    'model_execution_times', 'model_success_status']:
            if key in st.session_state:
                st.session_state[key] = [] if 'results' in key else {} if 'model_' in key else None
        st.session_state.total_cost = 0.0
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Search Configuration",
    "🔍 Execute Search",
    "✅ Validation",
    "📊 Results & Export",
    "❓ Help"
])

# Tab 1: Search Configuration (WITH PROFILES)
with tab1:
    st.header("Search Configuration")

    # ICP Profile Selection
    st.subheader("📋 ICP Profile Selection")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🏥 RMH Sydney", use_container_width=True, type="secondary"):
            st.session_state.current_profile_name = "rmh_sydney"
            st.rerun()

    with col2:
        if st.button("🦮 Guide Dogs Victoria", use_container_width=True, type="secondary"):
            st.session_state.current_profile_name = "guide_dogs_victoria"
            st.rerun()

    with col3:
        if st.button("🔧 Custom Profile", use_container_width=True, type="secondary"):
            st.session_state.current_profile_name = None
            st.session_state.current_criteria = None
            st.rerun()

    # Show profile details if selected
    if st.session_state.current_profile_name:
        st.divider()

        if st.session_state.current_profile_name == "rmh_sydney":
            st.markdown("### 🏥 RMH Sydney Profile")

            tier = st.radio(
                "Select Tier",
                ["A", "B", "C"],
                format_func=lambda x: {
                    "A": "Tier A - Perfect Match (Revenue $5-100M, 50+ employees, CSR focus)",
                    "B": "Tier B - Good Match (Revenue $2-200M, 20+ employees, Community involvement)",
                    "C": "Tier C - Potential Match (Revenue $1M+, Any size, Basic criteria)"
                }[x],
                key="rmh_tier_select"
            )
            st.session_state.current_tier = tier

        elif st.session_state.current_profile_name == "guide_dogs_victoria":
            st.markdown("### 🦮 Guide Dogs Victoria Profile")

            tier = st.radio(
                "Select Tier",
                ["A", "B", "C"],
                format_func=lambda x: {
                    "A": "Tier A - Strategic Partners (Revenue $500M+, 500+ employees, Certifications)",
                    "B": "Tier B - Exploratory Partners (Revenue $50-500M, 100-500 employees)",
                    "C": "Tier C - Potential Partners (Revenue $10M+, 50+ employees)"
                }[x],
                key="gdv_tier_select"
            )
            st.session_state.current_tier = tier

    st.divider()

    # Search Fields Configuration
    st.subheader("🔧 Search Criteria Configuration")

    # Load profile defaults if selected
    criteria = None
    if st.session_state.current_profile_name and icp_manager:
        profile = icp_manager.get_profile(st.session_state.current_profile_name)
        if profile:
            criteria = profile.tiers.get(st.session_state.current_tier)

    # Location Section
    with st.expander("🌍 Location Settings", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            country_options = ["Australia", "United States", "United Kingdom", "Canada", "New Zealand",
                               "Germany", "France", "Japan", "Singapore", "India"]
            default_countries = criteria.location.countries if criteria else ["Australia"]
            countries = st.multiselect(
                "Countries",
                country_options,
                default=default_countries
            )

        with col2:
            default_states = criteria.location.states if criteria else []
            states = st.text_input(
                "States/Regions",
                value=", ".join(default_states),
                placeholder="Victoria, New South Wales"
            )

        with col3:
            default_cities = criteria.location.cities if criteria else []
            cities = st.text_input(
                "Cities",
                value=", ".join(default_cities),
                placeholder="Sydney, Melbourne"
            )

        # Proximity search
        use_proximity = st.checkbox("Enable Proximity Search")
        if use_proximity:
            col1, col2 = st.columns(2)
            with col1:
                proximity_center = st.text_input("Center Location", placeholder="Sydney CBD")
            with col2:
                proximity_radius = st.number_input("Radius (km)", 10, 500, 50)
        else:
            proximity_center = None
            proximity_radius = None

    # Financial Section
    with st.expander("💰 Financial Criteria", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            default_min = criteria.financial.revenue_min / 1_000_000 if criteria and criteria.financial.revenue_min else 0
            revenue_min = st.number_input("Min Revenue (M)", 0, 10000, int(default_min))

            default_emp_min = criteria.organizational.employee_count_min if criteria else 0
            employee_min = st.number_input("Min Employees", 0, 100000, default_emp_min or 0)

        with col2:
            default_max = criteria.financial.revenue_max / 1_000_000 if criteria and criteria.financial.revenue_max else 100
            revenue_max = st.number_input("Max Revenue (M)", 0, 10000, int(default_max))

            default_emp_max = criteria.organizational.employee_count_max if criteria else 1000
            employee_max = st.number_input("Max Employees", 0, 100000, default_emp_max or 1000)

        with col3:
            default_currency = criteria.financial.revenue_currency if criteria else "AUD"
            currency = st.selectbox("Currency", ["AUD", "USD", "EUR", "GBP"],
                                    index=["AUD", "USD", "EUR", "GBP"].index(default_currency))

            default_giving = criteria.financial.giving_capacity_min / 1_000 if criteria and criteria.financial.giving_capacity_min else 0
            giving_capacity = st.number_input("Min Giving Capacity (K)", 0, 1000, int(default_giving))

        # Office types field
        office_types = st.multiselect(
            "Office Types",
            ["Headquarters", "Major Office", "Regional Office", "Branch Office"],
            default=criteria.organizational.office_types if criteria else []
        )

    # Industry & Business Section
    with st.expander("🏢 Industry & Business Type", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            business_options = ["B2B", "B2C", "B2B2C", "D2C", "Professional Services", "Enterprise", "SaaS"]
            default_types = criteria.business_types if criteria else ["B2B", "B2C"]
            business_types = st.multiselect(
                "Business Types",
                business_options,
                default=default_types
            )

        with col2:
            default_industries = []
            if criteria and criteria.industries:
                for ind in criteria.industries:
                    default_industries.append(ind.get('name', ''))

            industries = st.text_area(
                "Industries (one per line)",
                value="\n".join(default_industries),
                placeholder="Construction\nProperty\nHospitality"
            )

    # CSR & Behavioral Section
    with st.expander("💚 CSR & Behavioral Signals", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            csr_options = ["children", "community", "education", "health", "environment",
                           "disability", "inclusion", "diversity", "sustainability", "elderly",
                           "families", "accessibility", "wellbeing"]
            default_csr = criteria.behavioral.csr_focus_areas if criteria else []
            # Ensure defaults exist in options
            valid_csr_defaults = [d for d in default_csr if d in csr_options]

            csr_focus = st.multiselect(
                "CSR Focus Areas",
                csr_options,
                default=valid_csr_defaults
            )

            event_options = ["Office Move", "CSR Launch", "Expansion", "Anniversary", "Award",
                             "New Leadership", "IPO", "Merger", "Partnership"]
            default_events = criteria.behavioral.recent_events if criteria else []
            # Ensure defaults exist in options
            valid_event_defaults = [d for d in default_events if d in event_options]

            recent_events = st.multiselect(
                "Recent Events",
                event_options,
                default=valid_event_defaults
            )

        with col2:
            cert_options = ["B-Corp", "ISO 26000", "ISO 14001", "Carbon Neutral", "Fair Trade",
                            "Great Place to Work", "ESG Certified"]
            default_certs = criteria.behavioral.certifications if criteria else []
            # Ensure defaults exist in options
            valid_cert_defaults = [d for d in default_certs if d in cert_options]

            certifications = st.multiselect(
                "Certifications",
                cert_options,
                default=valid_cert_defaults
            )

            esg_maturity = st.selectbox(
                "ESG Maturity",
                ["Any", "Basic", "Developing", "Mature", "Leading"],
                index=0
            )

    # Exclusions Section (RESTORED)
    with st.expander("🚫 Exclusions", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            exclusion_options = ["Gambling", "Tobacco", "Fast Food", "Racing", "Alcohol",
                                 "Weapons", "Animal Testing"]
            default_exc_ind = criteria.excluded_industries if criteria else []
            # Ensure defaults exist in options
            valid_exc_defaults = [d for d in default_exc_ind if d in exclusion_options]

            excluded_industries = st.multiselect(
                "Excluded Industries",
                exclusion_options,
                default=valid_exc_defaults
            )

            behavior_exclusion_options = ["Recent Misconduct", "Bankruptcy", "Litigation",
                                          "Environmental Violations", "Labor Issues"]
            default_exc_behaviors = criteria.excluded_behaviors if criteria else []
            valid_behavior_defaults = [d for d in default_exc_behaviors if d in behavior_exclusion_options]

            excluded_behaviors = st.multiselect(
                "Excluded Behaviors",
                behavior_exclusion_options,
                default=valid_behavior_defaults
            )

        with col2:
            default_exc_comp = criteria.excluded_companies if criteria else []
            excluded_companies = st.text_area(
                "Excluded Companies (one per line)",
                value="\n".join(default_exc_comp),
                height=100,
                placeholder="McDonald's\nKFC\nBurger King"
            )

            location_exclusions = st.text_input(
                "Location Exclusions",
                placeholder="Rural areas, Remote regions"
            )

    # Advanced Search Options Section
    with st.expander("🔍 Advanced Search Options", expanded=False):
        use_free_text = st.checkbox("Add Free Text Criteria")

        if use_free_text:
            free_text = st.text_area(
                "Additional Search Criteria",
                placeholder="Example: Focus on companies that have won sustainability awards in the last 2 years",
                height=100
            )
        else:
            free_text = None

        keywords = st.text_input(
            "Keywords (comma-separated)",
            placeholder="innovation, award-winning, sustainable"
        )

    st.divider()

    # Action Buttons
    if st.button("✅ Confirm Criteria", type="primary", use_container_width=True):
        # Build criteria object
        built_criteria = SearchCriteria(
            location=LocationCriteria(
                countries=countries,
                states=[s.strip() for s in states.split(',')] if states else [],
                cities=[c.strip() for c in cities.split(',')] if cities else [],
                proximity={"location": proximity_center, "radius_km": proximity_radius} if use_proximity else None,
                exclusions=[e.strip() for e in location_exclusions.split(',')] if location_exclusions else []
            ),
            financial=FinancialCriteria(
                revenue_min=revenue_min * 1_000_000 if revenue_min > 0 else None,
                revenue_max=revenue_max * 1_000_000 if revenue_max > 0 else None,
                revenue_currency=currency,
                giving_capacity_min=giving_capacity * 1_000 if giving_capacity > 0 else None
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=employee_min if employee_min > 0 else None,
                employee_count_max=employee_max if employee_max > 0 else None,
                office_types=office_types if office_types else []
            ),
            behavioral=BehavioralSignals(
                csr_focus_areas=csr_focus,
                certifications=certifications,
                recent_events=recent_events,
                esg_maturity=esg_maturity if esg_maturity != "Any" else None
            ),
            business_types=business_types,
            industries=[{"name": ind.strip(), "priority": i + 1}
                        for i, ind in enumerate(industries.split('\n')) if ind.strip()],
            keywords=[k.strip() for k in keywords.split(',')] if keywords else [],
            custom_prompt=free_text,
            excluded_industries=excluded_industries,
            excluded_companies=[c.strip() for c in excluded_companies.split('\n') if c.strip()],
            excluded_behaviors=excluded_behaviors
        )

        st.session_state.current_criteria = built_criteria
        st.success("✅ Criteria confirmed! Go to 'Execute Search' tab.")

# Tab 2: Execute Search with ENHANCED TOKEN HANDLING
with tab2:
    st.header("Execute Company Search")

    if not st.session_state.current_criteria:
        st.warning("⚠️ Please configure search criteria first.")
    else:
        # Show criteria summary
        with st.expander("📋 Current Search Criteria", expanded=True):
            criteria = st.session_state.current_criteria
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Location:**")
                if criteria.location.countries:
                    st.write(f"Countries: {', '.join(criteria.location.countries)}")

            with col2:
                st.markdown("**Financial:**")
                if criteria.financial.revenue_min:
                    st.write(f"Revenue: ${criteria.financial.revenue_min / 1e6:.0f}M+")

            with col3:
                st.markdown("**Industries:**")
                if criteria.industries:
                    st.write(f"{len(criteria.industries)} industries")

        st.divider()

        # Show execution mode
        if st.session_state.parallel_execution_enabled:
            st.info(
                f"⚡ **Parallel Execution Mode Active**\n\nUsing {len(st.session_state.selected_models)} models: {', '.join(st.session_state.selected_models)}")
        else:
            st.info(f"**Single Model Mode**\n\nUsing: {st.session_state.selected_models[0]}")

        target_count = st.slider("Target Companies", 10, 1000, 50, step=10)

        # Show batch execution notice for large counts
        if target_count > 20:
            st.info(
                f"📦 Large search detected ({target_count} companies). Will use batch execution for optimal performance.")

        if st.button("🚀 Execute Search", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            # Clear previous model results
            st.session_state.model_results = {}
            st.session_state.model_execution_times = {}
            st.session_state.model_success_status = {}

            if st.session_state.parallel_execution_enabled and len(st.session_state.selected_models) > 1:
                # PARALLEL EXECUTION
                status_placeholder.info(
                    f"🚀 Executing parallel search with {len(st.session_state.selected_models)} models...")

                try:
                    # Run parallel search
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    result = loop.run_until_complete(
                        execute_parallel_search(
                            st.session_state.selected_models,
                            st.session_state.current_criteria,
                            target_count
                        )
                    )
                    loop.close()

                    # Extract results
                    companies = result.get('companies', [])
                    metadata = result.get('metadata', {})

                    # Update session state with model stats
                    st.session_state.model_success_status = metadata.get('model_stats', {})

                    if companies:
                        st.session_state.search_results = companies
                        st.session_state.total_cost += 0.02 * len(st.session_state.selected_models)

                        # Show success with details
                        batch_info = " (with batch execution)" if metadata.get('batch_execution_used', False) else ""
                        status_placeholder.success(
                            f"✅ Parallel search complete{batch_info}!\n\n"
                            f"**Total companies found:** {len(companies)}\n"
                            f"**Successful models:** {len(metadata.get('successful_models', []))}/{len(st.session_state.selected_models)}\n"
                            f"**Duplicates removed:** {metadata.get('duplicates_removed', 0)}\n"
                            f"**Execution time:** {metadata.get('execution_time', 0):.2f}s\n"
                            f"**Success rate:** {metadata.get('success_rate', 0):.1f}%"
                        )

                        # Show per-model breakdown
                        if metadata.get('model_stats'):
                            st.subheader("📊 Model Performance")
                            model_cols = st.columns(len(st.session_state.selected_models))
                            for i, (model, stats) in enumerate(metadata['model_stats'].items()):
                                with model_cols[i % len(model_cols)]:
                                    if 'success' in stats['status']:
                                        batch_note = " 📦" if stats.get('batch_execution', False) else ""
                                        st.metric(
                                            label=model.replace('gpt-4.1', 'Model') + batch_note,
                                            value=f"{stats['companies_found']} companies",
                                            delta=f"Target: {stats.get('target', target_count)}"
                                        )
                                    else:
                                        st.error(f"{model}: Failed")

                    else:
                        st.warning("No companies found. Try adjusting criteria.")

                    progress_bar.progress(1.0)

                except Exception as e:
                    st.error(f"Parallel search failed: {str(e)}")
                    traceback.print_exc()

            else:
                # SINGLE MODEL EXECUTION (updated with enhanced token handling)
                with st.spinner(f"Searching with {st.session_state.selected_models[0]}..."):
                    try:
                        agent = EnhancedSearchStrategistAgent(deployment_name=st.session_state.selected_models[0])

                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(
                            agent.generate_enhanced_strategy(
                                st.session_state.current_criteria,
                                target_count=target_count
                            )
                        )
                        loop.close()

                        companies = result.get("companies", [])

                        # Check if batch execution was used
                        batch_used = result.get("metadata", {}).get("batch_execution", False)

                        if companies:
                            st.session_state.search_results = companies
                            st.session_state.total_cost += 0.02

                            # Show success with batch info if applicable
                            if batch_used:
                                st.success(f"✅ Found {len(companies)} companies using batch execution!")
                            else:
                                st.success(f"✅ Found {len(companies)} companies!")
                        else:
                            st.warning("No companies found. Try adjusting criteria.")

                        progress_bar.progress(1.0)

                    except Exception as e:
                        st.error(f"Search failed: {str(e)}")
                        # Try with smaller batch as fallback
                        try:
                            st.info("Retrying with smaller batch size...")
                            agent = EnhancedSearchStrategistAgent(deployment_name=st.session_state.selected_models[0])

                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(
                                agent.generate_enhanced_strategy(
                                    st.session_state.current_criteria,
                                    target_count=min(10, target_count)
                                )
                            )
                            loop.close()

                            companies = result.get("companies", [])
                            if companies:
                                st.session_state.search_results = companies
                                st.session_state.total_cost += 0.02
                                st.warning(
                                    f"⚠️ Partial success: Found {len(companies)} companies (reduced from {target_count})")
                            else:
                                st.error("Search failed even with reduced batch size.")

                        except Exception as retry_error:
                            st.error(f"Retry also failed: {str(retry_error)}")

                        progress_bar.progress(1.0)

            progress_bar.empty()

        # Display results
        if st.session_state.search_results:
            st.divider()
            st.subheader("📊 Search Results")

            results_data = []
            for company in st.session_state.search_results[:100]:
                if hasattr(company, 'dict'):
                    c = company.dict()
                else:
                    c = company

                row_data = {
                    "Company": c.get('name', 'Unknown'),
                    "Industry": c.get('industry_category', 'Unknown'),
                    "Revenue": c.get('estimated_revenue', 'Unknown'),
                    "Employees": c.get('estimated_employees', 'Unknown'),
                    "ICP Score": c.get('icp_score', 0),
                    "ICP Tier": c.get('icp_tier', 'D')
                }

                # Add source model if parallel execution was used
                if st.session_state.parallel_execution_enabled:
                    row_data["Source Model"] = c.get('source_model', 'Unknown')

                results_data.append(row_data)

            df = pd.DataFrame(results_data)

            st.dataframe(df, use_container_width=True, height=400)

# Tab 3: Validation (with results display)
with tab3:
    st.header("Company Validation")

    if not st.session_state.search_results:
        st.info("No companies to validate. Run a search first.")
    else:
        st.write(f"**{len(st.session_state.search_results)} companies** ready for validation")

        # Show existing validation results if available
        if st.session_state.validation_results:
            st.info(
                f"ℹ️ {len(st.session_state.validation_results)} companies already validated. You can validate more or view existing results below.")

        # Validation mode selection
        st.subheader("🎯 Validation Mode")

        validation_mode = st.selectbox(
            "Select Validation Mode",
            [
                "Simple Check (2-3 credits)",
                "Smart Contact Extraction (3-5 credits)",
                "Smart CSR Verification (3-5 credits)",
                "Smart Financial Check (3-4 credits)",
                "Full Validation (10-15 credits)",
                "Raw Endpoint Access",
                "Custom Configuration"
            ],
            help="Choose validation depth based on your needs"
        )

        # Mode-specific information
        mode_info = {
            "Simple Check (2-3 credits)": {
                "description": "Quick existence and location verification",
                "extracts": ["Company exists", "Location verified", "Basic website"],
                "credits": 2.5
            },
            "Smart Contact Extraction (3-5 credits)": {
                "description": "Extract emails, phones, and contact names",
                "extracts": ["Email addresses", "Phone numbers", "Executive names", "LinkedIn profiles"],
                "credits": 4
            },
            "Smart CSR Verification (3-5 credits)": {
                "description": "Verify CSR programs and community involvement",
                "extracts": ["CSR programs", "Focus areas", "Certifications", "Giving evidence"],
                "credits": 4
            },
            "Smart Financial Check (3-4 credits)": {
                "description": "Verify revenue and employee information",
                "extracts": ["Revenue range", "Employee count", "Growth indicators", "Financial health"],
                "credits": 3.5
            },
            "Full Validation (10-15 credits)": {
                "description": "Comprehensive validation with all checks",
                "extracts": ["All of the above", "Recent news", "Risk signals", "Detailed analysis"],
                "credits": 12
            }
        }

        if validation_mode in mode_info:
            with st.expander("ℹ️ Mode Information", expanded=True):
                info = mode_info[validation_mode]
                st.write(f"**Description:** {info['description']}")
                st.write("**Extracts:**")
                for item in info['extracts']:
                    st.write(f"  • {item}")

        # Validation settings
        col1, col2 = st.columns(2)

        with col1:
            max_validate = st.number_input(
                "Companies to validate",
                1,
                min(50, len(st.session_state.search_results)),
                min(10, len(st.session_state.search_results))
            )

        with col2:
            # Cost estimate
            if validation_mode in mode_info:
                credits_per = mode_info[validation_mode]['credits']
            else:
                credits_per = 5  # Default

            est_credits = max_validate * credits_per
            est_cost = est_credits * 0.001

            st.metric("Estimated Credits", int(est_credits))
            st.metric("Estimated Cost", f"${est_cost:.3f}")

        # Validation button
        if st.button("✅ Start Validation", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            validated_companies = []

            for i, company in enumerate(st.session_state.search_results[:max_validate]):
                progress = (i + 1) / max_validate
                progress_bar.progress(progress)

                # Extract company details
                if hasattr(company, 'dict'):
                    company_dict = company.dict()
                elif isinstance(company, dict):
                    company_dict = company
                else:
                    company_dict = {'name': str(company)}

                company_name = company_dict.get('name', 'Unknown')
                status_text.text(f"Validating {i + 1}/{max_validate}: {company_name}")

                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    validation_result = loop.run_until_complete(
                        validate_company_with_serper(
                            company_dict,
                            validation_mode,
                            serper_key
                        )
                    )
                    loop.close()

                    validated_companies.append(validation_result)

                    # Rate limiting
                    if i < max_validate - 1:
                        time.sleep(0.5)

                except Exception as e:
                    st.error(f"Error validating {company_name}: {str(e)}")
                    validation_result = {
                        'company_name': company_name,
                        'validation_status': 'error',
                        'mode': validation_mode,
                        'credits_used': 0,
                        'validation_timestamp': datetime.now().isoformat(),
                        'error': str(e)
                    }
                    validated_companies.append(validation_result)

            st.session_state.validation_results = validated_companies

            # Calculate actual cost
            total_credits = sum(v.get('credits_used', 0) for v in validated_companies)
            actual_cost = total_credits * 0.001
            st.session_state.total_cost += actual_cost

            progress_bar.empty()
            status_text.empty()

            # Show completion message
            st.success(
                f"✅ Validated {len(validated_companies)} companies using {total_credits} credits (${actual_cost:.3f})")

            # Show validation results summary
            st.divider()
            st.subheader("Validation Results Summary")

            # Summary metrics
            verified = len([v for v in validated_companies if
                            v.get('validation_status', '').lower() in ['verified', 'verify', 'valid']])
            partial = len(
                [v for v in validated_companies if v.get('validation_status', '').lower() in ['partial', 'partially']])
            unverified = len([v for v in validated_companies if
                              v.get('validation_status', '').lower() in ['unverified', 'not verified']])
            errors = len(
                [v for v in validated_companies if v.get('validation_status', '').lower() in ['error', 'failed']])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Verified", verified)
            with col2:
                st.metric("Partial", partial)
            with col3:
                st.metric("Unverified", unverified)
            with col4:
                st.metric("Errors", errors)

            # Display detailed validation results
            st.divider()
            st.subheader("📋 Detailed Validation Results")

            # Create a DataFrame for display
            validation_display_data = []
            for val_result in validated_companies:
                display_row = {
                    "Company": val_result.get('company_name', 'Unknown'),
                    "Status": val_result.get('validation_status', 'unknown'),
                    "Mode": val_result.get('mode', validation_mode),
                    "Credits Used": val_result.get('credits_used', 0)
                }

                # Add mode-specific data based on validation type
                if "Contact" in validation_mode:
                    # Contact extraction results
                    emails = val_result.get('emails', [])
                    phones = val_result.get('phones', [])
                    names = val_result.get('names', [])
                    display_row["Emails Found"] = len(emails)
                    display_row["First Email"] = emails[0] if emails else ""
                    display_row["Phones Found"] = len(phones)
                    display_row["First Phone"] = phones[0] if phones else ""
                    display_row["Contacts Found"] = len(names)
                    display_row["First Contact"] = names[0] if names else ""

                elif "CSR" in validation_mode:
                    # CSR verification results
                    csr_programs = val_result.get('csr_programs', [])
                    certifications = val_result.get('certifications', [])
                    display_row["CSR Programs"] = ', '.join(csr_programs) if csr_programs else "None found"
                    display_row["Certifications"] = ', '.join(certifications) if certifications else "None found"
                    display_row["Has Foundation"] = "Yes" if val_result.get('has_foundation') else "No"

                elif "Financial" in validation_mode:
                    # Financial verification results
                    display_row["Revenue Range"] = val_result.get('revenue_range', 'Not found')
                    display_row["Employee Range"] = val_result.get('employee_range', 'Not found')
                    display_row["Stock Listed"] = "Yes" if val_result.get('stock_listed') else "No"
                    risk_signals = val_result.get('risk_signals', [])
                    display_row["Risk Signals"] = ', '.join(risk_signals) if risk_signals else "None"

                elif "Simple" in validation_mode:
                    # Simple check results
                    display_row["Location Verified"] = "Yes" if val_result.get('location_verified') else "No"
                    display_row["Address"] = val_result.get('address', 'Not found')
                    display_row["Phone"] = val_result.get('phone', 'Not found')

                elif "Full" in validation_mode:
                    # Full validation results - combine all
                    emails = val_result.get('emails', [])
                    display_row["Email"] = emails[0] if emails else "Not found"
                    display_row["Phone"] = val_result.get('phones', [None])[0] or val_result.get('phone', 'Not found')
                    display_row["Revenue"] = val_result.get('revenue_range', 'Not found')
                    display_row["Employees"] = val_result.get('employee_range', 'Not found')
                    csr_programs = val_result.get('csr_programs', [])
                    display_row["CSR"] = "Yes" if csr_programs else "No"
                    risk_signals = val_result.get('risk_signals', [])
                    display_row["Risks"] = len(risk_signals)

                # Add error message if validation failed
                if val_result.get('error'):
                    display_row["Error"] = val_result.get('error', '')

                validation_display_data.append(display_row)

            # Display the results table
            if validation_display_data:
                val_df = pd.DataFrame(validation_display_data)


                # Color code the status column
                def color_status(val):
                    if val.lower() in ['verified', 'verify', 'valid']:
                        return 'background-color: #90EE90'  # Light green
                    elif val.lower() in ['partial', 'partially']:
                        return 'background-color: #FFE4B5'  # Light orange
                    elif val.lower() in ['error', 'failed']:
                        return 'background-color: #FFB6C1'  # Light red
                    else:
                        return 'background-color: #D3D3D3'  # Light gray


                # Apply styling if Status column exists
                if 'Status' in val_df.columns:
                    styled_df = val_df.style.applymap(color_status, subset=['Status'])
                    st.dataframe(styled_df, use_container_width=True, height=400)
                else:
                    st.dataframe(val_df, use_container_width=True, height=400)

                # Show raw validation data in expander for debugging
                with st.expander("🔍 View Raw Validation Data"):
                    st.json(validated_companies)

    # Display existing validation results if available (outside the button click)
    if st.session_state.validation_results and len(st.session_state.validation_results) > 0:
        st.divider()
        st.subheader("📊 Existing Validation Results")

        # Summary of existing validations
        existing_verified = len([v for v in st.session_state.validation_results if
                                 v.get('validation_status', '').lower() in ['verified', 'verify', 'valid']])
        existing_partial = len([v for v in st.session_state.validation_results if
                                v.get('validation_status', '').lower() in ['partial', 'partially']])
        existing_unverified = len([v for v in st.session_state.validation_results if
                                   v.get('validation_status', '').lower() in ['unverified', 'not verified']])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Validated", len(st.session_state.validation_results))
        with col2:
            st.metric("Verified", existing_verified)
        with col3:
            st.metric("Partial", existing_partial)
        with col4:
            st.metric("Unverified", existing_unverified)

        # Create display table for existing results
        existing_display_data = []
        for val_result in st.session_state.validation_results:
            display_row = {
                "Company": val_result.get('company_name', 'Unknown'),
                "Status": val_result.get('validation_status', 'unknown'),
                "Mode": val_result.get('mode', 'Unknown'),
                "Credits Used": val_result.get('credits_used', 0),
                "Timestamp": val_result.get('validation_timestamp', '')
            }

            # Add key validation fields
            emails = val_result.get('emails', [])
            if emails:
                display_row["Email"] = emails[0]

            phones = val_result.get('phones', [])
            if phones:
                display_row["Phone"] = phones[0]
            elif val_result.get('phone'):
                display_row["Phone"] = val_result.get('phone')

            if val_result.get('revenue_range'):
                display_row["Revenue"] = val_result.get('revenue_range')

            if val_result.get('employee_range'):
                display_row["Employees"] = val_result.get('employee_range')

            csr_programs = val_result.get('csr_programs', [])
            if csr_programs:
                display_row["CSR Programs"] = ', '.join(csr_programs[:3])

            risk_signals = val_result.get('risk_signals', [])
            if risk_signals:
                display_row["Risk Signals"] = ', '.join(risk_signals[:3])

            existing_display_data.append(display_row)

        # Display the existing results
        if existing_display_data:
            existing_df = pd.DataFrame(existing_display_data)

            # Filter options for existing results
            col1, col2 = st.columns(2)
            with col1:
                filter_existing_status = st.selectbox(
                    "Filter by Status",
                    ["All", "Verified", "Partial", "Unverified"],
                    key="filter_existing_validation"
                )
            with col2:
                search_existing = st.text_input(
                    "Search Companies",
                    placeholder="Enter company name...",
                    key="search_existing_validation"
                )

            # Apply filters
            filtered_existing = existing_df.copy()

            if filter_existing_status != "All":
                status_map = {
                    "Verified": ['verified', 'verify', 'valid'],
                    "Partial": ['partial', 'partially'],
                    "Unverified": ['unverified', 'not verified']
                }
                if filter_existing_status in status_map:
                    filtered_existing = filtered_existing[
                        filtered_existing['Status'].str.lower().isin(status_map[filter_existing_status])
                    ]

            if search_existing:
                filtered_existing = filtered_existing[
                    filtered_existing['Company'].str.contains(search_existing, case=False, na=False)
                ]

            st.write(f"Showing {len(filtered_existing)} of {len(existing_df)} validated companies")


            # Color code the status column
            def color_status(val):
                if isinstance(val, str):
                    val_lower = val.lower()
                    if val_lower in ['verified', 'verify', 'valid']:
                        return 'background-color: #90EE90'  # Light green
                    elif val_lower in ['partial', 'partially']:
                        return 'background-color: #FFE4B5'  # Light orange
                    elif val_lower in ['error', 'failed']:
                        return 'background-color: #FFB6C1'  # Light red
                return 'background-color: #D3D3D3'  # Light gray


            # Apply styling and display
            if 'Status' in filtered_existing.columns:
                styled_existing = filtered_existing.style.applymap(color_status, subset=['Status'])
                st.dataframe(styled_existing, use_container_width=True, height=400)
            else:
                st.dataframe(filtered_existing, use_container_width=True, height=400)

            # Option to clear validation results
            if st.button("🗑️ Clear All Validation Results", key="clear_validation"):
                st.session_state.validation_results = []
                st.rerun()