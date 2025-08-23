# shared/data_models.py
"""
Shared data models used across all tabs
Extracted from the main streamlit_app.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# ENUMS
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


# ============================================================================
# CRITERIA CLASSES
# ============================================================================

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


# ============================================================================
# COMPANY ENTRY MODEL
# ============================================================================

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
    source_model: Optional[str] = Field(default=None)


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