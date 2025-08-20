# data_models.py
"""
Data models and types for the enhanced company search and validation system
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


# ============================================================================
# Enums
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
    SMALL = "small"  # 1-50 employees or <$10M revenue
    MEDIUM = "medium"  # 51-500 employees or $10M-$100M revenue
    ENTERPRISE = "enterprise"  # 500+ employees or $100M+ revenue
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for results"""
    ABSOLUTE = "absolute"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ICPTier(Enum):
    """ICP (Ideal Customer Profile) tiers"""
    A = "A"  # Perfect match
    B = "B"  # Good match
    C = "C"  # Acceptable match
    D = "D"  # Poor match
    UNTIERED = "Untiered"


class ValidationStatus(Enum):
    """Validation status types"""
    VERIFIED = "verified"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"
    PENDING = "pending"


class ESGMaturity(Enum):
    """ESG/CSR maturity levels"""
    LEADING = "Leading"
    MATURE = "Mature"
    DEVELOPING = "Developing"
    BASIC = "Basic"
    NONE = "None"
    UNKNOWN = "Unknown"


# ============================================================================
# Location Models
# ============================================================================

@dataclass
class LocationCriteria:
    """Location search criteria"""
    countries: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    cities: List[str] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    proximity: Optional[Dict[str, Any]] = None  # {"location": "Sydney CBD", "radius_km": 50}
    exclusions: List[str] = field(default_factory=list)


@dataclass
class Address:
    """Structured address information"""
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None
    formatted: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None  # {"lat": -33.8688, "lng": 151.2093}


# ============================================================================
# Financial Models
# ============================================================================

@dataclass
class FinancialCriteria:
    """Financial search criteria"""
    revenue_min: Optional[float] = None
    revenue_max: Optional[float] = None
    revenue_currency: str = "USD"
    giving_capacity_min: Optional[float] = None
    growth_rate_min: Optional[float] = None
    profitable: Optional[bool] = None
    funding_stage: Optional[str] = None  # "Seed", "Series A", "Series B", etc.


@dataclass
class FinancialData:
    """Company financial information"""
    revenue: Optional[str] = None
    revenue_currency: str = "USD"
    revenue_year: Optional[int] = None
    employees: Optional[str] = None
    growth_rate: Optional[float] = None
    profitability: Optional[str] = None
    funding_total: Optional[float] = None
    funding_stage: Optional[str] = None
    valuation: Optional[float] = None
    giving_capacity: Optional[float] = None


# ============================================================================
# Organizational Models
# ============================================================================

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
class OrganizationalData:
    """Company organizational information"""
    employee_count: Optional[int] = None
    employee_count_range: Optional[str] = None
    employees_by_location: Dict[str, int] = field(default_factory=dict)
    office_types: List[str] = field(default_factory=list)
    company_stage: Optional[str] = None
    founded_year: Optional[int] = None
    years_in_business: Optional[int] = None
    leadership: List[Dict[str, str]] = field(default_factory=list)  # [{"name": "John Doe", "title": "CEO"}]


# ============================================================================
# CSR/ESG Models
# ============================================================================

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
class CSRData:
    """Company CSR/ESG information"""
    programs: List[Dict[str, Any]] = field(default_factory=list)
    focus_areas: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    esg_score: Optional[float] = None
    esg_maturity: Optional[str] = None
    giving_history: List[Dict[str, Any]] = field(default_factory=list)
    community_involvement: List[str] = field(default_factory=list)
    sustainability_initiatives: List[str] = field(default_factory=list)


# ============================================================================
# Search Criteria (Combined)
# ============================================================================

@dataclass
class SearchCriteria:
    """Complete search criteria"""
    # Core criteria
    location: LocationCriteria
    financial: FinancialCriteria
    organizational: OrganizationalCriteria
    behavioral: BehavioralSignals

    # Industry and business
    business_types: List[str] = field(default_factory=list)
    industries: List[Dict[str, Any]] = field(default_factory=list)

    # Custom search
    keywords: List[str] = field(default_factory=list)
    custom_prompt: Optional[str] = None

    # Exclusions
    excluded_industries: List[str] = field(default_factory=list)
    excluded_companies: List[str] = field(default_factory=list)
    excluded_behaviors: List[str] = field(default_factory=list)


# ============================================================================
# Company Models (Pydantic for API compatibility)
# ============================================================================

class CompanyBase(BaseModel):
    """Base company information"""
    name: str = Field(description="Company name")
    confidence: str = Field(description="Confidence level")
    operates_in_country: bool = Field(description="Whether company operates in the specified country")
    business_type: str = Field(description="Type of business")
    industry_category: str = Field(description="Industry category")
    sub_industry: Optional[str] = Field(description="Sub-industry or niche", default=None)
    reasoning: str = Field(description="Brief reasoning for confidence level")


class CompanyLocation(BaseModel):
    """Company location information"""
    headquarters: Optional[Dict[str, Any]] = Field(default=None)
    office_locations: List[Any] = Field(default_factory=list)
    service_areas: List[str] = Field(default_factory=list)


class CompanyFinancials(BaseModel):
    """Company financial information"""
    estimated_revenue: Optional[str] = Field(default=None)
    revenue_currency: Optional[str] = Field(default="USD")
    estimated_employees: Optional[str] = Field(default=None)
    employees_by_location: Optional[Dict[str, str]] = Field(default=None)
    company_size: Optional[str] = Field(default="unknown")
    giving_history: List[Dict[str, Any]] = Field(default_factory=list)
    financial_health: Optional[str] = Field(default=None)


class CompanyCSR(BaseModel):
    """Company CSR/ESG information"""
    csr_programs: List[str] = Field(default_factory=list)
    csr_focus_areas: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    esg_score: Optional[float] = Field(default=None)
    esg_maturity: Optional[str] = Field(default=None)
    community_involvement: List[str] = Field(default_factory=list)


class CompanySignals(BaseModel):
    """Company signals and events"""
    recent_events: List[Any] = Field(default_factory=list)
    leadership_changes: List[Dict[str, Any]] = Field(default_factory=list)
    growth_signals: List[str] = Field(default_factory=list)


class CompanyICP(BaseModel):
    """Company ICP matching information"""
    icp_tier: Optional[str] = Field(default=None)
    icp_score: Optional[float] = Field(default=None)
    matched_criteria: List[str] = Field(default_factory=list)
    missing_criteria: List[str] = Field(default_factory=list)


class CompanyMetadata(BaseModel):
    """Company data quality metadata"""
    data_freshness: Optional[str] = Field(default=None)
    data_sources: List[str] = Field(default_factory=list)
    validation_notes: Optional[str] = Field(default=None)


class EnhancedCompanyEntry(
    CompanyBase,
    CompanyLocation,
    CompanyFinancials,
    CompanyCSR,
    CompanySignals,
    CompanyICP,
    CompanyMetadata
):
    """Complete enhanced company entry with all fields"""
    pass


# ============================================================================
# Validation Models
# ============================================================================

@dataclass
class ValidationResult:
    """Simplified validation result for API responses"""
    company_name: str
    validation_status: ValidationStatus
    confidence_level: ConfidenceLevel
    overall_score: float
    location_verified: bool
    financial_verified: bool
    csr_verified: bool
    risk_flags: List[str]
    evidence_count: int
    queries_used: int
    validation_time: float
    timestamp: str


# ============================================================================
# Use Case Models
# ============================================================================

@dataclass
class UseCaseConfig:
    """Configuration for specific use cases"""
    name: str
    description: str
    tier_a_criteria: SearchCriteria
    tier_b_criteria: Optional[SearchCriteria] = None
    tier_c_criteria: Optional[SearchCriteria] = None
    validation_weights: Dict[str, float] = field(default_factory=dict)
    required_validations: List[str] = field(default_factory=list)
    exclusion_keywords: List[str] = field(default_factory=list)


# Predefined use case configurations
RMH_SYDNEY_CONFIG = UseCaseConfig(
    name="RMH Sydney",
    description="Ronald McDonald House Sydney donor prospects",
    tier_a_criteria=SearchCriteria(
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
            {"name": "Construction/Trades", "priority": 1},
            {"name": "Property/Real Estate", "priority": 2},
            {"name": "Hospitality", "priority": 3}
        ],
        excluded_industries=["Fast Food", "Gambling"]
    ),
    validation_weights={
        "location": 0.35,
        "financial": 0.25,
        "csr": 0.30,
        "reputation": 0.10
    },
    required_validations=["location", "revenue", "csr_focus"],
    exclusion_keywords=["McDonald's competitor", "burger", "KFC"]
)

GUIDE_DOGS_VICTORIA_CONFIG = UseCaseConfig(
    name="Guide Dogs Victoria",
    description="Guide Dogs Victoria corporate partnership prospects",
    tier_a_criteria=SearchCriteria(
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
            {"name": "Health & Life Sciences", "priority": 1},
            {"name": "Financial Services", "priority": 2},
            {"name": "Technology", "priority": 3}
        ],
        excluded_industries=["Gambling", "Tobacco", "Racing"]
    ),
    validation_weights={
        "location": 0.25,
        "financial": 0.20,
        "csr": 0.35,
        "reputation": 0.20
    },
    required_validations=["headquarters", "revenue", "certifications", "csr_maturity"],
    exclusion_keywords=["gambling", "tobacco", "racing", "animal testing"]
)


# ============================================================================
# Helper Functions
# ============================================================================

def company_to_dict(company: Union[EnhancedCompanyEntry, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert company object to dictionary"""
    if isinstance(company, dict):
        return company
    elif hasattr(company, 'dict'):
        return company.dict()
    elif hasattr(company, '__dict__'):
        return company.__dict__
    else:
        return asdict(company)


def dict_to_company(data: Dict[str, Any]) -> EnhancedCompanyEntry:
    """Convert dictionary to company object"""
    return EnhancedCompanyEntry(**data)


def calculate_company_size(revenue: Optional[float], employees: Optional[int]) -> CompanySize:
    """Calculate company size based on revenue and employees"""
    if revenue and revenue >= 100_000_000:
        return CompanySize.ENTERPRISE
    elif employees and employees >= 500:
        return CompanySize.ENTERPRISE
    elif (revenue and revenue >= 10_000_000) or (employees and employees >= 51):
        return CompanySize.MEDIUM
    elif (revenue and revenue > 0) or (employees and employees > 0):
        return CompanySize.SMALL
    else:
        return CompanySize.UNKNOWN