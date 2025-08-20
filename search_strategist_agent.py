# enhanced_search_agent.py

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
import time
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field
from openai import AzureOpenAI


# Enhanced Business Type Enum
class BusinessType(Enum):
    B2C = "B2C"
    B2B = "B2B"
    B2B2C = "B2B2C"
    D2C = "D2C"
    MARKETPLACE = "Marketplace"
    HYBRID = "Hybrid"
    PROFESSIONAL_SERVICES = "Professional Services"
    REAL_ESTATE = "Real Estate"


# Enhanced Company Size Classifications
class CompanySize(Enum):
    SMALL = "small"  # 1-50 employees or <$10M revenue
    MEDIUM = "medium"  # 51-500 employees or $10M-$100M revenue
    ENTERPRISE = "enterprise"  # 500+ employees or $100M+ revenue
    UNKNOWN = "unknown"


# Enhanced location with proximity support
@dataclass
class LocationCriteria:
    countries: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    cities: List[str] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    proximity: Optional[Dict[str, Any]] = None  # {"location": "Sydney CBD", "radius_km": 50}
    exclusions: List[str] = field(default_factory=list)


@dataclass
class FinancialCriteria:
    revenue_min: Optional[float] = None
    revenue_max: Optional[float] = None
    revenue_currency: str = "USD"
    giving_capacity_min: Optional[float] = None
    growth_rate_min: Optional[float] = None
    profitable: Optional[bool] = None


@dataclass
class OrganizationalCriteria:
    employee_count_min: Optional[int] = None
    employee_count_max: Optional[int] = None
    employee_count_by_location: Optional[Dict[str, int]] = None  # {"Victoria": 150}
    office_types: List[str] = field(default_factory=list)  # ["HQ", "Regional", "Branch"]
    company_stage: Optional[str] = None  # "Startup", "Growth", "Mature"


@dataclass
class BehavioralSignals:
    csr_programs: List[str] = field(default_factory=list)
    csr_focus_areas: List[str] = field(default_factory=list)  # ["children", "education", "environment"]
    certifications: List[str] = field(default_factory=list)  # ["B-Corp", "ISO 26000"]
    recent_events: List[str] = field(default_factory=list)  # ["office move", "expansion", "CSR launch"]
    technology_stack: List[str] = field(default_factory=list)
    esg_maturity: Optional[str] = None  # "Basic", "Developing", "Mature", "Leading"


@dataclass
class SearchCriteria:
    # Core criteria
    location: LocationCriteria
    financial: FinancialCriteria
    organizational: OrganizationalCriteria
    behavioral: BehavioralSignals

    # Industry and business
    business_types: List[str] = field(default_factory=list)
    industries: List[Dict[str, Any]] = field(default_factory=list)  # [{"name": "Construction", "priority": 1}]

    # Custom search
    keywords: List[str] = field(default_factory=list)
    custom_prompt: Optional[str] = None

    # Exclusions
    excluded_industries: List[str] = field(default_factory=list)
    excluded_companies: List[str] = field(default_factory=list)
    excluded_behaviors: List[str] = field(default_factory=list)  # ["misconduct", "controversies"]


# Enhanced Company Entry with all new fields
class EnhancedCompanyEntry(BaseModel):
    # Basic info (existing)
    name: str = Field(description="Company name")
    confidence: str = Field(description="Confidence level: absolute, high, medium, low")
    operates_in_country: bool = Field(description="Whether company operates in the specified country")
    business_type: str = Field(description="Type of business")
    industry_category: str = Field(description="Industry category")
    sub_industry: Optional[str] = Field(description="Sub-industry or niche", default=None)

    # Geographic footprint
    headquarters: Optional[Dict[str, Any]] = Field(
        description="HQ location with address and coordinates",
        default=None
    )
    office_locations: List[Any] = Field(
        description="List of office locations - can be strings or dicts",
        default_factory=list
    )
    service_areas: List[str] = Field(
        description="Geographic areas where company provides services",
        default_factory=list
    )

    # Financial profile
    estimated_revenue: Optional[str] = Field(description="Estimated annual revenue range", default=None)
    revenue_currency: Optional[str] = Field(description="Currency of revenue", default="USD")
    estimated_employees: Optional[str] = Field(description="Estimated employee count range", default=None)
    employees_by_location: Optional[Dict[str, str]] = Field(
        description="Employee count by location",
        default=None
    )
    company_size: Optional[str] = Field(description="Company size classification", default="unknown")
    giving_history: List[Dict[str, Any]] = Field(
        description="Historical giving/donation data",
        default_factory=list
    )
    financial_health: Optional[str] = Field(description="Growth, stable, declining", default=None)

    # CSR/ESG Profile
    csr_programs: List[str] = Field(description="CSR programs and initiatives", default_factory=list)
    csr_focus_areas: List[str] = Field(description="Primary CSR focus areas", default_factory=list)
    certifications: List[str] = Field(description="Certifications like B-Corp", default_factory=list)
    esg_score: Optional[float] = Field(description="ESG score if available", default=None)
    esg_maturity: Optional[str] = Field(description="ESG maturity level", default=None)
    community_involvement: List[str] = Field(description="Community programs", default_factory=list)

    # Signals and triggers
    recent_events: List[Any] = Field(
        description="Recent events - can be strings or dicts",
        default_factory=list
    )
    leadership_changes: List[Dict[str, Any]] = Field(
        description="Recent leadership changes",
        default_factory=list
    )
    growth_signals: List[str] = Field(description="Indicators of growth", default_factory=list)

    # ICP Matching
    icp_tier: Optional[str] = Field(description="Tier A, B, C based on criteria match", default=None)
    icp_score: Optional[float] = Field(description="Overall ICP match score 0-100", default=None)
    matched_criteria: List[str] = Field(description="Which criteria were matched", default_factory=list)
    missing_criteria: List[str] = Field(description="Which criteria were not met", default_factory=list)

    # Data quality
    data_freshness: Optional[str] = Field(description="When data was last updated", default=None)
    data_sources: List[str] = Field(description="Sources used for this data", default_factory=list)
    validation_notes: Optional[str] = Field(description="Any validation notes", default=None)

    # Original fields kept for compatibility
    reasoning: str = Field(description="Brief reasoning for confidence level")


class OutputValidator:
    """Validates and fixes AI output formatting issues"""

    def __init__(self):
        self.retry_limit = 3

    def validate_and_fix(self, response: str, expected_type: type) -> Any:
        """Validate and attempt to fix malformed responses"""
        try:
            # First attempt: direct parsing
            if isinstance(response, str):
                parsed = json.loads(response)
            else:
                parsed = response
            return parsed
        except json.JSONDecodeError as e:
            # Try to fix common issues
            return self.fix_malformed_json(response, e)

    def fix_malformed_json(self, response: str, error: json.JSONDecodeError) -> Any:
        """Attempt to fix common JSON formatting issues"""
        # Handle truncated responses
        if "Unterminated string" in str(error) or response.count('{') > response.count('}'):
            # Try to complete the JSON
            if response.rstrip().endswith(','):
                response = response.rstrip()[:-1]

            # Count brackets and try to balance
            open_brackets = response.count('{') - response.count('}')
            open_squares = response.count('[') - response.count(']')

            if open_squares > 0:
                response += ']' * open_squares
            if open_brackets > 0:
                response += '}' * open_brackets

            try:
                return json.loads(response)
            except:
                pass

        # Try to extract JSON from mixed content
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        # Return empty structure if all else fails
        return {"companies": [], "error": "Failed to parse response"}


class StructuredOutputHandler:
    """Handles structured output generation with GPT-4"""

    def __init__(self, client: AzureOpenAI, deployment_name: str):
        self.client = client
        self.deployment_name = deployment_name
        self.validator = OutputValidator()

    async def get_structured_output(self, prompt: str, schema: Dict[str, Any], retry_count: int = 3) -> Any:
        """Get structured output with retries and validation"""
        last_error = None

        for attempt in range(retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system",
                         "content": "You are an expert at finding companies. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                return self.validator.validate_and_fix(content, dict)

            except Exception as e:
                last_error = e
                if attempt < retry_count - 1:
                    # Add error context to prompt for next attempt
                    prompt = f"{prompt}\n\nPrevious attempt failed with: {str(e)}. Please ensure valid JSON."
                    await asyncio.sleep(2 ** attempt)

        # Return partial results if all attempts fail
        return {"companies": [], "error": f"Failed after {retry_count} attempts: {str(last_error)}"}


class EnhancedSearchStrategistAgent:
    """Enhanced agent with comprehensive search capabilities"""

    def __init__(self, deployment_name: str = "gpt-4.1"):
        self.deployment_name = deployment_name
        self.client = None
        self.output_handler = None
        self.validator = OutputValidator()

    def _init_llm(self):
        """Initialize the LLM with Azure OpenAI"""
        try:
            self.client = AzureOpenAI(
                api_key="CUxPxhxqutsvRVHmGQcmH59oMim6mu55PjHTjSpM6y9UwIxwVZIuJQQJ99BFACL93NaXJ3w3AAABACOG3kI1",
                api_version="2024-02-01",
                azure_endpoint="https://amex-openai-2025.openai.azure.com/"
            )
            self.output_handler = StructuredOutputHandler(self.client, self.deployment_name)
            print(f"Successfully initialized Azure OpenAI client with deployment: {self.deployment_name}")
        except Exception as e:
            print(f"Error initializing Azure OpenAI: {str(e)}")
            raise

    def extract_criteria_from_text(self, free_text: str) -> Dict[str, Any]:
        """Extract structured criteria from free text using GPT-4"""
        if not self.client:
            self._init_llm()

        extraction_prompt = f"""
        Extract structured search criteria from this text and categorize them:

        Text: {free_text}

        Extract and return as JSON:
        {{
            "locations": {{
                "countries": [],
                "states": [],
                "cities": [],
                "regions": [],
                "proximity": null
            }},
            "financial": {{
                "revenue_min": null,
                "revenue_max": null,
                "revenue_currency": "USD",
                "giving_capacity_min": null
            }},
            "organizational": {{
                "employee_count_min": null,
                "employee_count_max": null,
                "office_types": []
            }},
            "behavioral": {{
                "csr_focus_areas": [],
                "certifications": [],
                "recent_events": []
            }},
            "industries": [],
            "keywords": [],
            "exclusions": {{
                "industries": [],
                "companies": [],
                "behaviors": []
            }},
            "confidence_scores": {{}}
        }}

        Be thorough but only extract what's explicitly mentioned.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured data from text."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error extracting criteria: {e}")
            return {}

    async def generate_enhanced_strategy(
            self,
            criteria: SearchCriteria,
            target_count: int = 100
    ) -> Dict[str, Any]:
        """Generate search strategy with enhanced criteria"""
        if not self.client:
            self._init_llm()

        # Build comprehensive search prompt
        prompt = self._build_enhanced_prompt(criteria, target_count)

        # Get structured output
        result = await self.output_handler.get_structured_output(
            prompt,
            schema={"type": "object", "properties": {"companies": {"type": "array"}}}
        )

        # Process and enhance company entries
        enhanced_companies = []
        for company_data in result.get("companies", []):
            try:
                company = EnhancedCompanyEntry(**company_data)
                # Calculate ICP score
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
                "deployment": self.deployment_name
            }
        }

    def _build_enhanced_prompt(self, criteria: SearchCriteria, target_count: int) -> str:
        """Build comprehensive prompt from criteria"""
        prompt_parts = [f"Find {target_count} companies matching these criteria:"]

        # Location criteria
        if criteria.location.cities:
            prompt_parts.append(f"Cities: {', '.join(criteria.location.cities)}")
        elif criteria.location.states:
            prompt_parts.append(f"States/Regions: {', '.join(criteria.location.states)}")
        elif criteria.location.countries:
            prompt_parts.append(f"Countries: {', '.join(criteria.location.countries)}")

        if criteria.location.proximity:
            prompt_parts.append(
                f"Within {criteria.location.proximity['radius_km']}km of {criteria.location.proximity['location']}")

        # Financial criteria
        if criteria.financial.revenue_min or criteria.financial.revenue_max:
            rev_range = f"{criteria.financial.revenue_currency} "
            if criteria.financial.revenue_min and criteria.financial.revenue_max:
                rev_range += f"{criteria.financial.revenue_min:,.0f} - {criteria.financial.revenue_max:,.0f}"
            elif criteria.financial.revenue_min:
                rev_range += f"{criteria.financial.revenue_min:,.0f}+"
            else:
                rev_range += f"up to {criteria.financial.revenue_max:,.0f}"
            prompt_parts.append(f"Revenue: {rev_range}")

        # Employee criteria
        if criteria.organizational.employee_count_min or criteria.organizational.employee_count_max:
            if criteria.organizational.employee_count_min and criteria.organizational.employee_count_max:
                emp_range = f"{criteria.organizational.employee_count_min} - {criteria.organizational.employee_count_max} employees"
            elif criteria.organizational.employee_count_min:
                emp_range = f"{criteria.organizational.employee_count_min}+ employees"
            else:
                emp_range = f"up to {criteria.organizational.employee_count_max} employees"
            prompt_parts.append(emp_range)

        # Industry priorities
        if criteria.industries:
            industry_list = sorted(criteria.industries, key=lambda x: x.get('priority', 999))
            prompt_parts.append("Industries (by priority):")
            for ind in industry_list:
                prompt_parts.append(f"  {ind.get('priority', '-')}. {ind['name']}")

        # CSR/ESG criteria
        if criteria.behavioral.csr_focus_areas:
            prompt_parts.append(f"CSR focus on: {', '.join(criteria.behavioral.csr_focus_areas)}")

        if criteria.behavioral.certifications:
            prompt_parts.append(f"Certifications: {', '.join(criteria.behavioral.certifications)}")

        # Custom prompt
        if criteria.custom_prompt:
            prompt_parts.append(f"\nAdditional requirements: {criteria.custom_prompt}")

        # Exclusions
        if criteria.excluded_industries:
            prompt_parts.append(f"\nExclude industries: {', '.join(criteria.excluded_industries)}")

        # Output format - EXPLICIT REQUIRED FIELDS
        prompt_parts.append("""

Return a JSON object with a "companies" array. Each company MUST have these fields:
{
    "companies": [
        {
            "name": "Company Name",
            "confidence": "high",
            "operates_in_country": true,
            "business_type": "B2C",
            "industry_category": "Retail",
            "reasoning": "Why this company matches",
            "estimated_revenue": "$50M-$100M",
            "estimated_employees": "200-500",
            "company_size": "medium",
            "headquarters": {"city": "London", "address": "123 Main St"},
            "office_locations": ["London", "Manchester"],
            "service_areas": ["UK", "Ireland"],
            "csr_programs": ["community support"],
            "csr_focus_areas": ["children", "education"],
            "certifications": [],
            "recent_events": [],
            "data_sources": ["Company website"]
        }
    ]
}

IMPORTANT: 
- Return ONLY valid JSON
- Each company MUST have: name, confidence, operates_in_country, business_type, industry_category, reasoning
- confidence must be one of: absolute, high, medium, low
- operates_in_country must be true or false
- office_locations can be a simple array of city names
        """)

        return "\n".join(prompt_parts)

    def _calculate_icp_score(self, company: EnhancedCompanyEntry, criteria: SearchCriteria) -> EnhancedCompanyEntry:
        """Calculate ICP score and tier for a company"""
        score = 0
        max_score = 0
        matched = []
        missing = []

        # Location match (20 points)
        max_score += 20
        if company.headquarters:
            # Check location matches
            location_matched = False
            if criteria.location.cities:
                if any(city.lower() in company.headquarters.get('city', '').lower() for city in
                       criteria.location.cities):
                    score += 20
                    matched.append("Location - City match")
                    location_matched = True
            elif criteria.location.countries:
                if company.operates_in_country:
                    score += 15
                    matched.append("Location - Country match")
                    location_matched = True

            if not location_matched:
                missing.append("Location match")

        # Revenue match (20 points)
        max_score += 20
        if company.estimated_revenue and (criteria.financial.revenue_min or criteria.financial.revenue_max):
            # Parse revenue from company
            rev_matched = True  # Simplified - in production, parse and compare
            if rev_matched:
                score += 20
                matched.append("Revenue range")
            else:
                missing.append("Revenue requirements")

        # Employee count (15 points)
        max_score += 15
        if company.estimated_employees and (
                criteria.organizational.employee_count_min or criteria.organizational.employee_count_max):
            emp_matched = True  # Simplified
            if emp_matched:
                score += 15
                matched.append("Employee count")
            else:
                missing.append("Employee requirements")

        # Industry match (15 points)
        max_score += 15
        if criteria.industries:
            industry_names = [ind['name'].lower() for ind in criteria.industries]
            if any(ind in company.industry_category.lower() for ind in industry_names):
                score += 15
                matched.append("Industry match")
            else:
                missing.append("Industry match")

        # CSR/ESG match (30 points)
        max_score += 30
        csr_score = 0

        # CSR focus areas (15 points)
        if criteria.behavioral.csr_focus_areas and company.csr_focus_areas:
            matching_areas = set(criteria.behavioral.csr_focus_areas) & set(company.csr_focus_areas)
            if matching_areas:
                csr_score += 15
                matched.append(f"CSR focus areas: {', '.join(matching_areas)}")
            else:
                missing.append("CSR focus area match")

        # Certifications (10 points)
        if criteria.behavioral.certifications and company.certifications:
            matching_certs = set(criteria.behavioral.certifications) & set(company.certifications)
            if matching_certs:
                csr_score += 10
                matched.append(f"Certifications: {', '.join(matching_certs)}")
            else:
                missing.append("Required certifications")

        # ESG maturity (5 points)
        if criteria.behavioral.esg_maturity and company.esg_maturity:
            if company.esg_maturity in ["Mature", "Leading"]:
                csr_score += 5
                matched.append("ESG maturity")

        score += csr_score

        # Calculate final score
        final_score = (score / max_score) * 100 if max_score > 0 else 0

        # Determine tier
        if final_score >= 80:
            tier = "A"
        elif final_score >= 60:
            tier = "B"
        elif final_score >= 40:
            tier = "C"
        else:
            tier = "D"

        # Update company object
        company.icp_score = round(final_score, 1)
        company.icp_tier = tier
        company.matched_criteria = matched
        company.missing_criteria = missing

        return company


# Backward compatibility
SearchStrategistAgent = EnhancedSearchStrategistAgent