# search_strategist_agent.py - FIXED VERSION with proper imports

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

# Azure OpenAI import - using correct syntax for v1.x
try:
    from openai import AzureOpenAI
except ImportError:
    # Fallback for older versions
    import openai

    AzureOpenAI = None
    print("Warning: AzureOpenAI not available, trying legacy mode")


# ============================================================================
# Data Models - Define all classes locally to avoid circular imports
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
    sub_industry: Optional[str] = Field(description="Sub-industry or niche", default=None)
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


# ============================================================================
# Agent Classes
# ============================================================================

class OutputValidator:
    """Validates and fixes AI output formatting issues"""

    def __init__(self):
        self.retry_limit = 3

    def validate_and_fix(self, response: str, expected_type: type) -> Any:
        """Validate and attempt to fix malformed responses"""
        try:
            if isinstance(response, str):
                parsed = json.loads(response)
            else:
                parsed = response
            return parsed
        except json.JSONDecodeError as e:
            return self.fix_malformed_json(response, e)

    def fix_malformed_json(self, response: str, error: json.JSONDecodeError) -> Any:
        """Attempt to fix common JSON formatting issues"""
        if "Unterminated string" in str(error) or response.count('{') > response.count('}'):
            if response.rstrip().endswith(','):
                response = response.rstrip()[:-1]

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

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        return {"companies": [], "error": "Failed to parse response"}


class StructuredOutputHandler:
    """Handles structured output generation with GPT-4"""

    def __init__(self, client, deployment_name: str):
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
                        {"role": "system", "content": "You are a company finder. Respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=4000,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                result = self.validator.validate_and_fix(content, dict)

                if result and "companies" in result and len(result["companies"]) > 0:
                    return result
                elif attempt < retry_count - 1:
                    prompt = self._simplify_prompt(prompt)
                    await asyncio.sleep(1)
                else:
                    return result

            except Exception as e:
                last_error = e
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)

        return {"companies": [], "error": f"Failed after {retry_count} attempts: {str(last_error)}"}

    def _simplify_prompt(self, prompt: str) -> str:
        """Simplify prompt for retry"""
        lines = prompt.splitlines()
        simplified = []

        for line in lines[:10]:
            simplified.append(line)

        simplified.append(
            "\nReturn 5 companies as JSON with these fields: name, confidence, operates_in_country, business_type, industry_category, reasoning")

        return "\n".join(simplified)


class EnhancedSearchStrategistAgent:
    """Enhanced agent with comprehensive search capabilities"""

    def __init__(self, deployment_name: str = "gpt-4.1"):
        self.deployment_name = deployment_name
        self.client = None
        self.output_handler = None
        self.validator = OutputValidator()
        self.initialized = False

    def _init_llm(self):
        """Initialize the LLM with Azure OpenAI"""
        if self.initialized:
            return

        try:
            # Get credentials from environment or use defaults
            api_key = os.getenv("AZURE_OPENAI_KEY",
                                "CUxPxhxqutsvRVHmGQcmH59oMim6mu55PjHTjSpM6y9UwIxwVZIuJQQJ99BFACL93NaXJ3w3AAABACOG3kI1")
            api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://amex-openai-2025.openai.azure.com/")

            if AzureOpenAI:
                # Use new v1.x syntax
                self.client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint
                )
            else:
                # Fallback to old syntax if needed
                import openai
                openai.api_type = "azure"
                openai.api_key = api_key
                openai.api_version = api_version
                openai.api_base = azure_endpoint
                self.client = openai

            self.output_handler = StructuredOutputHandler(self.client, self.deployment_name)
            self.initialized = True
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
            }}
        }}

        Be thorough but only extract what's explicitly mentioned.
        """

        try:
            if AzureOpenAI and hasattr(self.client, 'chat'):
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "Extract structured data from text. Return valid JSON."},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
            else:
                # Fallback for older OpenAI library
                response = openai.ChatCompletion.create(
                    engine=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "Extract structured data from text. Return valid JSON."},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    temperature=0.1
                )
                result = json.loads(response['choices'][0]['message']['content'])

            if not isinstance(result, dict):
                return self._get_default_extraction()

            for key in ['locations', 'financial', 'organizational', 'behavioral']:
                if key not in result or not isinstance(result[key], dict):
                    result[key] = self._get_default_extraction()[key]

            return result

        except Exception as e:
            print(f"Error extracting criteria: {e}")
            return self._get_default_extraction()

    def _get_default_extraction(self) -> Dict[str, Any]:
        """Return default extraction structure"""
        return {
            "locations": {
                "countries": [],
                "states": [],
                "cities": [],
                "regions": [],
                "proximity": None
            },
            "financial": {
                "revenue_min": None,
                "revenue_max": None,
                "revenue_currency": "USD",
                "giving_capacity_min": None
            },
            "organizational": {
                "employee_count_min": None,
                "employee_count_max": None,
                "office_types": []
            },
            "behavioral": {
                "csr_focus_areas": [],
                "certifications": [],
                "recent_events": []
            },
            "industries": [],
            "keywords": [],
            "exclusions": {
                "industries": [],
                "companies": [],
                "behaviors": []
            }
        }

    async def generate_enhanced_strategy(
            self,
            criteria: SearchCriteria,
            target_count: int = 100
    ) -> Dict[str, Any]:
        """Generate search strategy with enhanced criteria"""
        if not self.client:
            self._init_llm()

        prompt = self._build_enhanced_prompt_fixed(criteria, target_count)

        result = await self.output_handler.get_structured_output(
            prompt,
            schema={"type": "object", "properties": {"companies": {"type": "array"}}}
        )

        enhanced_companies = []
        for company_data in result.get("companies", []):
            try:
                company_data = self._ensure_company_fields(company_data)
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
                "deployment": self.deployment_name
            }
        }

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
            "validation_notes": None
        }

        for key, default_value in defaults.items():
            if key not in company_data:
                company_data[key] = default_value

        return company_data

    def _build_enhanced_prompt_fixed(self, criteria: SearchCriteria, target_count: int) -> str:
        """Build comprehensive prompt from criteria with better formatting"""
        prompt_parts = []

        prompt_parts.append(f"Find {target_count} companies with these requirements:")
        prompt_parts.append("")

        # Location
        location_specs = []
        if criteria.location.countries:
            location_specs.append(f"Country: {', '.join(criteria.location.countries)}")
        if criteria.location.states:
            location_specs.append(f"State: {', '.join(criteria.location.states)}")
        if criteria.location.cities:
            location_specs.append(f"City: {', '.join(criteria.location.cities)}")
        if criteria.location.regions:
            location_specs.append(f"Region: {', '.join(criteria.location.regions)}")

        if location_specs:
            prompt_parts.append("LOCATION:")
            prompt_parts.extend(location_specs)
            prompt_parts.append("")

        # Financial
        if criteria.financial.revenue_min or criteria.financial.revenue_max:
            prompt_parts.append("REVENUE:")
            if criteria.financial.revenue_min and criteria.financial.revenue_max:
                min_m = int(criteria.financial.revenue_min / 1_000_000)
                max_m = int(criteria.financial.revenue_max / 1_000_000)
                prompt_parts.append(f"Between {min_m} and {max_m} million {criteria.financial.revenue_currency}")
            elif criteria.financial.revenue_min:
                min_m = int(criteria.financial.revenue_min / 1_000_000)
                prompt_parts.append(f"Above {min_m} million {criteria.financial.revenue_currency}")
            else:
                max_m = int(criteria.financial.revenue_max / 1_000_000)
                prompt_parts.append(f"Up to {max_m} million {criteria.financial.revenue_currency}")
            prompt_parts.append("")

        # Employees
        if criteria.organizational.employee_count_min or criteria.organizational.employee_count_max:
            prompt_parts.append("EMPLOYEES:")
            if criteria.organizational.employee_count_min and criteria.organizational.employee_count_max:
                prompt_parts.append(
                    f"Between {criteria.organizational.employee_count_min} and {criteria.organizational.employee_count_max}")
            elif criteria.organizational.employee_count_min:
                prompt_parts.append(f"At least {criteria.organizational.employee_count_min}")
            else:
                prompt_parts.append(f"Up to {criteria.organizational.employee_count_max}")
            prompt_parts.append("")

        # Industries
        if criteria.industries:
            prompt_parts.append("INDUSTRIES:")
            for ind in criteria.industries[:5]:
                prompt_parts.append(f"- {ind['name']}")
            prompt_parts.append("")

        # Business types
        if criteria.business_types:
            prompt_parts.append(f"BUSINESS TYPES: {', '.join(criteria.business_types)}")
            prompt_parts.append("")

        # CSR
        if criteria.behavioral.csr_focus_areas:
            prompt_parts.append(f"CSR FOCUS: {', '.join(criteria.behavioral.csr_focus_areas)}")
            prompt_parts.append("")

        # Exclusions
        if criteria.excluded_industries:
            prompt_parts.append(f"EXCLUDE: {', '.join(criteria.excluded_industries)}")
            prompt_parts.append("")

        # JSON format instruction
        prompt_parts.append("Return companies as JSON in this exact format:")
        prompt_parts.append("""
{
  "companies": [
    {
      "name": "Example Company Pty Ltd",
      "confidence": "high",
      "operates_in_country": true,
      "business_type": "B2B",
      "industry_category": "Manufacturing",
      "reasoning": "Australian manufacturing company with 200 employees",
      "estimated_revenue": "50-100M",
      "estimated_employees": "100-500",
      "company_size": "medium",
      "headquarters": {"city": "Melbourne"},
      "office_locations": ["Melbourne"],
      "csr_programs": [],
      "csr_focus_areas": [],
      "certifications": [],
      "recent_events": [],
      "data_sources": ["public records"]
    }
  ]
}

Important: Return ONLY valid JSON. Each company must have ALL the fields shown above.""")

        return "\n".join(prompt_parts)

    def _calculate_icp_score(self, company: EnhancedCompanyEntry, criteria: SearchCriteria) -> EnhancedCompanyEntry:
        """Calculate ICP score and tier for a company"""
        score = 0
        max_score = 0
        matched = []
        missing = []

        # Location match (20 points)
        max_score += 20
        location_matched = False

        if company.headquarters or company.office_locations:
            company_locations = str(company.headquarters).lower() if company.headquarters else ""
            company_locations += " " + " ".join(str(loc).lower() for loc in company.office_locations)

            if criteria.location.cities:
                if any(city.lower() in company_locations for city in criteria.location.cities):
                    score += 20
                    matched.append("Location - City match")
                    location_matched = True

            if not location_matched and criteria.location.regions:
                if any(region.lower() in company_locations for region in criteria.location.regions):
                    score += 20
                    matched.append("Location - Region match")
                    location_matched = True

            if not location_matched and criteria.location.states:
                if any(state.lower() in company_locations for state in criteria.location.states):
                    score += 15
                    matched.append("Location - State match")
                    location_matched = True

            if not location_matched and company.operates_in_country:
                score += 10
                matched.append("Location - Country match")
                location_matched = True

        if not location_matched:
            missing.append("Location match")

        # Revenue match (20 points)
        max_score += 20
        if company.estimated_revenue and (criteria.financial.revenue_min or criteria.financial.revenue_max):
            score += 20
            matched.append("Revenue range")

        # Employee count (15 points)
        max_score += 15
        if company.estimated_employees and (
                criteria.organizational.employee_count_min or criteria.organizational.employee_count_max):
            score += 15
            matched.append("Employee count")

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

        if criteria.behavioral.csr_focus_areas and company.csr_focus_areas:
            matching_areas = set(criteria.behavioral.csr_focus_areas) & set(company.csr_focus_areas)
            if matching_areas:
                csr_score += 15
                matched.append(f"CSR focus areas: {', '.join(matching_areas)}")

        if criteria.behavioral.certifications and company.certifications:
            matching_certs = set(criteria.behavioral.certifications) & set(company.certifications)
            if matching_certs:
                csr_score += 10
                matched.append(f"Certifications: {', '.join(matching_certs)}")

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


# Backward compatibility alias
SearchStrategistAgent = EnhancedSearchStrategistAgent