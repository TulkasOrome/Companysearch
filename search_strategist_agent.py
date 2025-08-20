# search_strategist_agent.py

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field


# Business Type Enum
class BusinessType(Enum):
    B2C = "B2C"
    B2B = "B2B"
    B2B2C = "B2B2C"
    D2C = "D2C"
    MARKETPLACE = "Marketplace"
    HYBRID = "Hybrid"


# Company size classifications
class CompanySize(Enum):
    SMALL = "small"  # 1-50 employees or <$10M revenue
    MEDIUM = "medium"  # 51-500 employees or $10M-$100M revenue
    ENTERPRISE = "enterprise"  # 500+ employees or $100M+ revenue
    UNKNOWN = "unknown"  # No size data available


# Target countries configuration with population for proportional distribution
TARGET_COUNTRIES = {
    # North America
    "United States": {"languages": ["en"], "region": "North America", "population": 331000000},
    "Canada": {"languages": ["en", "fr"], "region": "North America", "population": 38000000},
    "Mexico": {"languages": ["es", "en"], "region": "North America", "population": 128000000},

    # Western Europe
    "United Kingdom": {"languages": ["en"], "region": "Western Europe", "population": 67000000},
    "Germany": {"languages": ["de", "en"], "region": "Western Europe", "population": 83000000},
    "France": {"languages": ["fr", "en"], "region": "Western Europe", "population": 67000000},
    "Netherlands": {"languages": ["nl", "en"], "region": "Western Europe", "population": 17500000},
    "Belgium": {"languages": ["nl", "fr", "en"], "region": "Western Europe", "population": 11500000},
    "Austria": {"languages": ["de", "en"], "region": "Western Europe", "population": 9000000},
    "Switzerland": {"languages": ["de", "fr", "it", "en"], "region": "Western Europe", "population": 8700000},
    "Luxembourg": {"languages": ["fr", "de", "en"], "region": "Western Europe", "population": 640000},

    # Southern Europe
    "Spain": {"languages": ["es", "en"], "region": "Southern Europe", "population": 47000000},
    "Italy": {"languages": ["it", "en"], "region": "Southern Europe", "population": 60000000},
    "Portugal": {"languages": ["pt", "en"], "region": "Southern Europe", "population": 10300000},
    "Greece": {"languages": ["el", "en"], "region": "Southern Europe", "population": 10700000},

    # Eastern Europe
    "Poland": {"languages": ["pl", "en"], "region": "Eastern Europe", "population": 38000000},
    "Romania": {"languages": ["ro", "en"], "region": "Eastern Europe", "population": 19000000},
    "Czech Republic": {"languages": ["cs", "en"], "region": "Eastern Europe", "population": 10700000},
    "Hungary": {"languages": ["hu", "en"], "region": "Eastern Europe", "population": 9700000},
    "Bulgaria": {"languages": ["bg", "en"], "region": "Eastern Europe", "population": 6900000},
    "Slovakia": {"languages": ["sk", "en"], "region": "Eastern Europe", "population": 5400000},
    "Croatia": {"languages": ["hr", "en"], "region": "Eastern Europe", "population": 4000000},

    # Nordic
    "Denmark": {"languages": ["da", "en"], "region": "Nordic", "population": 5800000},
    "Sweden": {"languages": ["sv", "en"], "region": "Nordic", "population": 10400000},
    "Finland": {"languages": ["fi", "en"], "region": "Nordic", "population": 5500000},
    "Norway": {"languages": ["no", "en"], "region": "Nordic", "population": 5400000},
    "Iceland": {"languages": ["is", "en"], "region": "Nordic", "population": 370000},

    # Asia Pacific
    "Japan": {"languages": ["ja", "en"], "region": "Asia Pacific", "population": 125000000},
    "South Korea": {"languages": ["ko", "en"], "region": "Asia Pacific", "population": 51700000},
    "China": {"languages": ["zh", "en"], "region": "Asia Pacific", "population": 1412000000},
    "India": {"languages": ["en", "hi"], "region": "Asia Pacific", "population": 1380000000},
    "Australia": {"languages": ["en"], "region": "Asia Pacific", "population": 25700000},
    "New Zealand": {"languages": ["en"], "region": "Asia Pacific", "population": 5100000},
    "Singapore": {"languages": ["en", "zh"], "region": "Asia Pacific", "population": 5700000},
    "Malaysia": {"languages": ["ms", "en"], "region": "Asia Pacific", "population": 32700000},
    "Thailand": {"languages": ["th", "en"], "region": "Asia Pacific", "population": 70000000},
    "Indonesia": {"languages": ["id", "en"], "region": "Asia Pacific", "population": 273000000},
    "Philippines": {"languages": ["en", "tl"], "region": "Asia Pacific", "population": 110000000},
    "Vietnam": {"languages": ["vi", "en"], "region": "Asia Pacific", "population": 97300000},

    # Middle East & Africa
    "United Arab Emirates": {"languages": ["ar", "en"], "region": "Middle East", "population": 9900000},
    "Saudi Arabia": {"languages": ["ar", "en"], "region": "Middle East", "population": 34800000},
    "Israel": {"languages": ["he", "en"], "region": "Middle East", "population": 9300000},
    "South Africa": {"languages": ["en", "af"], "region": "Africa", "population": 59300000},
    "Nigeria": {"languages": ["en"], "region": "Africa", "population": 206000000},
    "Egypt": {"languages": ["ar", "en"], "region": "Middle East", "population": 102000000},
    "Turkey": {"languages": ["tr", "en"], "region": "Middle East", "population": 84300000},

    # Latin America
    "Brazil": {"languages": ["pt", "en"], "region": "Latin America", "population": 212000000},
    "Argentina": {"languages": ["es", "en"], "region": "Latin America", "population": 45400000},
    "Chile": {"languages": ["es", "en"], "region": "Latin America", "population": 19100000},
    "Colombia": {"languages": ["es", "en"], "region": "Latin America", "population": 50900000},
    "Peru": {"languages": ["es", "en"], "region": "Latin America", "population": 33000000},

    # Other
    "Russia": {"languages": ["ru", "en"], "region": "Eastern Europe", "population": 144000000},
    "Ireland": {"languages": ["en"], "region": "Western Europe", "population": 5000000},
}

# Industry taxonomies by business type
INDUSTRY_TAXONOMIES = {
    BusinessType.B2C: {
        "Retail": ["Fashion & Apparel", "Electronics", "Home & Garden", "Sports & Outdoors",
                   "Books & Media", "Toys & Games", "Pet Supplies", "General Merchandise"],
        "Food & Beverage": ["Supermarkets", "Restaurants", "QSR/Fast Food", "Cafes & Coffee Shops",
                            "Specialty Food", "Liquor Stores", "Bakeries", "Food Delivery"],
        "Health & Beauty": ["Pharmacies", "Cosmetics", "Health Stores", "Optical",
                            "Beauty Salons", "Wellness Centers", "Medical Supplies"],
        "Hospitality": ["Hotels", "Resorts", "Vacation Rentals", "Travel Agencies",
                        "Entertainment Venues", "Theme Parks", "Cinemas"],
        "Services": ["Banking", "Insurance", "Telecommunications", "Utilities",
                     "Education", "Fitness Centers", "Car Rentals"]
    },
    BusinessType.B2B: {
        "Technology": ["Software", "SaaS", "IT Services", "Cloud Infrastructure",
                       "Cybersecurity", "Data Analytics", "AI/ML Solutions"],
        "Manufacturing": ["Industrial Equipment", "Components", "Raw Materials",
                          "Packaging", "Chemicals", "Textiles"],
        "Professional Services": ["Consulting", "Legal", "Accounting", "Marketing Agencies",
                                  "Staffing", "Engineering", "Architecture"],
        "Wholesale & Distribution": ["Food Distribution", "Industrial Supplies",
                                     "Medical Supplies", "Electronics Distribution"],
        "Business Operations": ["Logistics", "Facilities Management", "Payroll Services",
                                "Office Supplies", "Commercial Real Estate"]
    },
    BusinessType.D2C: {
        "Consumer Brands": ["Fashion Brands", "Beauty Brands", "Food & Beverage Brands",
                            "Electronics Brands", "Home Goods Brands", "Wellness Brands"],
        "Subscription Services": ["Meal Kits", "Beauty Boxes", "Media Subscriptions",
                                  "Software Subscriptions", "Fitness Programs"]
    },
    BusinessType.MARKETPLACE: {
        "E-commerce": ["General Marketplaces", "Fashion Marketplaces", "Food Delivery Platforms",
                       "Service Marketplaces", "B2B Marketplaces", "Local Marketplaces"]
    }
}


# Distribution modes
class DistributionMode(Enum):
    EQUAL = "equal"  # Equal distribution across countries
    POPULATION = "population"  # Proportional to population
    CUSTOM = "custom"  # User-defined per country


# Output schemas
class CompanyEntry(BaseModel):
    name: str = Field(description="Company name")
    confidence: str = Field(description="Confidence level: absolute, high, medium, low")
    operates_in_country: bool = Field(description="Whether company operates in the specified country")
    business_type: str = Field(description="Type of business: B2C, B2B, etc.")
    industry_category: str = Field(description="Industry category")
    estimated_revenue: Optional[str] = Field(description="Estimated annual revenue range", default=None)
    estimated_employees: Optional[str] = Field(description="Estimated employee count range", default=None)
    company_size: Optional[str] = Field(description="Company size: small, medium, enterprise, unknown",
                                        default="unknown")
    reasoning: str = Field(description="Brief reasoning for confidence level")

    def classify_size(self):
        """Classify company size based on revenue and employees"""
        # Try to classify by employees first
        if self.estimated_employees:
            emp_str = self.estimated_employees.lower()
            if any(x in emp_str for x in ["1-10", "11-50", "1-50"]):
                self.company_size = CompanySize.SMALL.value
            elif any(x in emp_str for x in ["51-200", "201-500", "51-500"]):
                self.company_size = CompanySize.MEDIUM.value
            elif any(x in emp_str for x in ["501-1000", "1000+", "5000+", "10000+"]):
                self.company_size = CompanySize.ENTERPRISE.value

        # If no employee classification, try revenue
        if self.company_size == "unknown" and self.estimated_revenue:
            rev_str = self.estimated_revenue.lower()
            if any(x in rev_str for x in ["$1m-$10m", "<$10m", "$5m"]):
                self.company_size = CompanySize.SMALL.value
            elif any(x in rev_str for x in ["$10m-$50m", "$50m-$100m", "$10m-$100m"]):
                self.company_size = CompanySize.MEDIUM.value
            elif any(x in rev_str for x in ["$100m", "$500m", "$1b", "billion"]):
                self.company_size = CompanySize.ENTERPRISE.value


class SearchStrategy(BaseModel):
    country: str = Field(description="Target country")
    business_type: str = Field(description="Business type being searched")
    industry: str = Field(description="Industry being searched")
    known_companies: List[CompanyEntry] = Field(description="Companies from AI knowledge")
    search_queries: List[str] = Field(description="Suggested search queries for validation")
    search_patterns: List[str] = Field(description="General patterns to search for")
    industry_keywords: Dict[str, List[str]] = Field(description="Industry-specific keywords by language")
    estimated_total: int = Field(description="Estimated total companies in this segment")


@dataclass
class SearchProgress:
    """Track progress across multiple searches"""
    total_companies_found: int = 0
    companies_by_country: Dict[str, int] = field(default_factory=dict)
    companies_by_confidence: Dict[str, int] = field(default_factory=dict)
    companies_by_size: Dict[str, int] = field(default_factory=dict)
    elapsed_time: float = 0
    api_calls: int = 0
    estimated_cost: float = 0
    deployments_used: List[str] = field(default_factory=list)
    avg_time_per_search: float = 0

    def update(self, strategy: SearchStrategy):
        """Update progress with results from a strategy"""
        self.total_companies_found += len(strategy.known_companies)
        country = strategy.country
        self.companies_by_country[country] = self.companies_by_country.get(country, 0) + len(strategy.known_companies)

        for company in strategy.known_companies:
            conf = company.confidence
            self.companies_by_confidence[conf] = self.companies_by_confidence.get(conf, 0) + 1

            size = company.company_size
            self.companies_by_size[size] = self.companies_by_size.get(size, 0) + 1

        self.api_calls += 1
        # Update average time
        if self.api_calls > 0:
            self.avg_time_per_search = self.elapsed_time / self.api_calls
        # Rough cost estimate: $0.01 per 1K tokens, assume ~2K tokens per call
        self.estimated_cost += 0.02


class SearchStrategistAgent:
    def __init__(self, deployment_name: str = "gpt-4.1"):
        self.deployment_name = deployment_name
        self.client = None  # Don't initialize yet
        self.progress = SearchProgress()

    def _init_llm(self):
        """Initialize the LLM with Azure OpenAI"""
        try:
            from openai import AzureOpenAI

            # Using the exact pattern from your NRF code
            self.client = AzureOpenAI(
                api_key="CUxPxhxqutsvRVHmGQcmH59oMim6mu55PjHTjSpM6y9UwIxwVZIuJQQJ99BFACL93NaXJ3w3AAABACOG3kI1",
                api_version="2024-02-01",
                azure_endpoint="https://amex-openai-2025.openai.azure.com/"
            )

            print(f"Successfully initialized Azure OpenAI client with deployment: {self.deployment_name}")
        except Exception as e:
            print(f"Error initializing Azure OpenAI: {str(e)}")
            raise

    async def generate_strategy(self,
                                country: str,
                                business_type: BusinessType,
                                industry: str,
                                sub_industry: Optional[str] = None,
                                target_count: int = 100,
                                include_financials: bool = True,
                                size_preferences: List[str] = None) -> SearchStrategy:
        """Generate a search strategy for finding companies"""
        start_time = time.time()

        # Initialize client if not already done
        if self.client is None:
            self._init_llm()

        # Use sub-industry if provided, otherwise use main industry
        search_industry = f"{industry} - {sub_industry}" if sub_industry else industry

        # Build size focus string
        size_focus = ""
        if size_preferences and "All Sizes" not in size_preferences:
            size_definitions = {
                "Small": "small companies (1-50 employees or <$10M revenue)",
                "Medium": "medium companies (51-500 employees or $10M-$100M revenue)",
                "Enterprise": "enterprise companies (500+ employees or $100M+ revenue)"
            }
            focused_sizes = [size_definitions.get(s, s) for s in size_preferences]
            size_focus = f"\nFocus particularly on {', '.join(focused_sizes)}."

        # Simple system message that mentions JSON
        system_message = f"You are an expert at identifying {business_type.value} companies globally. Always respond with valid JSON."

        # Simple user prompt with clear example - limit companies per request
        # Cap at 30 companies per request to avoid token limits
        actual_target = min(target_count, 30)

        user_message = f"""List {actual_target} {business_type.value} companies in {search_industry} industry in {country}.{size_focus}

For each company provide:
- name: Company name
- confidence: "absolute" for major brands, "high" for well-known, "medium" for fairly sure, "low" for uncertain
- operates_in_country: true/false
- business_type: "{business_type.value}"
- industry_category: "{search_industry}"
- estimated_revenue: {"revenue range like $10M-$50M" if include_financials else "null"}
- estimated_employees: {"employee range like 51-200" if include_financials else "null"}
- reasoning: Brief explanation

Example format:
{{
    "companies": [
        {{
            "name": "Example Company",
            "confidence": "high",
            "operates_in_country": true,
            "business_type": "{business_type.value}",
            "industry_category": "{search_industry}",
            "estimated_revenue": {"\"$10M-$50M\"" if include_financials else "null"},
            "estimated_employees": {"\"51-200\"" if include_financials else "null"},
            "reasoning": "Well-known company in this market"
        }}
    ]
}}"""

        try:
            # Call Azure OpenAI with json_object mode
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"}  # Force JSON response
            )

            # Parse the response
            content = response.choices[0].message.content

            # Try to parse JSON
            try:
                parsed_data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")

                # Try to fix truncated JSON
                if "Unterminated string" in str(e) or "Expecting property name" in str(e) or content.count(
                        '{') > content.count('}'):
                    print("Attempting to fix truncated JSON...")

                    # Method 1: Find the last complete company entry
                    last_complete_bracket = content.rfind('},')
                    if last_complete_bracket > 0:
                        # Truncate after the last complete company and properly close the JSON
                        fixed_content = content[:last_complete_bracket + 1] + ']}'
                        try:
                            parsed_data = json.loads(fixed_content)
                            print(f"Successfully recovered {len(parsed_data.get('companies', []))} companies")
                        except:
                            # Method 2: Try to find companies array and extract it
                            try:
                                companies_start = content.find('"companies"') + len('"companies"')
                                companies_start = content.find('[', companies_start)
                                if companies_start > 0:
                                    # Find matching bracket or last complete entry
                                    bracket_count = 0
                                    last_valid_pos = companies_start
                                    for i, char in enumerate(content[companies_start:], companies_start):
                                        if char == '[':
                                            bracket_count += 1
                                        elif char == ']':
                                            bracket_count -= 1
                                            if bracket_count == 0:
                                                last_valid_pos = i
                                                break
                                        elif char == '}' and content[i - 1] == '}':
                                            last_valid_pos = i

                                    companies_content = content[companies_start:last_valid_pos + 1]
                                    if not companies_content.endswith(']'):
                                        companies_content += ']'

                                    parsed_data = {"companies": json.loads(companies_content)}
                                    print(f"Successfully extracted {len(parsed_data.get('companies', []))} companies")
                                else:
                                    parsed_data = {"companies": []}
                            except:
                                parsed_data = {"companies": []}
                    else:
                        parsed_data = {"companies": []}
                else:
                    # Other JSON errors
                    parsed_data = {"companies": []}

            # Handle wrapped response - find the companies array
            companies = []
            if isinstance(parsed_data, dict):
                if "companies" in parsed_data:
                    companies = parsed_data["companies"]
                else:
                    # Find any key that contains a list
                    for key, value in parsed_data.items():
                        if isinstance(value, list):
                            companies = value
                            break

            # Convert to CompanyEntry objects
            company_entries = []
            for company_data in companies:
                try:
                    company = CompanyEntry(**company_data)
                    company.classify_size()  # Auto-classify size
                    company_entries.append(company)
                except Exception as e:
                    print(f"Error creating company entry: {e}")
                    continue

            # Create SearchStrategy
            strategy = SearchStrategy(
                country=country,
                business_type=business_type.value,
                industry=search_industry,
                known_companies=company_entries,
                search_queries=[f"{search_industry} {country}"],
                search_patterns=[f"{business_type.value} {search_industry} companies {country}"],
                industry_keywords={"en": [search_industry], "local": []},
                estimated_total=len(company_entries)
            )

            # Update progress
            self.progress.update(strategy)
            self.progress.elapsed_time += (time.time() - start_time)

            return strategy

        except Exception as e:
            print(f"Error generating strategy: {str(e)[:100]}")

            # Return empty strategy on error
            return SearchStrategy(
                country=country,
                business_type=business_type.value,
                industry=search_industry,
                known_companies=[],
                search_queries=[f"{search_industry} {country}"],
                search_patterns=[f"{business_type.value} {search_industry} companies {country}"],
                industry_keywords={"en": [search_industry], "local": []},
                estimated_total=0
            )

    def get_available_industries(self, business_type: BusinessType) -> Dict[str, List[str]]:
        """Get available industries and sub-industries for a business type"""
        return INDUSTRY_TAXONOMIES.get(business_type, {})

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        return {
            "total_companies": self.progress.total_companies_found,
            "by_country": dict(self.progress.companies_by_country),
            "by_confidence": dict(self.progress.companies_by_confidence),
            "by_size": dict(self.progress.companies_by_size),
            "api_calls": self.progress.api_calls,
            "elapsed_time": round(self.progress.elapsed_time, 2),
            "estimated_cost": round(self.progress.estimated_cost, 2)
        }


class ParallelSearchCoordinator:
    """Coordinates parallel searches across multiple deployments"""

    def __init__(self, deployments: List[str]):
        self.deployments = deployments
        self.agents = {dep: SearchStrategistAgent(dep) for dep in deployments}
        self.combined_progress = SearchProgress()

    def calculate_country_distribution(self, countries: List[str], total_target: int,
                                       mode: DistributionMode = DistributionMode.EQUAL) -> Dict[str, int]:
        """Calculate how many companies to search per country"""

        if mode == DistributionMode.EQUAL:
            per_country = total_target // len(countries)
            remainder = total_target % len(countries)
            distribution = {country: per_country for country in countries}
            # Distribute remainder
            for i, country in enumerate(countries[:remainder]):
                distribution[country] += 1

        elif mode == DistributionMode.POPULATION:
            # Get total population
            total_pop = sum(TARGET_COUNTRIES[c]["population"] for c in countries)
            distribution = {}
            allocated = 0

            for country in countries[:-1]:  # All but last
                pop = TARGET_COUNTRIES[country]["population"]
                count = int((pop / total_pop) * total_target)
                distribution[country] = max(10, count)  # Minimum 10 per country
                allocated += distribution[country]

            # Give remainder to last country
            distribution[countries[-1]] = total_target - allocated

        return distribution

    async def search_parallel(self,
                              countries: List[str],
                              business_type: BusinessType,
                              industry: str,
                              sub_industry: Optional[str] = None,
                              total_target: int = 1000,
                              distribution_mode: DistributionMode = DistributionMode.EQUAL,
                              custom_distribution: Optional[Dict[str, int]] = None,
                              include_financials: bool = True,
                              size_preferences: List[str] = None,
                              progress_callback=None) -> Dict[str, Any]:
        """Execute searches in parallel across multiple deployments"""

        start_time = time.time()

        # Calculate distribution
        if distribution_mode == DistributionMode.CUSTOM and custom_distribution:
            country_targets = custom_distribution
        else:
            country_targets = self.calculate_country_distribution(
                countries, total_target, distribution_mode
            )

        # Create work items (country, target_count pairs)
        work_items = [(country, count) for country, count in country_targets.items()]

        # Distribute work across deployments - one task per deployment at a time
        all_results = []

        # Process in batches of deployment count
        for i in range(0, len(work_items), len(self.deployments)):
            batch = work_items[i:i + len(self.deployments)]
            batch_tasks = []

            # Assign one country to each deployment
            for idx, (country, count) in enumerate(batch):
                if idx < len(self.deployments):
                    deployment = self.deployments[idx]
                    agent = self.agents[deployment]

                    # Create async task
                    task = agent.generate_strategy(
                        country=country,
                        business_type=business_type,
                        industry=industry,
                        sub_industry=sub_industry,
                        target_count=count,
                        include_financials=include_financials,
                        size_preferences=size_preferences
                    )
                    batch_tasks.append((task, country, deployment, count))

            # Execute batch and wait
            for task, country, deployment, count in batch_tasks:
                try:
                    strategy = await task

                    all_results.append({
                        "country": country,
                        "deployment": deployment,
                        "strategy": strategy,
                        "companies_found": len(strategy.known_companies),
                        "target": count
                    })

                    if progress_callback:
                        progress_callback(country, deployment, len(strategy.known_companies))

                except Exception as e:
                    print(f"Error searching {country} with {deployment}: {str(e)[:100]}")
                    all_results.append({
                        "country": country,
                        "deployment": deployment,
                        "strategy": None,
                        "companies_found": 0,
                        "target": count,
                        "error": str(e)
                    })

        # Combine progress from all agents
        for agent in self.agents.values():
            self.combined_progress.total_companies_found += agent.progress.total_companies_found
            self.combined_progress.api_calls += agent.progress.api_calls
            self.combined_progress.estimated_cost += agent.progress.estimated_cost

            for country, count in agent.progress.companies_by_country.items():
                self.combined_progress.companies_by_country[country] = \
                    self.combined_progress.companies_by_country.get(country, 0) + count

            for conf, count in agent.progress.companies_by_confidence.items():
                self.combined_progress.companies_by_confidence[conf] = \
                    self.combined_progress.companies_by_confidence.get(conf, 0) + count

            for size, count in agent.progress.companies_by_size.items():
                self.combined_progress.companies_by_size[size] = \
                    self.combined_progress.companies_by_size.get(size, 0) + count

        self.combined_progress.elapsed_time = time.time() - start_time
        self.combined_progress.deployments_used = self.deployments

        return {
            "results": all_results,
            "summary": {
                "total_companies_found": self.combined_progress.total_companies_found,
                "total_api_calls": self.combined_progress.api_calls,
                "total_cost": self.combined_progress.estimated_cost,
                "elapsed_time": self.combined_progress.elapsed_time,
                "deployments_used": len(self.deployments),
                "countries_searched": len(countries),
                "by_country": dict(self.combined_progress.companies_by_country),
                "by_confidence": dict(self.combined_progress.companies_by_confidence),
                "by_size": dict(self.combined_progress.companies_by_size)
            }
        }


# Basic test function
async def test_agent():
    """Test the Search Strategist Agent"""
    print("Testing Generic Search Strategist Agent")
    print("=" * 60)

    agent = SearchStrategistAgent()

    # Test Case 1: B2C Retail in UK
    print("\nTest 1: B2C Fashion Retail in United Kingdom")
    strategy = await agent.generate_strategy(
        country="United Kingdom",
        business_type=BusinessType.B2C,
        industry="Retail",
        sub_industry="Fashion & Apparel",
        target_count=20,
        include_financials=True
    )

    print(f"Found {len(strategy.known_companies)} companies")
    for company in strategy.known_companies[:5]:
        print(f"  - {company.name} ({company.confidence}) - Size: {company.company_size}")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_agent())