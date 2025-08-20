# validation_agent.py

import os
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
import time

from search_strategist_agent import CompanyEntry, SearchStrategy


class ValidationMode(Enum):
    """Validation modes available"""
    SKIP = "skip"  # Skip validation entirely
    PLACES_ONLY = "places_only"  # Places API only
    WEB_ONLY = "web_only"  # Web search only for B2C signals
    FULL = "full"  # Both Places and Web search


class ConfidenceFilter(Enum):
    """Which confidence levels to validate"""
    ALL = "all"
    LOW_ONLY = "low"
    MEDIUM_AND_LOW = "medium_and_low"
    HIGH_AND_BELOW = "high_medium_low"


@dataclass
class ValidationResult:
    """Result of company validation"""
    company_name: str
    country: str
    business_type: str
    industry: str
    original_confidence: str
    validation_status: str  # verified, rejected, unverified
    validation_mode: str
    confidence_after_validation: str

    # Serper Places data
    places_found: bool = False
    places_results: List[Dict[str, Any]] = None
    exact_match: bool = False
    place_details: Dict[str, Any] = None

    # Web search data
    web_search_performed: bool = False
    web_results: List[Dict[str, Any]] = None
    b2c_signals: List[str] = None
    b2b_signals: List[str] = None
    company_size_info: Dict[str, Any] = None

    # Metadata
    validation_reason: str = ""
    serper_queries_used: int = 0
    processing_time: float = 0.0
    timestamp: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class ValidationProgress:
    """Track validation progress"""
    total_processed: int = 0
    verified: int = 0
    rejected: int = 0
    unverified: int = 0
    serper_calls: int = 0
    estimated_cost: float = 0.0
    elapsed_time: float = 0.0

    def update(self, result: ValidationResult):
        self.total_processed += 1
        if result.validation_status == "verified":
            self.verified += 1
        elif result.validation_status == "rejected":
            self.rejected += 1
        else:
            self.unverified += 1
        self.serper_calls += result.serper_queries_used
        # Serper pricing estimate: $0.001 per query
        self.estimated_cost += result.serper_queries_used * 0.001


class SerperClient:
    """Client for Serper API interactions"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def places_search(self, query: str, country: str = None) -> List[Dict[str, Any]]:
        """Search using Serper Places API"""
        url = f"{self.base_url}/places"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        # Build location based on country
        data = {"q": query}

        # Add location for better results
        if country:
            data["location"] = country
            # Add country code if available
            country_codes = {
                "United States": "us", "United Kingdom": "uk", "Germany": "de",
                "France": "fr", "Spain": "es", "Italy": "it", "Japan": "jp",
                "Canada": "ca", "Australia": "au", "Brazil": "br", "India": "in",
                "Netherlands": "nl", "Belgium": "be", "Poland": "pl", "Sweden": "se"
            }
            if country in country_codes:
                data["gl"] = country_codes[country]

        try:
            async with self.session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("places", [])
                else:
                    error_text = await response.text()
                    print(f"Places API error {response.status}: {error_text}")
                    return []
        except Exception as e:
            print(f"Places API exception: {e}")
            return []

    async def web_search(self, query: str, country: str = None) -> List[Dict[str, Any]]:
        """Search using Serper Web Search API"""
        url = f"{self.base_url}/search"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        data = {
            "q": query,
            "num": 10
        }

        # Add country localization
        if country:
            country_codes = {
                "United States": "us", "United Kingdom": "uk", "Germany": "de",
                "France": "fr", "Spain": "es", "Italy": "it", "Japan": "jp",
                "Canada": "ca", "Australia": "au", "Brazil": "br", "India": "in",
                "Netherlands": "nl", "Belgium": "be", "Poland": "pl", "Sweden": "se"
            }
            if country in country_codes:
                data["gl"] = country_codes[country]

        try:
            async with self.session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("organic", [])
                else:
                    error_text = await response.text()
                    print(f"Web search API error {response.status}: {error_text}")
                    return []
        except Exception as e:
            print(f"Web search API exception: {e}")
            return []


class ValidationAgent:
    """Validates companies using Serper API and GPT-4.1 analysis"""

    def __init__(self, serper_api_key: str, gpt_deployment: str = "gpt-4.1"):
        self.serper_api_key = serper_api_key
        self.gpt_deployment = gpt_deployment
        self.progress = ValidationProgress()

        # Signal keywords for classification
        self.b2c_indicators = [
            "shop online", "buy now", "add to cart", "store locator",
            "opening hours", "visit us", "customer service", "products",
            "menu", "book now", "reservations", "retail", "consumer",
            "shopping", "boutique", "outlet", "showroom", "price",
            "sale", "discount", "free shipping", "customer reviews"
        ]

        self.b2b_indicators = [
            "request a demo", "enterprise", "solutions", "wholesale",
            "partners", "resellers", "distributors", "B2B", "business solutions",
            "for businesses", "integration", "API", "SaaS", "consulting",
            "agencies", "procurement", "vendor", "white label", "bulk pricing"
        ]

        # Company size indicators
        self.size_indicators = {
            "large": ["fortune 500", "multinational", "global", "billion", "10000+ employees"],
            "medium": ["regional", "national", "million revenue", "100-1000 employees"],
            "small": ["local", "startup", "small business", "under 100 employees"]
        }

    def normalize_company_name(self, name: str) -> str:
        """Normalize company name for matching"""
        # Remove common suffixes
        suffixes = [
            r'\s+(Ltd|Limited|Inc|Incorporated|Corp|Corporation|GmbH|AG|SA|SL|SpA|BV|NV|Oy|Ab|AS|A/S)\.?$',
            r'\s+(Group|Holdings|International|Global|Company|Co\.)\.?$'
        ]

        normalized = name
        for suffix in suffixes:
            normalized = re.sub(suffix, '', normalized, flags=re.IGNORECASE)

        return normalized.strip().lower()

    def is_exact_match(self, query_name: str, result_name: str) -> bool:
        """Check if names match with some flexibility"""
        query_clean = self.normalize_company_name(query_name)
        result_clean = self.normalize_company_name(result_name)

        # Exact match
        if query_clean == result_clean:
            return True

        # One contains the other (but not too different)
        if query_clean in result_clean or result_clean in query_clean:
            # Check it's not a completely different business
            exclude_terms = ["logistics", "solutions", "consulting", "wholesale", "b2b"]
            for term in exclude_terms:
                if term in result_clean and term not in query_clean:
                    return False
            return True

        # Check for very similar names (edit distance)
        # This is a simple check - could use Levenshtein distance for better results
        if len(query_clean) > 5 and len(result_clean) > 5:
            # Check if they share most of the same words
            query_words = set(query_clean.split())
            result_words = set(result_clean.split())
            if len(query_words.intersection(result_words)) >= len(query_words) * 0.7:
                return True

        return False

    def analyze_web_content(self, search_results: List[Dict[str, Any]], business_type: str) -> Tuple[
        List[str], List[str], Dict]:
        """Analyze web search results for business signals and company info"""
        b2c_signals_found = []
        b2b_signals_found = []
        company_info = {
            "size_indicators": [],
            "employee_count": None,
            "revenue": None,
            "description": None
        }

        for result in search_results:
            # Combine title, snippet, and any other text
            content = f"{result.get('title', '')} {result.get('snippet', '')}".lower()

            # Look for B2C signals
            for signal in self.b2c_indicators:
                if signal in content:
                    b2c_signals_found.append(signal)

            # Look for B2B signals
            for signal in self.b2b_indicators:
                if signal in content:
                    b2b_signals_found.append(signal)

            # Extract company size information
            for size, indicators in self.size_indicators.items():
                for indicator in indicators:
                    if indicator in content:
                        company_info["size_indicators"].append(size)

            # Look for employee count
            employee_match = re.search(r'(\d+[\d,]*)\s*employees', content)
            if employee_match and not company_info["employee_count"]:
                company_info["employee_count"] = employee_match.group(1)

            # Look for revenue
            revenue_match = re.search(r'\$(\d+[\d,\.]*)\s*(million|billion|M|B)', content)
            if revenue_match and not company_info["revenue"]:
                company_info["revenue"] = f"${revenue_match.group(1)} {revenue_match.group(2)}"

        # Deduplicate
        b2c_signals_found = list(set(b2c_signals_found))
        b2b_signals_found = list(set(b2b_signals_found))
        company_info["size_indicators"] = list(set(company_info["size_indicators"]))

        return b2c_signals_found, b2b_signals_found, company_info

    async def validate_company(self,
                               company: CompanyEntry,
                               country: str,
                               business_type: str,
                               mode: ValidationMode = ValidationMode.FULL,
                               serper_client: SerperClient = None) -> ValidationResult:
        """Validate a single company"""
        start_time = time.time()

        result = ValidationResult(
            company_name=company.name,
            country=country,
            business_type=business_type,
            industry=company.industry_category,
            original_confidence=company.confidence,
            validation_status="unverified",
            validation_mode=mode.value,
            confidence_after_validation=company.confidence,
            timestamp=datetime.now().isoformat()
        )

        # Skip validation mode
        if mode == ValidationMode.SKIP:
            result.validation_status = "verified" if company.confidence in ["absolute", "high"] else "unverified"
            result.validation_reason = "Validation skipped"
            result.processing_time = time.time() - start_time
            return result

        # Need Serper client for other modes
        if not serper_client:
            result.validation_reason = "No Serper client provided"
            return result

        # Places API validation
        if mode in [ValidationMode.PLACES_ONLY, ValidationMode.FULL]:
            places_query = f"{company.name} {country}"
            places_results = await serper_client.places_search(places_query, country)
            result.serper_queries_used += 1

            if places_results:
                result.places_found = True
                result.places_results = places_results

                # Check for exact match
                for place in places_results[:5]:  # Check top 5 results
                    place_name = place.get("title", "")
                    if self.is_exact_match(company.name, place_name):
                        result.exact_match = True
                        result.place_details = place
                        result.validation_status = "verified"
                        result.confidence_after_validation = "high"
                        result.validation_reason = f"Exact match found: {place_name}"
                        break

                if not result.exact_match:
                    # Check if any results are close enough
                    if len(places_results) > 0:
                        result.validation_status = "unverified"
                        result.confidence_after_validation = "medium"
                        result.validation_reason = f"Found {len(places_results)} places but no exact match"
                    else:
                        result.validation_status = "rejected"
                        result.confidence_after_validation = "low"
                        result.validation_reason = "Not found in Places search"
            else:
                result.places_found = False
                if mode == ValidationMode.PLACES_ONLY:
                    result.validation_status = "rejected"
                    result.confidence_after_validation = "low"
                    result.validation_reason = "No results in Places API"

        # Web search validation
        if mode in [ValidationMode.WEB_ONLY, ValidationMode.FULL]:
            # Skip web search if already verified by places
            if result.validation_status == "verified" and mode == ValidationMode.FULL:
                result.processing_time = time.time() - start_time
                return result

            # Construct web search query based on business type
            if business_type == "B2C":
                web_query = f'"{company.name}" {country} shop store retail consumer'
            else:
                web_query = f'"{company.name}" {country} {business_type}'

            web_results = await serper_client.web_search(web_query, country)
            result.web_search_performed = True
            result.web_results = web_results
            result.serper_queries_used += 1

            if web_results:
                # Analyze content
                b2c_signals, b2b_signals, company_info = self.analyze_web_content(web_results, business_type)
                result.b2c_signals = b2c_signals
                result.b2b_signals = b2b_signals
                result.company_size_info = company_info

                # Determine validation based on signals
                if business_type == "B2C":
                    if len(b2c_signals) > len(b2b_signals) and len(b2c_signals) > 0:
                        result.validation_status = "verified"
                        result.confidence_after_validation = "high" if len(b2c_signals) >= 3 else "medium"
                        result.validation_reason = f"B2C signals found: {', '.join(b2c_signals[:3])}"
                    elif len(b2b_signals) > len(b2c_signals):
                        result.validation_status = "rejected"
                        result.validation_reason = f"B2B signals found: {', '.join(b2b_signals[:3])}"
                    else:
                        result.validation_status = "unverified"
                        result.validation_reason = "Insufficient signals to determine business type"
                else:
                    # For B2B or other types, just verify existence
                    if len(web_results) > 0:
                        result.validation_status = "verified"
                        result.confidence_after_validation = "medium"
                        result.validation_reason = f"Found in web search with {len(web_results)} results"

        result.processing_time = time.time() - start_time
        self.progress.update(result)

        return result

    def should_validate(self, company: CompanyEntry, confidence_filter: ConfidenceFilter) -> bool:
        """Check if a company should be validated based on confidence filter"""
        if confidence_filter == ConfidenceFilter.ALL:
            return True
        elif confidence_filter == ConfidenceFilter.LOW_ONLY:
            return company.confidence == "low"
        elif confidence_filter == ConfidenceFilter.MEDIUM_AND_LOW:
            return company.confidence in ["medium", "low"]
        elif confidence_filter == ConfidenceFilter.HIGH_AND_BELOW:
            return company.confidence in ["high", "medium", "low"]
        return False

    async def validate_batch(self,
                             companies: List[CompanyEntry],
                             country: str,
                             business_type: str,
                             mode: ValidationMode = ValidationMode.FULL,
                             confidence_filter: ConfidenceFilter = ConfidenceFilter.ALL,
                             parallel_limit: int = 5) -> List[ValidationResult]:
        """Validate a batch of companies with rate limiting"""
        results = []

        # Filter companies based on confidence
        companies_to_validate = [c for c in companies if self.should_validate(c, confidence_filter)]
        companies_skipped = [c for c in companies if not self.should_validate(c, confidence_filter)]

        print(f"Validating {len(companies_to_validate)} companies (skipping {len(companies_skipped)})")

        # Add skipped companies as verified
        for company in companies_skipped:
            result = ValidationResult(
                company_name=company.name,
                country=country,
                business_type=business_type,
                industry=company.industry_category,
                original_confidence=company.confidence,
                validation_status="verified",
                validation_mode="skipped",
                confidence_after_validation=company.confidence,
                validation_reason="Skipped due to confidence filter",
                timestamp=datetime.now().isoformat()
            )
            results.append(result)

        # Skip validation mode or no companies to validate
        if mode == ValidationMode.SKIP or not companies_to_validate:
            for company in companies_to_validate:
                result = await self.validate_company(company, country, business_type, mode)
                results.append(result)
            return results

        # Validate with Serper
        async with SerperClient(self.serper_api_key) as serper_client:
            # Process in batches to respect rate limits
            for i in range(0, len(companies_to_validate), parallel_limit):
                batch = companies_to_validate[i:i + parallel_limit]

                # Create validation tasks
                tasks = []
                for company in batch:
                    task = self.validate_company(company, country, business_type, mode, serper_client)
                    tasks.append(task)

                # Wait for batch to complete
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)

                # Small delay between batches to avoid rate limits
                if i + parallel_limit < len(companies_to_validate):
                    await asyncio.sleep(1)

                # Progress update
                validated_so_far = len(companies_skipped) + i + len(batch)
                print(f"Progress: {validated_so_far}/{len(companies)} validated")

        return results

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get validation progress summary"""
        return {
            "total_processed": self.progress.total_processed,
            "verified": self.progress.verified,
            "rejected": self.progress.rejected,
            "unverified": self.progress.unverified,
            "serper_calls": self.progress.serper_calls,
            "estimated_cost": round(self.progress.estimated_cost, 3),
            "elapsed_time": round(self.progress.elapsed_time, 2),
            "verification_rate": round(self.progress.verified / max(1, self.progress.total_processed) * 100, 1)
        }


# Test function
async def test_validation_agent():
    """Test the validation agent"""
    from search_strategist_agent import CompanyEntry

    # Test companies
    test_companies = [
        CompanyEntry(
            name="Walmart",
            confidence="absolute",
            operates_in_country=True,
            business_type="B2C",
            industry_category="Retail",
            estimated_revenue="$500B+",
            estimated_employees="1000+",
            reasoning="Major US retailer"
        ),
        CompanyEntry(
            name="Local Shop Berlin",
            confidence="low",
            operates_in_country=True,
            business_type="B2C",
            industry_category="Retail",
            reasoning="Unknown local shop"
        )
    ]

    # Initialize agent
    serper_key = os.getenv("SERPER_API_KEY", "your-serper-key")
    agent = ValidationAgent(serper_key)

    # Test validation
    results = await agent.validate_batch(
        companies=test_companies,
        country="United States",
        business_type="B2C",
        mode=ValidationMode.FULL,
        confidence_filter=ConfidenceFilter.ALL
    )

    # Print results
    for result in results:
        print(f"\n{result.company_name}:")
        print(f"  Status: {result.validation_status}")
        print(f"  Confidence: {result.original_confidence} → {result.confidence_after_validation}")
        print(f"  Reason: {result.validation_reason}")
        print(f"  Serper queries: {result.serper_queries_used}")


if __name__ == "__main__":
    asyncio.run(test_validation_agent())