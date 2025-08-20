# enhanced_validation_agent.py

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import re
import time
from math import radians, cos, sin, asin, sqrt

from agents.search_strategist_agent import EnhancedCompanyEntry, SearchCriteria


class EnhancedValidationMode(Enum):
    """Enhanced validation modes"""
    SKIP = "skip"
    PLACES_ONLY = "places_only"
    WEB_ONLY = "web_only"
    NEWS_ONLY = "news_only"
    FULL = "full"
    COMPREHENSIVE = "comprehensive"  # Places + Web + News + Maps


@dataclass
class EnhancedValidationResult:
    """Enhanced validation result with comprehensive data"""
    # Basic info
    company_name: str
    country: str
    business_type: str
    industry: str
    original_confidence: str

    # Validation results
    validation_status: str  # verified, rejected, unverified, partial
    validation_mode: str
    confidence_after_validation: str
    validation_score: float = 0.0  # 0-100 score

    # Location validation
    location_verified: bool = False
    headquarters_confirmed: Optional[Dict[str, Any]] = None
    office_locations_found: List[Dict[str, Any]] = None
    proximity_matches: List[Dict[str, Any]] = None  # For proximity searches

    # Financial validation
    revenue_verified: bool = False
    revenue_data: Optional[Dict[str, Any]] = None
    employee_count_verified: bool = False
    employee_data: Optional[Dict[str, Any]] = None

    # CSR/ESG validation
    csr_programs_found: List[str] = None
    csr_evidence: List[Dict[str, Any]] = None
    certifications_verified: List[str] = None
    giving_history_found: List[Dict[str, Any]] = None

    # Recent events validation
    recent_events_found: List[Dict[str, Any]] = None
    leadership_changes: List[Dict[str, Any]] = None
    growth_signals: List[str] = None
    negative_signals: List[str] = None

    # Serper data
    serper_places_results: List[Dict[str, Any]] = None
    serper_web_results: List[Dict[str, Any]] = None
    serper_news_results: List[Dict[str, Any]] = None
    serper_maps_results: List[Dict[str, Any]] = None

    # Metadata
    validation_reason: str = ""
    validation_details: Dict[str, Any] = None
    serper_queries_used: int = 0
    processing_time: float = 0.0
    timestamp: str = ""
    data_sources: List[str] = None


class EnhancedSerperClient:
    """Enhanced Serper client with all API endpoints"""

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

    def _get_country_params(self, country: str) -> Dict[str, str]:
        """Get country-specific parameters for Serper"""
        country_configs = {
            "United States": {"gl": "us", "hl": "en"},
            "United Kingdom": {"gl": "uk", "hl": "en"},
            "Canada": {"gl": "ca", "hl": "en"},
            "Australia": {"gl": "au", "hl": "en"},
            "Germany": {"gl": "de", "hl": "de"},
            "France": {"gl": "fr", "hl": "fr"},
            "Spain": {"gl": "es", "hl": "es"},
            "Italy": {"gl": "it", "hl": "it"},
            "Japan": {"gl": "jp", "hl": "ja"},
            "Brazil": {"gl": "br", "hl": "pt"},
            "India": {"gl": "in", "hl": "en"},
            "Netherlands": {"gl": "nl", "hl": "nl"},
            "Sweden": {"gl": "se", "hl": "sv"},
            "Singapore": {"gl": "sg", "hl": "en"},
            "Mexico": {"gl": "mx", "hl": "es"},
        }
        return country_configs.get(country, {"gl": "us", "hl": "en"})

    async def places_search(self, query: str, location: str = None, country: str = None) -> List[Dict[str, Any]]:
        """Enhanced places search with location context"""
        url = f"{self.base_url}/places"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        data = {"q": query}

        # Add location context
        if location:
            data["location"] = location

        # Add country parameters
        if country:
            country_params = self._get_country_params(country)
            data.update(country_params)

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

    async def maps_search(self, query: str, location: str = None, country: str = None) -> List[Dict[str, Any]]:
        """Maps search for detailed location data"""
        url = f"{self.base_url}/maps"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        data = {"q": query}

        if location:
            data["location"] = location

        if country:
            country_params = self._get_country_params(country)
            data.update(country_params)

        try:
            async with self.session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("places", [])
                else:
                    return []
        except Exception as e:
            print(f"Maps API exception: {e}")
            return []

    async def web_search(self, query: str, country: str = None, num: int = 10) -> List[Dict[str, Any]]:
        """Enhanced web search"""
        url = f"{self.base_url}/search"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        data = {
            "q": query,
            "num": num
        }

        if country:
            country_params = self._get_country_params(country)
            data.update(country_params)

        try:
            async with self.session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("organic", [])
                else:
                    return []
        except Exception as e:
            print(f"Web search API exception: {e}")
            return []

    async def news_search(self, query: str, country: str = None, time_range: str = "month") -> List[Dict[str, Any]]:
        """News search for recent events"""
        url = f"{self.base_url}/news"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        data = {
            "q": query,
            "time": time_range  # "day", "week", "month", "year"
        }

        if country:
            country_params = self._get_country_params(country)
            data.update(country_params)

        try:
            async with self.session.post(url, json=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("news", [])
                else:
                    return []
        except Exception as e:
            print(f"News API exception: {e}")
            return []


class EnhancedValidationAgent:
    """Enhanced validation agent with comprehensive validation capabilities"""

    def __init__(self, serper_api_key: str):
        self.serper_api_key = serper_api_key

        # Enhanced signal keywords
        self.csr_indicators = [
            "corporate social responsibility", "csr", "sustainability", "esg",
            "community involvement", "charity", "donation", "sponsorship",
            "volunteer", "giving back", "social impact", "philanthropy",
            "environmental", "diversity", "inclusion", "ethics"
        ]

        self.financial_indicators = [
            "revenue", "turnover", "sales", "income", "profit", "funding",
            "valuation", "investment", "capital", "financial results",
            "annual report", "investor relations", "earnings"
        ]

        self.negative_indicators = [
            "bankruptcy", "lawsuit", "scandal", "fraud", "investigation",
            "violation", "fine", "penalty", "controversy", "complaint",
            "layoff", "closure", "restructuring", "lawsuit", "misconduct"
        ]

        self.growth_indicators = [
            "expansion", "growth", "acquisition", "merger", "new office",
            "hiring", "investment", "partnership", "launch", "opening",
            "record revenue", "milestone", "award", "recognition"
        ]

    def calculate_proximity(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in km"""
        # Haversine formula
        R = 6371  # Earth's radius in km

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))

        return R * c

    def extract_coordinates(self, place_data: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Extract coordinates from place data"""
        if "latitude" in place_data and "longitude" in place_data:
            return (place_data["latitude"], place_data["longitude"])
        elif "gps_coordinates" in place_data:
            coords = place_data["gps_coordinates"]
            if isinstance(coords, dict):
                return (coords.get("latitude"), coords.get("longitude"))
        return None

    def analyze_financial_data(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract financial information from search results"""
        financial_data = {
            "revenue_mentions": [],
            "employee_mentions": [],
            "funding_mentions": [],
            "financial_health": "unknown"
        }

        for result in search_results:
            content = f"{result.get('title', '')} {result.get('snippet', '')}".lower()

            # Look for revenue mentions
            revenue_patterns = [
                r'\$?([\d,]+\.?\d*)\s*(million|billion|m|b)\s*(revenue|turnover|sales)',
                r'(revenue|turnover|sales)\s*of\s*\$?([\d,]+\.?\d*)\s*(million|billion|m|b)',
                r'annual\s*(revenue|turnover|sales).*?\$?([\d,]+\.?\d*)\s*(million|billion|m|b)'
            ]

            for pattern in revenue_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    financial_data["revenue_mentions"].extend(matches)

            # Look for employee counts
            employee_patterns = [
                r'([\d,]+)\s*employees',
                r'([\d,]+)\s*staff',
                r'team\s*of\s*([\d,]+)',
                r'workforce\s*of\s*([\d,]+)'
            ]

            for pattern in employee_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    financial_data["employee_mentions"].extend(matches)

            # Look for funding information
            if any(term in content for term in ["funding", "investment", "raised", "series"]):
                financial_data["funding_mentions"].append(result.get('snippet', ''))

        # Analyze financial health
        positive_terms = ["growth", "profit", "expansion", "record", "increase"]
        negative_terms = ["loss", "decline", "layoff", "restructuring", "bankruptcy"]

        positive_count = sum(1 for result in search_results
                             for term in positive_terms
                             if term in result.get('snippet', '').lower())
        negative_count = sum(1 for result in search_results
                             for term in negative_terms
                             if term in result.get('snippet', '').lower())

        if positive_count > negative_count * 2:
            financial_data["financial_health"] = "positive"
        elif negative_count > positive_count * 2:
            financial_data["financial_health"] = "concerning"
        else:
            financial_data["financial_health"] = "stable"

        return financial_data

    def analyze_csr_evidence(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze CSR/ESG evidence from search results"""
        csr_data = {
            "programs": [],
            "focus_areas": [],
            "certifications": [],
            "evidence_snippets": []
        }

        # Define focus area keywords
        focus_areas_map = {
            "children": ["children", "kids", "youth", "school", "education"],
            "environment": ["environment", "sustainability", "green", "carbon", "climate"],
            "community": ["community", "local", "neighborhood", "volunteer"],
            "health": ["health", "medical", "hospital", "wellness", "mental health"],
            "diversity": ["diversity", "inclusion", "equity", "women", "minorities"]
        }

        # Certification keywords
        certifications = ["b-corp", "b corp", "iso 26000", "iso 14001", "leed", "fair trade"]

        for result in search_results:
            content = f"{result.get('title', '')} {result.get('snippet', '')}".lower()

            # Check for CSR indicators
            if any(indicator in content for indicator in self.csr_indicators):
                csr_data["evidence_snippets"].append({
                    "source": result.get('link', ''),
                    "snippet": result.get('snippet', ''),
                    "title": result.get('title', '')
                })

                # Extract focus areas
                for area, keywords in focus_areas_map.items():
                    if any(keyword in content for keyword in keywords):
                        if area not in csr_data["focus_areas"]:
                            csr_data["focus_areas"].append(area)

                # Check for certifications
                for cert in certifications:
                    if cert in content and cert not in csr_data["certifications"]:
                        csr_data["certifications"].append(cert)

        return csr_data

    async def validate_company_comprehensive(
            self,
            company: EnhancedCompanyEntry,
            criteria: SearchCriteria,
            mode: EnhancedValidationMode = EnhancedValidationMode.COMPREHENSIVE
    ) -> EnhancedValidationResult:
        """Comprehensive validation with all available data sources"""
        start_time = time.time()

        result = EnhancedValidationResult(
            company_name=company.name,
            country=criteria.location.countries[0] if criteria.location.countries else "Unknown",
            business_type=company.business_type,
            industry=company.industry_category,
            original_confidence=company.confidence,
            validation_status="unverified",
            validation_mode=mode.value,
            confidence_after_validation=company.confidence,
            timestamp=datetime.now().isoformat(),
            data_sources=[]
        )

        async with EnhancedSerperClient(self.serper_api_key) as client:
            validation_tasks = []

            # Places/Maps validation for location
            if mode in [EnhancedValidationMode.PLACES_ONLY, EnhancedValidationMode.FULL,
                        EnhancedValidationMode.COMPREHENSIVE]:
                validation_tasks.append(self._validate_location(company, criteria, client, result, mode))

            # Web search for company info and CSR
            if mode in [EnhancedValidationMode.WEB_ONLY, EnhancedValidationMode.FULL,
                        EnhancedValidationMode.COMPREHENSIVE]:
                validation_tasks.append(self._validate_web_presence(company, criteria, client, result))

            # News search for recent events
            if mode in [EnhancedValidationMode.NEWS_ONLY, EnhancedValidationMode.COMPREHENSIVE]:
                validation_tasks.append(self._validate_recent_events(company, criteria, client, result))

            # Execute all validation tasks
            if validation_tasks:
                await asyncio.gather(*validation_tasks)

            # Calculate final validation score
            result = self._calculate_validation_score(result, criteria)

            # Determine final status
            if result.validation_score >= 80:
                result.validation_status = "verified"
                result.confidence_after_validation = "high"
            elif result.validation_score >= 60:
                result.validation_status = "partial"
                result.confidence_after_validation = "medium"
            elif result.validation_score >= 40:
                result.validation_status = "unverified"
                result.confidence_after_validation = "low"
            else:
                result.validation_status = "rejected"
                result.confidence_after_validation = "low"

            result.processing_time = time.time() - start_time

        return result

    async def _validate_location(self, company: EnhancedCompanyEntry, criteria: SearchCriteria,
                                 client: EnhancedSerperClient, result: EnhancedValidationResult,
                                 mode: EnhancedValidationMode):
        """Validate company location using Places and Maps APIs"""
        # Search query
        location = criteria.location.cities[0] if criteria.location.cities else criteria.location.countries[0]
        query = f"{company.name} {location}"

        # Places search
        places_results = await client.places_search(query, location, criteria.location.countries[0])
        result.serper_places_results = places_results
        result.serper_queries_used += 1
        result.data_sources.append("serper_places")

        if places_results:
            # Check for exact matches
            for place in places_results[:5]:
                place_name = place.get("title", "").lower()
                company_name = company.name.lower()

                # Fuzzy matching
                if company_name in place_name or place_name in company_name:
                    result.location_verified = True
                    result.headquarters_confirmed = {
                        "name": place.get("title"),
                        "address": place.get("address"),
                        "coordinates": self.extract_coordinates(place),
                        "phone": place.get("phoneNumber"),
                        "type": place.get("type", [])
                    }
                    break

            # Check proximity if required
            if criteria.location.proximity and result.headquarters_confirmed:
                coords = result.headquarters_confirmed.get("coordinates")
                if coords:
                    # Calculate distance from target location
                    # This would require geocoding the target location
                    result.proximity_matches = [{
                        "location": result.headquarters_confirmed["address"],
                        "distance_km": "Calculated based on coordinates"
                    }]

        # Maps search for additional detail if needed
        if mode == EnhancedValidationMode.COMPREHENSIVE and not result.location_verified:
            maps_results = await client.maps_search(query, location)
            result.serper_maps_results = maps_results
            result.serper_queries_used += 1
            # Process maps results similarly

    async def _validate_web_presence(self, company: EnhancedCompanyEntry, criteria: SearchCriteria,
                                     client: EnhancedSerperClient, result: EnhancedValidationResult):
        """Validate company info and CSR through web search"""
        # General company search
        query = f'"{company.name}" {company.industry_category}'
        web_results = await client.web_search(query, criteria.location.countries[0])
        result.serper_web_results = web_results
        result.serper_queries_used += 1
        result.data_sources.append("serper_web")

        if web_results:
            # Analyze financial data
            financial_data = self.analyze_financial_data(web_results)
            if financial_data["revenue_mentions"]:
                result.revenue_verified = True
                result.revenue_data = financial_data
            if financial_data["employee_mentions"]:
                result.employee_count_verified = True
                result.employee_data = financial_data

            # Look for negative signals
            negative_found = []
            for res in web_results:
                content = f"{res.get('title', '')} {res.get('snippet', '')}".lower()
                for indicator in self.negative_indicators:
                    if indicator in content:
                        negative_found.append(indicator)

            if negative_found:
                result.negative_signals = list(set(negative_found))

        # CSR-specific search if required
        if criteria.behavioral.csr_focus_areas or criteria.behavioral.certifications:
            csr_query = f'"{company.name}" "corporate social responsibility" sustainability CSR'
            csr_results = await client.web_search(csr_query, criteria.location.countries[0])
            result.serper_queries_used += 1

            if csr_results:
                csr_data = self.analyze_csr_evidence(csr_results)
                result.csr_programs_found = csr_data["programs"]
                result.csr_evidence = csr_data["evidence_snippets"]
                result.certifications_verified = csr_data["certifications"]

    async def _validate_recent_events(self, company: EnhancedCompanyEntry, criteria: SearchCriteria,
                                      client: EnhancedSerperClient, result: EnhancedValidationResult):
        """Validate recent events and news"""
        query = f'"{company.name}"'
        news_results = await client.news_search(query, criteria.location.countries[0], time_range="month")
        result.serper_news_results = news_results
        result.serper_queries_used += 1
        result.data_sources.append("serper_news")

        if news_results:
            recent_events = []
            leadership_changes = []
            growth_signals = []

            for news in news_results:
                title = news.get("title", "").lower()
                snippet = news.get("snippet", "").lower()
                content = f"{title} {snippet}"

                # Categorize news
                if any(term in content for term in ["ceo", "president", "executive", "appoint", "hire"]):
                    leadership_changes.append({
                        "title": news.get("title"),
                        "date": news.get("date"),
                        "source": news.get("source")
                    })

                if any(term in content for term in self.growth_indicators):
                    growth_signals.append(news.get("title"))

                # Look for specific events
                if any(term in content for term in criteria.behavioral.recent_events):
                    recent_events.append({
                        "title": news.get("title"),
                        "date": news.get("date"),
                        "source": news.get("source"),
                        "link": news.get("link")
                    })

            result.recent_events_found = recent_events
            result.leadership_changes = leadership_changes
            result.growth_signals = growth_signals

    def _calculate_validation_score(self, result: EnhancedValidationResult,
                                    criteria: SearchCriteria) -> EnhancedValidationResult:
        """Calculate comprehensive validation score"""
        score = 0
        max_score = 0
        details = {}

        # Location verification (25 points)
        max_score += 25
        if result.location_verified:
            score += 25
            details["location"] = "Verified"
        elif result.serper_places_results:
            score += 10
            details["location"] = "Partial match"
        else:
            details["location"] = "Not verified"

        # Financial verification (20 points)
        max_score += 20
        if result.revenue_verified and result.employee_count_verified:
            score += 20
            details["financial"] = "Fully verified"
        elif result.revenue_verified or result.employee_count_verified:
            score += 10
            details["financial"] = "Partially verified"
        else:
            details["financial"] = "Not verified"

        # CSR verification (30 points if CSR criteria exist)
        if criteria.behavioral.csr_focus_areas or criteria.behavioral.certifications:
            max_score += 30
            csr_score = 0

            # CSR programs found (15 points)
            if result.csr_evidence:
                csr_score += 15

            # Matching focus areas (10 points)
            if result.csr_programs_found and criteria.behavioral.csr_focus_areas:
                matching_areas = set(result.csr_programs_found) & set(criteria.behavioral.csr_focus_areas)
                if matching_areas:
                    csr_score += 10

            # Certifications (5 points)
            if result.certifications_verified:
                csr_score += 5

            score += csr_score
            details["csr"] = f"{csr_score}/30 points"

        # Recent events (15 points)
        max_score += 15
        if result.recent_events_found:
            score += 15
            details["recent_events"] = "Found"
        elif result.serper_news_results:
            score += 5
            details["recent_events"] = "News found but no matching events"
        else:
            details["recent_events"] = "No recent news"

        # Negative signals penalty (up to -20 points)
        if result.negative_signals:
            penalty = min(20, len(result.negative_signals) * 5)
            score -= penalty
            details["negative_signals"] = f"-{penalty} points"

        # Growth signals bonus (up to +10 points)
        if result.growth_signals:
            bonus = min(10, len(result.growth_signals) * 2)
            score += bonus
            max_score += 10
            details["growth_bonus"] = f"+{bonus} points"

        # Calculate final score
        result.validation_score = max(0, (score / max_score) * 100) if max_score > 0 else 0
        result.validation_details = details

        return result

    async def validate_batch_enhanced(
            self,
            companies: List[EnhancedCompanyEntry],
            criteria: SearchCriteria,
            mode: EnhancedValidationMode = EnhancedValidationMode.FULL,
            parallel_limit: int = 5
    ) -> List[EnhancedValidationResult]:
        """Validate a batch of companies with enhanced criteria"""
        results = []

        # Process in batches
        for i in range(0, len(companies), parallel_limit):
            batch = companies[i:i + parallel_limit]

            # Create validation tasks
            tasks = []
            for company in batch:
                task = self.validate_company_comprehensive(company, criteria, mode)
                tasks.append(task)

            # Execute batch
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Rate limiting
            if i + parallel_limit < len(companies):
                await asyncio.sleep(1)

            # Progress
            print(f"Validated {min(i + parallel_limit, len(companies))}/{len(companies)} companies")

        return results


# Backward compatibility
ValidationAgent = EnhancedValidationAgent
ValidationMode = EnhancedValidationMode
ValidationResult = EnhancedValidationResult