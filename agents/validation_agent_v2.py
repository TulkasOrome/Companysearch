# validation_agent_v2.py
"""
Enhanced Validation Agent V2 with Multiple Validation Modes
Supports Simple, Raw Endpoint, Smart (Contact/CSR/Financial), and Custom validation
"""

import asyncio
import aiohttp
import time
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

# Use absolute imports when this is used as a module
try:
    from core.serper_client import EnhancedSerperClient, SerperEndpoint
    from core.validation_strategies import ValidationTier
except ImportError:
    # Fall back to creating minimal required classes
    class SerperEndpoint(Enum):
        SEARCH = "search"
        PLACES = "places"
        MAPS = "maps"
        NEWS = "news"
        SHOPPING = "shopping"


    class ValidationTier(Enum):
        QUICK = "quick"
        STANDARD = "standard"
        COMPREHENSIVE = "comprehensive"

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Available validation modes"""
    SIMPLE = "simple"  # Just verify existence (2-3 credits)
    RAW_ENDPOINT = "raw"  # Direct endpoint access (1-2 credits)
    SMART_CONTACT = "smart_contact"  # Contact extraction focus (3-5 credits)
    SMART_CSR = "smart_csr"  # CSR verification focus (3-5 credits)
    SMART_FINANCIAL = "smart_financial"  # Financial verification (3-4 credits)
    FULL = "full"  # Complete validation (10-15 credits)
    CUSTOM = "custom"  # User-defined endpoints


@dataclass
class ValidationConfig:
    """Configuration for validation process"""
    serper_api_key: str
    max_parallel_queries: int = 5
    rate_limit_per_second: float = 10
    timeout: int = 30
    max_retries: int = 3
    cache_duration_hours: int = 24

    # Validation thresholds
    min_confidence_for_quick: float = 0.8
    min_score_for_approval: float = 60

    # Cost controls
    max_cost_per_company: float = 0.015  # $0.015 (15 credits)
    max_total_cost: float = 10.0  # $10

    # Feature flags
    enable_caching: bool = True
    enable_batch_optimization: bool = True
    enable_smart_routing: bool = True


@dataclass
class ValidationCriteria:
    """Criteria for validation"""
    # Location requirements
    must_be_in_locations: List[str] = field(default_factory=list)
    must_be_within_radius: Optional[Tuple[str, float]] = None

    # Financial requirements
    min_revenue: Optional[float] = None
    max_revenue: Optional[float] = None
    revenue_currency: str = "USD"
    min_employees: Optional[int] = None
    max_employees: Optional[int] = None

    # CSR requirements
    required_csr_areas: List[str] = field(default_factory=list)
    required_certifications: List[str] = field(default_factory=list)

    # Event triggers
    required_events: List[str] = field(default_factory=list)

    # Exclusions
    excluded_industries: List[str] = field(default_factory=list)
    excluded_keywords: List[str] = field(default_factory=list)
    negative_signals: List[str] = field(default_factory=list)


@dataclass
class EndpointConfig:
    """Configuration for specific endpoint usage"""
    endpoint: SerperEndpoint
    enabled: bool = True
    max_queries: int = 1
    query_templates: List[str] = field(default_factory=list)


@dataclass
class ComprehensiveValidation:
    """Complete validation results"""
    company_name: str
    validation_mode: ValidationMode
    validation_timestamp: str

    # Core validation results
    exists: bool = False
    location_verified: bool = False
    confidence_score: float = 0.0
    validation_status: str = "unverified"

    # Extracted data
    basic_info: Dict[str, Any] = field(default_factory=dict)
    contact_info: Dict[str, Any] = field(default_factory=dict)
    financial_info: Dict[str, Any] = field(default_factory=dict)
    csr_info: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    credits_used: int = 0
    queries_executed: List[str] = field(default_factory=list)
    raw_responses: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class EnhancedValidationAgent:
    """
    Enhanced validation agent with multiple validation modes
    """

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.session = None
        self.total_credits = 0
        self.validation_cache = {}
        self.start_time = datetime.now()

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _make_serper_request(self, endpoint: str, query: str, options: Dict = None) -> Dict[str, Any]:
        """Make a synchronous Serper API request"""
        import http.client
        import json

        try:
            conn = http.client.HTTPSConnection("google.serper.dev")

            payload = {"q": query, "gl": "au"}
            if options:
                payload.update(options)

            headers = {
                'X-API-KEY': self.config.serper_api_key,
                'Content-Type': 'application/json'
            }

            conn.request("POST", f"/{endpoint}", json.dumps(payload), headers)
            res = conn.getresponse()
            data = res.read()

            result = json.loads(data.decode("utf-8"))
            self.total_credits += result.get('credits', 1)

            return result

        except Exception as e:
            logger.error(f"Serper request failed: {e}")
            return {"error": str(e)}

    def _fuzzy_name_match(self, name1: str, name2: str, threshold: float = 0.7) -> bool:
        """Fuzzy name matching for company names"""
        name1_clean = name1.lower().strip()
        name2_clean = name2.lower().strip()

        # Direct substring match
        if name1_clean in name2_clean or name2_clean in name1_clean:
            return True

        # Remove common suffixes
        suffixes = ['pty ltd', 'limited', 'ltd', 'inc', 'corporation', 'corp', 'group', 'holdings']
        for suffix in suffixes:
            name1_clean = name1_clean.replace(suffix, '').strip()
            name2_clean = name2_clean.replace(suffix, '').strip()

        if name1_clean in name2_clean or name2_clean in name1_clean:
            return True

        # Word overlap check
        words1 = set(name1_clean.split())
        words2 = set(name2_clean.split())

        if not words1 or not words2:
            return False

        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))

        return (overlap / total) >= threshold if total > 0 else False

    async def validate_simple(
            self,
            company_name: str,
            location: str,
            country: str = "Australia"
    ) -> ComprehensiveValidation:
        """
        Simple validation mode - just verify existence
        Uses 2-3 credits maximum
        """
        logger.info(f"Simple validation for {company_name}")

        validation = ComprehensiveValidation(
            company_name=company_name,
            validation_mode=ValidationMode.SIMPLE,
            validation_timestamp=datetime.now().isoformat()
        )

        # Try Places API first
        places_query = f"{company_name} {location}"
        places_result = self._make_serper_request(
            "places",
            places_query,
            {"location": location}
        )
        validation.queries_executed.append(f"places: {places_query}")
        validation.raw_responses.append({"endpoint": "places", "response": places_result})

        if places_result.get('places'):
            for place in places_result['places'][:3]:
                if self._fuzzy_name_match(company_name, place.get('title', '')):
                    validation.exists = True
                    validation.confidence_score = 80

                    # Check location match
                    if location.lower() in place.get('address', '').lower():
                        validation.location_verified = True
                        validation.confidence_score = 95

                    validation.basic_info = {
                        "name": place.get('title'),
                        "address": place.get('address'),
                        "phone": place.get('phoneNumber'),
                        "website": place.get('website'),
                        "rating": place.get('rating')
                    }
                    break

        # Fallback to web search if not found
        if not validation.exists:
            web_query = f'"{company_name}" {location} company'
            web_result = self._make_serper_request("search", web_query, {"num": 3})
            validation.queries_executed.append(f"search: {web_query}")
            validation.raw_responses.append({"endpoint": "search", "response": web_result})

            if web_result.get('organic'):
                for result in web_result['organic']:
                    if self._fuzzy_name_match(company_name, result.get('title', '')):
                        validation.exists = True
                        validation.confidence_score = 60
                        if location.lower() in result.get('snippet', '').lower():
                            validation.location_verified = True
                            validation.confidence_score = 75
                        break

        # Set validation status
        if validation.confidence_score >= 80:
            validation.validation_status = "verified"
        elif validation.confidence_score >= 60:
            validation.validation_status = "partial"
        else:
            validation.validation_status = "unverified"

        validation.credits_used = len(validation.queries_executed)

        return validation

    async def validate_raw_endpoint(
            self,
            endpoint: SerperEndpoint,
            query: str,
            options: Dict = None
    ) -> ComprehensiveValidation:
        """
        Raw endpoint mode - direct access to any Serper endpoint
        Returns raw response data
        """
        logger.info(f"Raw endpoint validation: {endpoint.value}")

        validation = ComprehensiveValidation(
            company_name=query,
            validation_mode=ValidationMode.RAW_ENDPOINT,
            validation_timestamp=datetime.now().isoformat()
        )

        result = self._make_serper_request(endpoint.value, query, options)
        validation.queries_executed.append(f"{endpoint.value}: {query}")
        validation.raw_responses.append({"endpoint": endpoint.value, "response": result})

        # Store raw results in basic_info
        validation.basic_info = result
        validation.exists = bool(
            result.get('places') or
            result.get('organic') or
            result.get('news') or
            result.get('maps')
        )
        validation.confidence_score = 100 if validation.exists else 0
        validation.validation_status = "raw_data"
        validation.credits_used = 1

        return validation

    async def validate_smart_contact(
            self,
            company_name: str,
            location: Optional[str] = None
    ) -> ComprehensiveValidation:
        """
        Smart contact extraction mode
        Focuses on emails, phones, and key personnel
        """
        logger.info(f"Smart contact validation for {company_name}")

        validation = ComprehensiveValidation(
            company_name=company_name,
            validation_mode=ValidationMode.SMART_CONTACT,
            validation_timestamp=datetime.now().isoformat()
        )

        contacts = {
            "emails": [],
            "phones": [],
            "names": [],
            "titles": []
        }

        # 1. Email search
        email_query = f'"{company_name}" email contact @'
        email_result = self._make_serper_request("search", email_query, {"num": 5})
        validation.queries_executed.append(f"search: {email_query}")
        validation.raw_responses.append({"endpoint": "search", "response": email_result})

        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if email_result.get('organic'):
            for result in email_result['organic']:
                text = f"{result.get('title', '')} {result.get('snippet', '')}"
                emails = re.findall(email_pattern, text)
                contacts['emails'].extend(emails)

        # 2. LinkedIn search for executives
        linkedin_query = f'site:linkedin.com "{company_name}" CEO OR Director OR Manager'
        linkedin_result = self._make_serper_request("search", linkedin_query, {"num": 5})
        validation.queries_executed.append(f"search: {linkedin_query}")
        validation.raw_responses.append({"endpoint": "search", "response": linkedin_result})

        # Extract names from LinkedIn
        if linkedin_result.get('organic'):
            for result in linkedin_result['organic']:
                title = result.get('title', '')
                # Pattern: "Name - Title - Company | LinkedIn"
                name_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', title)
                if name_match:
                    contacts['names'].append(name_match.group(1))

        # 3. Places for phone numbers
        if location:
            places_query = f"{company_name} {location}"
        else:
            places_query = company_name

        places_result = self._make_serper_request("places", places_query)
        validation.queries_executed.append(f"places: {places_query}")
        validation.raw_responses.append({"endpoint": "places", "response": places_result})

        if places_result.get('places'):
            for place in places_result['places'][:2]:
                if place.get('phoneNumber'):
                    contacts['phones'].append(place['phoneNumber'])

        # Deduplicate
        contacts['emails'] = list(set(contacts['emails']))[:10]
        contacts['names'] = list(set(contacts['names']))[:10]
        contacts['phones'] = list(set(contacts['phones']))[:5]

        validation.contact_info = contacts
        validation.exists = bool(contacts['emails'] or contacts['phones'] or contacts['names'])
        validation.confidence_score = min(100,
                                          len(contacts['emails']) * 15 +
                                          len(contacts['names']) * 10 +
                                          len(contacts['phones']) * 20
                                          )

        if validation.confidence_score >= 60:
            validation.validation_status = "verified"
        else:
            validation.validation_status = "partial"

        validation.credits_used = len(validation.queries_executed)

        return validation

    async def validate_smart_csr(
            self,
            company_name: str,
            required_areas: Optional[List[str]] = None
    ) -> ComprehensiveValidation:
        """
        Smart CSR validation mode
        Focuses on CSR programs, certifications, and community involvement
        """
        logger.info(f"Smart CSR validation for {company_name}")

        validation = ComprehensiveValidation(
            company_name=company_name,
            validation_mode=ValidationMode.SMART_CSR,
            validation_timestamp=datetime.now().isoformat()
        )

        csr_data = {
            "has_csr": False,
            "has_foundation": False,
            "focus_areas": [],
            "certifications": [],
            "recent_activities": [],
            "giving_evidence": [],
            "sustainability_report": False
        }

        # 1. Check for foundation/CSR program
        foundation_query = f'"{company_name} Foundation" OR "{company_name} CSR" sustainability'
        foundation_result = self._make_serper_request("search", foundation_query, {"num": 5})
        validation.queries_executed.append(f"search: {foundation_query}")
        validation.raw_responses.append({"endpoint": "search", "response": foundation_result})

        if foundation_result.get('organic'):
            for result in foundation_result['organic']:
                text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()

                if f"{company_name.lower()} foundation" in text:
                    csr_data['has_foundation'] = True
                    csr_data['has_csr'] = True

                if 'sustainability report' in text or 'esg report' in text:
                    csr_data['sustainability_report'] = True
                    csr_data['has_csr'] = True

        # 2. Search for CSR focus areas
        csr_query = f'"{company_name}" community donation charity sponsorship'
        csr_result = self._make_serper_request("search", csr_query, {"num": 5})
        validation.queries_executed.append(f"search: {csr_query}")
        validation.raw_responses.append({"endpoint": "search", "response": csr_result})

        # Detect focus areas
        csr_keywords = {
            "children": ["children", "kids", "youth", "school"],
            "community": ["community", "local", "volunteer"],
            "environment": ["environment", "sustainability", "green"],
            "health": ["health", "medical", "hospital"],
            "education": ["education", "scholarship", "literacy"]
        }

        if csr_result.get('organic'):
            for result in csr_result['organic']:
                text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()

                for area, keywords in csr_keywords.items():
                    if any(keyword in text for keyword in keywords):
                        if area not in csr_data['focus_areas']:
                            csr_data['focus_areas'].append(area)

                # Look for giving evidence
                if '$' in text and any(word in text for word in ['donated', 'raised', 'contributed']):
                    csr_data['giving_evidence'].append(result.get('snippet', '')[:100])
                    csr_data['has_csr'] = True

        # 3. Check certifications
        cert_query = f'"{company_name}" "B Corp" OR "ISO 26000" OR "carbon neutral"'
        cert_result = self._make_serper_request("search", cert_query, {"num": 3})
        validation.queries_executed.append(f"search: {cert_query}")
        validation.raw_responses.append({"endpoint": "search", "response": cert_result})

        cert_patterns = ["b corp", "b-corp", "iso 26000", "iso 14001", "carbon neutral"]
        if cert_result.get('organic'):
            for result in cert_result['organic']:
                text = result.get('snippet', '').lower()
                for cert in cert_patterns:
                    if cert in text and cert not in csr_data['certifications']:
                        csr_data['certifications'].append(cert)

        # 4. Recent CSR news
        news_query = f'"{company_name}" donation charity community'
        news_result = self._make_serper_request("news", news_query, {"time": "year"})
        validation.queries_executed.append(f"news: {news_query}")
        validation.raw_responses.append({"endpoint": "news", "response": news_result})

        if news_result.get('news'):
            for item in news_result['news'][:5]:
                if any(word in item.get('title', '').lower() for word in ['donation', 'charity', 'sponsor']):
                    csr_data['recent_activities'].append({
                        "title": item.get('title'),
                        "date": item.get('date')
                    })
                    csr_data['has_csr'] = True

        validation.csr_info = csr_data
        validation.exists = csr_data['has_csr']

        # Calculate confidence
        confidence = 0
        if csr_data['has_csr']:
            confidence += 30
        if csr_data['has_foundation']:
            confidence += 20
        if csr_data['sustainability_report']:
            confidence += 15
        confidence += min(len(csr_data['focus_areas']) * 5, 20)
        confidence += min(len(csr_data['certifications']) * 10, 15)

        validation.confidence_score = min(confidence, 100)

        if validation.confidence_score >= 60:
            validation.validation_status = "verified"
        else:
            validation.validation_status = "partial"

        validation.credits_used = len(validation.queries_executed)

        return validation

    async def validate_smart_financial(
            self,
            company_name: str,
            check_listed: bool = True
    ) -> ComprehensiveValidation:
        """
        Smart financial validation mode
        Focuses on revenue, employees, and growth indicators
        """
        logger.info(f"Smart financial validation for {company_name}")

        validation = ComprehensiveValidation(
            company_name=company_name,
            validation_mode=ValidationMode.SMART_FINANCIAL,
            validation_timestamp=datetime.now().isoformat()
        )

        financial_data = {
            "revenue": None,
            "revenue_currency": None,
            "employees": None,
            "growth_stage": None,
            "stock_listed": False,
            "asx_code": None,
            "recent_events": []
        }

        # 1. Search for revenue and employees
        financial_query = f'"{company_name}" annual revenue employees million'
        financial_result = self._make_serper_request("search", financial_query, {"num": 5})
        validation.queries_executed.append(f"search: {financial_query}")
        validation.raw_responses.append({"endpoint": "search", "response": financial_result})

        # Extract revenue
        revenue_pattern = r'\$?([\d,]+\.?\d*)\s*(million|billion|m|b)'
        employee_pattern = r'([\d,]+)\s*(?:employees|staff)'

        if financial_result.get('organic'):
            for result in financial_result['organic']:
                text = f"{result.get('title', '')} {result.get('snippet', '')}"

                if not financial_data['revenue']:
                    revenue_matches = re.findall(revenue_pattern, text, re.IGNORECASE)
                    if revenue_matches:
                        financial_data['revenue'] = f"${revenue_matches[0][0]}{revenue_matches[0][1].upper()}"
                        financial_data['revenue_currency'] = 'AUD' if 'aud' in text.lower() else 'USD'

                if not financial_data['employees']:
                    employee_matches = re.findall(employee_pattern, text, re.IGNORECASE)
                    if employee_matches:
                        financial_data['employees'] = employee_matches[0]

        # 2. Check if publicly listed
        if check_listed:
            asx_query = f'"{company_name}" ASX listed share price'
            asx_result = self._make_serper_request("search", asx_query, {"num": 3})
            validation.queries_executed.append(f"search: {asx_query}")
            validation.raw_responses.append({"endpoint": "search", "response": asx_result})

            if asx_result.get('organic'):
                for result in asx_result['organic']:
                    text = result.get('snippet', '').lower()
                    asx_match = re.search(r'asx:\s*([a-z]{3,4})', text, re.IGNORECASE)
                    if asx_match:
                        financial_data['stock_listed'] = True
                        financial_data['asx_code'] = asx_match.group(1).upper()
                        break

        # 3. Recent financial news
        news_query = f'"{company_name}" funding expansion acquisition'
        news_result = self._make_serper_request("news", news_query, {"time": "year"})
        validation.queries_executed.append(f"news: {news_query}")
        validation.raw_responses.append({"endpoint": "news", "response": news_result})

        growth_keywords = ['funding', 'investment', 'expansion', 'acquisition', 'ipo']
        if news_result.get('news'):
            for item in news_result['news'][:5]:
                title = item.get('title', '').lower()
                for keyword in growth_keywords:
                    if keyword in title:
                        financial_data['recent_events'].append({
                            "type": keyword,
                            "title": item.get('title'),
                            "date": item.get('date')
                        })
                        break

        # Determine growth stage
        if financial_data['recent_events']:
            event_types = [e['type'] for e in financial_data['recent_events']]
            if 'ipo' in event_types:
                financial_data['growth_stage'] = 'Public'
            elif 'funding' in event_types:
                financial_data['growth_stage'] = 'Growth'
            elif 'expansion' in event_types:
                financial_data['growth_stage'] = 'Expansion'
            else:
                financial_data['growth_stage'] = 'Established'

        validation.financial_info = financial_data
        validation.exists = bool(financial_data['revenue'] or financial_data['employees'])

        # Calculate confidence
        confidence = 0
        if financial_data['revenue']:
            confidence += 35
        if financial_data['employees']:
            confidence += 25
        if financial_data['stock_listed']:
            confidence += 20
        confidence += min(len(financial_data['recent_events']) * 5, 20)

        validation.confidence_score = min(confidence, 100)

        if validation.confidence_score >= 60:
            validation.validation_status = "verified"
        else:
            validation.validation_status = "partial"

        validation.credits_used = len(validation.queries_executed)

        return validation

    async def validate_full(
            self,
            company_name: str,
            location: str,
            criteria: Optional[ValidationCriteria] = None
    ) -> ComprehensiveValidation:
        """
        Full validation mode - comprehensive check across all dimensions
        Uses 10-15 credits
        """
        logger.info(f"Full validation for {company_name}")

        # Run all validation modes and combine results
        simple = await self.validate_simple(company_name, location)
        contact = await self.validate_smart_contact(company_name, location)
        csr = await self.validate_smart_csr(company_name)
        financial = await self.validate_smart_financial(company_name)

        # Combine results
        validation = ComprehensiveValidation(
            company_name=company_name,
            validation_mode=ValidationMode.FULL,
            validation_timestamp=datetime.now().isoformat()
        )

        # Merge data
        validation.basic_info = simple.basic_info
        validation.contact_info = contact.contact_info
        validation.csr_info = csr.csr_info
        validation.financial_info = financial.financial_info

        # Combine queries and responses
        validation.queries_executed = (
                simple.queries_executed +
                contact.queries_executed +
                csr.queries_executed +
                financial.queries_executed
        )
        validation.raw_responses = (
                simple.raw_responses +
                contact.raw_responses +
                csr.raw_responses +
                financial.raw_responses
        )

        # Calculate overall confidence
        validation.exists = simple.exists or contact.exists
        validation.location_verified = simple.location_verified

        # Weighted average of confidence scores
        weights = [0.3, 0.2, 0.25, 0.25]  # simple, contact, csr, financial
        scores = [simple.confidence_score, contact.confidence_score,
                  csr.confidence_score, financial.confidence_score]
        validation.confidence_score = sum(w * s for w, s in zip(weights, scores))

        if validation.confidence_score >= 80:
            validation.validation_status = "verified"
        elif validation.confidence_score >= 60:
            validation.validation_status = "partial"
        else:
            validation.validation_status = "unverified"

        validation.credits_used = len(validation.queries_executed)

        return validation

    async def validate_custom(
            self,
            company_name: str,
            endpoints: List[EndpointConfig]
    ) -> ComprehensiveValidation:
        """
        Custom validation mode - user-defined endpoints and queries
        """
        logger.info(f"Custom validation for {company_name}")

        validation = ComprehensiveValidation(
            company_name=company_name,
            validation_mode=ValidationMode.CUSTOM,
            validation_timestamp=datetime.now().isoformat()
        )

        for config in endpoints:
            if not config.enabled:
                continue

            for query_template in config.query_templates[:config.max_queries]:
                # Replace placeholder with company name
                query = query_template.replace("{company}", company_name)

                result = self._make_serper_request(
                    config.endpoint.value,
                    query
                )

                validation.queries_executed.append(f"{config.endpoint.value}: {query}")
                validation.raw_responses.append({
                    "endpoint": config.endpoint.value,
                    "response": result
                })

        # Store all results in basic_info for custom processing
        validation.basic_info = {
            "raw_responses": validation.raw_responses,
            "queries": validation.queries_executed
        }

        validation.exists = len(validation.raw_responses) > 0
        validation.confidence_score = 50  # Neutral confidence for custom
        validation.validation_status = "custom"
        validation.credits_used = len(validation.queries_executed)

        return validation

    async def validate_batch(
            self,
            companies: List[Dict[str, Any]],
            mode: ValidationMode,
            location: str = "Australia",
            parallel_limit: int = 5
    ) -> List[ComprehensiveValidation]:
        """
        Validate multiple companies in batch
        """
        results = []

        for i in range(0, len(companies), parallel_limit):
            batch = companies[i:i + parallel_limit]

            tasks = []
            for company in batch:
                company_name = company.get('name', '')

                if mode == ValidationMode.SIMPLE:
                    task = self.validate_simple(company_name, location)
                elif mode == ValidationMode.SMART_CONTACT:
                    task = self.validate_smart_contact(company_name, location)
                elif mode == ValidationMode.SMART_CSR:
                    task = self.validate_smart_csr(company_name)
                elif mode == ValidationMode.SMART_FINANCIAL:
                    task = self.validate_smart_financial(company_name)
                elif mode == ValidationMode.FULL:
                    task = self.validate_full(company_name, location)
                else:
                    continue

                tasks.append(task)

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Progress update
            logger.info(f"Validated {min(i + parallel_limit, len(companies))}/{len(companies)} companies")

            # Rate limiting
            if i + parallel_limit < len(companies):
                await asyncio.sleep(1)

        return results

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation activities"""
        runtime = (datetime.now() - self.start_time).total_seconds()

        return {
            "total_credits_used": self.total_credits,
            "estimated_cost": self.total_credits * 0.001,
            "runtime_seconds": runtime,
            "cache_size": len(self.validation_cache)
        }


# Backward compatibility
ValidationOrchestrator = EnhancedValidationAgent