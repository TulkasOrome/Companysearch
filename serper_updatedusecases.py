#!/usr/bin/env python3
"""
serper_validation_improved.py - Improved validation modes with 3 companies per test
Fixes issues found in initial test and adds better name matching and CSR detection
"""

import asyncio
import json
import http.client
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict


class ValidationMode(Enum):
    """Validation modes available"""
    SIMPLE = "simple"  # Just verify existence
    RAW_ENDPOINT = "raw"  # Direct endpoint access
    SMART_CONTACT = "smart_contact"  # Contact extraction focus
    SMART_CSR = "smart_csr"  # CSR verification focus
    SMART_FINANCIAL = "smart_financial"  # Financial verification
    FULL = "full"  # Complete validation


class SerperEndpoint(Enum):
    """Available Serper endpoints"""
    SEARCH = "search"
    PLACES = "places"
    MAPS = "maps"
    NEWS = "news"
    SHOPPING = "shopping"


@dataclass
class ValidationResult:
    """Unified validation result structure"""
    company_name: str
    mode: ValidationMode
    success: bool
    confidence: float
    credits_used: int
    data: Dict[str, Any]
    raw_responses: List[Dict[str, Any]]
    timestamp: str


class ImprovedSerperValidator:
    """Improved validation with better matching and detection"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.total_credits = 0

        # Test companies for each mode
        self.test_companies = {
            "simple": [
                {"name": "Merivale", "location": "Sydney"},
                {"name": "Richard Crookes Constructions", "location": "Sydney"},
                {"name": "Dyldam Developments", "location": "Parramatta"}
            ],
            "contact": [
                "Coles Group",
                "Woolworths Group",
                "Westpac Banking Corporation"
            ],
            "csr": [
                "Commonwealth Bank",
                "Telstra Corporation",
                "BHP Group"
            ],
            "financial": [
                "Qantas Airways",
                "Harvey Norman",
                "JB Hi-Fi"
            ]
        }

    def _make_serper_request(self, endpoint: str, query: str, options: Dict = None) -> Dict[str, Any]:
        """Make a raw Serper API request"""
        try:
            conn = http.client.HTTPSConnection("google.serper.dev")

            payload = {"q": query, "gl": "au"}
            if options:
                payload.update(options)

            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }

            conn.request("POST", f"/{endpoint}", json.dumps(payload), headers)
            res = conn.getresponse()
            data = res.read()

            result = json.loads(data.decode("utf-8"))
            self.total_credits += result.get('credits', 1)

            return result

        except Exception as e:
            return {"error": str(e)}

    def _fuzzy_name_match(self, name1: str, name2: str, threshold: float = 0.7) -> bool:
        """Improved fuzzy name matching"""
        name1_clean = name1.lower().strip()
        name2_clean = name2.lower().strip()

        # Direct substring match
        if name1_clean in name2_clean or name2_clean in name1_clean:
            return True

        # Remove common suffixes and try again
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

    async def simple_validation(self, company_name: str, location: str) -> ValidationResult:
        """
        IMPROVED SIMPLE MODE: Better name matching and fallback strategies
        """
        print(f"\n{'=' * 60}")
        print(f"SIMPLE VALIDATION: {company_name}")
        print(f"{'=' * 60}")

        start_credits = self.total_credits
        raw_responses = []

        # Try multiple name variations for Places API
        print("1. Checking existence via Places API...")

        name_variations = [
            f"{company_name} {location}",
            company_name,
            f"{company_name} pty ltd",
            f"{company_name} limited"
        ]

        exists = False
        location_match = False
        confidence = 0
        place_found = None

        for variation in name_variations[:2]:  # Try first 2 to save credits
            places_result = self._make_serper_request(
                "places",
                variation,
                {"location": location}
            )
            raw_responses.append({"endpoint": "places", "query": variation, "response": places_result})

            if places_result.get('places'):
                for place in places_result['places'][:3]:
                    place_name = place.get('title', '')
                    if self._fuzzy_name_match(company_name, place_name):
                        exists = True
                        place_found = place
                        confidence += 50

                        # Check location match
                        address = place.get('address', '').lower()
                        if location.lower() in address:
                            location_match = True
                            confidence += 30

                        print(f"  ✓ Found: {place.get('title')}")
                        print(f"  📍 Address: {place.get('address')}")
                        print(f"  📞 Phone: {place.get('phoneNumber', 'N/A')}")
                        break

                if exists:
                    break

        # Fallback to web search if not found in Places
        if not exists:
            print("2. Fallback to Web Search...")
            web_result = self._make_serper_request(
                "search",
                f'"{company_name}" {location} Australia company',
                {"num": 5}
            )
            raw_responses.append({"endpoint": "search", "response": web_result})

            if web_result.get('organic'):
                for result in web_result['organic'][:3]:
                    title = result.get('title', '').lower()
                    snippet = result.get('snippet', '').lower()

                    if self._fuzzy_name_match(company_name, title):
                        exists = True
                        confidence += 30
                        print(f"  ✓ Found in web search: {result.get('title')}")

                        if location.lower() in snippet:
                            location_match = True
                            confidence += 20
                        break

        # Industry confirmation
        if exists and confidence < 70:
            print("3. Confirming industry...")
            industry_result = self._make_serper_request(
                "search",
                f'"{company_name}" industry sector business',
                {"num": 3}
            )
            raw_responses.append({"endpoint": "search", "response": industry_result})

            if industry_result.get('organic'):
                confidence += 10
                print(f"  ✓ Industry presence confirmed")

        # Compile result
        validation_data = {
            "exists": exists,
            "location_match": location_match,
            "confidence": min(confidence, 100),
            "basic_info": {
                "name": place_found.get('title') if place_found else company_name,
                "address": place_found.get('address') if place_found else None,
                "phone": place_found.get('phoneNumber') if place_found else None,
                "website": place_found.get('website') if place_found else None,
                "rating": place_found.get('rating') if place_found else None
            }
        }

        print(f"\n📊 Simple Validation Result:")
        print(f"  Exists: {'✓' if exists else '✗'}")
        print(f"  Location Match: {'✓' if location_match else '✗'}")
        print(f"  Confidence: {confidence}%")
        print(f"  Credits Used: {self.total_credits - start_credits}")

        return ValidationResult(
            company_name=company_name,
            mode=ValidationMode.SIMPLE,
            success=exists,
            confidence=confidence,
            credits_used=self.total_credits - start_credits,
            data=validation_data,
            raw_responses=raw_responses,
            timestamp=datetime.now().isoformat()
        )

    async def smart_contact_validation(self, company_name: str) -> ValidationResult:
        """
        IMPROVED CONTACT MODE: Better email extraction and name parsing
        """
        print(f"\n{'=' * 60}")
        print(f"SMART CONTACT VALIDATION: {company_name}")
        print(f"{'=' * 60}")

        start_credits = self.total_credits
        raw_responses = []
        contacts = {"emails": [], "phones": [], "names": [], "titles": []}

        # 1. Enhanced email search with multiple patterns
        print("1. Searching for email contacts...")
        email_queries = [
            f'"{company_name}" email contact @',
            f'"{company_name}" "contact us" email',
            f'"{company_name}" enquiries @'
        ]

        for query in email_queries[:2]:  # Use first 2 queries
            email_result = self._make_serper_request(
                "search",
                query,
                {"num": 5}
            )
            raw_responses.append({"endpoint": "search", "query": query, "response": email_result})

            # Enhanced email extraction
            email_patterns = [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                r'mailto:([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
                r'\b([A-Za-z0-9._%+-]+)\s*\[\s*at\s*\]\s*([A-Za-z0-9.-]+)\s*\[\s*dot\s*\]\s*([A-Z|a-z]{2,})\b'
            ]

            if email_result.get('organic'):
                for result in email_result['organic']:
                    text = f"{result.get('title', '')} {result.get('snippet', '')}"

                    for pattern in email_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        for match in matches:
                            if isinstance(match, tuple):
                                # Handle obfuscated emails
                                email = f"{match[0]}@{match[1]}.{match[2]}"
                            else:
                                email = match

                            # Filter out generic/invalid emails
                            if '@' in email and not any(
                                    x in email.lower() for x in ['example.com', 'email.com', 'sentry.io']):
                                contacts['emails'].append(email.lower())

        # 2. Enhanced LinkedIn search
        print("2. Searching LinkedIn for executives...")
        linkedin_result = self._make_serper_request(
            "search",
            f'site:linkedin.com "{company_name}" CEO OR Director OR Manager OR Head',
            {"num": 10}
        )
        raw_responses.append({"endpoint": "search", "response": linkedin_result})

        # Improved name extraction from LinkedIn
        if linkedin_result.get('organic'):
            for result in linkedin_result['organic']:
                title = result.get('title', '')
                snippet = result.get('snippet', '')

                # Pattern for LinkedIn: "Name - Title - Company | LinkedIn"
                name_title_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*[-–]\s*([^-|]+)', title)
                if name_title_match:
                    name = name_title_match.group(1).strip()
                    job_title = name_title_match.group(2).strip()

                    if len(name.split()) >= 2 and len(name.split()) <= 4:
                        contacts['names'].append(name)
                        contacts['titles'].append(job_title)

                # Also check snippet for names with titles
                exec_patterns = [
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),?\s+(?:CEO|CFO|CTO|Director|Manager|Head)',
                    r'(?:CEO|CFO|CTO|Director|Manager|Head)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
                ]

                for pattern in exec_patterns:
                    matches = re.findall(pattern, snippet)
                    contacts['names'].extend(matches)

        # 3. Enhanced Places search
        print("3. Getting contact details from Places...")
        places_result = self._make_serper_request("places", company_name)
        raw_responses.append({"endpoint": "places", "response": places_result})

        if places_result.get('places'):
            for place in places_result['places'][:2]:
                if self._fuzzy_name_match(company_name, place.get('title', '')):
                    if place.get('phoneNumber'):
                        contacts['phones'].append(place['phoneNumber'])
                    if place.get('website'):
                        # Sometimes email is in website metadata
                        contacts['emails'].append(
                            f"info@{place['website'].replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]}")

        # 4. News search for recent mentions with contacts
        print("4. Checking news for spokespersons...")
        news_result = self._make_serper_request(
            "news",
            f'"{company_name}" announced said spokesperson',
            {"time": "month"}
        )
        raw_responses.append({"endpoint": "news", "response": news_result})

        if news_result.get('news'):
            for item in news_result['news'][:5]:
                snippet = item.get('snippet', '')

                # Look for quoted names
                quote_patterns = [
                    r'"[^"]+"\s+said\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+said\s+in\s+a\s+statement'
                ]

                for pattern in quote_patterns:
                    matches = re.findall(pattern, snippet)
                    contacts['names'].extend(matches)

        # Deduplicate and clean
        contacts['emails'] = list(set(contacts['emails']))[:10]
        contacts['names'] = list(set(contacts['names']))[:10]
        contacts['phones'] = list(set(contacts['phones']))[:5]
        contacts['titles'] = list(set(contacts['titles']))[:10]

        print(f"\n📊 Contact Extraction Results:")
        print(f"  Emails Found: {len(contacts['emails'])}")
        if contacts['emails']:
            for email in contacts['emails'][:3]:
                print(f"    • {email}")
        print(f"  Names Found: {len(contacts['names'])}")
        if contacts['names']:
            for name in contacts['names'][:3]:
                print(f"    • {name}")
        print(f"  Titles Found: {len(contacts['titles'])}")
        if contacts['titles']:
            for title in contacts['titles'][:3]:
                print(f"    • {title}")
        print(f"  Phones Found: {len(contacts['phones'])}")
        if contacts['phones']:
            for phone in contacts['phones']:
                print(f"    • {phone}")
        print(f"  Credits Used: {self.total_credits - start_credits}")

        success = bool(contacts['emails'] or contacts['phones'] or contacts['names'])
        confidence = min(100,
                         len(contacts['emails']) * 15 +
                         len(contacts['names']) * 10 +
                         len(contacts['phones']) * 20 +
                         len(contacts['titles']) * 5)

        return ValidationResult(
            company_name=company_name,
            mode=ValidationMode.SMART_CONTACT,
            success=success,
            confidence=confidence,
            credits_used=self.total_credits - start_credits,
            data=contacts,
            raw_responses=raw_responses,
            timestamp=datetime.now().isoformat()
        )

    async def smart_csr_validation(self, company_name: str) -> ValidationResult:
        """
        IMPROVED CSR MODE: Better detection of CSR programs and foundations
        """
        print(f"\n{'=' * 60}")
        print(f"SMART CSR VALIDATION: {company_name}")
        print(f"{'=' * 60}")

        start_credits = self.total_credits
        raw_responses = []
        csr_data = {
            "has_csr": False,
            "has_foundation": False,
            "focus_areas": [],
            "certifications": [],
            "recent_activities": [],
            "giving_evidence": [],
            "esg_score": None,
            "sustainability_report": False
        }

        # 1. Check for company foundation or formal CSR program
        print("1. Checking for foundation/formal CSR program...")
        foundation_result = self._make_serper_request(
            "search",
            f'"{company_name} Foundation" OR "{company_name} CSR" OR "{company_name} sustainability report"',
            {"num": 5}
        )
        raw_responses.append({"endpoint": "search", "response": foundation_result})

        if foundation_result.get('organic'):
            for result in foundation_result['organic']:
                title = result.get('title', '').lower()
                snippet = result.get('snippet', '').lower()
                text = f"{title} {snippet}"

                # Check for foundation
                if f"{company_name.lower()} foundation" in text:
                    csr_data['has_foundation'] = True
                    csr_data['has_csr'] = True
                    print(f"  ✓ Found {company_name} Foundation")

                # Check for sustainability report
                if 'sustainability report' in text or 'esg report' in text:
                    csr_data['sustainability_report'] = True
                    csr_data['has_csr'] = True
                    print(f"  ✓ Found sustainability/ESG report")

                # Check for CSR programs
                if any(term in text for term in
                       ['corporate social responsibility', 'csr program', 'community program']):
                    csr_data['has_csr'] = True

        # 2. Search for CSR focus areas and activities
        print("2. Identifying CSR focus areas...")
        csr_result = self._make_serper_request(
            "search",
            f'"{company_name}" community donation charity sponsorship volunteer environment',
            {"num": 10}
        )
        raw_responses.append({"endpoint": "search", "response": csr_result})

        # Enhanced CSR keyword detection
        csr_keywords = {
            "children": ["children", "kids", "youth", "school", "education", "students"],
            "community": ["community", "local", "neighborhood", "volunteer", "civic"],
            "environment": ["environment", "sustainability", "green", "carbon", "climate", "renewable"],
            "health": ["health", "medical", "hospital", "wellness", "mental health", "healthcare"],
            "disability": ["disability", "accessible", "inclusion", "special needs"],
            "indigenous": ["indigenous", "aboriginal", "first nations", "reconciliation"],
            "arts": ["arts", "culture", "music", "creative", "museum"],
            "education": ["education", "scholarship", "literacy", "stem", "training"]
        }

        if csr_result.get('organic'):
            for result in csr_result['organic']:
                text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()

                # Detect focus areas
                for area, keywords in csr_keywords.items():
                    if any(keyword in text for keyword in keywords):
                        if area not in csr_data['focus_areas']:
                            csr_data['focus_areas'].append(area)

                # Look for giving evidence
                giving_patterns = [
                    r'\$[\d,]+(?:\.\d+)?\s*(?:million|thousand|k|m)?\s*(?:donated|raised|contributed)',
                    r'donated\s*\$[\d,]+',
                    r'raised\s*\$[\d,]+',
                    r'contributed\s*\$[\d,]+'
                ]

                for pattern in giving_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        csr_data['giving_evidence'].extend(matches)
                        csr_data['has_csr'] = True

        # 3. Check for certifications and memberships
        print("3. Checking for certifications...")
        cert_result = self._make_serper_request(
            "search",
            f'"{company_name}" "B Corp" OR "ISO 26000" OR "ISO 14001" OR "carbon neutral" OR "GRI" certification',
            {"num": 5}
        )
        raw_responses.append({"endpoint": "search", "response": cert_result})

        cert_keywords = {
            "B-Corp": ["b corp", "b-corp", "certified b corporation"],
            "ISO 26000": ["iso 26000", "iso26000"],
            "ISO 14001": ["iso 14001", "iso14001"],
            "Carbon Neutral": ["carbon neutral", "net zero", "carbon negative"],
            "GRI": ["gri standards", "global reporting initiative"],
            "UN Global Compact": ["un global compact", "ungc"]
        }

        if cert_result.get('organic'):
            for result in cert_result['organic']:
                text = result.get('snippet', '').lower()

                for cert_name, patterns in cert_keywords.items():
                    if any(pattern in text for pattern in patterns):
                        if cert_name not in csr_data['certifications']:
                            csr_data['certifications'].append(cert_name)
                            print(f"  ✓ Found certification: {cert_name}")

        # 4. Recent CSR news and activities
        print("4. Checking recent CSR activities...")
        news_result = self._make_serper_request(
            "news",
            f'"{company_name}" donation charity sponsor community partnership',
            {"time": "year"}
        )
        raw_responses.append({"endpoint": "news", "response": news_result})

        if news_result.get('news'):
            for item in news_result['news'][:10]:
                title = item.get('title', '').lower()
                snippet = item.get('snippet', '').lower()

                # Check for CSR-related news
                csr_terms = ['donation', 'charity', 'sponsor', 'volunteer', 'community', 'foundation', 'scholarship']
                if any(term in title or term in snippet for term in csr_terms):
                    csr_data['recent_activities'].append({
                        "title": item.get('title'),
                        "date": item.get('date'),
                        "source": item.get('source')
                    })
                    csr_data['has_csr'] = True

        # Limit activities to most recent 5
        csr_data['recent_activities'] = csr_data['recent_activities'][:5]
        csr_data['giving_evidence'] = list(set(csr_data['giving_evidence']))[:5]

        print(f"\n📊 CSR Validation Results:")
        print(f"  Has CSR Programs: {'✓' if csr_data['has_csr'] else '✗'}")
        print(f"  Has Foundation: {'✓' if csr_data['has_foundation'] else '✗'}")
        print(f"  Sustainability Report: {'✓' if csr_data['sustainability_report'] else '✗'}")
        print(f"  Focus Areas: {', '.join(csr_data['focus_areas']) if csr_data['focus_areas'] else 'None found'}")
        print(
            f"  Certifications: {', '.join(csr_data['certifications']) if csr_data['certifications'] else 'None found'}")
        print(f"  Recent Activities: {len(csr_data['recent_activities'])}")
        print(f"  Giving Evidence: {len(csr_data['giving_evidence'])} instances")
        print(f"  Credits Used: {self.total_credits - start_credits}")

        # Calculate confidence based on CSR evidence
        confidence = 0
        if csr_data['has_csr']:
            confidence += 30
        if csr_data['has_foundation']:
            confidence += 20
        if csr_data['sustainability_report']:
            confidence += 15
        confidence += min(len(csr_data['focus_areas']) * 5, 20)
        confidence += min(len(csr_data['certifications']) * 10, 20)
        confidence += min(len(csr_data['recent_activities']) * 3, 15)
        confidence = min(confidence, 100)

        return ValidationResult(
            company_name=company_name,
            mode=ValidationMode.SMART_CSR,
            success=csr_data['has_csr'],
            confidence=confidence,
            credits_used=self.total_credits - start_credits,
            data=csr_data,
            raw_responses=raw_responses,
            timestamp=datetime.now().isoformat()
        )

    async def smart_financial_validation(self, company_name: str) -> ValidationResult:
        """
        NEW: SMART FINANCIAL MODE - Revenue, employees, growth indicators
        """
        print(f"\n{'=' * 60}")
        print(f"SMART FINANCIAL VALIDATION: {company_name}")
        print(f"{'=' * 60}")

        start_credits = self.total_credits
        raw_responses = []
        financial_data = {
            "revenue": None,
            "revenue_currency": None,
            "employees": None,
            "growth_stage": None,
            "recent_funding": [],
            "financial_events": [],
            "stock_listed": False,
            "asx_code": None
        }

        # 1. Search for revenue and employee data
        print("1. Searching for revenue and employee data...")
        financial_result = self._make_serper_request(
            "search",
            f'"{company_name}" annual revenue employees million billion AUD',
            {"num": 10}
        )
        raw_responses.append({"endpoint": "search", "response": financial_result})

        # Revenue extraction patterns
        revenue_patterns = [
            r'(?:revenue|turnover|sales).*?\$?([\d,]+\.?\d*)\s*(million|billion|m|b)',
            r'\$?([\d,]+\.?\d*)\s*(million|billion|m|b)\s*(?:revenue|turnover|sales)',
            r'(?:AUD|AU\$|A\$)\s*([\d,]+\.?\d*)\s*(million|billion|m|b)',
        ]

        # Employee extraction patterns
        employee_patterns = [
            r'([\d,]+)\s*(?:employees|staff|workers)',
            r'(?:employs|workforce of)\s*([\d,]+)',
            r'team of\s*([\d,]+)',
        ]

        if financial_result.get('organic'):
            for result in financial_result['organic']:
                text = f"{result.get('title', '')} {result.get('snippet', '')}"

                # Extract revenue
                if not financial_data['revenue']:
                    for pattern in revenue_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            match = matches[0]
                            amount = match[0] if isinstance(match, tuple) else match
                            unit = match[1] if isinstance(match, tuple) and len(match) > 1 else ''

                            # Convert to number
                            amount = amount.replace(',', '')
                            multiplier = 1000000 if 'm' in unit.lower() else 1000000000
                            financial_data['revenue'] = f"${amount}{unit.upper()}"
                            financial_data[
                                'revenue_currency'] = 'AUD' if 'aud' in text.lower() or 'a$' in text.lower() else 'USD'
                            print(f"  ✓ Revenue: {financial_data['revenue']} {financial_data['revenue_currency']}")
                            break

                # Extract employees
                if not financial_data['employees']:
                    for pattern in employee_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            financial_data['employees'] = matches[0]
                            print(f"  ✓ Employees: {financial_data['employees']}")
                            break

        # 2. Check if publicly listed (ASX)
        print("2. Checking if publicly listed...")
        asx_result = self._make_serper_request(
            "search",
            f'"{company_name}" ASX listed share price stock code',
            {"num": 3}
        )
        raw_responses.append({"endpoint": "search", "response": asx_result})

        if asx_result.get('organic'):
            for result in asx_result['organic']:
                text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()

                # Look for ASX code
                asx_pattern = r'asx:\s*([A-Z]{3,4})'
                matches = re.findall(asx_pattern, text, re.IGNORECASE)
                if matches:
                    financial_data['stock_listed'] = True
                    financial_data['asx_code'] = matches[0].upper()
                    print(f"  ✓ Listed on ASX: {financial_data['asx_code']}")

        # 3. Recent financial news
        print("3. Checking recent financial news...")
        news_result = self._make_serper_request(
            "news",
            f'"{company_name}" funding investment acquisition merger IPO expansion',
            {"time": "year"}
        )
        raw_responses.append({"endpoint": "news", "response": news_result})

        growth_indicators = ['funding', 'investment', 'acquisition', 'merger', 'expansion', 'ipo', 'growth']

        if news_result.get('news'):
            for item in news_result['news'][:5]:
                title = item.get('title', '').lower()

                for indicator in growth_indicators:
                    if indicator in title:
                        financial_data['financial_events'].append({
                            "type": indicator,
                            "title": item.get('title'),
                            "date": item.get('date')
                        })
                        break

        # Determine growth stage
        if financial_data['financial_events']:
            recent_events = [e['type'] for e in financial_data['financial_events']]
            if 'ipo' in recent_events:
                financial_data['growth_stage'] = 'Public'
            elif 'funding' in recent_events or 'investment' in recent_events:
                financial_data['growth_stage'] = 'Growth'
            elif 'expansion' in recent_events or 'acquisition' in recent_events:
                financial_data['growth_stage'] = 'Expansion'
            else:
                financial_data['growth_stage'] = 'Established'

        print(f"\n📊 Financial Validation Results:")
        print(f"  Revenue: {financial_data['revenue'] or 'Not found'}")
        print(f"  Employees: {financial_data['employees'] or 'Not found'}")
        print(f"  Listed: {'✓ (ASX:' + financial_data['asx_code'] + ')' if financial_data['stock_listed'] else '✗'}")
        print(f"  Growth Stage: {financial_data['growth_stage'] or 'Unknown'}")
        print(f"  Recent Events: {len(financial_data['financial_events'])}")
        print(f"  Credits Used: {self.total_credits - start_credits}")

        # Calculate confidence
        confidence = 0
        if financial_data['revenue']:
            confidence += 35
        if financial_data['employees']:
            confidence += 25
        if financial_data['stock_listed']:
            confidence += 20
        confidence += min(len(financial_data['financial_events']) * 5, 20)

        success = bool(financial_data['revenue'] or financial_data['employees'])

        return ValidationResult(
            company_name=company_name,
            mode=ValidationMode.SMART_FINANCIAL,
            success=success,
            confidence=min(confidence, 100),
            credits_used=self.total_credits - start_credits,
            data=financial_data,
            raw_responses=raw_responses,
            timestamp=datetime.now().isoformat()
        )

    async def run_all_tests(self):
        """Run 3 companies through each validation mode"""
        print("=" * 80)
        print("IMPROVED SERPER VALIDATION TEST")
        print(f"Started: {datetime.now()}")
        print("=" * 80)

        all_results = []

        # Test 1: Simple Validation (3 companies)
        print("\n" + "=" * 80)
        print("TEST SET 1: SIMPLE VALIDATION")
        print("=" * 80)

        for company in self.test_companies["simple"]:
            result = await self.simple_validation(company["name"], company["location"])
            all_results.append(result)
            await asyncio.sleep(1)  # Rate limiting

        # Test 2: Contact Extraction (3 companies)
        print("\n" + "=" * 80)
        print("TEST SET 2: SMART CONTACT EXTRACTION")
        print("=" * 80)

        for company in self.test_companies["contact"]:
            result = await self.smart_contact_validation(company)
            all_results.append(result)
            await asyncio.sleep(1)

        # Test 3: CSR Validation (3 companies)
        print("\n" + "=" * 80)
        print("TEST SET 3: SMART CSR VALIDATION")
        print("=" * 80)

        for company in self.test_companies["csr"]:
            result = await self.smart_csr_validation(company)
            all_results.append(result)
            await asyncio.sleep(1)

        # Test 4: Financial Validation (3 companies)
        print("\n" + "=" * 80)
        print("TEST SET 4: SMART FINANCIAL VALIDATION")
        print("=" * 80)

        for company in self.test_companies["financial"]:
            result = await self.smart_financial_validation(company)
            all_results.append(result)
            await asyncio.sleep(1)

        # Generate comprehensive report
        self.generate_report(all_results)

        return all_results

    def generate_report(self, results: List[ValidationResult]):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST REPORT")
        print("=" * 80)

        # Group results by mode
        mode_results = {}
        for result in results:
            mode = result.mode.value
            if mode not in mode_results:
                mode_results[mode] = []
            mode_results[mode].append(result)

        # Overall statistics
        total_tests = len(results)
        successful = sum(1 for r in results if r.success)
        total_credits = sum(r.credits_used for r in results)

        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Successful: {successful}/{total_tests} ({successful / total_tests * 100:.1f}%)")
        print(f"  Total Credits Used: {total_credits}")
        print(f"  Average Credits per Test: {total_credits / total_tests:.1f}")

        # Per-mode analysis
        print(f"\n📈 PER-MODE ANALYSIS:")
        for mode, mode_results_list in mode_results.items():
            success_rate = sum(1 for r in mode_results_list if r.success) / len(mode_results_list) * 100
            avg_credits = sum(r.credits_used for r in mode_results_list) / len(mode_results_list)
            avg_confidence = sum(r.confidence for r in mode_results_list) / len(mode_results_list)

            print(f"\n  {mode.upper()}:")
            print(f"    Tests: {len(mode_results_list)}")
            print(f"    Success Rate: {success_rate:.0f}%")
            print(f"    Avg Credits: {avg_credits:.1f}")
            print(f"    Avg Confidence: {avg_confidence:.0f}%")

            # Individual results
            for result in mode_results_list:
                status = "✓" if result.success else "✗"
                print(
                    f"      {status} {result.company_name:30s} Credits: {result.credits_used:2d} Confidence: {result.confidence:3.0f}%")

        # Key findings
        print(f"\n🔍 KEY FINDINGS:")

        # Best performing mode
        best_mode = max(mode_results.items(),
                        key=lambda x: sum(r.success for r in x[1]) / len(x[1]))
        print(
            f"  Best Success Rate: {best_mode[0]} ({sum(r.success for r in best_mode[1]) / len(best_mode[1]) * 100:.0f}%)")

        # Most efficient mode
        efficient_mode = min(mode_results.items(),
                             key=lambda x: sum(r.credits_used for r in x[1]) / len(x[1]))
        print(
            f"  Most Credit Efficient: {efficient_mode[0]} ({sum(r.credits_used for r in efficient_mode[1]) / len(efficient_mode[1]):.1f} credits/test)")

        # Save detailed results
        output = {
            "test_time": datetime.now().isoformat(),
            "total_credits": total_credits,
            "success_rate": successful / total_tests,
            "results": [asdict(r) for r in results]
        }

        filename = f"validation_improved_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: {filename}")


async def main():
    """Main test runner"""
    SERPER_API_KEY = "99c44b79892f5f7499accf2d7c26d93313880937"

    validator = ImprovedSerperValidator(SERPER_API_KEY)
    results = await validator.run_all_tests()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    asyncio.run(main())