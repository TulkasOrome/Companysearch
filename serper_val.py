#!/usr/bin/env python3
"""
serper_validation_modes_test.py - Test different validation modes with minimal examples
Tests Simple, Raw Endpoint, Smart, and Custom validation approaches
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


class SerperValidationTester:
    """Test different validation modes with Serper API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.total_credits = 0

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

    async def simple_validation(self, company_name: str, location: str) -> ValidationResult:
        """
        SIMPLE MODE: Minimal validation - just confirm existence
        Uses: 2-3 credits max
        """
        print(f"\n{'=' * 60}")
        print(f"SIMPLE VALIDATION: {company_name}")
        print(f"{'=' * 60}")

        start_credits = self.total_credits
        raw_responses = []

        # Step 1: Places search to confirm existence and location
        print("1. Checking existence via Places API...")
        places_result = self._make_serper_request(
            "places",
            f"{company_name} {location}",
            {"location": location}
        )
        raw_responses.append({"endpoint": "places", "response": places_result})

        exists = False
        location_match = False
        confidence = 0

        if places_result.get('places'):
            place = places_result['places'][0]
            place_name = place.get('title', '').lower()
            if company_name.lower() in place_name or place_name in company_name.lower():
                exists = True
                confidence += 50
                if location.lower() in place.get('address', '').lower():
                    location_match = True
                    confidence += 30
                print(f"  ✓ Found: {place.get('title')}")
                print(f"  📍 Address: {place.get('address')}")

        # Step 2: Quick web search for industry confirmation
        print("2. Confirming industry via Web Search...")
        web_result = self._make_serper_request(
            "search",
            f'"{company_name}" company Australia',
            {"num": 3}
        )
        raw_responses.append({"endpoint": "search", "response": web_result})

        industry_confirmed = False
        if web_result.get('organic'):
            for result in web_result['organic'][:3]:
                if company_name.lower() in result.get('title', '').lower():
                    industry_confirmed = True
                    confidence += 20
                    print(f"  ✓ Web presence confirmed")
                    break

        # Compile result
        validation_data = {
            "exists": exists,
            "location_match": location_match,
            "industry_confirmed": industry_confirmed,
            "confidence": min(confidence, 100),
            "basic_info": {
                "address": places_result['places'][0].get('address') if places_result.get('places') else None,
                "phone": places_result['places'][0].get('phoneNumber') if places_result.get('places') else None,
                "website": places_result['places'][0].get('website') if places_result.get('places') else None
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

    async def raw_endpoint_query(self, endpoint: SerperEndpoint, query: str, options: Dict = None) -> ValidationResult:
        """
        RAW ENDPOINT MODE: Direct access to any endpoint
        Returns raw response from Serper
        """
        print(f"\n{'=' * 60}")
        print(f"RAW ENDPOINT QUERY: {endpoint.value}")
        print(f"Query: {query}")
        print(f"{'=' * 60}")

        start_credits = self.total_credits

        # Make raw request
        result = self._make_serper_request(endpoint.value, query, options)

        # Display key results based on endpoint
        if endpoint == SerperEndpoint.PLACES and result.get('places'):
            print(f"Found {len(result['places'])} places")
            for place in result['places'][:3]:
                print(f"  • {place.get('title')} - {place.get('address')}")

        elif endpoint == SerperEndpoint.NEWS and result.get('news'):
            print(f"Found {len(result['news'])} news items")
            for item in result['news'][:3]:
                print(f"  • {item.get('title')[:60]}...")

        elif endpoint == SerperEndpoint.SEARCH and result.get('organic'):
            print(f"Found {len(result['organic'])} results")
            for item in result['organic'][:3]:
                print(f"  • {item.get('title')}")

        print(f"\nCredits Used: {self.total_credits - start_credits}")

        return ValidationResult(
            company_name=query,
            mode=ValidationMode.RAW_ENDPOINT,
            success=bool(result.get('places') or result.get('organic') or result.get('news')),
            confidence=100,  # Raw mode doesn't calculate confidence
            credits_used=self.total_credits - start_credits,
            data=result,
            raw_responses=[{"endpoint": endpoint.value, "response": result}],
            timestamp=datetime.now().isoformat()
        )

    async def smart_contact_validation(self, company_name: str) -> ValidationResult:
        """
        SMART CONTACT MODE: Focused on extracting contact information
        Uses targeted queries for emails and key personnel
        """
        print(f"\n{'=' * 60}")
        print(f"SMART CONTACT VALIDATION: {company_name}")
        print(f"{'=' * 60}")

        start_credits = self.total_credits
        raw_responses = []
        contacts = {"emails": [], "phones": [], "names": []}

        # 1. Targeted email search
        print("1. Searching for email contacts...")
        email_result = self._make_serper_request(
            "search",
            f'"{company_name}" email contact @',
            {"num": 5}
        )
        raw_responses.append({"endpoint": "search", "response": email_result})

        # Extract emails from results
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if email_result.get('organic'):
            for result in email_result['organic']:
                text = f"{result.get('title', '')} {result.get('snippet', '')}"
                found_emails = re.findall(email_pattern, text)
                contacts['emails'].extend(found_emails)

        # 2. LinkedIn search for executives
        print("2. Searching LinkedIn for executives...")
        linkedin_result = self._make_serper_request(
            "search",
            f'site:linkedin.com "{company_name}" CEO OR Director OR Manager',
            {"num": 5}
        )
        raw_responses.append({"endpoint": "search", "response": linkedin_result})

        # Extract names from LinkedIn results
        if linkedin_result.get('organic'):
            for result in linkedin_result['organic']:
                title = result.get('title', '')
                # LinkedIn titles often start with person's name
                name_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', title)
                if name_match:
                    contacts['names'].append(name_match.group(1))

        # 3. Places for phone numbers
        print("3. Getting phone numbers from Places...")
        places_result = self._make_serper_request("places", company_name)
        raw_responses.append({"endpoint": "places", "response": places_result})

        if places_result.get('places'):
            for place in places_result['places'][:2]:
                if place.get('phoneNumber'):
                    contacts['phones'].append(place['phoneNumber'])

        # Deduplicate
        contacts['emails'] = list(set(contacts['emails']))
        contacts['names'] = list(set(contacts['names']))
        contacts['phones'] = list(set(contacts['phones']))

        print(f"\n📊 Contact Extraction Results:")
        print(f"  Emails Found: {len(contacts['emails'])}")
        if contacts['emails']:
            for email in contacts['emails'][:3]:
                print(f"    • {email}")
        print(f"  Names Found: {len(contacts['names'])}")
        if contacts['names']:
            for name in contacts['names'][:3]:
                print(f"    • {name}")
        print(f"  Phones Found: {len(contacts['phones'])}")
        if contacts['phones']:
            for phone in contacts['phones']:
                print(f"    • {phone}")
        print(f"  Credits Used: {self.total_credits - start_credits}")

        success = bool(contacts['emails'] or contacts['phones'] or contacts['names'])
        confidence = min(100,
                         len(contacts['emails']) * 20 +
                         len(contacts['names']) * 15 +
                         len(contacts['phones']) * 25)

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
        SMART CSR MODE: Focused on CSR programs and community involvement
        Critical for RMH and Guide Dogs use cases
        """
        print(f"\n{'=' * 60}")
        print(f"SMART CSR VALIDATION: {company_name}")
        print(f"{'=' * 60}")

        start_credits = self.total_credits
        raw_responses = []
        csr_data = {
            "has_csr": False,
            "focus_areas": [],
            "certifications": [],
            "recent_activities": [],
            "giving_evidence": []
        }

        # 1. Search for CSR programs
        print("1. Searching for CSR programs...")
        csr_result = self._make_serper_request(
            "search",
            f'"{company_name}" "corporate social responsibility" OR CSR OR sustainability community',
            {"num": 5}
        )
        raw_responses.append({"endpoint": "search", "response": csr_result})

        # Analyze CSR focus areas
        csr_keywords = {
            "children": ["children", "kids", "youth", "school", "education"],
            "community": ["community", "local", "neighborhood", "volunteer"],
            "environment": ["environment", "sustainability", "green", "carbon"],
            "health": ["health", "medical", "hospital", "wellness"],
            "disability": ["disability", "accessible", "inclusion"]
        }

        if csr_result.get('organic'):
            for result in csr_result['organic']:
                text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
                if 'csr' in text or 'social responsibility' in text:
                    csr_data['has_csr'] = True

                for area, keywords in csr_keywords.items():
                    if any(keyword in text for keyword in keywords):
                        if area not in csr_data['focus_areas']:
                            csr_data['focus_areas'].append(area)

        # 2. Check for certifications
        print("2. Checking for certifications...")
        cert_result = self._make_serper_request(
            "search",
            f'"{company_name}" "B-Corp" OR "ISO 26000" OR "carbon neutral" certification',
            {"num": 3}
        )
        raw_responses.append({"endpoint": "search", "response": cert_result})

        cert_patterns = ["b-corp", "b corp", "iso 26000", "iso 14001", "carbon neutral"]
        if cert_result.get('organic'):
            for result in cert_result['organic']:
                text = result.get('snippet', '').lower()
                for cert in cert_patterns:
                    if cert in text:
                        csr_data['certifications'].append(cert)

        # 3. Recent CSR news
        print("3. Checking recent CSR activities...")
        news_result = self._make_serper_request(
            "news",
            f'"{company_name}" donation OR charity OR community OR sponsorship',
            {"time": "year"}
        )
        raw_responses.append({"endpoint": "news", "response": news_result})

        if news_result.get('news'):
            for item in news_result['news'][:5]:
                title = item.get('title', '').lower()
                if any(word in title for word in ['donation', 'charity', 'sponsor', 'community']):
                    csr_data['recent_activities'].append({
                        "title": item.get('title'),
                        "date": item.get('date')
                    })

        # Deduplicate
        csr_data['certifications'] = list(set(csr_data['certifications']))

        print(f"\n📊 CSR Validation Results:")
        print(f"  Has CSR Programs: {'✓' if csr_data['has_csr'] else '✗'}")
        print(f"  Focus Areas: {', '.join(csr_data['focus_areas']) if csr_data['focus_areas'] else 'None found'}")
        print(
            f"  Certifications: {', '.join(csr_data['certifications']) if csr_data['certifications'] else 'None found'}")
        print(f"  Recent Activities: {len(csr_data['recent_activities'])}")
        print(f"  Credits Used: {self.total_credits - start_credits}")

        # Calculate confidence based on CSR evidence
        confidence = 0
        if csr_data['has_csr']:
            confidence += 40
        confidence += len(csr_data['focus_areas']) * 15
        confidence += len(csr_data['certifications']) * 20
        confidence += min(len(csr_data['recent_activities']) * 10, 30)
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

    async def run_all_tests(self):
        """Run one company through each validation mode"""
        print("=" * 80)
        print("SERPER VALIDATION MODES TEST")
        print(f"Started: {datetime.now()}")
        print("=" * 80)

        results = []

        # Test 1: Simple Validation
        print("\nTest 1: SIMPLE VALIDATION")
        print("-" * 40)
        simple_result = await self.simple_validation("Merivale", "Sydney")
        results.append(simple_result)

        # Test 2: Raw Endpoint Query - News search
        print("\nTest 2: RAW ENDPOINT QUERY")
        print("-" * 40)
        raw_result = await self.raw_endpoint_query(
            SerperEndpoint.NEWS,
            "Woolworths sustainability",
            {"time": "month"}
        )
        results.append(raw_result)

        # Test 3: Smart Contact Extraction
        print("\nTest 3: SMART CONTACT EXTRACTION")
        print("-" * 40)
        contact_result = await self.smart_contact_validation("Richard Crookes Constructions")
        results.append(contact_result)

        # Test 4: Smart CSR Validation
        print("\nTest 4: SMART CSR VALIDATION")
        print("-" * 40)
        csr_result = await self.smart_csr_validation("Coles Group")
        results.append(csr_result)

        # Summary Report
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        total_credits = sum(r.credits_used for r in results)
        successful = sum(1 for r in results if r.success)

        print(f"\nTests Run: {len(results)}")
        print(f"Successful: {successful}/{len(results)}")
        print(f"Total Credits Used: {total_credits}")
        print(f"Average Credits per Test: {total_credits / len(results):.1f}")

        print("\nPer-Mode Results:")
        for result in results:
            status = "✓" if result.success else "✗"
            print(f"  {status} {result.mode.value:20s} - {result.company_name[:30]:30s} "
                  f"Credits: {result.credits_used:3d} Confidence: {result.confidence:3.0f}%")

        # Save results
        output = {
            "test_time": datetime.now().isoformat(),
            "total_credits": total_credits,
            "results": [asdict(r) for r in results]
        }

        filename = f"validation_modes_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {filename}")

        return results


async def main():
    """Main test runner"""
    SERPER_API_KEY = "99c44b79892f5f7499accf2d7c26d93313880937"

    tester = SerperValidationTester(SERPER_API_KEY)
    results = await tester.run_all_tests()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    asyncio.run(main())