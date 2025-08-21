#!/usr/bin/env python3
"""
serper_validation_test.py - Comprehensive test of Serper API for company validation
Tests search, places, news, and website scraping endpoints
Attempts to extract contact information from results
"""

import asyncio
import json
import http.client
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import time
import traceback

# Add project root to path
project_root = Path(__file__).parent if '__file__' in globals() else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import your search agent
from agents.search_strategist_agent import (
    EnhancedSearchStrategistAgent,
    SearchCriteria,
    LocationCriteria,
    FinancialCriteria,
    OrganizationalCriteria,
    BehavioralSignals
)

# Try to import ICP manager
try:
    from enhanced_icp_manager import ICPManager

    ICP_AVAILABLE = True
except ImportError:
    ICP_AVAILABLE = False
    print("Warning: ICP Manager not available, using manual criteria")


class SerperAPITester:
    """Test Serper API endpoints for company validation"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.results = []
        self.total_credits_used = 0

    def search_web(self, query: str, location: str = "au") -> Dict[str, Any]:
        """
        Perform web search using Serper
        Returns organic results, related searches, etc.
        """
        print(f"  🔍 Web Search: {query}")

        try:
            conn = http.client.HTTPSConnection("google.serper.dev")
            payload = json.dumps({
                "q": query,
                "gl": location,
                "num": 10  # Get more results for better coverage
            })
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }

            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = res.read()

            result = json.loads(data.decode("utf-8"))
            credits = result.get('credits', 1)
            self.total_credits_used += credits

            print(f"    ✅ Found {len(result.get('organic', []))} results (Credits: {credits})")
            return result

        except Exception as e:
            print(f"    ❌ Web search failed: {e}")
            return {"error": str(e), "organic": []}

    def search_places(self, company_name: str, location: str = "Sydney") -> Dict[str, Any]:
        """
        Search for company in Google Places
        Returns address, phone, website, ratings, etc.
        """
        print(f"  📍 Places Search: {company_name} in {location}")

        try:
            conn = http.client.HTTPSConnection("google.serper.dev")
            payload = json.dumps({
                "q": f"{company_name} {location}",
                "gl": "au",
                "location": location
            })
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }

            conn.request("POST", "/places", payload, headers)
            res = conn.getresponse()
            data = res.read()

            result = json.loads(data.decode("utf-8"))
            credits = result.get('credits', 2)
            self.total_credits_used += credits

            places = result.get('places', [])
            print(f"    ✅ Found {len(places)} places (Credits: {credits})")

            # Extract key info from first result if available
            if places:
                place = places[0]
                print(f"      📞 Phone: {place.get('phoneNumber', 'N/A')}")
                print(f"      🏢 Address: {place.get('address', 'N/A')}")
                print(f"      🌐 Website: {place.get('website', 'N/A')}")

            return result

        except Exception as e:
            print(f"    ❌ Places search failed: {e}")
            return {"error": str(e), "places": []}

    def search_news(self, company_name: str, time_range: str = "month") -> Dict[str, Any]:
        """
        Search for recent news about the company
        time_range: "day", "week", "month", "year"
        """
        print(f"  📰 News Search: {company_name} (last {time_range})")

        try:
            conn = http.client.HTTPSConnection("google.serper.dev")
            payload = json.dumps({
                "q": company_name,
                "gl": "au",
                "tbm": "nws",  # News search
                "tbs": f"qdr:{time_range[0]}"  # Time filter
            })
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }

            conn.request("POST", "/news", payload, headers)
            res = conn.getresponse()
            data = res.read()

            result = json.loads(data.decode("utf-8"))
            credits = result.get('credits', 1)
            self.total_credits_used += credits

            news = result.get('news', [])
            print(f"    ✅ Found {len(news)} news items (Credits: {credits})")

            # Show first few headlines
            for item in news[:3]:
                print(f"      📄 {item.get('title', 'N/A')[:60]}...")

            return result

        except Exception as e:
            print(f"    ❌ News search failed: {e}")
            return {"error": str(e), "news": []}

    def scrape_website(self, url: str) -> Dict[str, Any]:
        """
        Scrape website content using Serper's scraping endpoint
        Attempts to extract contact information
        """
        print(f"  🌐 Scraping Website: {url}")

        try:
            conn = http.client.HTTPSConnection("scrape.serper.dev")
            payload = json.dumps({
                "url": url
            })
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }

            conn.request("POST", "/", payload, headers)
            res = conn.getresponse()
            data = res.read()

            result = json.loads(data.decode("utf-8"))
            credits = result.get('credits', 2)
            self.total_credits_used += credits

            # Extract text content
            text = result.get('text', '')
            metadata = result.get('metadata', {})

            print(f"    ✅ Scraped {len(text)} characters (Credits: {credits})")

            # Try to extract contact info from scraped content
            contacts = self.extract_contact_info(text, metadata)

            return {
                "url": url,
                "text": text[:1000],  # First 1000 chars for display
                "metadata": metadata,
                "contacts": contacts,
                "credits": credits
            }

        except Exception as e:
            print(f"    ❌ Website scraping failed: {e}")
            return {"error": str(e), "url": url, "contacts": {}}

    def extract_contact_info(self, text: str, metadata: Dict = None) -> Dict[str, Any]:
        """
        Extract contact information from scraped text
        """
        contacts = {
            "emails": [],
            "phones": [],
            "names": [],
            "addresses": []
        }

        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contacts["emails"] = list(set(emails))  # Remove duplicates

        # Phone extraction (Australian format)
        phone_patterns = [
            r'\+61\s?\d{1,2}\s?\d{4}\s?\d{4}',  # +61 2 9999 9999
            r'\(0\d\)\s?\d{4}\s?\d{4}',  # (02) 9999 9999
            r'0\d\s?\d{4}\s?\d{4}',  # 02 9999 9999
            r'1[38]00\s?\d{3}\s?\d{3}',  # 1300/1800 numbers
        ]

        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            contacts["phones"].extend(phones)
        contacts["phones"] = list(set(contacts["phones"]))

        # Name extraction (look for contact patterns)
        name_patterns = [
            r'Contact:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'CEO:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'Director:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'Manager:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'For (?:more )?information[,:]?\s*(?:contact\s*)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]

        for pattern in name_patterns:
            names = re.findall(pattern, text)
            contacts["names"].extend(names)
        contacts["names"] = list(set(contacts["names"]))

        # Address extraction (basic)
        address_pattern = r'\d+\s+[A-Za-z\s]+(?:Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Place|Pl|Boulevard|Blvd)[,\s]+[A-Za-z\s]+(?:NSW|VIC|QLD|WA|SA|TAS|ACT|NT)'
        addresses = re.findall(address_pattern, text)
        contacts["addresses"] = list(set(addresses))

        # Summary
        if contacts["emails"] or contacts["phones"]:
            print(f"      📧 Emails found: {len(contacts['emails'])}")
            print(f"      📞 Phones found: {len(contacts['phones'])}")
            print(f"      👤 Names found: {len(contacts['names'])}")
            print(f"      📍 Addresses found: {len(contacts['addresses'])}")
        else:
            print(f"      ⚠️ No contact information extracted")

        return contacts

    async def validate_company(self, company: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single company using all Serper endpoints
        """
        company_name = company.get('name', 'Unknown')
        print(f"\n🏢 Validating: {company_name}")
        print("-" * 60)

        validation_result = {
            "company_name": company_name,
            "company_data": company,
            "timestamp": datetime.now().isoformat(),
            "serper_results": {},
            "extracted_contacts": {},
            "validation_status": "pending"
        }

        # 1. Web Search - general search and contact search
        print("\n1️⃣ Web Search:")
        web_results = self.search_web(f"{company_name} Australia")
        validation_result["serper_results"]["web_search"] = web_results

        # Try specific contact search
        contact_search = self.search_web(f"{company_name} contact email phone Australia")
        validation_result["serper_results"]["contact_search"] = contact_search

        # 2. Places Search
        print("\n2️⃣ Places Search:")
        # Try to get location from company data
        location = "Sydney"  # Default
        if company.get('headquarters'):
            hq = company['headquarters']
            if isinstance(hq, dict):
                location = hq.get('city', 'Sydney')
        elif company.get('office_locations'):
            locations = company['office_locations']
            if locations and isinstance(locations, list):
                location = locations[0] if isinstance(locations[0], str) else 'Sydney'

        places_results = self.search_places(company_name, location)
        validation_result["serper_results"]["places"] = places_results

        # 3. News Search
        print("\n3️⃣ News Search:")
        news_results = self.search_news(company_name)
        validation_result["serper_results"]["news"] = news_results

        # 4. Website Scraping
        print("\n4️⃣ Website Scraping:")
        website_url = None

        # Try to get website from places result
        if places_results.get('places'):
            place = places_results['places'][0]
            website_url = place.get('website')

        # Or from web search
        if not website_url and web_results.get('organic'):
            for result in web_results['organic'][:3]:
                link = result.get('link', '')
                # Look for official website (not social media, directories, etc.)
                if company_name.lower().replace(' ', '') in link.lower().replace('-', '').replace('_', ''):
                    website_url = link
                    break

        if website_url:
            scrape_results = self.scrape_website(website_url)
            validation_result["serper_results"]["website_scrape"] = scrape_results
            validation_result["extracted_contacts"] = scrape_results.get('contacts', {})
        else:
            print("    ⚠️ No website found to scrape")

        # 5. Compile all contact information
        print("\n5️⃣ Compiling Contact Information:")
        all_contacts = {
            "phones": [],
            "emails": [],
            "addresses": [],
            "names": [],
            "website": website_url
        }

        # From places
        if places_results.get('places'):
            place = places_results['places'][0]
            if place.get('phoneNumber'):
                all_contacts['phones'].append(place['phoneNumber'])
            if place.get('address'):
                all_contacts['addresses'].append(place['address'])

        # From website scrape
        if validation_result.get('extracted_contacts'):
            contacts = validation_result['extracted_contacts']
            all_contacts['emails'].extend(contacts.get('emails', []))
            all_contacts['phones'].extend(contacts.get('phones', []))
            all_contacts['names'].extend(contacts.get('names', []))
            all_contacts['addresses'].extend(contacts.get('addresses', []))

        # Remove duplicates
        for key in ['phones', 'emails', 'addresses', 'names']:
            all_contacts[key] = list(set(all_contacts[key]))

        validation_result["compiled_contacts"] = all_contacts

        # Print summary
        print(f"\n  📊 Summary for {company_name}:")
        print(f"    📞 Phones: {', '.join(all_contacts['phones'][:3]) if all_contacts['phones'] else 'None found'}")
        print(f"    📧 Emails: {', '.join(all_contacts['emails'][:3]) if all_contacts['emails'] else 'None found'}")
        print(f"    👤 Names: {', '.join(all_contacts['names'][:3]) if all_contacts['names'] else 'None found'}")
        print(f"    📍 Addresses: {all_contacts['addresses'][0] if all_contacts['addresses'] else 'None found'}")
        print(f"    🌐 Website: {website_url or 'None found'}")

        # Determine validation status
        if all_contacts['phones'] or all_contacts['emails']:
            validation_result["validation_status"] = "verified"
        elif website_url:
            validation_result["validation_status"] = "partial"
        else:
            validation_result["validation_status"] = "unverified"

        return validation_result

    async def get_rmh_companies(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get RMH Tier A companies using the search agent
        """
        print("\n" + "=" * 80)
        print("GETTING RMH TIER A COMPANIES")
        print("=" * 80)

        if ICP_AVAILABLE:
            # Use ICP manager
            icp_manager = ICPManager()
            profile = icp_manager.get_profile("rmh_sydney")
            criteria = profile.tiers.get("A")
        else:
            # Manual criteria for RMH Tier A
            criteria = SearchCriteria(
                location=LocationCriteria(
                    countries=["Australia"],
                    cities=["Sydney"],
                    regions=["Greater Western Sydney"]
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
                excluded_industries=["Fast Food", "Gambling"],
                excluded_companies=["McDonald's", "KFC", "Burger King"]
            )

        # Get companies from search agent
        agent = EnhancedSearchStrategistAgent(deployment_name="gpt-4.1")

        print(f"Searching for {count} RMH Tier A companies...")
        result = await agent.generate_enhanced_strategy(criteria, target_count=count)

        companies = result.get('companies', [])
        print(f"✅ Found {len(companies)} companies")

        # Convert to dict if needed
        company_dicts = []
        for company in companies[:count]:
            if hasattr(company, 'dict'):
                company_dicts.append(company.dict())
            else:
                company_dicts.append(company)

        return company_dicts

    async def run_comprehensive_test(self):
        """
        Run comprehensive validation test
        """
        print("=" * 80)
        print("SERPER API COMPREHENSIVE VALIDATION TEST")
        print(f"Started: {datetime.now()}")
        print("=" * 80)

        # Get RMH companies
        companies = await self.get_rmh_companies(10)

        if not companies:
            print("❌ No companies found to validate")
            return

        # Validate each company
        print("\n" + "=" * 80)
        print("VALIDATING COMPANIES")
        print("=" * 80)

        validation_results = []

        for i, company in enumerate(companies, 1):
            print(f"\n[{i}/{len(companies)}] " + "=" * 60)

            try:
                result = await self.validate_company(company)
                validation_results.append(result)

                # Rate limiting
                if i < len(companies):
                    print("\n⏳ Waiting 2 seconds before next company...")
                    await asyncio.sleep(2)

            except Exception as e:
                print(f"❌ Error validating company: {e}")
                traceback.print_exc()
                validation_results.append({
                    "company_name": company.get('name', 'Unknown'),
                    "error": str(e),
                    "validation_status": "error"
                })

        # Generate report
        self.generate_report(validation_results)

        return validation_results

    def generate_report(self, validation_results: List[Dict[str, Any]]):
        """
        Generate summary report of validation results
        """
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)

        # Overall statistics
        total = len(validation_results)
        verified = sum(1 for r in validation_results if r.get('validation_status') == 'verified')
        partial = sum(1 for r in validation_results if r.get('validation_status') == 'partial')
        unverified = sum(1 for r in validation_results if r.get('validation_status') == 'unverified')
        errors = sum(1 for r in validation_results if r.get('validation_status') == 'error')

        print(f"\n📊 Overall Statistics:")
        print(f"  Total Companies: {total}")
        print(f"  ✅ Verified: {verified} ({verified / total * 100:.1f}%)")
        print(f"  ⚠️ Partial: {partial} ({partial / total * 100:.1f}%)")
        print(f"  ❌ Unverified: {unverified} ({unverified / total * 100:.1f}%)")
        print(f"  🚫 Errors: {errors}")
        print(f"  💳 Total Credits Used: {self.total_credits_used}")

        # Contact extraction statistics
        total_phones = 0
        total_emails = 0
        total_names = 0
        total_websites = 0

        for result in validation_results:
            contacts = result.get('compiled_contacts', {})
            if contacts.get('phones'):
                total_phones += len(contacts['phones'])
            if contacts.get('emails'):
                total_emails += len(contacts['emails'])
            if contacts.get('names'):
                total_names += len(contacts['names'])
            if contacts.get('website'):
                total_websites += 1

        print(f"\n📋 Contact Information Extracted:")
        print(f"  📞 Total Phone Numbers: {total_phones}")
        print(f"  📧 Total Email Addresses: {total_emails}")
        print(f"  👤 Total Contact Names: {total_names}")
        print(f"  🌐 Total Websites: {total_websites}")

        # Detailed results
        print(f"\n📝 Detailed Results:")
        print("-" * 80)

        for result in validation_results:
            company_name = result.get('company_name', 'Unknown')
            status = result.get('validation_status', 'unknown')
            contacts = result.get('compiled_contacts', {})

            status_emoji = {
                'verified': '✅',
                'partial': '⚠️',
                'unverified': '❌',
                'error': '🚫'
            }.get(status, '❓')

            print(f"\n{status_emoji} {company_name}")
            print(f"  Status: {status}")

            if contacts:
                if contacts.get('phones'):
                    print(f"  📞 Phone: {contacts['phones'][0]}")
                if contacts.get('emails'):
                    print(f"  📧 Email: {contacts['emails'][0]}")
                if contacts.get('names'):
                    print(f"  👤 Contact: {contacts['names'][0]}")
                if contacts.get('addresses'):
                    print(f"  📍 Address: {contacts['addresses'][0][:50]}...")
                if contacts.get('website'):
                    print(f"  🌐 Website: {contacts['website']}")

            if result.get('error'):
                print(f"  ⚠️ Error: {result['error']}")

        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'serper_validation_results_{timestamp}.json'

        with open(filename, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)

        print(f"\n💾 Full results saved to: {filename}")

        # Recommendations
        print(f"\n💡 Recommendations:")
        if verified < total * 0.5:
            print("  - Low verification rate. Consider:")
            print("    • Improving company name matching in searches")
            print("    • Adding more search variations")
            print("    • Using additional data sources")

        if total_emails == 0:
            print("  - No emails extracted. Website scraping may need:")
            print("    • Better email pattern matching")
            print("    • Searching contact/about pages specifically")
            print("    • Using dedicated email finder services")

        if total_names == 0:
            print("  - No contact names found. Consider:")
            print("    • Searching LinkedIn for company employees")
            print("    • Looking for leadership/team pages")
            print("    • Using business directories")


async def main():
    """
    Main test runner
    """
    # Your Serper API key
    SERPER_API_KEY = "99c44b79892f5f7499accf2d7c26d93313880937"  # Replace with your actual key

    # Create tester
    tester = SerperAPITester(SERPER_API_KEY)

    # Run comprehensive test
    results = await tester.run_comprehensive_test()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    asyncio.run(main())