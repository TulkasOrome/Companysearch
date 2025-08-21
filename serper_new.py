#!/usr/bin/env python3
"""
email_contact_extraction_test.py - Focused test for extracting emails and contact names
Tests multiple mechanisms and strategies for finding contact information
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
from urllib.parse import urlparse, urljoin

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


class EmailContactExtractor:
    """Advanced email and contact extraction using multiple methods"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.results = []
        self.total_credits_used = 0

        # Enhanced patterns for extraction
        self.email_patterns = [
            # Standard email
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            # Email with quotes
            r'"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})"',
            # Email in mailto links
            r'mailto:([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
            # Obfuscated emails
            r'\b[A-Za-z0-9._%+-]+\s*\[\s*at\s*\]\s*[A-Za-z0-9.-]+\s*\[\s*dot\s*\]\s*[A-Z|a-z]{2,}\b',
            r'\b[A-Za-z0-9._%+-]+\s*\(\s*at\s*\)\s*[A-Za-z0-9.-]+\s*\(\s*dot\s*\)\s*[A-Z|a-z]{2,}\b',
        ]

        self.name_patterns = [
            # Executive titles
            r'(?:CEO|Chief Executive Officer|Managing Director|MD|Director|Manager|President|Founder|Owner|Principal|Partner)(?:\s*[:\-–])?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),?\s*(?:CEO|Chief Executive Officer|Managing Director|MD|Director|Manager|President|Founder|Owner)',

            # Contact patterns
            r'Contact(?:\s*[:\-–])?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'For (?:more )?(?:information|enquiries|queries|questions)(?:\s*[,:\-–])?\s*(?:contact|speak to|call|email)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'(?:Sales|Marketing|Business Development|Customer Service)(?:\s*[:\-–])?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',

            # Team/About us patterns
            r'(?:Meet|Our Team|Leadership|Management|Board|Executive).*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*(?:leads|manages|oversees|heads)',

            # Email context
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*<[^>]+@[^>]+>',
            r'Email\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*at',

            # LinkedIn style
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\|\s*(?:CEO|Director|Manager|Founder)',
        ]

    def serper_search(self, query: str, endpoint: str = "search") -> Dict[str, Any]:
        """Generic Serper API call"""
        try:
            conn = http.client.HTTPSConnection("google.serper.dev")
            payload = json.dumps({
                "q": query,
                "gl": "au",
                "num": 10
            })
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }

            conn.request("POST", f"/{endpoint}", payload, headers)
            res = conn.getresponse()
            data = res.read()

            result = json.loads(data.decode("utf-8"))
            self.total_credits_used += result.get('credits', 1)
            return result

        except Exception as e:
            print(f"    ❌ Search failed: {e}")
            return {"error": str(e)}

    def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a specific URL"""
        try:
            conn = http.client.HTTPSConnection("scrape.serper.dev")
            payload = json.dumps({"url": url})
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }

            conn.request("POST", "/", payload, headers)
            res = conn.getresponse()
            data = res.read()

            result = json.loads(data.decode("utf-8"))
            self.total_credits_used += result.get('credits', 2)
            return result

        except Exception as e:
            return {"error": str(e), "text": "", "metadata": {}}

    def extract_emails_from_text(self, text: str) -> List[str]:
        """Extract emails using multiple patterns"""
        emails = set()

        for pattern in self.email_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                # Clean up obfuscated emails
                match = match.replace('[at]', '@').replace('(at)', '@')
                match = match.replace('[dot]', '.').replace('(dot)', '.')
                match = match.replace(' ', '')
                # Validate email format
                if '@' in match and '.' in match.split('@')[1]:
                    emails.add(match.lower())

        # Filter out common non-personal emails
        filtered_emails = []
        exclude_patterns = ['example.com', 'sentry.io', 'wordpress', 'w3.org', 'schema.org']
        for email in emails:
            if not any(pattern in email for pattern in exclude_patterns):
                filtered_emails.append(email)

        return filtered_emails

    def extract_names_from_text(self, text: str) -> List[str]:
        """Extract contact names using multiple patterns"""
        names = set()

        for pattern in self.name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                # Clean up the name
                name = match.strip()
                # Validate it's a real name (2-4 parts, each starting with capital)
                parts = name.split()
                if 2 <= len(parts) <= 4:
                    if all(part[0].isupper() for part in parts if part):
                        # Filter out common false positives
                        if not any(word in name.lower() for word in
                                   ['contact us', 'click here', 'read more', 'learn more',
                                    'find out', 'get started', 'our team', 'the team']):
                            names.add(name)

        return list(names)

    async def method1_targeted_searches(self, company_name: str) -> Dict[str, Any]:
        """Method 1: Targeted search queries for emails and contacts"""
        print("\n  🔍 Method 1: Targeted Search Queries")
        results = {"emails": [], "names": [], "sources": []}

        # Search queries targeting contact info
        queries = [
            f'"{company_name}" email @',
            f'"{company_name}" contact email',
            f'site:linkedin.com "{company_name}" CEO OR Director OR Manager',
            f'"{company_name}" "contact us" email',
            f'"{company_name}" management team executives',
            f'"{company_name}" @*.com.au OR @*.com',
        ]

        for query in queries:
            print(f"    Searching: {query[:50]}...")
            search_result = self.serper_search(query)

            if 'organic' in search_result:
                for result in search_result['organic']:
                    # Check title and snippet for emails and names
                    text = f"{result.get('title', '')} {result.get('snippet', '')}"

                    # Extract emails
                    found_emails = self.extract_emails_from_text(text)
                    results['emails'].extend(found_emails)

                    # Extract names
                    found_names = self.extract_names_from_text(text)
                    results['names'].extend(found_names)

                    if found_emails or found_names:
                        results['sources'].append(result.get('link', 'Unknown'))

        # Deduplicate
        results['emails'] = list(set(results['emails']))
        results['names'] = list(set(results['names']))

        print(f"    Found: {len(results['emails'])} emails, {len(results['names'])} names")
        return results

    async def method2_page_specific_scraping(self, company_name: str, base_url: str = None) -> Dict[str, Any]:
        """Method 2: Scrape specific pages (contact, about, team)"""
        print("\n  🌐 Method 2: Page-Specific Scraping")
        results = {"emails": [], "names": [], "pages_scraped": []}

        if not base_url:
            # First, find the company website
            search = self.serper_search(f'"{company_name}" official website')
            if 'organic' in search and search['organic']:
                base_url = search['organic'][0].get('link')

        if not base_url:
            print("    ❌ No website found")
            return results

        # Parse base URL
        parsed = urlparse(base_url)
        base_domain = f"{parsed.scheme}://{parsed.netloc}"

        # Target pages to scrape
        target_pages = [
            '',  # Homepage
            '/contact',
            '/contact-us',
            '/contactus',
            '/about',
            '/about-us',
            '/aboutus',
            '/team',
            '/our-team',
            '/leadership',
            '/management',
            '/people',
            '/staff',
            '/get-in-touch',
            '/enquiries',
            '/enquiry',
        ]

        pages_to_scrape = []
        for page in target_pages:
            pages_to_scrape.append(urljoin(base_domain, page))

        # Also try to find these pages via search
        page_search = self.serper_search(f'site:{parsed.netloc} contact OR about OR team')
        if 'organic' in page_search:
            for result in page_search['organic'][:5]:
                link = result.get('link', '')
                if link and link not in pages_to_scrape:
                    pages_to_scrape.append(link)

        # Scrape each page (limit to 5 to control credits)
        for url in pages_to_scrape[:5]:
            print(f"    Scraping: {url}")
            scraped = self.scrape_url(url)

            if 'text' in scraped:
                text = scraped['text']
                results['pages_scraped'].append(url)

                # Extract emails
                found_emails = self.extract_emails_from_text(text)
                results['emails'].extend(found_emails)

                # Extract names
                found_names = self.extract_names_from_text(text)
                results['names'].extend(found_names)

                # Also check metadata
                metadata = scraped.get('metadata', {})
                meta_text = json.dumps(metadata)
                found_emails = self.extract_emails_from_text(meta_text)
                results['emails'].extend(found_emails)

        # Deduplicate
        results['emails'] = list(set(results['emails']))
        results['names'] = list(set(results['names']))

        print(f"    Scraped {len(results['pages_scraped'])} pages")
        print(f"    Found: {len(results['emails'])} emails, {len(results['names'])} names")
        return results

    async def method3_social_media_search(self, company_name: str) -> Dict[str, Any]:
        """Method 3: Search social media profiles for contact info"""
        print("\n  📱 Method 3: Social Media Search")
        results = {"emails": [], "names": [], "profiles": []}

        # Search social media platforms
        platforms = [
            ('linkedin.com', 'CEO OR Director OR Founder OR Manager'),
            ('facebook.com', 'contact OR email OR message'),
            ('twitter.com', 'contact OR email'),
            ('instagram.com', 'contact OR email'),
        ]

        for platform, keywords in platforms:
            query = f'site:{platform} "{company_name}" {keywords}'
            print(f"    Searching {platform}...")

            search_result = self.serper_search(query)

            if 'organic' in search_result:
                for result in search_result['organic'][:3]:  # Limit to top 3
                    text = f"{result.get('title', '')} {result.get('snippet', '')}"
                    link = result.get('link', '')

                    # Extract emails
                    found_emails = self.extract_emails_from_text(text)
                    results['emails'].extend(found_emails)

                    # Extract names (especially from LinkedIn)
                    if 'linkedin.com' in link:
                        # LinkedIn titles often have the person's name
                        title = result.get('title', '')
                        # Pattern: "Name - Title - Company | LinkedIn"
                        name_match = re.match(r'^([^-|]+)\s*[-|]', title)
                        if name_match:
                            name = name_match.group(1).strip()
                            if len(name.split()) >= 2:
                                results['names'].append(name)

                    # Regular name extraction
                    found_names = self.extract_names_from_text(text)
                    results['names'].extend(found_names)

                    if found_emails or found_names:
                        results['profiles'].append(link)

        # Deduplicate
        results['emails'] = list(set(results['emails']))
        results['names'] = list(set(results['names']))

        print(f"    Found: {len(results['emails'])} emails, {len(results['names'])} names")
        return results

    async def method4_news_articles(self, company_name: str) -> Dict[str, Any]:
        """Method 4: Extract from news articles and press releases"""
        print("\n  📰 Method 4: News Articles & Press Releases")
        results = {"emails": [], "names": [], "articles": []}

        # Search for news and press releases
        queries = [
            f'"{company_name}" announces appoints CEO president',
            f'"{company_name}" press release contact',
            f'"{company_name}" media enquiries contact',
            f'"{company_name}" spokesman spokesperson said',
        ]

        for query in queries:
            print(f"    Searching: {query[:50]}...")
            search_result = self.serper_search(query)

            if 'organic' in search_result:
                for result in search_result['organic'][:5]:
                    text = f"{result.get('title', '')} {result.get('snippet', '')}"

                    # Extract emails
                    found_emails = self.extract_emails_from_text(text)
                    results['emails'].extend(found_emails)

                    # Extract names with quotes (common in news)
                    quote_patterns = [
                        r'"[^"]+"\s*(?:said|says|stated|explained|announced)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*(?:said|says|stated|explained|announced)\s*[,:]?\s*"',
                        r'According to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                    ]

                    for pattern in quote_patterns:
                        matches = re.findall(pattern, text)
                        results['names'].extend(matches)

                    # Regular name extraction
                    found_names = self.extract_names_from_text(text)
                    results['names'].extend(found_names)

                    if found_emails or found_names or matches:
                        results['articles'].append(result.get('link', 'Unknown'))

        # Also try news endpoint
        news_result = self.serper_search(company_name, "news")
        if 'news' in news_result:
            for article in news_result['news'][:5]:
                text = f"{article.get('title', '')} {article.get('snippet', '')}"

                # Extract names from news
                found_names = self.extract_names_from_text(text)
                results['names'].extend(found_names)

        # Deduplicate
        results['emails'] = list(set(results['emails']))
        results['names'] = list(set(results['names']))

        print(f"    Found: {len(results['emails'])} emails, {len(results['names'])} names")
        return results

    async def method5_directory_listings(self, company_name: str) -> Dict[str, Any]:
        """Method 5: Search business directories"""
        print("\n  📚 Method 5: Business Directory Search")
        results = {"emails": [], "names": [], "directories": []}

        # Business directories to search
        directories = [
            'yellowpages.com.au',
            'whitepages.com.au',
            'truelocal.com.au',
            'hotfrog.com.au',
            'startlocal.com.au',
            'yelp.com.au',
            'abn.business.gov.au',
            'dnb.com',  # Dun & Bradstreet
            'kompass.com',
            'australianbusinessdirectory.com.au',
        ]

        for directory in directories:
            query = f'site:{directory} "{company_name}"'
            print(f"    Searching {directory}...")

            search_result = self.serper_search(query)

            if 'organic' in search_result:
                for result in search_result['organic'][:2]:  # Limit per directory
                    text = f"{result.get('title', '')} {result.get('snippet', '')}"

                    # Extract emails
                    found_emails = self.extract_emails_from_text(text)
                    results['emails'].extend(found_emails)

                    # Extract names
                    found_names = self.extract_names_from_text(text)
                    results['names'].extend(found_names)

                    if found_emails or found_names:
                        results['directories'].append(result.get('link', 'Unknown'))

        # Deduplicate
        results['emails'] = list(set(results['emails']))
        results['names'] = list(set(results['names']))

        print(f"    Found: {len(results['emails'])} emails, {len(results['names'])} names")
        return results

    async def extract_all_contacts(self, company_name: str, website_url: str = None) -> Dict[str, Any]:
        """Run all extraction methods for a company"""
        print(f"\n{'=' * 60}")
        print(f"🏢 Extracting contacts for: {company_name}")
        print(f"{'=' * 60}")

        all_results = {
            "company_name": company_name,
            "timestamp": datetime.now().isoformat(),
            "emails": [],
            "names": [],
            "methods_used": [],
            "credits_used": 0
        }

        start_credits = self.total_credits_used

        # Run all methods
        method1 = await self.method1_targeted_searches(company_name)
        all_results['emails'].extend(method1['emails'])
        all_results['names'].extend(method1['names'])
        all_results['methods_used'].append({
            "method": "Targeted Searches",
            "emails_found": len(method1['emails']),
            "names_found": len(method1['names'])
        })

        method2 = await self.method2_page_specific_scraping(company_name, website_url)
        all_results['emails'].extend(method2['emails'])
        all_results['names'].extend(method2['names'])
        all_results['methods_used'].append({
            "method": "Page-Specific Scraping",
            "emails_found": len(method2['emails']),
            "names_found": len(method2['names']),
            "pages_scraped": len(method2.get('pages_scraped', []))
        })

        method3 = await self.method3_social_media_search(company_name)
        all_results['emails'].extend(method3['emails'])
        all_results['names'].extend(method3['names'])
        all_results['methods_used'].append({
            "method": "Social Media Search",
            "emails_found": len(method3['emails']),
            "names_found": len(method3['names'])
        })

        method4 = await self.method4_news_articles(company_name)
        all_results['emails'].extend(method4['emails'])
        all_results['names'].extend(method4['names'])
        all_results['methods_used'].append({
            "method": "News Articles",
            "emails_found": len(method4['emails']),
            "names_found": len(method4['names'])
        })

        method5 = await self.method5_directory_listings(company_name)
        all_results['emails'].extend(method5['emails'])
        all_results['names'].extend(method5['names'])
        all_results['methods_used'].append({
            "method": "Directory Listings",
            "emails_found": len(method5['emails']),
            "names_found": len(method5['names'])
        })

        # Deduplicate final results
        all_results['emails'] = list(set(all_results['emails']))
        all_results['names'] = list(set(all_results['names']))
        all_results['credits_used'] = self.total_credits_used - start_credits

        # Summary
        print(f"\n  📊 FINAL RESULTS for {company_name}:")
        print(f"    ✉️ Unique Emails: {len(all_results['emails'])}")
        if all_results['emails']:
            for email in all_results['emails'][:5]:  # Show first 5
                print(f"       • {email}")

        print(f"    👤 Unique Names: {len(all_results['names'])}")
        if all_results['names']:
            for name in all_results['names'][:5]:  # Show first 5
                print(f"       • {name}")

        print(f"    💳 Credits Used: {all_results['credits_used']}")

        return all_results

    async def test_companies(self, companies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Test contact extraction for multiple companies"""
        results = []

        for i, company in enumerate(companies, 1):
            print(f"\n[{i}/{len(companies)}] Testing company...")

            company_name = company.get('name', 'Unknown')
            website = None

            # Try to get website from company data
            if company.get('headquarters') and isinstance(company['headquarters'], dict):
                website = company['headquarters'].get('website')

            result = await self.extract_all_contacts(company_name, website)
            results.append(result)

            # Rate limiting
            if i < len(companies):
                print("\n⏳ Waiting 3 seconds before next company...")
                await asyncio.sleep(3)

        return results

    async def get_test_companies(self) -> List[Dict[str, Any]]:
        """Get a mix of test companies"""
        # Use some known Australian companies for testing
        test_companies = [
            {"name": "Richard Crookes Constructions", "headquarters": {"city": "Sydney"}},
            {"name": "Merivale", "headquarters": {"city": "Sydney"}},
            {"name": "Dyldam Developments", "headquarters": {"city": "Parramatta"}},
            {"name": "Coles Group", "headquarters": {"city": "Melbourne"}},
            {"name": "Woolworths Group", "headquarters": {"city": "Sydney"}},
        ]

        return test_companies

    def generate_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive report"""
        print("\n" + "=" * 80)
        print("EMAIL & CONTACT EXTRACTION REPORT")
        print("=" * 80)

        # Overall statistics
        total_companies = len(results)
        companies_with_emails = sum(1 for r in results if r['emails'])
        companies_with_names = sum(1 for r in results if r['names'])
        total_emails = sum(len(r['emails']) for r in results)
        total_names = sum(len(r['names']) for r in results)
        total_credits = sum(r['credits_used'] for r in results)

        print(f"\n📊 Overall Statistics:")
        print(f"  Companies Tested: {total_companies}")
        print(
            f"  Companies with Emails: {companies_with_emails} ({companies_with_emails / total_companies * 100:.1f}%)")
        print(f"  Companies with Names: {companies_with_names} ({companies_with_names / total_companies * 100:.1f}%)")
        print(f"  Total Unique Emails: {total_emails}")
        print(f"  Total Unique Names: {total_names}")
        print(f"  Total Credits Used: {total_credits}")
        print(f"  Avg Credits per Company: {total_credits / total_companies:.1f}")

        # Method effectiveness
        print(f"\n📈 Method Effectiveness:")
        method_stats = {}

        for result in results:
            for method in result['methods_used']:
                method_name = method['method']
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        'emails_total': 0,
                        'names_total': 0,
                        'companies_successful': 0
                    }

                method_stats[method_name]['emails_total'] += method['emails_found']
                method_stats[method_name]['names_total'] += method['names_found']
                if method['emails_found'] > 0 or method['names_found'] > 0:
                    method_stats[method_name]['companies_successful'] += 1

        for method, stats in method_stats.items():
            print(f"\n  {method}:")
            print(f"    Emails Found: {stats['emails_total']}")
            print(f"    Names Found: {stats['names_total']}")
            print(f"    Success Rate: {stats['companies_successful']}/{total_companies} companies")

        # Detailed results
        print(f"\n📝 Detailed Results by Company:")
        print("-" * 80)

        for result in results:
            print(f"\n  {result['company_name']}:")
            print(f"    Emails: {len(result['emails'])}")
            if result['emails']:
                for email in result['emails'][:3]:
                    print(f"      • {email}")
            print(f"    Names: {len(result['names'])}")
            if result['names']:
                for name in result['names'][:3]:
                    print(f"      • {name}")
            print(f"    Credits: {result['credits_used']}")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'email_contact_extraction_results_{timestamp}.json'

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Full results saved to: {filename}")

        # Recommendations
        print(f"\n💡 Recommendations:")
        if companies_with_emails < total_companies * 0.5:
            print("  - Email extraction rate is low. Consider:")
            print("    • Using email finder APIs (Hunter.io, Clearbit, etc.)")
            print("    • Manual verification for high-value prospects")
            print("    • Purchasing business contact databases")

        if companies_with_names < total_companies * 0.7:
            print("  - Name extraction rate could be improved. Consider:")
            print("    • LinkedIn Sales Navigator for executive names")
            print("    • Industry directories and associations")
            print("    • Company annual reports and ASX filings")


async def main():
    """Main test runner"""
    # Your Serper API key
    SERPER_API_KEY = "99c44b79892f5f7499accf2d7c26d93313880937"  # Replace with your actual key

    print("=" * 80)
    print("EMAIL & CONTACT EXTRACTION TEST")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    # Create extractor
    extractor = EmailContactExtractor(SERPER_API_KEY)

    # Get test companies
    print("\n📋 Getting test companies...")
    companies = await extractor.get_test_companies()
    print(f"Testing {len(companies)} companies")

    # Run extraction
    results = await extractor.test_companies(companies)

    # Generate report
    extractor.generate_report(results)

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    asyncio.run(main())