# serper_validation_integration.py
"""
Serper validation integration for Streamlit app
Provides actual validation using Serper API instead of simulated data
"""

import asyncio
import json
import http.client
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import time


class SerperValidator:
    """Actual Serper validation implementation"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.total_credits_used = 0

        # Email patterns
        self.email_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})"',
            r'mailto:([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
        ]

        # Name patterns
        self.name_patterns = [
            r'(?:CEO|Chief Executive Officer|Managing Director|MD|Director|Manager|President|Founder|Owner|Principal|Partner)(?:\s*[:\-–])?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+),?\s*(?:CEO|Chief Executive Officer|Managing Director|MD|Director|Manager|President|Founder|Owner)',
            r'Contact(?:\s*[:\-–])?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*<[^>]+@[^>]+>',
        ]

    def serper_search(self, query: str, endpoint: str = "search") -> Dict[str, Any]:
        """Make actual Serper API call"""
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
            return {"error": str(e), "organic": [], "places": [], "news": []}

    def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape website using Serper"""
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
        """Extract real emails from text"""
        emails = set()

        for pattern in self.email_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                match = match.replace('[at]', '@').replace('(at)', '@')
                match = match.replace('[dot]', '.').replace('(dot)', '.')
                match = match.replace(' ', '')

                # Validate email format
                if '@' in match and '.' in match.split('@')[1]:
                    # Filter out common non-personal emails
                    exclude = ['example.com', 'sentry.io', 'wordpress', 'w3.org', 'schema.org']
                    if not any(ex in match for ex in exclude):
                        emails.add(match.lower())

        return list(emails)

    def extract_names_from_text(self, text: str) -> List[str]:
        """Extract real contact names from text"""
        names = set()

        for pattern in self.name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                name = match.strip()

                # Validate it's a real name
                parts = name.split()
                if 2 <= len(parts) <= 4:
                    if all(part[0].isupper() for part in parts if part):
                        # Filter out false positives
                        false_positives = ['contact us', 'click here', 'read more',
                                           'learn more', 'find out', 'get started']
                        if not any(fp in name.lower() for fp in false_positives):
                            names.add(name)

        return list(names)

    def extract_phones_from_text(self, text: str) -> List[str]:
        """Extract phone numbers from text"""
        phones = set()

        # Phone patterns (Australian format)
        phone_patterns = [
            r'\+61\s?\d{1,2}\s?\d{4}\s?\d{4}',  # +61 2 9999 9999
            r'\(0\d\)\s?\d{4}\s?\d{4}',  # (02) 9999 9999
            r'0\d\s?\d{4}\s?\d{4}',  # 02 9999 9999
            r'1[38]00\s?\d{3}\s?\d{3}',  # 1300/1800 numbers
        ]

        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            phones.update(matches)

        return list(phones)

    async def validate_simple(self, company_name: str, location: str = "Australia") -> Dict[str, Any]:
        """Simple validation - just check existence (2-3 credits)"""
        # Places search
        places_result = self.serper_search(f"{company_name} {location}", "places")

        result = {
            'company_name': company_name,
            'validation_status': 'unverified',
            'mode': 'Simple Check',
            'credits_used': 2,
            'validation_timestamp': datetime.now().isoformat()
        }

        if places_result.get('places'):
            place = places_result['places'][0]
            place_name = place.get('title', '').lower()

            # Check name match
            if company_name.lower() in place_name or place_name in company_name.lower():
                result['validation_status'] = 'verified'
                result['phone'] = place.get('phoneNumber')
                result['address'] = place.get('address')
                result['website'] = place.get('website')

        return result

    async def validate_contact(self, company_name: str, location: str = "Australia") -> Dict[str, Any]:
        """Smart contact extraction (3-5 credits)"""
        result = {
            'company_name': company_name,
            'validation_status': 'unverified',
            'mode': 'Smart Contact Extraction',
            'credits_used': 0,
            'validation_timestamp': datetime.now().isoformat(),
            'emails': [],
            'phones': [],
            'names': []
        }

        # 1. Search for contact info
        contact_search = self.serper_search(f'"{company_name}" contact email phone')
        result['credits_used'] += 1

        if contact_search.get('organic'):
            for item in contact_search['organic'][:5]:
                text = f"{item.get('title', '')} {item.get('snippet', '')}"

                # Extract emails
                emails = self.extract_emails_from_text(text)
                result['emails'].extend(emails)

                # Extract names
                names = self.extract_names_from_text(text)
                result['names'].extend(names)

                # Extract phones
                phones = self.extract_phones_from_text(text)
                result['phones'].extend(phones)

        # 2. LinkedIn search for executives
        linkedin_search = self.serper_search(f'site:linkedin.com "{company_name}" CEO OR Director')
        result['credits_used'] += 1

        if linkedin_search.get('organic'):
            for item in linkedin_search['organic'][:3]:
                title = item.get('title', '')
                # Extract name from LinkedIn title format
                name_match = re.match(r'^([^-|]+)\s*[-|]', title)
                if name_match:
                    name = name_match.group(1).strip()
                    if len(name.split()) >= 2:
                        result['names'].append(name)

        # 3. Try to find website and scrape
        website_search = self.serper_search(f'"{company_name}" official website')
        result['credits_used'] += 1

        website_url = None
        if website_search.get('organic'):
            for item in website_search['organic'][:3]:
                link = item.get('link', '')
                if company_name.lower().replace(' ', '') in link.lower().replace('-', '').replace('_', ''):
                    website_url = link
                    break

        if website_url:
            # Scrape website
            scraped = self.scrape_url(website_url)
            result['credits_used'] += 2

            if scraped.get('text'):
                text = scraped['text']

                # Extract from scraped content
                emails = self.extract_emails_from_text(text)
                result['emails'].extend(emails)

                names = self.extract_names_from_text(text)
                result['names'].extend(names)

                phones = self.extract_phones_from_text(text)
                result['phones'].extend(phones)

        # Deduplicate
        result['emails'] = list(set(result['emails']))
        result['names'] = list(set(result['names']))
        result['phones'] = list(set(result['phones']))

        # Set first found values for display
        if result['emails']:
            result['email'] = result['emails'][0]
            result['validation_status'] = 'verified'
        if result['phones']:
            result['phone'] = result['phones'][0]
            result['validation_status'] = 'verified'
        if result['names']:
            result['contact_name'] = result['names'][0]

        return result

    async def validate_csr(self, company_name: str, location: str = "Australia") -> Dict[str, Any]:
        """Smart CSR verification (3-5 credits)"""
        result = {
            'company_name': company_name,
            'validation_status': 'unverified',
            'mode': 'Smart CSR Verification',
            'credits_used': 0,
            'validation_timestamp': datetime.now().isoformat(),
            'csr_programs': [],
            'certifications': [],
            'csr_verified': False
        }

        # 1. Search for CSR programs
        csr_search = self.serper_search(f'"{company_name}" CSR "corporate social responsibility" sustainability')
        result['credits_used'] += 1

        csr_keywords = {
            'children': ['children', 'kids', 'youth', 'school', 'education'],
            'community': ['community', 'local', 'neighborhood', 'volunteer'],
            'environment': ['environment', 'sustainability', 'green', 'carbon', 'climate'],
            'health': ['health', 'medical', 'hospital', 'wellness'],
            'diversity': ['diversity', 'inclusion', 'equity', 'women', 'minorities']
        }

        certifications = ['b-corp', 'b corp', 'iso 26000', 'iso 14001', 'carbon neutral']

        if csr_search.get('organic'):
            for item in csr_search['organic']:
                text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()

                # Check for CSR programs
                for area, keywords in csr_keywords.items():
                    if any(kw in text for kw in keywords):
                        if area not in result['csr_programs']:
                            result['csr_programs'].append(area)

                # Check for certifications
                for cert in certifications:
                    if cert in text and cert not in result['certifications']:
                        result['certifications'].append(cert)

        # 2. Search for foundation or giving programs
        foundation_search = self.serper_search(f'"{company_name}" foundation donation charity giving')
        result['credits_used'] += 1

        if foundation_search.get('organic'):
            for item in foundation_search['organic'][:3]:
                text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
                if any(word in text for word in ['foundation', 'charity', 'donation', 'sponsorship']):
                    result['csr_verified'] = True
                    if 'charitable giving' not in result['csr_programs']:
                        result['csr_programs'].append('charitable giving')

        if result['csr_programs'] or result['certifications']:
            result['validation_status'] = 'verified'

        return result

    async def validate_financial(self, company_name: str, location: str = "Australia") -> Dict[str, Any]:
        """Smart financial verification (3-4 credits)"""
        result = {
            'company_name': company_name,
            'validation_status': 'unverified',
            'mode': 'Smart Financial Check',
            'credits_used': 0,
            'validation_timestamp': datetime.now().isoformat(),
            'revenue_verified': False,
            'employee_verified': False
        }

        # 1. Search for financial info
        financial_search = self.serper_search(f'"{company_name}" revenue employees annual report')
        result['credits_used'] += 1

        # Revenue patterns
        revenue_patterns = [
            r'\$?([\d,]+\.?\d*)\s*(million|billion|m|b)\s*(revenue|turnover|sales)',
            r'(revenue|turnover|sales)\s*of\s*\$?([\d,]+\.?\d*)\s*(million|billion|m|b)',
        ]

        # Employee patterns
        employee_patterns = [
            r'([\d,]+)\s*employees',
            r'([\d,]+)\s*staff',
            r'team\s*of\s*([\d,]+)',
        ]

        if financial_search.get('organic'):
            for item in financial_search['organic']:
                text = f"{item.get('title', '')} {item.get('snippet', '')}"

                # Check for revenue
                for pattern in revenue_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        result['revenue_verified'] = True
                        match = matches[0]
                        if isinstance(match, tuple):
                            amount = match[0] if match[0] else match[1]
                            unit = match[1] if match[1] in ['million', 'billion', 'm', 'b'] else match[2]
                        else:
                            amount = match
                            unit = "unknown"
                        result['revenue_range'] = f"${amount} {unit}"
                        break

                # Check for employees
                for pattern in employee_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        result['employee_verified'] = True
                        result['employee_range'] = matches[0] if isinstance(matches[0], str) else matches[0][0]
                        break

        # 2. Check ASX listings
        asx_search = self.serper_search(f'site:asx.com.au "{company_name}"')
        result['credits_used'] += 1

        if asx_search.get('organic'):
            result['asx_listed'] = True
            result['validation_status'] = 'verified'

        if result['revenue_verified'] or result['employee_verified']:
            result['validation_status'] = 'verified'

        return result

    async def validate_full(self, company_name: str, location: str = "Australia") -> Dict[str, Any]:
        """Full validation combining all methods (10-15 credits)"""
        # Run all validations
        simple = await self.validate_simple(company_name, location)
        contact = await self.validate_contact(company_name, location)
        csr = await self.validate_csr(company_name, location)
        financial = await self.validate_financial(company_name, location)

        # Combine results
        result = {
            'company_name': company_name,
            'validation_status': 'unverified',
            'mode': 'Full Validation',
            'credits_used': sum([
                simple.get('credits_used', 0),
                contact.get('credits_used', 0),
                csr.get('credits_used', 0),
                financial.get('credits_used', 0)
            ]),
            'validation_timestamp': datetime.now().isoformat()
        }

        # Merge data
        if contact.get('email'):
            result['email'] = contact['email']
        if contact.get('phone'):
            result['phone'] = contact['phone']
        if contact.get('contact_name'):
            result['contact_name'] = contact['contact_name']

        result['emails'] = contact.get('emails', [])
        result['phones'] = contact.get('phones', [])
        result['names'] = contact.get('names', [])

        result['csr_programs'] = csr.get('csr_programs', [])
        result['certifications'] = csr.get('certifications', [])
        result['csr_verified'] = csr.get('csr_verified', False)

        result['revenue_verified'] = financial.get('revenue_verified', False)
        result['revenue_range'] = financial.get('revenue_range')
        result['employee_verified'] = financial.get('employee_verified', False)
        result['employee_range'] = financial.get('employee_range')

        # Determine overall status
        if any([
            contact.get('validation_status') == 'verified',
            csr.get('validation_status') == 'verified',
            financial.get('validation_status') == 'verified'
        ]):
            result['validation_status'] = 'verified'
        elif simple.get('validation_status') == 'verified':
            result['validation_status'] = 'partial'

        # Add risk signals (search for negative news)
        news_search = self.serper_search(f'"{company_name}" scandal lawsuit controversy', "news")
        result['credits_used'] += 1

        risk_signals = []
        if news_search.get('news'):
            for item in news_search['news'][:3]:
                text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
                if any(word in text for word in ['lawsuit', 'scandal', 'fraud', 'investigation']):
                    risk_signals.append(item.get('title', '')[:100])

        result['risk_signals'] = risk_signals

        return result


async def validate_company_with_serper(
        company: Dict[str, Any],
        validation_mode: str,
        serper_api_key: str
) -> Dict[str, Any]:
    """
    Main function to validate a company using Serper

    Args:
        company: Company data dictionary
        validation_mode: Type of validation to perform
        serper_api_key: Serper API key

    Returns:
        Validation result dictionary
    """
    validator = SerperValidator(serper_api_key)

    # Extract company info
    if hasattr(company, 'dict'):
        company_dict = company.dict()
    elif isinstance(company, dict):
        company_dict = company
    else:
        company_dict = {'name': str(company)}

    company_name = company_dict.get('name', 'Unknown')

    # Determine location
    location = "Australia"  # Default
    if company_dict.get('headquarters'):
        hq = company_dict['headquarters']
        if isinstance(hq, dict):
            location = hq.get('city', 'Australia')
    elif company_dict.get('office_locations'):
        locations = company_dict['office_locations']
        if locations and isinstance(locations, list):
            location = locations[0] if isinstance(locations[0], str) else 'Australia'

    # Run appropriate validation
    if "Simple" in validation_mode:
        result = await validator.validate_simple(company_name, location)
    elif "Contact" in validation_mode:
        result = await validator.validate_contact(company_name, location)
    elif "CSR" in validation_mode:
        result = await validator.validate_csr(company_name, location)
    elif "Financial" in validation_mode:
        result = await validator.validate_financial(company_name, location)
    elif "Full" in validation_mode:
        result = await validator.validate_full(company_name, location)
    else:
        # Default to simple
        result = await validator.validate_simple(company_name, location)

    return result