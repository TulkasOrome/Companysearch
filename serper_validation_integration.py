# serper_validation_integration.py
"""
Fixed Serper validation integration for the Streamlit app
Ensures validation_status is properly set as lowercase 'verified'
"""

import asyncio
import aiohttp
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime


async def validate_company_with_serper(
        company: Dict[str, Any],
        mode: str,
        api_key: str
) -> Dict[str, Any]:
    """
    Validate a company using Serper API

    Args:
        company: Company data dictionary
        mode: Validation mode string
        api_key: Serper API key

    Returns:
        Validation result dictionary with proper status
    """

    company_name = company.get('name', 'Unknown')
    location = company.get('headquarters', {}).get('city', '') if isinstance(company.get('headquarters'), dict) else ''

    # Map mode strings to actual validation types
    mode_map = {
        "Simple Check (2-3 credits)": "simple",
        "Smart Contact Extraction (3-5 credits)": "contact",
        "Smart CSR Verification (3-5 credits)": "csr",
        "Smart Financial Check (3-4 credits)": "financial",
        "Full Validation (10-15 credits)": "full",
        "Raw Endpoint Access": "raw",
        "Custom Configuration": "custom"
    }

    actual_mode = mode_map.get(mode, "simple")

    # Initialize result with default status
    result = {
        'company_name': company_name,
        'validation_status': 'unverified',  # Default to unverified
        'mode': actual_mode,
        'credits_used': 0,
        'validation_timestamp': datetime.now().isoformat()
    }

    try:
        # Create aiohttp session
        async with aiohttp.ClientSession() as session:

            if actual_mode == "simple":
                # Simple validation - just check existence
                queries = [
                    ('places', f"{company_name} {location}")
                ]

            elif actual_mode == "contact":
                # Contact extraction
                queries = [
                    ('search', f'"{company_name}" email contact @'),
                    ('search', f'site:linkedin.com "{company_name}" CEO OR Director'),
                    ('places', f"{company_name} {location}")
                ]

            elif actual_mode == "csr":
                # CSR verification
                queries = [
                    ('search', f'"{company_name} Foundation" OR "{company_name} CSR" sustainability'),
                    ('search', f'"{company_name}" community donation charity sponsorship'),
                    ('search', f'"{company_name}" "B Corp" OR "ISO 26000" OR "carbon neutral"'),
                    ('news', f'"{company_name}" donation charity community')
                ]

            elif actual_mode == "financial":
                # Financial verification
                queries = [
                    ('search', f'"{company_name}" annual revenue employees million'),
                    ('search', f'"{company_name}" ASX listed share price'),
                    ('news', f'"{company_name}" funding expansion acquisition')
                ]

            elif actual_mode == "full":
                # Full validation - combine all
                queries = [
                    ('places', f"{company_name} {location}"),
                    ('search', f'"{company_name}" revenue employees'),
                    ('search', f'"{company_name}" email contact @'),
                    ('search', f'"{company_name}" CSR sustainability community'),
                    ('news', f'"{company_name}"')
                ]

            else:
                # Default to simple
                queries = [('places', f"{company_name}")]

            # Execute queries
            all_results = []
            for endpoint, query in queries:
                serper_result = await make_serper_request(session, api_key, endpoint, query)
                all_results.append((endpoint, serper_result))
                result['credits_used'] += 1

            # Process results based on mode
            if actual_mode == "simple":
                result.update(process_simple_results(all_results, company_name))

            elif actual_mode == "contact":
                result.update(process_contact_results(all_results, company_name))

            elif actual_mode == "csr":
                result.update(process_csr_results(all_results, company_name))

            elif actual_mode == "financial":
                result.update(process_financial_results(all_results, company_name))

            elif actual_mode == "full":
                # Combine all processing
                simple = process_simple_results(all_results, company_name)
                contact = process_contact_results(all_results, company_name)
                csr = process_csr_results(all_results, company_name)
                financial = process_financial_results(all_results, company_name)

                # Merge results and determine best status
                result.update({
                    'location_verified': simple.get('location_verified', False),
                    'emails': contact.get('emails', []),
                    'phones': contact.get('phones', []),
                    'names': contact.get('names', []),
                    'csr_programs': csr.get('csr_programs', []),
                    'certifications': csr.get('certifications', []),
                    'revenue_range': financial.get('revenue_range', ''),
                    'employee_range': financial.get('employee_range', ''),
                    'risk_signals': financial.get('risk_signals', [])
                })

                # Determine overall status - ENSURE IT'S LOWERCASE 'verified'
                if simple.get('validation_status') == 'verified' or contact.get('validation_status') == 'verified':
                    result['validation_status'] = 'verified'
                elif any([simple.get('validation_status') == 'partial', contact.get('validation_status') == 'partial']):
                    result['validation_status'] = 'partial'
                else:
                    result['validation_status'] = 'unverified'

    except Exception as e:
        result['error'] = str(e)
        result['validation_status'] = 'error'

    # ENSURE validation_status is always lowercase and one of the expected values
    if result['validation_status'] not in ['verified', 'partial', 'unverified', 'error']:
        result['validation_status'] = 'unverified'

    return result


async def make_serper_request(
        session: aiohttp.ClientSession,
        api_key: str,
        endpoint: str,
        query: str
) -> Dict[str, Any]:
    """Make a request to Serper API"""

    url = f"https://google.serper.dev/{endpoint}"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    data = {"q": query}

    # Add endpoint-specific parameters
    if endpoint == "news":
        data["time"] = "month"
    elif endpoint == "places":
        data["gl"] = "au"  # Default to Australia

    try:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"API error {response.status}"}
    except Exception as e:
        return {"error": str(e)}


def process_simple_results(results: List[tuple], company_name: str) -> Dict[str, Any]:
    """Process results for simple validation"""

    output = {
        'validation_status': 'unverified',
        'location_verified': False
    }

    for endpoint, data in results:
        if endpoint == 'places' and 'places' in data:
            for place in data['places'][:3]:
                if fuzzy_match(company_name, place.get('title', '')):
                    output['validation_status'] = 'verified'  # LOWERCASE
                    output['location_verified'] = True
                    output['address'] = place.get('address', '')
                    output['phone'] = place.get('phoneNumber', '')
                    break

    return output


def process_contact_results(results: List[tuple], company_name: str) -> Dict[str, Any]:
    """Process results for contact extraction"""

    output = {
        'emails': [],
        'phones': [],
        'names': [],
        'validation_status': 'unverified'  # Default status
    }

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    for endpoint, data in results:
        if endpoint == 'search' and 'organic' in data:
            for result in data['organic']:
                text = f"{result.get('title', '')} {result.get('snippet', '')}"

                # Extract emails
                emails = re.findall(email_pattern, text)
                output['emails'].extend(emails)

                # Extract names from LinkedIn results
                if 'linkedin.com' in result.get('link', ''):
                    title = result.get('title', '')
                    name_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', title)
                    if name_match:
                        output['names'].append(name_match.group(1))

        elif endpoint == 'places' and 'places' in data:
            for place in data['places'][:2]:
                if place.get('phoneNumber'):
                    output['phones'].append(place['phoneNumber'])

    # Deduplicate
    output['emails'] = list(set(output['emails']))[:10]
    output['names'] = list(set(output['names']))[:10]
    output['phones'] = list(set(output['phones']))[:5]

    # Set status based on findings
    if output['emails'] or output['phones']:
        output['validation_status'] = 'verified'  # LOWERCASE
    elif output['names']:
        output['validation_status'] = 'partial'  # LOWERCASE

    return output


def process_csr_results(results: List[tuple], company_name: str) -> Dict[str, Any]:
    """Process results for CSR verification"""

    output = {
        'csr_programs': [],
        'certifications': [],
        'has_foundation': False,
        'validation_status': 'unverified'  # Default status
    }

    csr_keywords = {
        'children': ['children', 'kids', 'youth', 'school'],
        'community': ['community', 'local', 'volunteer'],
        'environment': ['environment', 'sustainability', 'green'],
        'health': ['health', 'medical', 'hospital']
    }

    cert_patterns = ['b corp', 'b-corp', 'iso 26000', 'iso 14001', 'carbon neutral']

    for endpoint, data in results:
        if endpoint == 'search' and 'organic' in data:
            for result in data['organic']:
                text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()

                # Check for foundation
                if f"{company_name.lower()} foundation" in text:
                    output['has_foundation'] = True

                # Check CSR areas
                for area, keywords in csr_keywords.items():
                    if any(keyword in text for keyword in keywords):
                        if area not in output['csr_programs']:
                            output['csr_programs'].append(area)

                # Check certifications
                for cert in cert_patterns:
                    if cert in text and cert not in output['certifications']:
                        output['certifications'].append(cert)

        elif endpoint == 'news' and 'news' in data:
            for item in data['news'][:5]:
                title = item.get('title', '').lower()
                if any(word in title for word in ['donation', 'charity', 'sponsor']):
                    output['validation_status'] = 'verified'  # LOWERCASE

    # Set final status
    if output['csr_programs'] or output['certifications'] or output['has_foundation']:
        if output['validation_status'] != 'verified':
            output['validation_status'] = 'verified'  # LOWERCASE

    return output


def process_financial_results(results: List[tuple], company_name: str) -> Dict[str, Any]:
    """Process results for financial verification"""

    output = {
        'revenue_range': '',
        'employee_range': '',
        'stock_listed': False,
        'risk_signals': [],
        'validation_status': 'unverified',  # Default status
        'financial_health': 'Unknown'
    }

    revenue_pattern = r'\$?([\d,]+\.?\d*)\s*(million|billion|m|b)'
    employee_pattern = r'([\d,]+)\s*(?:employees|staff)'

    negative_signals = ['bankruptcy', 'lawsuit', 'scandal', 'fraud', 'investigation']

    for endpoint, data in results:
        if endpoint == 'search' and 'organic' in data:
            for result in data['organic']:
                text = f"{result.get('title', '')} {result.get('snippet', '')}"

                # Extract revenue
                if not output['revenue_range']:
                    revenue_matches = re.findall(revenue_pattern, text, re.IGNORECASE)
                    if revenue_matches:
                        output['revenue_range'] = f"${revenue_matches[0][0]}{revenue_matches[0][1].upper()}"
                        output['validation_status'] = 'verified'  # LOWERCASE

                # Extract employees
                if not output['employee_range']:
                    employee_matches = re.findall(employee_pattern, text, re.IGNORECASE)
                    if employee_matches:
                        output['employee_range'] = employee_matches[0]
                        if output['validation_status'] != 'verified':
                            output['validation_status'] = 'partial'  # LOWERCASE

                # Check for ASX listing
                if 'asx' in text.lower():
                    output['stock_listed'] = True

        elif endpoint == 'news' and 'news' in data:
            for item in data['news']:
                text = item.get('snippet', '').lower()
                # Check for negative signals
                for signal in negative_signals:
                    if signal in text:
                        output['risk_signals'].append(signal)

    output['risk_signals'] = list(set(output['risk_signals']))

    return output


def fuzzy_match(name1: str, name2: str, threshold: float = 0.7) -> bool:
    """Fuzzy name matching for company names"""

    name1_clean = name1.lower().strip()
    name2_clean = name2.lower().strip()

    # Direct substring match
    if name1_clean in name2_clean or name2_clean in name1_clean:
        return True

    # Remove common suffixes
    suffixes = ['pty ltd', 'limited', 'ltd', 'inc', 'corporation', 'corp', 'group']
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