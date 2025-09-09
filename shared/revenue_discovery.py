# shared/revenue_discovery.py
"""
Revenue Discovery Module - Based on the working example
Validates company revenue using Serper API after initial GPT search
ENHANCED: With parallel execution for speed (5 concurrent sessions)
"""

import asyncio
import aiohttp
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================
SERPER_API_KEY = "99c44b79892f5f7499accf2d7c26d93313880937"
RATE_LIMIT_DELAY = 0.1  # Reduced delay since we're running in parallel
MAX_QUERIES_PER_COMPANY = 3  # Reduced for efficiency
PARALLEL_SESSIONS = 5  # Number of concurrent Serper sessions


# ==================== Data Model ====================

@dataclass
class CompanyRevenueResult:
    """Result for company revenue discovery"""
    company_name: str

    # Revenue data
    revenue_amount: str = ""
    revenue_currency: str = ""
    revenue_year: str = ""
    revenue_confidence: float = 0.0
    all_revenues_found: str = ""

    # Metadata
    queries_used: int = 0
    status: str = "pending"
    error_message: str = ""
    session_id: int = 0  # Track which session processed this

    def meets_criteria(self, min_revenue: Optional[float], max_revenue: Optional[float], currency: str = "AUD") -> bool:
        """Check if revenue meets criteria"""
        if not self.revenue_amount or self.revenue_confidence < 0.5:
            return False

        # Extract numeric value from revenue_amount (e.g., "$245.122B" -> 245122000000)
        try:
            amount_str = self.revenue_amount.replace('$', '').replace('A$', '')

            # Handle billions/millions
            multiplier = 1
            if 'B' in amount_str.upper() or 'BILLION' in amount_str.upper():
                multiplier = 1_000_000_000
                amount_str = re.sub(r'[Bb]illion|B', '', amount_str)
            elif 'M' in amount_str.upper() or 'MILLION' in amount_str.upper():
                multiplier = 1_000_000
                amount_str = re.sub(r'[Mm]illion|M', '', amount_str)

            numeric_value = float(amount_str.strip()) * multiplier

            # Check against criteria
            if min_revenue and numeric_value < min_revenue:
                return False
            if max_revenue and numeric_value > max_revenue:
                return False

            return True

        except Exception as e:
            logger.error(f"Error parsing revenue {self.revenue_amount}: {e}")
            return False


# ==================== Revenue Extractor ====================

class RevenueExtractor:
    """Extract revenue from Serper API responses"""

    @staticmethod
    def extract_revenue(text: str) -> List[Dict[str, Any]]:
        """Extract revenue with patterns that match actual API responses"""
        if not text:
            return []

        revenues = []
        text = str(text)

        # Patterns that work with Serper responses
        patterns = [
            # Pattern for: "$245.122B" or "A$420.90 Billion"
            (r'(?:A\$|\$|USD\s*\$?|AUD\s*\$?)\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M)\b', 'currency'),

            # Pattern for: "revenue of $X" or "revenue was $X"
            (r'revenue\s+(?:of|was|is|reached|reaching)?\s*(?:A\$|\$)?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M)',
             'revenue_phrase'),

            # Pattern for: "$X in revenue" or "$X in annual revenue"
            (r'(?:A\$|\$)\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M)\s+in\s+(?:annual\s+)?revenue', 'in_revenue'),

            # Pattern for year-specific revenue
            (r'(?:fiscal\s+year\s+|FY\s*|year\s+)?(\d{4})\s+.*?(?:A\$|\$)\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M)',
             'with_year'),

            # Simple money amounts (catch-all)
            (r'(?:A\$|\$)\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M)\b', 'simple'),

            # Numbers without currency symbol but with "billion/million revenue" context
            (r'([\d,]+(?:\.\d+)?)\s*(billion|million)\s+(?:in\s+)?(?:annual\s+)?revenue', 'no_currency'),
        ]

        seen = set()  # Avoid duplicates

        for pattern, pattern_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                full_match = match.group(0)

                # Skip if we've seen this exact match
                if full_match in seen:
                    continue
                seen.add(full_match)

                # Extract components based on pattern type
                if pattern_type == 'with_year':
                    year = match.group(1)
                    amount = match.group(2).replace(',', '')
                    unit = match.group(3)
                elif pattern_type in ['currency', 'revenue_phrase', 'in_revenue', 'simple', 'no_currency']:
                    amount = match.group(1).replace(',', '')
                    unit = match.group(2)
                    year = RevenueExtractor._extract_nearby_year(text, match.start())
                else:
                    continue

                # Normalize unit
                unit = unit.lower()
                if unit == 'b':
                    unit = 'billion'
                elif unit == 'm':
                    unit = 'million'

                # Calculate numeric value
                try:
                    numeric_value = float(amount)
                    if unit == 'billion':
                        numeric_value *= 1_000_000_000
                    elif unit == 'million':
                        numeric_value *= 1_000_000

                    # Determine currency
                    currency = 'USD'
                    if 'A$' in full_match or 'AUD' in full_match:
                        currency = 'AUD'

                    revenues.append({
                        'amount': numeric_value,
                        'formatted': f"{'A$' if currency == 'AUD' else '$'}{amount}{unit[0].upper()}",
                        'year': year or 'recent',
                        'currency': currency,
                        'confidence': 0.9 if year else 0.7,
                        'source_text': full_match
                    })
                except ValueError:
                    continue

        # Sort by confidence and recency
        revenues.sort(key=lambda x: (x['confidence'], x.get('year', '0')), reverse=True)

        return revenues

    @staticmethod
    def _extract_nearby_year(text: str, position: int, window: int = 50) -> Optional[str]:
        """Extract year near the revenue mention"""
        start = max(0, position - window)
        end = min(len(text), position + window)
        snippet = text[start:end]

        # Look for 4-digit years (2020-2029 range)
        year_pattern = r'\b(202[0-9]|201[0-9])\b'
        matches = re.findall(year_pattern, snippet)

        if matches:
            return max(matches)  # Return most recent year
        return None


# ==================== Revenue Discovery Agent (Single Session) ====================

class RevenueDiscoverySession:
    """Single session for revenue discovery - can be used in parallel"""

    def __init__(self, session_id: int, api_key: str = SERPER_API_KEY):
        self.session_id = session_id
        self.api_key = api_key
        self.session = None
        self.extractor = RevenueExtractor()
        self.queries_count = 0

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10),  # Increased limit for parallel
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_web(self, query: str, location: str = "us") -> Dict[str, Any]:
        """Web search using Serper API"""
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        data = {
            "q": query,
            "gl": location,
            "hl": "en",
            "num": 10
        }

        try:
            async with self.session.post(url, json=data, headers=headers) as response:
                self.queries_count += 1
                if response.status == 200:
                    return await response.json()
                logger.error(f"Session {self.session_id}: API error {response.status}")
                return {"error": f"API error {response.status}"}
        except Exception as e:
            logger.error(f"Session {self.session_id}: Search error: {e}")
            return {"error": str(e)}

    def clean_company_name(self, name: str) -> str:
        """Clean company name for better matching"""
        suffixes = ['Corporation', 'Corp', 'Inc', 'Limited', 'Ltd', 'LLC',
                    'Group', 'Holdings', 'Company', 'Co', 'Pty']

        clean_name = name
        for suffix in suffixes:
            clean_name = re.sub(rf'\s+{suffix}\.?\s*$', '', clean_name, flags=re.IGNORECASE)

        return clean_name.strip()

    async def discover_revenue(self, company_name: str, location: str = "Australia") -> CompanyRevenueResult:
        """Discover revenue for a single company"""

        result = CompanyRevenueResult(
            company_name=company_name,
            status='searching',
            session_id=self.session_id
        )

        if not company_name or company_name == 'nan':
            result.status = 'failed'
            result.error_message = 'No company name provided'
            return result

        logger.debug(f"Session {self.session_id}: Discovering revenue for {company_name}")

        all_revenues = []

        try:
            # Determine location parameters
            clean_name = self.clean_company_name(company_name)
            is_australian = any(term in company_name for term in
                                ['Australia', 'Australian', 'Pty', 'ASX']) or location == "Australia"
            gl_location = "au" if is_australian else "us"

            # Query 1: Simple revenue search
            query1 = f"{company_name} revenue"
            result1 = await self.search_web(query1, gl_location)
            result.queries_used += 1

            if "organic" in result1:
                for item in result1.get("organic", [])[:5]:
                    title = item.get('title', '')
                    snippet = item.get('snippet', '')

                    # Check if this result is about our company
                    if clean_name.lower() in title.lower() or clean_name.lower() in snippet.lower():
                        combined_text = f"{title} {snippet}"
                        revenues = self.extractor.extract_revenue(combined_text)
                        all_revenues.extend(revenues)

            # Query 2: Annual revenue with year (if needed)
            if not all_revenues and result.queries_used < MAX_QUERIES_PER_COMPANY:
                await asyncio.sleep(RATE_LIMIT_DELAY)
                query2 = f"{company_name} annual revenue 2024 2023 financial"
                result2 = await self.search_web(query2, gl_location)
                result.queries_used += 1

                if "organic" in result2:
                    for item in result2.get("organic", [])[:3]:
                        if clean_name.lower() in item.get('title', '').lower() or \
                                clean_name.lower() in item.get('snippet', '').lower():
                            combined_text = f"{item.get('title', '')} {item.get('snippet', '')}"
                            revenues = self.extractor.extract_revenue(combined_text)
                            all_revenues.extend(revenues)

            # Query 3: Financial reports (if still no revenue found)
            if not all_revenues and result.queries_used < MAX_QUERIES_PER_COMPANY:
                await asyncio.sleep(RATE_LIMIT_DELAY)
                query3 = f'"{company_name}" investor relations annual report'
                result3 = await self.search_web(query3, gl_location)
                result.queries_used += 1

                if "organic" in result3:
                    for item in result3.get("organic", [])[:3]:
                        combined_text = f"{item.get('title', '')} {item.get('snippet', '')}"
                        revenues = self.extractor.extract_revenue(combined_text)
                        all_revenues.extend(revenues)

            # Process results
            if all_revenues:
                # Remove duplicates based on amount and year
                unique_revenues = {}
                for rev in all_revenues:
                    key = f"{rev['amount']}_{rev.get('year', 'unknown')}"
                    if key not in unique_revenues or rev['confidence'] > unique_revenues[key]['confidence']:
                        unique_revenues[key] = rev

                # Sort by confidence and select best
                sorted_revenues = sorted(unique_revenues.values(),
                                         key=lambda x: (x['confidence'], x.get('year', '0')),
                                         reverse=True)

                best_revenue = sorted_revenues[0]
                result.revenue_amount = best_revenue['formatted']
                result.revenue_currency = best_revenue.get('currency', 'USD')
                result.revenue_year = best_revenue.get('year', 'recent')
                result.revenue_confidence = best_revenue['confidence']

                # Store all revenues found
                result.all_revenues_found = "; ".join([
                    f"{r['formatted']} ({r.get('year', 'recent')})"
                    for r in sorted_revenues[:5]
                ])

                result.status = 'completed_with_revenue'
                logger.info(f"  ✓ Found revenue: {result.revenue_amount} ({result.revenue_year})")
            else:
                result.status = 'completed_no_revenue'
                logger.info(f"  ✗ No revenue found for {company_name}")

        except Exception as e:
            result.status = 'failed'
            result.error_message = str(e)
            logger.error(f"Session {self.session_id}: Error processing {company_name}: {e}")

        return result


# ==================== Parallel Revenue Discovery Manager ====================

class ParallelRevenueDiscoveryManager:
    """Manages multiple parallel revenue discovery sessions"""

    def __init__(self, num_sessions: int = PARALLEL_SESSIONS, api_key: str = SERPER_API_KEY):
        self.num_sessions = num_sessions
        self.api_key = api_key
        self.sessions = []
        self.total_queries = 0

    async def __aenter__(self):
        # Create multiple sessions
        for i in range(self.num_sessions):
            session = await RevenueDiscoverySession(i, self.api_key).__aenter__()
            self.sessions.append(session)
        logger.info(f"Initialized {self.num_sessions} parallel Serper sessions")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Close all sessions
        for session in self.sessions:
            await session.__aexit__(exc_type, exc_val, exc_tb)

    async def discover_revenues_batch(
            self,
            companies: List[Tuple[Any, str]],  # List of (company, location) tuples
    ) -> List[CompanyRevenueResult]:
        """Process a batch of companies in parallel"""

        tasks = []
        session_assignments = []

        # Distribute companies across sessions
        for i, (company, location) in enumerate(companies):
            # Extract company name
            if hasattr(company, 'name'):
                company_name = company.name
            elif isinstance(company, dict):
                company_name = company.get('name', '')
            else:
                company_name = str(company)

            # Assign to a session (round-robin)
            session_idx = i % self.num_sessions
            session = self.sessions[session_idx]

            # Create task
            task = session.discover_revenue(company_name, location)
            tasks.append(task)
            session_assignments.append((company, session_idx))

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for result, (company, session_idx) in zip(results, session_assignments):
            if isinstance(result, Exception):
                # Create error result
                if hasattr(company, 'name'):
                    company_name = company.name
                else:
                    company_name = str(company)

                error_result = CompanyRevenueResult(
                    company_name=company_name,
                    status='failed',
                    error_message=str(result),
                    session_id=session_idx
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        # Update total queries
        self.total_queries = sum(session.queries_count for session in self.sessions)

        return processed_results


# ==================== Main Batch Processing Function ====================

async def validate_company_revenues(
        companies: List[Any],  # Can be EnhancedCompanyEntry objects or dicts
        criteria: Dict[str, Any],
        parallel_limit: int = PARALLEL_SESSIONS
) -> Tuple[List[Any], List[Any]]:
    """
    Validate revenues for a batch of companies using parallel sessions
    Returns: (valid_companies, excluded_companies)
    """
    valid_companies = []
    excluded_companies = []

    # Extract revenue criteria
    min_revenue = criteria.get('financial', {}).get('revenue_min')
    max_revenue = criteria.get('financial', {}).get('revenue_max')
    revenue_currency = criteria.get('financial', {}).get('revenue_currency', 'AUD')

    # Get location for searches
    countries = criteria.get('location', {}).get('countries', ['Australia'])
    location = countries[0] if countries else "Australia"

    start_time = time.time()
    logger.info(f"Starting parallel revenue discovery for {len(companies)} companies with {parallel_limit} sessions")

    async with ParallelRevenueDiscoveryManager(num_sessions=parallel_limit) as manager:
        # Process in batches that match our parallel capacity
        batch_size = parallel_limit * 2  # Process 2x parallel capacity at once for efficiency

        for batch_start in range(0, len(companies), batch_size):
            batch_end = min(batch_start + batch_size, len(companies))
            batch = companies[batch_start:batch_end]

            # Prepare company-location pairs
            company_location_pairs = [(company, location) for company in batch]

            # Process batch in parallel
            logger.info(
                f"Processing batch {batch_start // batch_size + 1}: companies {batch_start + 1}-{batch_end}/{len(companies)}")
            batch_results = await manager.discover_revenues_batch(company_location_pairs)

            # Process results
            for company, revenue_result in zip(batch, batch_results):
                # Create a dictionary with revenue data to attach to company
                revenue_data = {
                    'discovered_revenue_amount': revenue_result.revenue_amount,
                    'discovered_revenue_currency': revenue_result.revenue_currency,
                    'discovered_revenue_year': revenue_result.revenue_year,
                    'discovered_revenue_confidence': revenue_result.revenue_confidence,
                    'discovered_all_revenues': revenue_result.all_revenues_found
                }

                # For EnhancedCompanyEntry objects, we need to convert to dict, add revenue data, then recreate
                if hasattr(company, 'dict'):
                    # It's a Pydantic model
                    company_dict = company.dict()
                    company_dict.update(revenue_data)

                    # Update the estimated_revenue field if we found a better one
                    if revenue_result.revenue_amount and revenue_result.revenue_confidence > 0.7:
                        company_dict['estimated_revenue'] = revenue_result.revenue_amount

                    # Recreate the object with updated data
                    from shared.data_models import EnhancedCompanyEntry
                    updated_company = EnhancedCompanyEntry(**company_dict)

                    # Store revenue data as a separate attribute (won't be in the model fields)
                    updated_company._revenue_discovered = revenue_data

                    company_to_check = updated_company
                elif isinstance(company, dict):
                    # It's already a dict
                    company.update(revenue_data)

                    # Update the estimated_revenue field if we found a better one
                    if revenue_result.revenue_amount and revenue_result.revenue_confidence > 0.7:
                        company['estimated_revenue'] = revenue_result.revenue_amount

                    company_to_check = company
                else:
                    # Unknown type, try to handle gracefully
                    company_to_check = company

                # Check if meets criteria
                if revenue_result.status == 'completed_no_revenue':
                    # No revenue found - keep if no strict revenue requirement
                    if min_revenue and min_revenue > 0:
                        excluded_companies.append(company_to_check)
                        logger.debug(f"Excluding {revenue_result.company_name}: No revenue found")
                    else:
                        valid_companies.append(company_to_check)
                elif revenue_result.meets_criteria(min_revenue, max_revenue, revenue_currency):
                    valid_companies.append(company_to_check)
                    logger.debug(f"✓ {revenue_result.company_name} meets criteria: {revenue_result.revenue_amount}")
                else:
                    excluded_companies.append(company_to_check)
                    logger.debug(f"✗ {revenue_result.company_name} excluded: {revenue_result.revenue_amount}")

            # Small delay between batches to be nice to the API
            if batch_end < len(companies):
                await asyncio.sleep(0.5)

        # Log statistics
        elapsed_time = time.time() - start_time
        total_queries = manager.total_queries
        logger.info(f"Revenue discovery complete:")
        logger.info(f"  - Time: {elapsed_time:.1f}s")
        logger.info(f"  - Companies processed: {len(companies)}")
        logger.info(f"  - Valid: {len(valid_companies)}")
        logger.info(f"  - Excluded: {len(excluded_companies)}")
        logger.info(f"  - Total Serper queries: {total_queries}")
        logger.info(f"  - Queries per second: {total_queries / elapsed_time:.1f}")
        logger.info(f"  - Companies per second: {len(companies) / elapsed_time:.1f}")

    return valid_companies, excluded_companies