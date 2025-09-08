# shared/search_utils.py
"""
Search utilities with exhaustive criteria verification before relaxation
Ensures all segments are tried at each level before compromising criteria
"""

import asyncio
import aiohttp
import json
import time
import os
import re
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from shared.data_models import SearchCriteria, EnhancedCompanyEntry
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CriteriaPriority(Enum):
    """Priority levels for different criteria"""
    CRITICAL = 1  # Must have - never relax
    HIGH = 2  # Very important - relax last
    MEDIUM = 3  # Important - relax after low
    LOW = 4  # Nice to have - relax first
    OPTIONAL = 5  # Bonus - can ignore


class SearchProgressTracker:
    """Enhanced tracker with exhaustion verification"""

    def __init__(self):
        self.found_companies: Set[str] = set()
        self.found_company_cores: Set[str] = set()
        self.model_progress: Dict[str, Dict[str, Any]] = {}
        self.letter_coverage: Dict[str, List[Any]] = {letter: [] for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
        self.total_found = 0
        self.start_time = time.time()

        # Exhaustion tracking
        self.segments_tried: Dict[int, Dict[str, Set[str]]] = {}  # relaxation_level -> segment_type -> segments
        self.attempts_per_level: Dict[int, int] = {}
        self.yield_per_attempt: Dict[int, List[int]] = {}
        self.all_companies = []

    def add_companies(self, model: str, companies: List[Any], segment_info: Optional[Dict[str, Any]] = None) -> List[
        Any]:
        """Add companies and track segment coverage"""
        unique_companies = []

        # Track segment if provided
        if segment_info:
            level = segment_info.get('relaxation_level', 0)
            seg_type = segment_info.get('type', 'unknown')
            seg_value = segment_info.get('value', 'unknown')

            if level not in self.segments_tried:
                self.segments_tried[level] = {}
            if seg_type not in self.segments_tried[level]:
                self.segments_tried[level][seg_type] = set()
            self.segments_tried[level][seg_type].add(seg_value)

        for company in companies:
            if hasattr(company, 'name'):
                company_name = company.name.lower().strip()
                original_name = company.name
            else:
                company_name = company.get('name', '').lower().strip()
                original_name = company.get('name', '')

            # Create core name for deduplication
            name_core = self._get_name_core(company_name)

            if name_core not in self.found_company_cores and company_name not in self.found_companies:
                self.found_companies.add(company_name)
                self.found_company_cores.add(name_core)
                unique_companies.append(company)
                self.all_companies.append(company)
                self.total_found += 1

                # Track letter coverage
                first_letter = original_name[0].upper() if original_name else '?'
                if first_letter in self.letter_coverage:
                    self.letter_coverage[first_letter].append(company)

        # Update model progress
        if model not in self.model_progress:
            self.model_progress[model] = {'found': 0, 'duplicates': 0, 'by_segment': {}}

        self.model_progress[model]['found'] += len(unique_companies)
        self.model_progress[model]['duplicates'] += len(companies) - len(unique_companies)

        return unique_companies

    def record_attempt(self, relaxation_level: int, companies_found: int):
        """Record an attempt at a specific relaxation level"""
        if relaxation_level not in self.attempts_per_level:
            self.attempts_per_level[relaxation_level] = 0
            self.yield_per_attempt[relaxation_level] = []

        self.attempts_per_level[relaxation_level] += 1
        self.yield_per_attempt[relaxation_level].append(companies_found)

    def get_consecutive_low_yields(self, relaxation_level: int, threshold: int = 2) -> int:
        """Get number of consecutive low-yield attempts"""
        if relaxation_level not in self.yield_per_attempt:
            return 0

        yields = self.yield_per_attempt[relaxation_level]
        consecutive = 0

        for y in reversed(yields):
            if y < threshold:
                consecutive += 1
            else:
                break

        return consecutive

    def _get_name_core(self, company_name: str) -> str:
        """Get core name for deduplication"""
        name_core = company_name
        for suffix in ['pty ltd', 'limited', 'ltd', 'inc', 'corporation', 'corp', 'llc', 'plc',
                       '& co', 'and company', 'group', 'holdings', 'international', 'global']:
            name_core = name_core.replace(suffix, '').strip()
        return name_core

    def get_exclusion_list(self, max_items: int = 100) -> List[str]:
        """Get list of company names to exclude"""
        return list(self.found_companies)[-max_items:]

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        elapsed = time.time() - self.start_time
        letter_coverage = {letter: len(companies) for letter, companies in self.letter_coverage.items()}
        return {
            'total_found': self.total_found,
            'unique_companies': len(self.found_companies),
            'elapsed_time': elapsed,
            'models': self.model_progress,
            'letter_coverage': letter_coverage,
            'segments_tried': self.segments_tried,
            'attempts_per_level': self.attempts_per_level
        }


class ExhaustionVerifier:
    """Verifies that criteria have been exhausted before relaxation"""

    @staticmethod
    def verify_segment_exhaustion(
            criteria: SearchCriteria,
            relaxation_level: int,
            tracker: SearchProgressTracker
    ) -> Dict[str, Any]:
        """Verify we've truly exhausted current criteria level"""

        exhaustion_report = {
            'level': relaxation_level,
            'segments_tried': {},
            'segments_remaining': {},
            'coverage_percentage': 0,
            'exhausted': False,
            'reason': ''
        }

        tried_at_level = tracker.segments_tried.get(relaxation_level, {})

        # Check alphabet coverage
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        letters_tried = tried_at_level.get('letter', set())
        letters_remaining = set(alphabet) - letters_tried

        exhaustion_report['segments_tried']['letters'] = sorted(list(letters_tried))
        exhaustion_report['segments_remaining']['letters'] = sorted(list(letters_remaining))

        # Check industry coverage (if specified)
        if criteria.industries:
            industries_in_criteria = {ind['name'] for ind in criteria.industries}
            industries_tried = tried_at_level.get('industry', set())
            industries_remaining = industries_in_criteria - industries_tried

            exhaustion_report['segments_tried']['industries'] = sorted(list(industries_tried))
            exhaustion_report['segments_remaining']['industries'] = sorted(list(industries_remaining))

        # Check location coverage
        if criteria.location.cities:
            cities_in_criteria = set(criteria.location.cities)
            cities_tried = tried_at_level.get('location', set())
            cities_remaining = cities_in_criteria - cities_tried

            exhaustion_report['segments_tried']['cities'] = sorted(list(cities_tried))
            exhaustion_report['segments_remaining']['cities'] = sorted(list(cities_remaining))

        # Calculate coverage percentage
        total_segments = 26  # alphabet
        if criteria.industries:
            total_segments += len(criteria.industries)
        if criteria.location.cities:
            total_segments += len(criteria.location.cities)

        total_tried = len(letters_tried)
        if criteria.industries:
            total_tried += len(exhaustion_report['segments_tried'].get('industries', []))
        if criteria.location.cities:
            total_tried += len(exhaustion_report['segments_tried'].get('cities', []))

        exhaustion_report['coverage_percentage'] = (total_tried / total_segments * 100) if total_segments > 0 else 0

        # Determine if exhausted
        total_remaining = (
                len(letters_remaining) +
                len(exhaustion_report['segments_remaining'].get('industries', [])) +
                len(exhaustion_report['segments_remaining'].get('cities', []))
        )

        if total_remaining == 0:
            exhaustion_report['exhausted'] = True
            exhaustion_report['reason'] = 'All segments tried'
        elif exhaustion_report['coverage_percentage'] >= 90:
            exhaustion_report['exhausted'] = True
            exhaustion_report['reason'] = f"{exhaustion_report['coverage_percentage']:.1f}% coverage achieved"
        else:
            exhaustion_report['exhausted'] = False
            exhaustion_report['reason'] = f"Only {exhaustion_report['coverage_percentage']:.1f}% coverage"

        return exhaustion_report

    @staticmethod
    def should_relax_criteria(
            tracker: SearchProgressTracker,
            criteria: SearchCriteria,
            target_count: int,
            current_count: int,
            relaxation_level: int,
            min_attempts: int = 3
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Determine if we should relax criteria"""

        remaining_needed = target_count - current_count

        # Step 1: Check segment coverage
        exhaustion = ExhaustionVerifier.verify_segment_exhaustion(criteria, relaxation_level, tracker)

        if not exhaustion['exhausted']:
            logger.info(f"Level {relaxation_level} coverage: {exhaustion['coverage_percentage']:.1f}%")
            if exhaustion['segments_remaining'].get('letters'):
                logger.info(f"  Untried letters: {exhaustion['segments_remaining']['letters'][:10]}...")
            return False, "Segments remain untried", exhaustion

        # Step 2: Verify attempt count
        attempts_at_level = tracker.attempts_per_level.get(relaxation_level, 0)

        if attempts_at_level < min_attempts:
            reason = f"Need more attempts ({attempts_at_level}/{min_attempts})"
            return False, reason, exhaustion

        # Step 3: Check yield trend
        yields = tracker.yield_per_attempt.get(relaxation_level, [])

        if len(yields) >= 2:
            # If yield is improving, don't relax yet
            if yields[-1] > yields[-2] * 1.2:  # 20% improvement
                reason = f"Yield improving: {yields[-2]} → {yields[-1]}"
                return False, reason, exhaustion

        # Step 4: Check total yield at this level
        total_at_level = sum(yields) if yields else 0

        if total_at_level < remaining_needed * 0.05:  # Less than 5% of what we need
            reason = f"Low total yield: {total_at_level} companies"
            return True, reason, exhaustion

        # Step 5: Diminishing returns check
        if len(yields) >= 3:
            recent_average = sum(yields[-3:]) / 3
            if recent_average < 2:  # Less than 2 companies per attempt average
                reason = f"Diminishing returns: {recent_average:.1f} avg per attempt"
                return True, reason, exhaustion

        # Step 6: If we have good yield but all segments tried
        if exhaustion['exhausted'] and total_at_level >= 10:
            reason = f"Level exhausted with {total_at_level} companies found"
            return True, reason, exhaustion

        return False, "Continue at current level", exhaustion


class EnhancedSearchStrategistAgent:
    """Enhanced agent with exhaustive search capabilities"""

    def __init__(self, deployment_name: str = "gpt-4.1"):
        self.deployment_name = deployment_name
        self.client = None
        self.initialized = False
        self.max_companies_per_call = 20

    def _init_llm(self):
        """Initialize the LLM with Azure OpenAI"""
        if self.initialized:
            return

        try:
            from openai import AzureOpenAI

            api_key = os.getenv("AZURE_OPENAI_KEY",
                                "CUxPxhxqutsvRVHmGQcmH59oMim6mu55PjHTjSpM6y9UwIxwVZIuJQQJ99BFACL93NaXJ3w3AAABACOG3kI1")
            api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://amex-openai-2025.openai.azure.com/")

            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint
            )

            self.initialized = True
            logger.info(f"Initialized {self.deployment_name}")

        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI: {str(e)}")
            raise

    async def search_with_segment(
            self,
            segment_type: str,
            segment_value: str,
            criteria: SearchCriteria,
            target_count: int = 20,
            exclusion_list: Optional[List[str]] = None,
            relaxation_level: int = 0
    ) -> Dict[str, Any]:
        """Search with specific segmentation strategy"""
        if not self.client:
            self._init_llm()

        # Apply smart relaxation based on use case
        relaxed_criteria, relaxed_items = apply_smart_relaxation(
            criteria,
            relaxation_level,
            getattr(criteria, 'use_case', 'general')
        )

        # Build prompt
        prompt = self._build_segment_prompt(
            segment_type,
            segment_value,
            relaxed_criteria,
            target_count,
            exclusion_list,
            relaxed_items
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a company finder. Return valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3 + (relaxation_level * 0.05),
                    max_tokens=8192,
                    response_format={"type": "json_object"}
                )
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            # Process companies
            companies = []
            for company_data in result.get("companies", []):
                try:
                    company_data = self._ensure_company_fields(company_data)
                    company_data['search_segment'] = f"{segment_type}:{segment_value}"
                    company_data['source_model'] = self.deployment_name
                    company_data['relaxation_level'] = relaxation_level
                    company = EnhancedCompanyEntry(**company_data)
                    company = self._calculate_icp_score(company, criteria)
                    companies.append(company)
                except Exception as e:
                    logger.warning(f"Error processing company: {e}")
                    continue

            logger.info(
                f"{self.deployment_name} found {len(companies)} for {segment_type}:{segment_value} (L{relaxation_level})")

            return {
                "companies": companies,
                "segment": f"{segment_type}:{segment_value}",
                "count": len(companies)
            }

        except Exception as e:
            logger.error(f"Search error for segment {segment_type}:{segment_value}: {e}")
            return {"companies": [], "segment": f"{segment_type}:{segment_value}", "error": str(e)}

    def _build_segment_prompt(
            self,
            segment_type: str,
            segment_value: str,
            criteria: SearchCriteria,
            target_count: int,
            exclusion_list: Optional[List[str]] = None,
            relaxed_items: List[str] = None
    ) -> str:
        """Build prompt based on segmentation strategy"""
        prompt_parts = []

        prompt_parts.append("=" * 60)
        prompt_parts.append(f"FIND {target_count} COMPANIES")

        # Show relaxation if applied
        if relaxed_items:
            prompt_parts.append(f"RELAXATION APPLIED: {', '.join(relaxed_items)}")

        # Segment-specific requirements
        if segment_type == "letter":
            prompt_parts.append(f"⛔ CRITICAL: Every company name MUST start with '{segment_value}'")
            prompt_parts.append(f"⛔ Do NOT include companies starting with any other letter")
        elif segment_type == "industry":
            prompt_parts.append(f"⛔ CRITICAL: Focus on {segment_value} industry")
            prompt_parts.append(f"⛔ Include related sub-industries and niches")
        elif segment_type == "location":
            prompt_parts.append(f"⛔ CRITICAL: Companies in {segment_value}")
            prompt_parts.append(f"⛔ Include companies with offices/operations there")
        elif segment_type == "size":
            prompt_parts.append(f"⛔ CRITICAL: {segment_value} companies")
        elif segment_type == "general":
            prompt_parts.append(f"⛔ Find diverse companies matching criteria")

        prompt_parts.append("=" * 60)

        # Add exclusions
        if exclusion_list and len(exclusion_list) > 0:
            prompt_parts.append("\n🚫 EXCLUDE these companies:")
            for company in exclusion_list[:30]:
                prompt_parts.append(f"   ❌ {company}")

        # Add search criteria
        prompt_parts.append("\n📋 SEARCH CRITERIA:")

        # Location
        if criteria.location.countries:
            prompt_parts.append(f"Countries: {', '.join(criteria.location.countries)}")
        if criteria.location.cities:
            prompt_parts.append(f"Cities: {', '.join(criteria.location.cities[:5])}")

        # Financial
        if criteria.financial.revenue_min or criteria.financial.revenue_max:
            if criteria.financial.revenue_min and criteria.financial.revenue_max:
                prompt_parts.append(
                    f"Revenue: ${criteria.financial.revenue_min / 1e6:.0f}M-${criteria.financial.revenue_max / 1e6:.0f}M")
            elif criteria.financial.revenue_min:
                prompt_parts.append(f"Revenue: >${criteria.financial.revenue_min / 1e6:.0f}M")

        # Employees
        if criteria.organizational.employee_count_min:
            prompt_parts.append(f"Minimum employees: {criteria.organizational.employee_count_min}")

        # Industries
        if segment_type != "industry" and criteria.industries:
            ind_names = [ind['name'] for ind in criteria.industries[:5]]
            prompt_parts.append(f"Industries: {', '.join(ind_names)}")

        # Business types
        if criteria.business_types:
            prompt_parts.append(f"Business types: {', '.join(criteria.business_types)}")

        # CSR (if not relaxed)
        if criteria.behavioral.csr_focus_areas:
            prompt_parts.append(f"CSR areas: {', '.join(criteria.behavioral.csr_focus_areas[:3])}")

        prompt_parts.append("\n📊 RETURN JSON FORMAT:")
        prompt_parts.append("""{"companies":[{
"name":"Company Name",
"confidence":"high",
"operates_in_country":true,
"business_type":"B2B",
"industry_category":"Industry",
"reasoning":"Why this matches",
"estimated_revenue":"50-100M",
"revenue_category":"medium",
"estimated_employees":"100-500",
"headquarters":{"city":"City"},
"csr_focus_areas":[]
}]}""")

        return "\n".join(prompt_parts)

    def _ensure_company_fields(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required fields exist"""
        defaults = {
            "name": "Unknown Company",
            "confidence": "low",
            "operates_in_country": True,
            "business_type": "Unknown",
            "industry_category": "Unknown",
            "reasoning": "No reasoning provided",
            "sub_industry": None,
            "headquarters": None,
            "office_locations": [],
            "service_areas": [],
            "estimated_revenue": None,
            "revenue_category": "unknown",
            "revenue_currency": "USD",
            "estimated_employees": None,
            "employees_by_location": None,
            "company_size": "unknown",
            "giving_history": [],
            "financial_health": None,
            "csr_programs": [],
            "csr_focus_areas": [],
            "certifications": [],
            "esg_score": None,
            "esg_maturity": None,
            "community_involvement": [],
            "recent_events": [],
            "leadership_changes": [],
            "growth_signals": [],
            "icp_tier": None,
            "icp_score": None,
            "matched_criteria": [],
            "missing_criteria": [],
            "data_freshness": None,
            "data_sources": [],
            "validation_notes": None
        }

        for key, default_value in defaults.items():
            if key not in company_data:
                company_data[key] = default_value

        return company_data

    def _calculate_icp_score(self, company: EnhancedCompanyEntry, criteria: SearchCriteria) -> EnhancedCompanyEntry:
        """Calculate ICP score with relaxation awareness"""
        score = 0
        max_score = 100

        # Base scoring
        if company.operates_in_country:
            score += 20
        if company.estimated_revenue:
            score += 20
        if company.estimated_employees:
            score += 15
        if company.business_type in criteria.business_types:
            score += 15
        if company.csr_programs or company.csr_focus_areas:
            score += 15
        if company.confidence in ["high", "absolute"]:
            score += 15

        # Get relaxation level (handle both dict and object)
        if hasattr(company, 'relaxation_level'):
            relaxation_level = company.relaxation_level
        elif isinstance(company, dict):
            relaxation_level = company.get('relaxation_level', 0)
        else:
            relaxation_level = 0

        # Penalize based on relaxation level
        relaxation_penalty = relaxation_level * 5
        score = max(0, score - relaxation_penalty)

        # Determine tier with relaxation consideration
        if score >= 80 and relaxation_level == 0:
            tier = "A"
        elif score >= 70 and relaxation_level <= 1:
            tier = "B"
        elif score >= 60 and relaxation_level <= 2:
            tier = "C"
        elif score >= 50 and relaxation_level <= 3:
            tier = "D"
        elif score >= 40:
            tier = "E"
        else:
            tier = "F"

        company.icp_score = score
        company.icp_tier = tier

        return company


def apply_smart_relaxation(
        criteria: SearchCriteria,
        level: int,
        use_case: str = "general"
) -> Tuple[SearchCriteria, List[str]]:
    """Apply smart relaxation based on criteria priority and use case"""
    import copy

    relaxed = copy.deepcopy(criteria)
    relaxed_items = []

    # Define relaxation plans by use case
    if "rmh" in use_case.lower():
        relaxation_plan = {
            1: ["csr_focus_areas", "certifications", "recent_events"],
            2: ["industries"],
            3: ["revenue_expand_20"],
            4: ["employee_reduce_25"],
            5: ["revenue_expand_50"],
            6: ["location_cities"],
        }
    elif "guide_dogs" in use_case.lower():
        relaxation_plan = {
            1: ["recent_events"],
            2: ["esg_maturity"],
            3: ["business_types"],
            4: ["industries"],
            5: ["certifications"],
            6: ["employee_reduce_50"],
        }
    else:
        relaxation_plan = {
            1: ["csr_focus_areas", "certifications"],
            2: ["industries", "business_types"],
            3: ["revenue_expand_50"],
            4: ["employee_reduce_50"],
            5: ["location_cities"],
            6: ["most_restrictions"],
        }

    # Apply relaxations up to the specified level
    for relax_level in range(1, min(level + 1, 7)):
        items = relaxation_plan.get(relax_level, [])

        for item in items:
            if item == "csr_focus_areas":
                relaxed.behavioral.csr_focus_areas = []
                relaxed_items.append("CSR focus areas removed")
            elif item == "certifications":
                relaxed.behavioral.certifications = []
                relaxed_items.append("Certifications removed")
            elif item == "recent_events":
                relaxed.behavioral.recent_events = []
                relaxed_items.append("Recent events removed")
            elif item == "industries":
                relaxed.industries = []
                relaxed_items.append("Industry restrictions removed")
            elif item == "business_types":
                relaxed.business_types = []
                relaxed_items.append("Business type restrictions removed")
            elif item == "revenue_expand_20":
                if relaxed.financial.revenue_min:
                    relaxed.financial.revenue_min *= 0.8
                if relaxed.financial.revenue_max:
                    relaxed.financial.revenue_max *= 1.2
                relaxed_items.append("Revenue range expanded 20%")
            elif item == "revenue_expand_50":
                if relaxed.financial.revenue_min:
                    relaxed.financial.revenue_min *= 0.5
                if relaxed.financial.revenue_max:
                    relaxed.financial.revenue_max *= 2
                relaxed_items.append("Revenue range expanded 50%")
            elif item == "employee_reduce_25":
                if relaxed.organizational.employee_count_min:
                    relaxed.organizational.employee_count_min = int(
                        relaxed.organizational.employee_count_min * 0.75
                    )
                relaxed_items.append("Minimum employees reduced 25%")
            elif item == "employee_reduce_50":
                if relaxed.organizational.employee_count_min:
                    relaxed.organizational.employee_count_min = max(
                        1,
                        relaxed.organizational.employee_count_min // 2
                    )
                relaxed_items.append("Minimum employees reduced 50%")
            elif item == "location_cities":
                relaxed.location.cities = []
                relaxed_items.append("City restrictions removed")
            elif item == "most_restrictions":
                relaxed.organizational.employee_count_min = None
                relaxed.financial.revenue_min = None
                relaxed.excluded_industries = []
                relaxed_items.append("Most restrictions removed")

    return relaxed, relaxed_items


async def get_untried_segments(
        criteria: SearchCriteria,
        relaxation_level: int,
        tracker: SearchProgressTracker
) -> List[Dict[str, Any]]:
    """Get segments that haven't been tried at this relaxation level"""

    untried = []
    tried_at_level = tracker.segments_tried.get(relaxation_level, {})

    # Check alphabet
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letters_tried = tried_at_level.get('letter', set())

    for letter in alphabet:
        if letter not in letters_tried:
            untried.append({
                'type': 'letter',
                'value': letter,
                'priority': 1  # Alphabet has highest priority
            })

    # Check industries
    if criteria.industries:
        industries_tried = tried_at_level.get('industry', set())
        for industry in criteria.industries:
            ind_name = industry['name']
            if ind_name not in industries_tried:
                untried.append({
                    'type': 'industry',
                    'value': ind_name,
                    'priority': 2
                })

    # Check locations
    if criteria.location.cities:
        locations_tried = tried_at_level.get('location', set())
        for city in criteria.location.cities:
            if city not in locations_tried:
                untried.append({
                    'type': 'location',
                    'value': city,
                    'priority': 3
                })

    # Sort by priority
    untried.sort(key=lambda x: x['priority'])

    return untried


async def execute_parallel_search(
        models: List[str],
        criteria: SearchCriteria,
        target_count: int,
        serper_key: Optional[str] = None,
        enable_recursive: bool = True,
        progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Execute parallel search with exhaustive verification before relaxation"""

    logger.info(f"Starting exhaustive search for {target_count} companies with {len(models)} models")

    tracker = SearchProgressTracker()
    all_companies = []
    relaxation_level = 0
    max_relaxation = 6

    while len(all_companies) < target_count and relaxation_level <= max_relaxation:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"RELAXATION LEVEL {relaxation_level}")
        logger.info(f"Found so far: {len(all_companies)}/{target_count}")

        level_companies = []
        attempts = 0
        max_attempts_per_level = 5
        min_attempts_before_relaxation = 3

        while attempts < max_attempts_per_level:
            attempts += 1
            logger.info(f"\nLevel {relaxation_level}, Attempt {attempts}")

            # Get untried segments
            untried_segments = await get_untried_segments(criteria, relaxation_level, tracker)

            if not untried_segments and attempts > 1:
                logger.info("All segments tried at this level")
                break

            # If no untried segments, retry some segments
            if not untried_segments:
                # Retry alphabet with different approach
                untried_segments = [
                    {'type': 'letter', 'value': chr(65 + (attempts % 26)), 'priority': 1}
                    for _ in range(min(5, len(models)))
                ]

            # Search untried segments
            tasks = []
            for i, segment in enumerate(untried_segments[:len(models) * 3]):  # Limit parallel tasks
                model = models[i % len(models)]
                agent = EnhancedSearchStrategistAgent(deployment_name=model)

                companies_per_segment = min(
                    20,
                    (target_count - len(all_companies)) // max(1, len(untried_segments)) + 5
                )

                task = agent.search_with_segment(
                    segment_type=segment['type'],
                    segment_value=segment['value'],
                    criteria=criteria,
                    target_count=companies_per_segment,
                    exclusion_list=tracker.get_exclusion_list(),
                    relaxation_level=relaxation_level
                )
                tasks.append((model, segment, task))

            # Execute tasks
            if tasks:
                results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)

                round_companies = []
                for (model, segment, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        logger.error(f"Task failed: {result}")
                        continue

                    if isinstance(result, dict) and 'companies' in result:
                        companies = result['companies']
                        segment_info = {
                            'type': segment['type'],
                            'value': segment['value'],
                            'relaxation_level': relaxation_level
                        }
                        unique_companies = tracker.add_companies(model, companies, segment_info)
                        round_companies.extend(unique_companies)
                        level_companies.extend(unique_companies)
                        all_companies.extend(unique_companies)

                # Record attempt
                tracker.record_attempt(relaxation_level, len(round_companies))

                logger.info(f"Attempt {attempts} found {len(round_companies)} companies")

                if progress_callback:
                    progress_callback(tracker.get_progress_summary())

            # Check if we have enough
            if len(all_companies) >= target_count:
                logger.info(f"Target reached: {len(all_companies)}/{target_count}")
                break

            # Check for consecutive low yields
            if tracker.get_consecutive_low_yields(relaxation_level) >= 3:
                logger.info("3 consecutive low-yield attempts")
                break

        # Level summary
        logger.info(f"\nLevel {relaxation_level} Summary:")
        logger.info(f"  - Attempts: {attempts}")
        logger.info(f"  - Companies found at this level: {len(level_companies)}")
        logger.info(f"  - Total companies: {len(all_companies)}")

        # Check if we should relax
        if len(all_companies) >= target_count:
            break

        if not enable_recursive:
            logger.info("Recursive search disabled, stopping")
            break

        # Exhaustion verification
        should_relax, reason, exhaustion = ExhaustionVerifier.should_relax_criteria(
            tracker, criteria, target_count, len(all_companies),
            relaxation_level, min_attempts_before_relaxation
        )

        logger.info(f"\nExhaustion Check:")
        logger.info(f"  Coverage: {exhaustion['coverage_percentage']:.1f}%")
        logger.info(f"  Decision: {'RELAX' if should_relax else 'CONTINUE'} - {reason}")

        if should_relax:
            logger.info(f"✓ Level {relaxation_level} EXHAUSTED: {reason}")
            relaxation_level += 1
        else:
            logger.info(f"✗ Level {relaxation_level} NOT exhausted: {reason}")
            # Continue at same level but with different approach
            if attempts >= max_attempts_per_level:
                logger.info("Max attempts reached, forcing relaxation")
                relaxation_level += 1

    # Final summary
    summary = tracker.get_progress_summary()
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SEARCH COMPLETE")
    logger.info(f"Total found: {len(all_companies)}")
    logger.info(f"Relaxation levels used: {relaxation_level}")
    logger.info(f"Time elapsed: {summary['elapsed_time']:.1f}s")

    return {
        'companies': all_companies[:target_count],
        'metadata': {
            'total_found': len(all_companies),
            'letter_coverage': summary.get('letter_coverage', {}),
            'model_performance': summary['models'],
            'execution_time': summary['elapsed_time'],
            'relaxation_levels_used': relaxation_level,
            'strategy': 'EXHAUSTIVE_WITH_VERIFICATION',
            'segments_tried': summary.get('segments_tried', {}),
            'attempts_per_level': summary.get('attempts_per_level', {})
        }
    }