# shared/search_utils.py
"""
Search utilities with exhaustive criteria verification before relaxation
Ensures all segments are tried at each level before compromising criteria
UPDATED: Fixed tier B/C support with appropriate relaxation strategies
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

        # EXCLUDED ENTITY TYPES - entities that should never be included
        self.excluded_entity_types = [
            'tafe', 'institute of tafe', 'university', 'college', 'school',
            'government', 'department of', 'ministry', 'council', 'municipality',
            'superannuation', 'super fund', 'pension fund',
            'educational institution', 'training provider', 'rto',
            'charity', 'foundation', 'not-for-profit', 'nfp', 'ngo',
            'association', 'society', 'club'
        ]

    def add_companies(self, model: str, companies: List[Any], segment_info: Optional[Dict[str, Any]] = None) -> List[
        Any]:
        """Add companies and track segment coverage with entity type filtering"""
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

            # ENTITY TYPE CHECK - Skip non-company entities
            if self._is_excluded_entity(company_name):
                logger.debug(f"Excluding non-company entity: {original_name}")
                continue

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

    def _is_excluded_entity(self, company_name: str) -> bool:
        """Check if the entity name indicates it's not a company"""
        name_lower = company_name.lower()
        for excluded_term in self.excluded_entity_types:
            if excluded_term in name_lower:
                return True
        return False

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
        """Get core name for deduplication - ENHANCED with more suffixes"""
        name_core = company_name

        # Extended suffix list for better deduplication
        suffixes = [
            'pty ltd', 'pty. ltd.', 'proprietary limited', 'limited', 'ltd', 'ltd.',
            'inc', 'inc.', 'incorporated', 'corporation', 'corp', 'corp.',
            'llc', 'l.l.c.', 'plc', 'p.l.c.',
            'gmbh', 'ag', 's.a.', 'b.v.', 'n.v.',
            '& co', '& co.', 'and company', 'and co',
            'group', 'holdings', 'international', 'global',
            '& associates', '& partners', 'partners',
            'australia', 'aust', 'aus', 'nz', 'new zealand',
            'victoria', 'vic', 'nsw', 'qld', 'queensland', 'wa', 'sa', 'tas', 'act', 'nt'
        ]

        for suffix in suffixes:
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
            relaxation_level: int = 0,
            tier: str = "A"  # ADD TIER PARAMETER
    ) -> Dict[str, Any]:
        """Search with specific segmentation strategy"""
        if not self.client:
            self._init_llm()

        # Apply smart relaxation based on use case AND TIER
        relaxed_criteria, relaxed_items = apply_smart_relaxation(
            criteria,
            relaxation_level,
            getattr(criteria, 'use_case', 'general'),
            tier  # PASS TIER TO RELAXATION
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
                        {"role": "system",
                         "content": "You are a company finder. Return valid JSON only. Only return actual companies, not educational institutions, government entities, or non-profit organizations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3 + (relaxation_level * 0.05),
                    max_tokens=8192,
                    response_format={"type": "json_object"}
                )
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            # Process companies with validation
            companies = []
            for company_data in result.get("companies", []):
                try:
                    company_data = self._ensure_company_fields(company_data)

                    # VALIDATE AGAINST CRITERIA
                    if not self._validate_company_against_criteria(company_data, relaxed_criteria):
                        logger.debug(f"Rejecting company {company_data.get('name')} - doesn't match criteria")
                        continue

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

    def _validate_company_against_criteria(self, company_data: Dict[str, Any], criteria: SearchCriteria) -> bool:
        """Validate that a company matches the search criteria - FIXED STATE MATCHING"""

        # Import the geography module
        from shared.australian_geography import get_state_from_location, validate_location_in_state

        # STATE VALIDATION - ENHANCED WITH CITY-STATE MAPPING
        if criteria.location.states:
            # Extract all location information from company
            location_fields = []

            # Check headquarters
            if company_data.get('headquarters'):
                hq = company_data['headquarters']
                if isinstance(hq, dict):
                    location_fields.append(hq.get('city', ''))
                    location_fields.append(hq.get('state', ''))
                    location_fields.append(hq.get('address', ''))
                else:
                    location_fields.append(str(hq))

            # Check office locations
            if company_data.get('office_locations'):
                for office in company_data['office_locations']:
                    if isinstance(office, dict):
                        location_fields.append(office.get('city', ''))
                        location_fields.append(office.get('state', ''))
                    else:
                        location_fields.append(str(office))

            # Check service areas
            if company_data.get('service_areas'):
                location_fields.extend(company_data['service_areas'])

            # Combine all location info
            combined_location = ' '.join(str(f) for f in location_fields if f)

            # Validate using the geography module
            if not validate_location_in_state(combined_location, criteria.location.states):
                logger.debug(
                    f"State mismatch: {company_data.get('name')} location '{combined_location}' not in {criteria.location.states}")
                return False

        # INDUSTRY VALIDATION - Keep existing logic but make it less strict for Tier B/C
        if criteria.industries:
            company_industry = company_data.get('industry_category', '').lower()
            industry_match = False

            for ind in criteria.industries:
                ind_name = ind.get('name', '').lower()
                # More flexible matching
                if ind_name in company_industry or company_industry in ind_name:
                    industry_match = True
                    break
                # Check for common variations
                if self._match_industry_variations(company_industry, ind_name):
                    industry_match = True
                    break

            # For Tier B/C, be less strict about industry matching
            tier = getattr(criteria, 'tier', 'A')
            if not industry_match and tier == 'A':
                logger.debug(
                    f"Industry mismatch: {company_data.get('name')} industry '{company_industry}' not in criteria")
                return False
            elif not industry_match and tier in ['B', 'C']:
                # Log but don't reject for Tier B/C
                logger.debug(f"Industry soft mismatch for Tier {tier}: {company_data.get('name')}")

        # ENTITY TYPE VALIDATION - Keep existing
        company_name = company_data.get('name', '').lower()
        excluded_entity_terms = [
            'tafe', 'institute of tafe', 'university', 'college', 'school',
            'government', 'department of', 'ministry', 'council', 'municipality',
            'superannuation', 'super fund', 'pension fund',
            'educational institution', 'training provider', 'rto',
            'charity', 'foundation', 'not-for-profit', 'nfp', 'ngo',
            'association', 'society', 'club'
        ]

        for excluded_term in excluded_entity_terms:
            if excluded_term in company_name:
                logger.debug(f"Entity type exclusion: {company_data.get('name')} contains '{excluded_term}'")
                return False

        return True

    def _get_state_abbrev(self, state: str) -> str:
        """Get state abbreviation for Australian states"""
        state_abbrevs = {
            'victoria': 'vic',
            'new south wales': 'nsw',
            'queensland': 'qld',
            'western australia': 'wa',
            'south australia': 'sa',
            'tasmania': 'tas',
            'northern territory': 'nt',
            'australian capital territory': 'act'
        }
        return state_abbrevs.get(state, state)

    def _match_industry_variations(self, company_industry: str, target_industry: str) -> bool:
        """Check for common industry variations"""
        industry_variations = {
            'technology': ['tech', 'it', 'software', 'digital', 'information technology'],
            'financial services': ['finance', 'banking', 'insurance', 'investment', 'fintech'],
            'health': ['healthcare', 'medical', 'pharmaceutical', 'biotech', 'health services'],
            'construction': ['building', 'engineering', 'infrastructure', 'property development'],
            'property': ['real estate', 'realty', 'property management'],
            'hospitality': ['hotels', 'tourism', 'accommodation', 'restaurants', 'food service']
        }

        # Check if both industries match any variation group
        for main_industry, variations in industry_variations.items():
            if (target_industry in variations or target_industry == main_industry) and \
                    (company_industry in variations or company_industry == main_industry):
                return True

        return False

    def _build_segment_prompt(
            self,
            segment_type: str,
            segment_value: str,
            criteria: SearchCriteria,
            target_count: int,
            exclusion_list: Optional[List[str]] = None,
            relaxed_items: List[str] = None
    ) -> str:
        """Build prompt based on segmentation strategy - ENHANCED WITH STATE/CITY MAPPING"""
        prompt_parts = []

        prompt_parts.append("=" * 60)
        prompt_parts.append(f"FIND {target_count} COMPANIES")
        prompt_parts.append("⛔ ONLY return actual commercial companies")
        prompt_parts.append(
            "⛔ DO NOT include: TAFEs, universities, schools, government entities, councils, charities, foundations, associations, or superannuation funds")

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

        # ENHANCED LOCATION SECTION WITH CITY-STATE MAPPING
        if criteria.location.countries:
            prompt_parts.append(f"⛔ Countries: {', '.join(criteria.location.countries)}")

        # STATE ENFORCEMENT WITH CITY EXAMPLES - CRITICAL FIX
        if criteria.location.states:
            from shared.australian_geography import get_cities_in_state

            prompt_parts.append(
                f"⛔ MUST be headquartered in these STATES/REGIONS: {', '.join(criteria.location.states)}")

            # Add city examples for each state
            for state in criteria.location.states:
                major_cities = get_cities_in_state(state, major_only=True)
                if major_cities:
                    city_list = ', '.join([c.title() for c in major_cities[:6]])
                    prompt_parts.append(f"   ✓ {state} includes: {city_list}")

            prompt_parts.append("⛔ Companies in ANY city within these states are VALID")
            prompt_parts.append("⛔ DO NOT exclude companies just because they list a city instead of a state")

            # Add specific guidance for Australian states
            if "Victoria" in criteria.location.states:
                prompt_parts.append("✓ Victoria companies include those in Melbourne, Geelong, Ballarat, Bendigo, etc.")
                prompt_parts.append("❌ EXCLUDE companies from Sydney, Brisbane, Perth, Adelaide (NOT Victoria)")
            elif "New South Wales" in criteria.location.states:
                prompt_parts.append("✓ NSW companies include those in Sydney, Newcastle, Wollongong, Parramatta, etc.")
                prompt_parts.append("❌ EXCLUDE companies from Melbourne, Brisbane, Perth, Adelaide (NOT NSW)")
            elif "Queensland" in criteria.location.states:
                prompt_parts.append(
                    "✓ Queensland companies include those in Brisbane, Gold Coast, Cairns, Townsville, etc.")
                prompt_parts.append("❌ EXCLUDE companies from Sydney, Melbourne, Perth, Adelaide (NOT Queensland)")

        # City-specific criteria (if specified separately)
        if criteria.location.cities:
            prompt_parts.append(f"Specific cities: {', '.join(criteria.location.cities[:5])}")

        # Add proximity if specified
        if criteria.location.proximity:
            location = criteria.location.proximity.get('location')
            radius = criteria.location.proximity.get('radius_km')
            prompt_parts.append(f"⛔ Within {radius}km of {location}")

        # Financial criteria
        if criteria.financial.revenue_min or criteria.financial.revenue_max:
            if criteria.financial.revenue_min and criteria.financial.revenue_max:
                prompt_parts.append(
                    f"Revenue: ${criteria.financial.revenue_min / 1e6:.0f}M-${criteria.financial.revenue_max / 1e6:.0f}M {criteria.financial.revenue_currency}")
            elif criteria.financial.revenue_min:
                prompt_parts.append(
                    f"Revenue: >${criteria.financial.revenue_min / 1e6:.0f}M {criteria.financial.revenue_currency}")
            elif criteria.financial.revenue_max:
                prompt_parts.append(
                    f"Revenue: <${criteria.financial.revenue_max / 1e6:.0f}M {criteria.financial.revenue_currency}")

        # Employee criteria
        if criteria.organizational.employee_count_min:
            prompt_parts.append(f"Minimum employees: {criteria.organizational.employee_count_min}")
        if criteria.organizational.employee_count_max:
            prompt_parts.append(f"Maximum employees: {criteria.organizational.employee_count_max}")

        # ENHANCED INDUSTRY ENFORCEMENT
        if segment_type != "industry" and criteria.industries:
            ind_names = [ind['name'] for ind in criteria.industries[:5]]
            prompt_parts.append(f"⛔ Industries: {', '.join(ind_names)}")

            # Get tier for less strict enforcement
            tier = getattr(criteria, 'tier', 'A')
            if tier == 'A':
                prompt_parts.append(f"⛔ ONLY include companies in these specific industries")
            else:
                prompt_parts.append(f"⛔ PREFER companies in these industries, but related industries acceptable")

            # Add specific guidance for common industries
            for ind in criteria.industries[:3]:
                ind_name = ind['name'].lower()
                if 'technology' in ind_name or 'tech' in ind_name:
                    prompt_parts.append(
                        "✓ Technology includes: software, IT services, SaaS, digital, tech consulting, fintech")
                elif 'health' in ind_name:
                    prompt_parts.append(
                        "✓ Health includes: hospitals, medical devices, pharmaceuticals, healthcare services, aged care")
                elif 'financial' in ind_name:
                    prompt_parts.append(
                        "✓ Financial Services includes: banks, insurance, investment, accounting, wealth management")
                elif 'construction' in ind_name:
                    prompt_parts.append(
                        "✓ Construction includes: builders, contractors, civil engineering, infrastructure, property development")
                elif 'property' in ind_name or 'real estate' in ind_name:
                    prompt_parts.append(
                        "✓ Property includes: real estate, property management, REITs, property development")
                elif 'hospitality' in ind_name:
                    prompt_parts.append(
                        "✓ Hospitality includes: hotels, resorts, restaurants, tourism, accommodation, entertainment")

        # Business types
        if criteria.business_types:
            prompt_parts.append(f"Business types: {', '.join(criteria.business_types)}")

        # CSR (if not relaxed)
        if criteria.behavioral.csr_focus_areas and "CSR focus areas removed" not in (relaxed_items or []):
            prompt_parts.append(f"CSR focus areas: {', '.join(criteria.behavioral.csr_focus_areas[:3])}")

        # Office types
        if criteria.organizational.office_types:
            prompt_parts.append(f"Office types: {', '.join(criteria.organizational.office_types)}")

        prompt_parts.append("\n📊 RETURN JSON FORMAT:")
        prompt_parts.append("""{"companies":[{
    "name":"Company Name (actual company only)",
    "confidence":"high",
    "operates_in_country":true,
    "business_type":"B2B or B2C etc",
    "industry_category":"Must be relevant to criteria",
    "reasoning":"Why this matches criteria",
    "estimated_revenue":"50-100M",
    "revenue_category":"medium",
    "estimated_employees":"100-500",
    "headquarters":{"city":"Melbourne", "state":"Victoria"},
    "office_locations":[],
    "csr_focus_areas":[]
    }]}""")

        prompt_parts.append("\n⛔ IMPORTANT REMINDERS:")
        prompt_parts.append("- Include the STATE in headquarters (e.g., 'Victoria' for Melbourne companies)")
        prompt_parts.append("- Companies in any city within the specified states are VALID")
        prompt_parts.append("- Do NOT reject companies just because they're listed by city name")
        prompt_parts.append("- Return ONLY actual commercial companies, not institutions or charities")

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
        """Calculate ICP score with relaxation awareness - ENHANCED WITH STATE/INDUSTRY SCORING"""
        score = 0
        max_score = 100
        matched = []
        missing = []

        # STATE MATCH (25 points) - NEW ENHANCED SCORING
        if criteria.location.states and company.headquarters:
            hq_str = str(company.headquarters).lower() if isinstance(company.headquarters, dict) else str(
                company.headquarters).lower()
            state_matched = False

            for state in criteria.location.states:
                if state.lower() in hq_str or self._get_state_abbrev(state.lower()) in hq_str:
                    score += 25
                    matched.append(f"State match: {state}")
                    state_matched = True
                    break

            if not state_matched:
                missing.append("State match")

        # INDUSTRY MATCH (20 points) - ENHANCED SCORING
        if criteria.industries:
            industry_matched = False
            company_industry = company.industry_category.lower() if company.industry_category else ""

            for ind in criteria.industries:
                ind_name = ind.get('name', '').lower()
                if ind_name in company_industry or self._match_industry_variations(company_industry, ind_name):
                    score += 20
                    matched.append(f"Industry match: {ind.get('name')}")
                    industry_matched = True
                    break

            if not industry_matched:
                missing.append("Industry match")

        # Location (city) match (15 points)
        if criteria.location.cities and company.headquarters:
            hq_str = str(company.headquarters).lower() if isinstance(company.headquarters, dict) else str(
                company.headquarters).lower()
            for city in criteria.location.cities:
                if city.lower() in hq_str:
                    score += 15
                    matched.append(f"City match: {city}")
                    break

        # Base scoring for other criteria
        if company.operates_in_country:
            score += 10
            matched.append("Country match")

        if company.estimated_revenue:
            score += 10
            matched.append("Revenue data available")

        if company.estimated_employees:
            score += 10
            matched.append("Employee data available")

        if company.business_type in criteria.business_types:
            score += 5
            matched.append(f"Business type: {company.business_type}")

        if company.csr_programs or company.csr_focus_areas:
            score += 5
            matched.append("CSR programs")

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
        company.matched_criteria = matched
        company.missing_criteria = missing

        return company


def apply_smart_relaxation(
        criteria: SearchCriteria,
        level: int,
        use_case: str = "general",
        tier: str = "A"  # NEW TIER PARAMETER
) -> Tuple[SearchCriteria, List[str]]:
    """Apply smart relaxation based on criteria priority, use case, and TIER"""
    import copy

    relaxed = copy.deepcopy(criteria)
    relaxed_items = []

    # TIER-AWARE RELAXATION STRATEGIES
    if "rmh" in use_case.lower():
        if tier == "A":
            # Tier A: Full relaxation strategy for exhaustive search
            relaxation_plan = {
                1: ["recent_events"],  # Drop recent events first
                2: ["certifications"],  # Drop certifications
                3: ["csr_focus_areas"],  # Drop CSR requirements
                4: ["business_types"],  # Drop business type restrictions
                5: ["revenue_expand_20"],  # Expand revenue by 20%
                6: ["employee_reduce_25"],  # Reduce employee count by 25%
                7: ["industries"],  # Drop industry restrictions
                8: ["location_cities"],  # Drop city restrictions (keep state)
                # Never drop states for RMH - Sydney/NSW is core requirement
            }
        elif tier == "B":
            # Tier B: Gentler relaxation, preserve NSW
            relaxation_plan = {
                1: ["recent_events", "certifications"],
                2: ["csr_focus_areas"],
                3: ["business_types"],
                4: ["revenue_expand_20"],
                5: ["employee_reduce_25"],
                # Keep NSW state requirement
            }
        else:  # Tier C
            # Tier C: Very gentle relaxation
            relaxation_plan = {
                1: ["recent_events", "certifications", "csr_focus_areas"],
                2: ["business_types"],
                3: ["employee_reduce_50"],
                # Keep basic location and revenue requirements
            }

    elif "guide_dogs" in use_case.lower():
        if tier == "A":
            # Tier A: Full relaxation strategy
            relaxation_plan = {
                1: ["recent_events"],
                2: ["esg_maturity"],
                3: ["certifications"],  # Can relax certifications for Tier A
                4: ["business_types"],
                5: ["csr_focus_areas"],  # Can eventually relax CSR
                6: ["employee_reduce_25"],
                7: ["revenue_expand_20"],
                8: ["industries"],  # Can relax industries later
                9: ["location_cities"],  # Drop cities but keep Victoria
                # Never drop Victoria state requirement for Guide Dogs
            }
        elif tier == "B":
            # Tier B: Preserve Victoria and moderate requirements
            relaxation_plan = {
                1: ["recent_events", "esg_maturity", "certifications"],
                2: ["csr_focus_areas"],  # Can relax CSR for Tier B
                3: ["business_types"],
                4: ["employee_reduce_25"],
                # Keep Victoria state and revenue range
            }
        else:  # Tier C
            # Tier C: Very minimal relaxation, preserve Victoria
            relaxation_plan = {
                1: ["recent_events", "certifications", "esg_maturity", "csr_focus_areas"],
                2: ["business_types"],
                3: ["employee_reduce_50"],
                # Keep Victoria and basic requirements
            }
    else:
        # Generic relaxation plan
        relaxation_plan = {
            1: ["csr_focus_areas", "certifications"],
            2: ["business_types"],
            3: ["revenue_expand_50"],
            4: ["employee_reduce_50"],
            5: ["industries"],
            6: ["location_cities"],
            7: ["location_states"],
        }

    # Apply relaxations up to the specified level
    for relax_level in range(1, min(level + 1, len(relaxation_plan) + 1)):
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
            elif item == "esg_maturity":
                relaxed.behavioral.esg_maturity = None
                relaxed_items.append("ESG maturity removed")
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
            elif item == "location_states":
                # Only remove states if explicitly in the plan (shouldn't happen for B/C tiers)
                relaxed.location.states = []
                relaxed_items.append("State restrictions removed")

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


async def execute_parallel_search_with_revenue(
        models: List[str],
        criteria: SearchCriteria,
        target_count: int,
        serper_key: Optional[str] = None,
        enable_recursive: bool = True,
        progress_callback: Optional[callable] = None,
        tier: str = "A",
        validate_revenue: bool = True  # NEW PARAMETER
) -> Dict[str, Any]:
    """Execute parallel search with exhaustive verification and optional revenue discovery"""

    logger.info(f"Starting exhaustive search for {target_count} companies with {len(models)} models for Tier {tier}")

    # Log revenue validation status
    if validate_revenue:
        logger.info("💰 Revenue validation ENABLED - will verify company revenues using Serper API")
    else:
        logger.info("📊 Revenue validation DISABLED - accepting all companies without revenue checks")

    # Import revenue discovery module only if needed
    if validate_revenue:
        from shared.revenue_discovery import validate_company_revenues

    # Check if revenue validation is needed
    has_revenue_criteria = (
            criteria.financial.revenue_min is not None or
            criteria.financial.revenue_max is not None
    )
    should_validate_revenue = validate_revenue and has_revenue_criteria

    if has_revenue_criteria and not validate_revenue:
        logger.info("⚠️ Revenue criteria set but validation disabled - criteria will be ignored")

    # Store tier in criteria for downstream use
    criteria.tier = tier

    tracker = SearchProgressTracker()
    all_companies = []

    # ENHANCED: Store excluded companies by relaxation level and reason
    excluded_by_revenue = {
        'too_low': [],  # Companies with revenue below minimum
        'too_high': [],  # Companies with revenue above maximum
        'no_data': []  # Companies with no revenue found
    }

    # Track which excluded companies have been re-evaluated at each level
    re_evaluated_at_level = {}

    relaxation_level = 0
    revenue_validation_attempts = 0
    max_revenue_attempts = 3  # Maximum times to retry after revenue exclusions

    # Adjust max relaxation based on tier
    if tier == "B":
        max_relaxation = 5
    elif tier == "C":
        max_relaxation = 3
    else:
        max_relaxation = 9

    while len(all_companies) < target_count and relaxation_level <= max_relaxation:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"RELAXATION LEVEL {relaxation_level} (Tier {tier})")
        logger.info(f"Found so far: {len(all_companies)}/{target_count}")

        # ENHANCED: Check if we should re-evaluate excluded companies
        if relaxation_level > 0 and should_validate_revenue:
            companies_to_reevaluate = []

            # Apply smart relaxation to get current criteria
            relaxed_criteria, relaxed_items = apply_smart_relaxation(
                criteria,
                relaxation_level,
                getattr(criteria, 'use_case', 'general'),
                tier
            )

            # Check if revenue criteria have been relaxed
            revenue_relaxed = False
            original_min = criteria.financial.revenue_min
            original_max = criteria.financial.revenue_max
            relaxed_min = relaxed_criteria.financial.revenue_min
            relaxed_max = relaxed_criteria.financial.revenue_max

            if (original_min != relaxed_min) or (original_max != relaxed_max):
                revenue_relaxed = True
                # FIXED: Handle None values for min/max
                if relaxed_min is not None and relaxed_max is not None:
                    logger.info(f"📊 Revenue criteria relaxed: ${relaxed_min / 1e6:.0f}M - ${relaxed_max / 1e6:.0f}M")
                elif relaxed_min is not None:
                    logger.info(f"📊 Revenue criteria relaxed: Min ${relaxed_min / 1e6:.0f}M (no max)")
                elif relaxed_max is not None:
                    logger.info(f"📊 Revenue criteria relaxed: Max ${relaxed_max / 1e6:.0f}M (no min)")
                else:
                    logger.info(f"📊 Revenue criteria relaxed: No specific limits")

            # Re-evaluate excluded companies if revenue criteria changed
            if revenue_relaxed and relaxation_level not in re_evaluated_at_level:
                re_evaluated_at_level[relaxation_level] = True

                # Check companies excluded for being too low
                if relaxed_min is not None and original_min is not None and relaxed_min < original_min:
                    for company in excluded_by_revenue['too_low']:
                        # Extract the discovered revenue amount
                        if hasattr(company, 'dict'):
                            c_dict = company.dict()
                        else:
                            c_dict = company

                        revenue_str = c_dict.get('discovered_revenue_amount', '')
                        if revenue_str:
                            # Parse the revenue amount
                            try:
                                amount_str = revenue_str.replace('$', '').replace('A$', '')
                                multiplier = 1_000_000_000 if 'B' in amount_str else 1_000_000 if 'M' in amount_str else 1
                                amount_str = re.sub(r'[BMbm]', '', amount_str)
                                revenue_value = float(amount_str) * multiplier

                                # Check if now meets relaxed criteria
                                if revenue_value >= relaxed_min:
                                    companies_to_reevaluate.append(company)
                                    logger.info(
                                        f"♻️ Re-including {c_dict.get('name')}: ${revenue_value / 1e6:.0f}M now meets relaxed minimum")
                            except:
                                pass

                # Check companies excluded for being too high
                if relaxed_max is not None and original_max is not None and relaxed_max > original_max:
                    for company in excluded_by_revenue['too_high']:
                        if hasattr(company, 'dict'):
                            c_dict = company.dict()
                        else:
                            c_dict = company

                        revenue_str = c_dict.get('discovered_revenue_amount', '')
                        if revenue_str:
                            try:
                                amount_str = revenue_str.replace('$', '').replace('A$', '')
                                multiplier = 1_000_000_000 if 'B' in amount_str else 1_000_000 if 'M' in amount_str else 1
                                amount_str = re.sub(r'[BMbm]', '', amount_str)
                                revenue_value = float(amount_str) * multiplier

                                if revenue_value <= relaxed_max:
                                    companies_to_reevaluate.append(company)
                                    logger.info(
                                        f"♻️ Re-including {c_dict.get('name')}: ${revenue_value / 1e6:.0f}M now meets relaxed maximum")
                            except:
                                pass

                # Check companies with no revenue data if criteria are significantly relaxed
                if relaxation_level >= 3 and "revenue_expand_50" in str(relaxed_items):
                    # At higher relaxation levels, include some companies with no revenue data
                    include_percentage = min(0.3, relaxation_level * 0.1)  # Include up to 30% of no-data companies
                    num_to_include = int(len(excluded_by_revenue['no_data']) * include_percentage)

                    if num_to_include > 0:
                        companies_to_include = excluded_by_revenue['no_data'][:num_to_include]
                        companies_to_reevaluate.extend(companies_to_include)
                        logger.info(
                            f"♻️ Re-including {num_to_include} companies with no revenue data at relaxation level {relaxation_level}")

                # Add re-evaluated companies back to the pool
                if companies_to_reevaluate:
                    # Add them with updated relaxation level
                    for company in companies_to_reevaluate:
                        if hasattr(company, 'relaxation_level'):
                            company.relaxation_level = relaxation_level
                        elif isinstance(company, dict):
                            company['relaxation_level'] = relaxation_level

                        # Track them properly
                        segment_info = {
                            'type': 're-evaluated',
                            'value': f'level_{relaxation_level}',
                            'relaxation_level': relaxation_level
                        }
                        unique = tracker.add_companies('re-evaluation', [company], segment_info)
                        all_companies.extend(unique)

                    logger.info(f"✅ Re-included {len(companies_to_reevaluate)} previously excluded companies")

                    # Remove them from excluded lists so they won't be re-evaluated again
                    for company in companies_to_reevaluate:
                        for list_key in excluded_by_revenue:
                            if company in excluded_by_revenue[list_key]:
                                excluded_by_revenue[list_key].remove(company)

        level_companies = []
        attempts = 0
        max_attempts_per_level = 5
        min_attempts_before_relaxation = 3

        while attempts < max_attempts_per_level:
            attempts += 1
            logger.info(f"\nLevel {relaxation_level}, Attempt {attempts}")

            # Check if we already have enough companies
            if len(all_companies) >= target_count:
                logger.info(f"Target reached: {len(all_companies)}/{target_count}")
                break

            # Get untried segments
            untried_segments = await get_untried_segments(criteria, relaxation_level, tracker)

            if not untried_segments and attempts > 1:
                logger.info("All segments tried at this level")
                break

            # If no untried segments, retry some segments
            if not untried_segments:
                untried_segments = [
                    {'type': 'letter', 'value': chr(65 + (attempts % 26)), 'priority': 1}
                    for _ in range(min(5, len(models)))
                ]

            # Search untried segments
            tasks = []
            for i, segment in enumerate(untried_segments[:len(models) * 3]):
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
                    relaxation_level=relaxation_level,
                    tier=tier
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

                        # ==================== REVENUE DISCOVERY WITH CATEGORIZATION ====================
                        if should_validate_revenue and len(companies) > 0:
                            logger.info(f"\n🔍 Starting revenue discovery for {len(companies)} companies from {model}")

                            # Use current relaxation level criteria for revenue validation
                            relaxed_criteria, _ = apply_smart_relaxation(
                                criteria,
                                relaxation_level,
                                getattr(criteria, 'use_case', 'general'),
                                tier
                            )

                            # Convert criteria to dict format for revenue validation
                            criteria_dict = {
                                'financial': {
                                    'revenue_min': relaxed_criteria.financial.revenue_min,
                                    'revenue_max': relaxed_criteria.financial.revenue_max,
                                    'revenue_currency': relaxed_criteria.financial.revenue_currency
                                },
                                'location': {
                                    'countries': relaxed_criteria.location.countries
                                }
                            }

                            # Validate revenues
                            valid_companies, excluded = await validate_company_revenues(
                                companies,
                                criteria_dict,
                                parallel_limit=5
                            )

                            logger.info(
                                f"Revenue validation results: {len(valid_companies)} valid, {len(excluded)} excluded")

                            # ENHANCED: Categorize excluded companies
                            for exc_company in excluded:
                                if hasattr(exc_company, 'dict'):
                                    c_dict = exc_company.dict()
                                else:
                                    c_dict = exc_company

                                revenue_str = c_dict.get('discovered_revenue_amount', '')

                                if not revenue_str:
                                    excluded_by_revenue['no_data'].append(exc_company)
                                else:
                                    try:
                                        # Parse revenue to determine why it was excluded
                                        amount_str = revenue_str.replace('$', '').replace('A$', '')
                                        multiplier = 1_000_000_000 if 'B' in amount_str else 1_000_000 if 'M' in amount_str else 1
                                        amount_str = re.sub(r'[BMbm]', '', amount_str)
                                        revenue_value = float(amount_str) * multiplier

                                        if relaxed_criteria.financial.revenue_min and revenue_value < relaxed_criteria.financial.revenue_min:
                                            excluded_by_revenue['too_low'].append(exc_company)
                                            logger.debug(
                                                f"  Excluded (too low): {c_dict.get('name')} - ${revenue_value / 1e6:.0f}M")
                                        elif relaxed_criteria.financial.revenue_max and revenue_value > relaxed_criteria.financial.revenue_max:
                                            excluded_by_revenue['too_high'].append(exc_company)
                                            logger.debug(
                                                f"  Excluded (too high): {c_dict.get('name')} - ${revenue_value / 1e6:.0f}M")
                                        else:
                                            excluded_by_revenue['no_data'].append(exc_company)
                                    except:
                                        excluded_by_revenue['no_data'].append(exc_company)

                            # Use only valid companies
                            companies = valid_companies
                        elif not should_validate_revenue:
                            logger.info(
                                f"💡 Skipping revenue validation for {len(companies)} companies (validation disabled)")
                        # ==================== END REVENUE DISCOVERY ====================

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

                if should_validate_revenue:
                    logger.info(f"Attempt {attempts} found {len(round_companies)} companies (after revenue validation)")
                else:
                    logger.info(f"Attempt {attempts} found {len(round_companies)} companies")

                if progress_callback:
                    progress_data = tracker.get_progress_summary()
                    if should_validate_revenue:
                        total_excluded = sum(len(v) for v in excluded_by_revenue.values())
                        progress_data['revenue_excluded'] = total_excluded
                        progress_data['revenue_excluded_breakdown'] = {
                            'too_low': len(excluded_by_revenue['too_low']),
                            'too_high': len(excluded_by_revenue['too_high']),
                            'no_data': len(excluded_by_revenue['no_data'])
                        }
                    progress_callback(progress_data)

            # Check if we have enough
            if len(all_companies) >= target_count:
                logger.info(f"Target reached: {len(all_companies)}/{target_count}")
                break

            # If many companies were excluded for revenue, try to find more
            if should_validate_revenue:
                total_excluded = sum(len(v) for v in excluded_by_revenue.values())
                if total_excluded > len(all_companies) * 0.3:
                    if revenue_validation_attempts < max_revenue_attempts:
                        revenue_validation_attempts += 1
                        logger.info(f"⚠️ High revenue exclusion rate ({total_excluded} excluded). "
                                    f"Attempting to find more companies (attempt {revenue_validation_attempts}/{max_revenue_attempts})")
                        # Continue searching without incrementing relaxation level
                        continue

            # Check for consecutive low yields
            if tracker.get_consecutive_low_yields(relaxation_level) >= 3:
                logger.info("3 consecutive low-yield attempts")
                break

        # Level summary
        if should_validate_revenue:
            total_excluded = sum(len(v) for v in excluded_by_revenue.values())
        else:
            total_excluded = 0

        logger.info(f"\nLevel {relaxation_level} Summary:")
        logger.info(f"  - Attempts: {attempts}")
        logger.info(f"  - Companies found at this level: {len(level_companies)}")
        logger.info(f"  - Total companies: {len(all_companies)}")

        if should_validate_revenue and total_excluded > 0:
            logger.info(f"  - Revenue excluded (total): {total_excluded}")
            logger.info(f"    - Too low: {len(excluded_by_revenue['too_low'])}")
            logger.info(f"    - Too high: {len(excluded_by_revenue['too_high'])}")
            logger.info(f"    - No data: {len(excluded_by_revenue['no_data'])}")

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
            revenue_validation_attempts = 0  # Reset revenue attempts for new level
        else:
            logger.info(f"✗ Level {relaxation_level} NOT exhausted: {reason}")
            if attempts >= max_attempts_per_level:
                logger.info("Max attempts reached, forcing relaxation")
                relaxation_level += 1
                revenue_validation_attempts = 0

    # Final summary
    summary = tracker.get_progress_summary()

    if should_validate_revenue:
        total_excluded = sum(len(v) for v in excluded_by_revenue.values())
    else:
        total_excluded = 0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"SEARCH COMPLETE")
    logger.info(f"Total found: {len(all_companies)}")

    if should_validate_revenue and total_excluded > 0:
        logger.info(f"Revenue excluded: {total_excluded}")
        logger.info(f"  - Too low: {len(excluded_by_revenue['too_low'])} companies")
        logger.info(f"  - Too high: {len(excluded_by_revenue['too_high'])} companies")
        logger.info(f"  - No data: {len(excluded_by_revenue['no_data'])} companies")
    elif not should_validate_revenue and has_revenue_criteria:
        logger.info("Revenue validation was DISABLED - no companies excluded for revenue")

    logger.info(f"Relaxation levels used: {relaxation_level}")
    logger.info(f"Time elapsed: {summary['elapsed_time']:.1f}s")

    # Build metadata
    metadata = {
        'total_found': len(all_companies),
        'letter_coverage': summary.get('letter_coverage', {}),
        'model_performance': summary['models'],
        'execution_time': summary['elapsed_time'],
        'relaxation_levels_used': relaxation_level,
        'segments_tried': summary.get('segments_tried', {}),
        'attempts_per_level': summary.get('attempts_per_level', {}),
        'tier': tier,
        'revenue_validation_enabled': validate_revenue
    }

    if should_validate_revenue:
        metadata['revenue_excluded'] = total_excluded
        metadata['revenue_excluded_breakdown'] = {
            'too_low': len(excluded_by_revenue['too_low']),
            'too_high': len(excluded_by_revenue['too_high']),
            'no_data': len(excluded_by_revenue['no_data'])
        }
        metadata['strategy'] = 'EXHAUSTIVE_WITH_REVENUE_VALIDATION_AND_RECOVERY'
    else:
        metadata['revenue_excluded'] = 0
        metadata['strategy'] = 'EXHAUSTIVE_WITHOUT_REVENUE_VALIDATION'

    return {
        'companies': all_companies[:target_count],
        'metadata': metadata
    }