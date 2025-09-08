# shared/search_utils.py
"""
Search utilities with STRONG deduplication strategy and segment enforcement
Enhanced parallel execution with exclusion lists and hard boundaries
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
from shared.data_models import SearchCriteria, EnhancedCompanyEntry
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchProgressTracker:
    """Tracks progress across all parallel searches"""

    def __init__(self):
        self.found_companies: Set[str] = set()
        self.found_company_cores: Set[str] = set()
        self.model_progress: Dict[str, Dict[str, Any]] = {}
        self.total_found = 0
        self.start_time = time.time()

    def add_companies(self, model: str, companies: List[Any]) -> List[Any]:
        """Add companies and return only unique ones"""
        unique_companies = []

        for company in companies:
            if hasattr(company, 'name'):
                company_name = company.name.lower().strip()
            else:
                company_name = company.get('name', '').lower().strip()

            # Create core name for deduplication
            name_core = self._get_name_core(company_name)

            if name_core not in self.found_company_cores and company_name not in self.found_companies:
                self.found_companies.add(company_name)
                self.found_company_cores.add(name_core)
                unique_companies.append(company)
                self.total_found += 1

        # Update model progress
        if model not in self.model_progress:
            self.model_progress[model] = {'found': 0, 'duplicates': 0}

        self.model_progress[model]['found'] += len(unique_companies)
        self.model_progress[model]['duplicates'] += len(companies) - len(unique_companies)

        return unique_companies

    def _get_name_core(self, company_name: str) -> str:
        """Get core name for deduplication"""
        name_core = company_name
        for suffix in ['pty ltd', 'limited', 'ltd', 'inc', 'corporation', 'corp', 'llc', 'plc',
                       '& co', 'and company', 'group', 'holdings', 'international', 'global']:
            name_core = name_core.replace(suffix, '').strip()
        return name_core

    def get_exclusion_list(self, max_items: int = 100) -> List[str]:
        """Get list of company names to exclude (limited for prompt size)"""
        # Return most recent companies to exclude
        return list(self.found_companies)[-max_items:]

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        elapsed = time.time() - self.start_time
        return {
            'total_found': self.total_found,
            'unique_companies': len(self.found_companies),
            'elapsed_time': elapsed,
            'models': self.model_progress
        }


class EnhancedSearchStrategistAgent:
    """Enhanced agent with STRONG deduplication and segment enforcement"""

    def __init__(self, deployment_name: str = "gpt-4.1"):
        self.deployment_name = deployment_name
        self.client = None
        self.initialized = False
        self.max_companies_per_call = 15
        self.segment_info = None  # Will store segment assignment

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

    async def generate_enhanced_strategy(
            self,
            criteria: SearchCriteria,
            target_count: int = 100,
            unique_instructions: Optional[Dict[str, Any]] = None,
            exclusion_list: Optional[List[str]] = None,
            relaxation_level: int = 0
    ) -> Dict[str, Any]:
        """Generate search strategy with STRONG uniqueness enforcement"""
        if not self.client:
            self._init_llm()

        # Store segment info for validation
        self.segment_info = unique_instructions

        # Build prompt with strong enforcement
        prompt = self._build_enforced_prompt(
            criteria,
            target_count,
            unique_instructions,
            exclusion_list,
            relaxation_level
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system",
                         "content": "You are a company finder. You MUST follow ALL segmentation rules EXACTLY. Companies outside your assigned segment will be REJECTED."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=16384,
                    response_format={"type": "json_object"}
                )
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            # Post-process and validate companies
            validated_companies = self._validate_segment_compliance(
                result.get("companies", []),
                unique_instructions,
                criteria
            )

            enhanced_companies = []
            for company_data in validated_companies:
                try:
                    company_data = self._ensure_company_fields(company_data)
                    company_data['source_model'] = self.deployment_name
                    company_data['relaxation_level'] = relaxation_level

                    if unique_instructions:
                        company_data['search_segment'] = unique_instructions.get('segment_id', 'general')
                        company_data['segment_type'] = unique_instructions.get('strategy', 'unknown')

                    company = EnhancedCompanyEntry(**company_data)
                    company = self._calculate_icp_score(company, criteria)
                    enhanced_companies.append(company)
                except Exception as e:
                    logger.warning(f"Error processing company: {e}")
                    continue

            logger.info(
                f"{self.deployment_name} found {len(enhanced_companies)} companies (segment: {unique_instructions.get('segment_id', 'none')})")

            return {
                "companies": enhanced_companies,
                "metadata": {
                    "total_found": len(enhanced_companies),
                    "deployment": self.deployment_name,
                    "unique_instructions": unique_instructions,
                    "relaxation_level": relaxation_level,
                    "segment_compliant": True
                }
            }

        except Exception as e:
            logger.error(f"Search error in {self.deployment_name}: {e}")
            return {"companies": [], "error": str(e), "deployment": self.deployment_name}

    def _build_enforced_prompt(
            self,
            criteria: SearchCriteria,
            target_count: int,
            unique_instructions: Optional[Dict[str, Any]],
            exclusion_list: Optional[List[str]],
            relaxation_level: int
    ) -> str:
        """Build prompt with STRONG enforcement of uniqueness"""
        prompt_parts = []

        # CRITICAL: Start with the STRONGEST enforcement
        prompt_parts.append("=" * 60)
        prompt_parts.append("🚨 CRITICAL MANDATORY RULES - VIOLATION WILL RESULT IN REJECTION 🚨")
        prompt_parts.append("=" * 60)

        # Add segment-specific HARD boundaries
        if unique_instructions:
            strategy = unique_instructions.get('strategy', 'unknown')

            if strategy == 'ALPHABET_PRIMARY':
                letter = unique_instructions['letter']
                prompt_parts.append(f"⛔ HARD RULE #1: You MUST ONLY return companies whose names START with '{letter}'")
                prompt_parts.append(f"⛔ HARD RULE #2: Companies NOT starting with '{letter}' are FORBIDDEN")
                prompt_parts.append(f"⛔ HARD RULE #3: Check EVERY company name - it MUST begin with '{letter}'")
                prompt_parts.append(
                    f"⛔ If you cannot find {target_count} companies starting with '{letter}', return fewer companies")
                prompt_parts.append(f"⛔ DO NOT include companies starting with other letters to fill the quota")

            elif strategy == 'GEOGRAPHIC_PRIMARY':
                locations = unique_instructions['assigned_locations']
                prompt_parts.append(f"⛔ HARD RULE #1: You MUST ONLY return companies in: {', '.join(locations)}")
                prompt_parts.append(f"⛔ HARD RULE #2: Companies from OTHER locations are FORBIDDEN")
                prompt_parts.append(f"⛔ HARD RULE #3: Verify EVERY company is in your assigned locations")

            elif strategy == 'INDUSTRY_PRIMARY':
                industries = unique_instructions['assigned_industries']
                prompt_parts.append(f"⛔ HARD RULE #1: You MUST ONLY return companies in: {', '.join(industries)}")
                prompt_parts.append(f"⛔ HARD RULE #2: Companies from OTHER industries are FORBIDDEN")
                prompt_parts.append(f"⛔ HARD RULE #3: Verify EVERY company is in your assigned industries")

            # Add rank offset if specified
            if unique_instructions.get('rank_offset', 0) > 0:
                offset = unique_instructions['rank_offset']
                prompt_parts.append(f"⛔ ADDITIONAL RULE: Skip the first {offset} most obvious companies")
                prompt_parts.append(f"⛔ Focus on companies ranked {offset + 1} onwards in prominence")

        # Add exclusion list with STRONG enforcement
        if exclusion_list and len(exclusion_list) > 0:
            prompt_parts.append("\n" + "=" * 60)
            prompt_parts.append("🚫 EXCLUDED COMPANIES - DO NOT INCLUDE THESE:")
            prompt_parts.append("=" * 60)
            # Limit to prevent token overflow
            for company in exclusion_list[:50]:
                prompt_parts.append(f"  ❌ {company}")
            if len(exclusion_list) > 50:
                prompt_parts.append(f"  ... and {len(exclusion_list) - 50} more")
            prompt_parts.append("⛔ Including ANY of these companies will result in REJECTION")

        prompt_parts.append("\n" + "=" * 60)
        prompt_parts.append("SEARCH CRITERIA (with relaxation level " + str(relaxation_level) + "):")
        prompt_parts.append("=" * 60)

        # Apply relaxation to criteria (but NOT to segment boundaries)
        relaxed_criteria = self._apply_relaxation(criteria, relaxation_level)

        # Add the actual search criteria
        prompt_parts.append(f"Find {target_count} companies matching:")

        # Location (unless overridden by segment)
        if unique_instructions and unique_instructions.get('strategy') != 'GEOGRAPHIC_PRIMARY':
            if relaxed_criteria.location.countries:
                prompt_parts.append(f"Countries: {', '.join(relaxed_criteria.location.countries)}")

        # Financial criteria (relaxed)
        if relaxation_level < 2:
            if relaxed_criteria.financial.revenue_categories:
                prompt_parts.append(f"Revenue categories: {', '.join(relaxed_criteria.financial.revenue_categories)}")
        else:
            prompt_parts.append("Revenue: ANY (relaxed)")

        # Employee criteria (relaxed)
        if relaxation_level < 5:
            if relaxed_criteria.organizational.employee_count_min:
                prompt_parts.append(f"Employees: {relaxed_criteria.organizational.employee_count_min}+")
        else:
            prompt_parts.append("Employees: ANY (relaxed)")

        # Industries (unless overridden by segment)
        if unique_instructions and unique_instructions.get('strategy') != 'INDUSTRY_PRIMARY':
            if relaxation_level < 3 and relaxed_criteria.industries:
                ind_names = [ind['name'] for ind in relaxed_criteria.industries[:5]]
                prompt_parts.append(f"Industries: {', '.join(ind_names)}")
            elif relaxation_level >= 3:
                prompt_parts.append("Industries: ANY (relaxed)")

        # CSR criteria (relaxed first)
        if relaxation_level < 1 and relaxed_criteria.behavioral.csr_focus_areas:
            prompt_parts.append(f"CSR: {', '.join(relaxed_criteria.behavioral.csr_focus_areas)}")

        # CRITICAL: End with enforcement reminder
        prompt_parts.append("\n" + "=" * 60)
        prompt_parts.append("⚠️ FINAL REMINDER - YOU MUST:")
        prompt_parts.append("=" * 60)

        if unique_instructions:
            strategy = unique_instructions.get('strategy', 'unknown')
            if strategy == 'ALPHABET_PRIMARY':
                letter = unique_instructions['letter']
                prompt_parts.append(f"1. ONLY include companies starting with '{letter}'")
                prompt_parts.append(f"2. REJECT all companies NOT starting with '{letter}'")
            elif strategy == 'GEOGRAPHIC_PRIMARY':
                prompt_parts.append(
                    f"1. ONLY include companies in {', '.join(unique_instructions['assigned_locations'])}")
                prompt_parts.append(f"2. REJECT all companies from other locations")
            elif strategy == 'INDUSTRY_PRIMARY':
                prompt_parts.append(
                    f"1. ONLY include companies in {', '.join(unique_instructions['assigned_industries'])}")
                prompt_parts.append(f"2. REJECT all companies from other industries")

        if exclusion_list:
            prompt_parts.append(f"3. EXCLUDE all {len(exclusion_list)} companies in the exclusion list")

        prompt_parts.append(f"4. Return EXACTLY {target_count} companies if possible")
        prompt_parts.append("5. Return fewer companies rather than violate these rules")

        # JSON format
        prompt_parts.append("\nReturn JSON format:")
        prompt_parts.append("""{"companies":[{
"name":"Company Name",
"confidence":"high",
"operates_in_country":true,
"business_type":"B2B",
"industry_category":"Industry",
"reasoning":"Brief reason",
"estimated_revenue":"50-100M",
"revenue_category":"medium",
"estimated_employees":"100-500",
"headquarters":{"city":"City"},
"csr_focus_areas":[]
}]}""")

        return "\n".join(prompt_parts)

    def _apply_relaxation(self, criteria: SearchCriteria, level: int) -> SearchCriteria:
        """Apply relaxation to criteria while preserving segment boundaries"""
        import copy
        from shared.data_models import determine_revenue_categories_from_range

        relaxed = copy.deepcopy(criteria)

        if level >= 1:
            # Remove CSR requirements
            relaxed.behavioral.csr_focus_areas = []
            relaxed.behavioral.certifications = []
            relaxed.behavioral.recent_events = []

        if level >= 2:
            # Expand revenue categories
            relaxed.financial.revenue_categories = ["very_high", "high", "medium", "low", "very_low", "unknown"]

        if level >= 3:
            # Remove industry restrictions (but NOT if using industry segmentation)
            if self.segment_info and self.segment_info.get('strategy') != 'INDUSTRY_PRIMARY':
                relaxed.industries = []

        if level >= 4:
            # Expand geographic scope (but NOT if using geographic segmentation)
            if self.segment_info and self.segment_info.get('strategy') != 'GEOGRAPHIC_PRIMARY':
                if relaxed.location.cities:
                    relaxed.location.cities = []

        if level >= 5:
            # Lower employee minimums
            if relaxed.organizational.employee_count_min:
                relaxed.organizational.employee_count_min = max(1, relaxed.organizational.employee_count_min // 2)

        return relaxed

    def _validate_segment_compliance(
            self,
            companies: List[Dict[str, Any]],
            unique_instructions: Optional[Dict[str, Any]],
            criteria: SearchCriteria
    ) -> List[Dict[str, Any]]:
        """Validate and filter companies for segment compliance"""
        if not unique_instructions:
            return companies

        strategy = unique_instructions.get('strategy', 'unknown')
        compliant_companies = []

        for company in companies:
            company_name = company.get('name', '').strip()
            is_compliant = True

            if strategy == 'ALPHABET_PRIMARY':
                letter = unique_instructions['letter']
                # Check if company name starts with the assigned letter
                if not company_name.upper().startswith(letter.upper()):
                    logger.warning(f"Rejecting {company_name} - doesn't start with {letter}")
                    is_compliant = False

            elif strategy == 'GEOGRAPHIC_PRIMARY':
                locations = unique_instructions['assigned_locations']
                # Check if company location matches assigned locations
                hq = company.get('headquarters', {})
                if isinstance(hq, dict):
                    city = hq.get('city', '').lower()
                    # Simple check - could be enhanced
                    location_found = any(loc.lower() in city for loc in locations)
                    if not location_found:
                        logger.warning(f"Rejecting {company_name} - not in {locations}")
                        is_compliant = False

            elif strategy == 'INDUSTRY_PRIMARY':
                industries = unique_instructions['assigned_industries']
                company_industry = company.get('industry_category', '').lower()
                industry_found = any(ind.lower() in company_industry for ind in industries)
                if not industry_found:
                    logger.warning(f"Rejecting {company_name} - not in {industries}")
                    is_compliant = False

            if is_compliant:
                compliant_companies.append(company)

        logger.info(f"Segment compliance: {len(compliant_companies)}/{len(companies)} companies passed")
        return compliant_companies

    def _ensure_company_fields(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required fields exist in company data"""
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
            "validation_notes": None,
            "source_model": None,
            "search_segment": None,
            "relaxation_level": 0
        }

        for key, default_value in defaults.items():
            if key not in company_data:
                company_data[key] = default_value

        return company_data

    def _calculate_icp_score(self, company: EnhancedCompanyEntry, criteria: SearchCriteria) -> EnhancedCompanyEntry:
        """Calculate ICP score and tier for a company"""
        score = 0
        max_score = 100

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

        # Determine tier
        if score >= 80:
            tier = "A"
        elif score >= 60:
            tier = "B"
        elif score >= 40:
            tier = "C"
        else:
            tier = "D"

        company.icp_score = score
        company.icp_tier = tier

        return company


async def execute_parallel_search_with_recursion(
        models: List[str],
        criteria: SearchCriteria,
        target_count: int,
        enable_recursive: bool = True,
        progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Execute parallel search with recursive relaxation and STRONG deduplication"""

    logger.info(f"Starting parallel search for {target_count} companies with {len(models)} models")

    # Initialize progress tracker
    tracker = SearchProgressTracker()

    # Determine strategy
    strategy, strategy_description, strategy_params = determine_scale_strategy(
        len(models),
        target_count,
        criteria
    )

    logger.info(f"Using strategy: {strategy} - {strategy_description}")

    all_found_companies = []
    relaxation_level = 0
    max_relaxation = 5
    remaining_target = target_count

    while remaining_target > 0 and (not enable_recursive or relaxation_level <= max_relaxation):

        if relaxation_level > 0:
            logger.info(f"Applying relaxation level {relaxation_level}")

        # Calculate targets for this round
        base_per_model = remaining_target // len(models)

        # Create async tasks for each model
        async def search_with_model(model_idx: int, model: str) -> List[Any]:
            """Async function to search with a single model"""
            model_target = base_per_model + (1 if model_idx < (remaining_target % len(models)) else 0)

            logger.info(f"Model {model} starting search for {model_target} companies")

            # Generate unique instructions that PERSIST through relaxation
            unique_instructions = generate_scale_instructions(
                model_idx,
                len(models),
                criteria,
                strategy,
                strategy_params,
                model_target,
                1  # call number
            )

            # Add relaxation info to instructions
            unique_instructions['relaxation_level'] = relaxation_level
            unique_instructions['enforce_segment'] = True

            # Get exclusion list from tracker
            exclusion_list = tracker.get_exclusion_list(max_items=100)

            try:
                agent = EnhancedSearchStrategistAgent(deployment_name=model)

                result = await agent.generate_enhanced_strategy(
                    criteria,
                    target_count=model_target,
                    unique_instructions=unique_instructions,
                    exclusion_list=exclusion_list,
                    relaxation_level=relaxation_level
                )

                companies = result.get('companies', [])

                # Add to tracker and get unique companies
                unique_companies = tracker.add_companies(model, companies)

                logger.info(
                    f"Model {model} found {len(unique_companies)} unique companies ({len(companies) - len(unique_companies)} duplicates)")

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(tracker.get_progress_summary())

                return unique_companies

            except Exception as e:
                logger.error(f"Model {model} failed: {str(e)}")
                return []

        # Execute all models in parallel
        tasks = [search_with_model(idx, model) for idx, model in enumerate(models)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Collect results from this round
        round_companies = []
        for company_list in results:
            round_companies.extend(company_list)

        all_found_companies.extend(round_companies)
        remaining_target = target_count - len(all_found_companies)

        logger.info(
            f"Round {relaxation_level + 1} complete: Found {len(round_companies)} companies, Total: {len(all_found_companies)}/{target_count}")

        # Check if we should continue
        if not enable_recursive or remaining_target <= 0:
            break

        # Check if this round was unsuccessful enough to warrant relaxation
        if len(round_companies) < remaining_target * 0.2:  # Less than 20% of what we need
            relaxation_level += 1
        else:
            # Try same level again with updated exclusions
            pass

    # Final summary
    summary = tracker.get_progress_summary()
    logger.info(f"Search complete: Found {len(all_found_companies)} total companies")
    logger.info(f"Model performance: {summary['models']}")

    return {
        'companies': all_found_companies[:target_count],  # Trim to exact target
        'metadata': {
            'parallel_execution': True,
            'strategy': strategy,
            'models_used': models,
            'relaxation_levels_used': relaxation_level,
            'total_found': len(all_found_companies),
            'duplicates_prevented': summary['total_found'] - len(all_found_companies),
            'execution_time': summary['elapsed_time'],
            'model_performance': summary['models']
        }
    }


def determine_scale_strategy(
        num_models: int,
        target_count: int,
        criteria: SearchCriteria
) -> Tuple[str, str, Dict[str, Any]]:
    """Determine the best strategy for scale searches with hard segmentation"""

    strategy_params = {
        'max_per_call': 15,
        'enforce_boundaries': True,  # NEW: Enforce hard boundaries
        'use_exclusions': True  # NEW: Use exclusion lists
    }

    # Calculate calls needed
    strategy_params['total_calls_needed'] = (target_count + 14) // 15
    strategy_params['calls_per_model'] = (strategy_params['total_calls_needed'] + num_models - 1) // num_models

    # For any significant search, use alphabet as primary
    # It's the most reliable for hard boundaries
    if target_count >= 100:
        return (
            "ALPHABET_PRIMARY",
            f"Alphabet segmentation for {target_count} companies with hard boundaries",
            strategy_params
        )
    else:
        # For smaller searches, can use geographic or industry if applicable
        if criteria.location.cities and len(criteria.location.cities) >= num_models:
            return (
                "GEOGRAPHIC_PRIMARY",
                f"Geographic segmentation across {len(criteria.location.cities)} cities",
                strategy_params
            )
        elif criteria.industries and len(criteria.industries) >= num_models:
            return (
                "INDUSTRY_PRIMARY",
                f"Industry segmentation across {len(criteria.industries)} industries",
                strategy_params
            )
        else:
            return (
                "ALPHABET_PRIMARY",
                "Alphabet segmentation for complete coverage",
                strategy_params
            )


def generate_scale_instructions(
        model_index: int,
        total_models: int,
        criteria: SearchCriteria,
        strategy: str,
        strategy_params: Dict[str, Any],
        target_per_model: int,
        call_number: int = 1
) -> Dict[str, Any]:
    """Generate unique instructions with HARD boundaries for scale searches"""

    instructions = {
        'agent_number': model_index + 1,
        'total_agents': total_models,
        'strategy': strategy,
        'segment_id': f"agent_{model_index + 1}_call_{call_number}",
        'call_number': call_number,
        'enforce_boundaries': True,  # NEW: Always enforce
        'total_calls': strategy_params.get('calls_per_model', 1)
    }

    if strategy == "ALPHABET_PRIMARY":
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # Divide alphabet evenly among models
        letters_per_model = len(alphabet) / total_models
        start_idx = int(model_index * letters_per_model)
        end_idx = int((model_index + 1) * letters_per_model) - 1

        if model_index == total_models - 1:
            end_idx = len(alphabet) - 1

        if start_idx == end_idx:
            instructions['letter'] = alphabet[start_idx]
            instructions['segment_description'] = f"ONLY companies starting with '{alphabet[start_idx]}'"
        else:
            # For range, still assign specific letters
            assigned_letters = alphabet[start_idx:end_idx + 1]
            instructions['letter'] = assigned_letters[call_number % len(assigned_letters)]
            instructions['letter_range'] = f"{alphabet[start_idx]}-{alphabet[end_idx]}"
            instructions[
                'segment_description'] = f"ONLY companies starting with '{instructions['letter']}' (from range {instructions['letter_range']})"

        # Add sub-segment for large searches
        if target_per_model > 50:
            sub_segments = ['large', 'medium', 'small']
            instructions['sub_segment'] = sub_segments[(call_number - 1) % len(sub_segments)]

    elif strategy == "GEOGRAPHIC_PRIMARY":
        locations = criteria.location.cities if criteria.location.cities else criteria.location.countries

        locs_per_model = max(1, len(locations) // total_models)
        start_idx = model_index * locs_per_model
        end_idx = start_idx + locs_per_model

        if model_index == total_models - 1:
            end_idx = len(locations)

        instructions['assigned_locations'] = locations[start_idx:end_idx]
        instructions['rank_offset'] = (call_number - 1) * 50
        instructions['segment_description'] = f"ONLY companies in {', '.join(instructions['assigned_locations'])}"

    elif strategy == "INDUSTRY_PRIMARY":
        industries = [ind['name'] for ind in criteria.industries]

        ind_per_model = max(1, len(industries) // total_models)
        start_idx = model_index * ind_per_model
        end_idx = start_idx + ind_per_model

        if model_index == total_models - 1:
            end_idx = len(industries)

        instructions['assigned_industries'] = industries[start_idx:end_idx]
        instructions['rank_offset'] = (call_number - 1) * 50
        instructions['segment_description'] = f"ONLY companies in {', '.join(instructions['assigned_industries'])}"

    return instructions


# Wrapper function for compatibility
async def execute_parallel_search(
        models: List[str],
        criteria: SearchCriteria,
        target_count: int,
        serper_key: Optional[str] = None,
        enable_recursive: bool = True,
        progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Wrapper for backward compatibility"""
    return await execute_parallel_search_with_recursion(
        models,
        criteria,
        target_count,
        enable_recursive,
        progress_callback
    )