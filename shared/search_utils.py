# shared/search_utils.py
"""
Search utilities with deduplication strategy optimized for LARGE SCALE (10K+) searches
"""

import asyncio
import json
import time
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict
from shared.data_models import SearchCriteria, EnhancedCompanyEntry


class EnhancedSearchStrategistAgent:
    """Enhanced agent with comprehensive search capabilities and deduplication strategy"""

    def __init__(self, deployment_name: str = "gpt-4.1"):
        self.deployment_name = deployment_name
        self.client = None
        self.initialized = False

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
            print(f"Successfully initialized Azure OpenAI client with deployment: {self.deployment_name}")

        except Exception as e:
            print(f"Error initializing Azure OpenAI: {str(e)}")
            raise

    async def generate_enhanced_strategy(
            self,
            criteria: SearchCriteria,
            target_count: int = 100,
            unique_instructions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate search strategy with unique instructions to avoid duplicates"""
        if not self.client:
            self._init_llm()

        # Build prompt with unique instructions if provided
        prompt = self._build_enhanced_prompt_with_uniqueness(
            criteria,
            target_count,
            unique_instructions
        )

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system",
                     "content": "You are a company finder. Respond with valid JSON only. Follow all uniqueness instructions EXACTLY to avoid duplicating companies found by other agents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=16384,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            enhanced_companies = []
            for company_data in result.get("companies", []):
                try:
                    company_data = self._ensure_company_fields(company_data)
                    # Add source model and uniqueness tracking
                    company_data['source_model'] = self.deployment_name
                    if unique_instructions:
                        company_data['search_segment'] = unique_instructions.get('segment_id', 'general')

                    company = EnhancedCompanyEntry(**company_data)
                    company = self._calculate_icp_score(company, criteria)
                    enhanced_companies.append(company)
                except Exception as e:
                    print(f"Error processing company: {e}")
                    continue

            return {
                "companies": enhanced_companies,
                "search_criteria": asdict(criteria) if hasattr(criteria, '__dict__') else criteria,
                "metadata": {
                    "total_found": len(enhanced_companies),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "deployment": self.deployment_name,
                    "unique_instructions": unique_instructions,
                    "batch_execution": target_count > 20
                }
            }

        except Exception as e:
            print(f"Search error in {self.deployment_name}: {e}")
            return {"companies": [], "error": str(e), "deployment": self.deployment_name}

    def _build_enhanced_prompt_with_uniqueness(
            self,
            criteria: SearchCriteria,
            target_count: int,
            unique_instructions: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt with unique instructions to avoid duplicates"""
        prompt_parts = []

        # Add uniqueness instructions FIRST and PROMINENTLY
        if unique_instructions:
            prompt_parts.append("=" * 50)
            prompt_parts.append("⚠️ CRITICAL DEDUPLICATION INSTRUCTIONS - MUST FOLLOW:")
            prompt_parts.append("=" * 50)

            strategy = unique_instructions.get('strategy', 'unknown')

            if strategy == 'ALPHABET_PRIMARY':
                # Alphabet with sub-segments for large scale
                letter = unique_instructions['letter']
                sub_segment = unique_instructions.get('sub_segment', 'all')

                prompt_parts.append(f"🔴 MANDATORY: ONLY find companies whose names START with letter {letter}")

                if sub_segment != 'all':
                    prompt_parts.append(f"🔴 ADDITIONAL FILTER: Focus on {sub_segment} within letter {letter}")
                    if sub_segment == 'large':
                        prompt_parts.append("   - Prioritize companies with >$100M revenue or >1000 employees")
                    elif sub_segment == 'medium':
                        prompt_parts.append("   - Prioritize companies with $10-100M revenue or 100-1000 employees")
                    elif sub_segment == 'small':
                        prompt_parts.append("   - Prioritize companies with <$10M revenue or <100 employees")
                    elif sub_segment == 'established':
                        prompt_parts.append("   - Prioritize companies established >10 years ago")
                    elif sub_segment == 'emerging':
                        prompt_parts.append("   - Prioritize newer companies established <10 years ago")

                prompt_parts.append(f"🔴 REJECT ANY company not starting with {letter}")
                prompt_parts.append(f"🔴 You are responsible for finding ALL {letter} companies")

            elif strategy == 'GEOGRAPHIC_PRIMARY':
                # Geographic with rank offset for scale
                locations = unique_instructions['assigned_locations']
                rank_offset = unique_instructions.get('rank_offset', 0)

                prompt_parts.append(f"🔴 MANDATORY: ONLY find companies in: {', '.join(locations)}")
                if rank_offset > 0:
                    prompt_parts.append(f"🔴 Skip the first {rank_offset} most obvious companies")
                    prompt_parts.append(f"🔴 Focus on companies ranked {rank_offset + 1} onwards")
                prompt_parts.append(f"🔴 EXCLUDE companies from any other locations")

            elif strategy == 'INDUSTRY_PRIMARY':
                # Industry with rank offset for scale
                industries = unique_instructions['assigned_industries']
                rank_offset = unique_instructions.get('rank_offset', 0)

                prompt_parts.append(f"🔴 MANDATORY: ONLY find companies in: {', '.join(industries)}")
                if rank_offset > 0:
                    prompt_parts.append(f"🔴 Skip the first {rank_offset} most prominent companies")
                    prompt_parts.append(f"🔴 Focus on lesser-known players ranked {rank_offset + 1} onwards")
                prompt_parts.append(f"🔴 EXCLUDE companies from other industries")

            elif strategy == 'RANK_DISCOVERY':
                # Pure rank-based for deep discovery
                rank_start = unique_instructions['rank_start']
                rank_end = unique_instructions['rank_end']
                tier_description = unique_instructions.get('tier_description', '')

                prompt_parts.append(f"🔴 MANDATORY: Find companies ranked {rank_start} to {rank_end}")
                prompt_parts.append(f"🔴 Skip the top {rank_start - 1} most obvious/prominent companies")
                prompt_parts.append(f"🔴 Focus on: {tier_description}")
                prompt_parts.append(f"🔴 Avoid well-known market leaders")

            elif strategy == 'HYBRID_SCALE':
                # Hybrid approach for massive scale searches
                primary = unique_instructions.get('primary_dimension')
                secondary = unique_instructions.get('secondary_dimension')

                if primary == 'alphabet':
                    letters = unique_instructions['letter_range']
                    prompt_parts.append(f"🔴 PRIMARY: Companies starting with {letters}")

                if secondary == 'industry_focus':
                    industries = unique_instructions['industry_subset']
                    prompt_parts.append(f"🔴 SECONDARY: Prioritize these industries: {', '.join(industries)}")
                elif secondary == 'size_focus':
                    size = unique_instructions['size_subset']
                    prompt_parts.append(f"🔴 SECONDARY: Focus on {size} companies")
                elif secondary == 'location_focus':
                    locs = unique_instructions['location_subset']
                    prompt_parts.append(f"🔴 SECONDARY: Prioritize companies in: {', '.join(locs)}")

            # Add call number for large batches
            if unique_instructions.get('call_number'):
                call_num = unique_instructions['call_number']
                total_calls = unique_instructions.get('total_calls', 1)
                prompt_parts.append(f"🔴 BATCH: This is call {call_num} of {total_calls} for this segment")
                prompt_parts.append(f"🔴 Find DIFFERENT companies than previous calls")

            prompt_parts.append("=" * 50)
            prompt_parts.append("")

        # Now add the base search criteria
        prompt_parts.append(f"Find {target_count} companies matching these criteria:")

        # Location
        if criteria.location.countries or criteria.location.cities:
            locations = []
            if criteria.location.countries:
                locations.append(f"Countries: {', '.join(criteria.location.countries[:3])}")
            if criteria.location.cities:
                locations.append(f"Cities: {', '.join(criteria.location.cities[:5])}")
            prompt_parts.append("LOCATION: " + "; ".join(locations))

        # Financial
        if criteria.financial.revenue_min or criteria.financial.revenue_max:
            if criteria.financial.revenue_min and criteria.financial.revenue_max:
                prompt_parts.append(
                    f"REVENUE: ${int(criteria.financial.revenue_min / 1e6)}-{int(criteria.financial.revenue_max / 1e6)}M {criteria.financial.revenue_currency}")
            elif criteria.financial.revenue_min:
                prompt_parts.append(
                    f"REVENUE: >${int(criteria.financial.revenue_min / 1e6)}M {criteria.financial.revenue_currency}")

        # Employees
        if criteria.organizational.employee_count_min:
            prompt_parts.append(f"EMPLOYEES: {criteria.organizational.employee_count_min}+")

        # Industries - respect unique instructions if they override
        if criteria.industries and not (
                unique_instructions and unique_instructions.get('strategy') == 'INDUSTRY_PRIMARY'):
            ind_names = [ind['name'] for ind in criteria.industries[:5]]
            prompt_parts.append(f"INDUSTRIES: {', '.join(ind_names)}")

        # Business types
        if criteria.business_types:
            prompt_parts.append(f"TYPES: {', '.join(criteria.business_types[:3])}")

        # CSR
        if criteria.behavioral.csr_focus_areas:
            prompt_parts.append(f"CSR: {', '.join(criteria.behavioral.csr_focus_areas[:3])}")

        # Final reminder about uniqueness
        if unique_instructions:
            prompt_parts.append("\n⚠️ REMEMBER: You MUST follow the deduplication instructions above!")
            prompt_parts.append(
                f"⚠️ You are Agent {unique_instructions.get('agent_number', 'X')} of {unique_instructions.get('total_agents', 'Y')}")
            prompt_parts.append(
                f"⚠️ Your specific role: {unique_instructions.get('segment_description', 'Follow instructions above')}")

        # JSON format
        prompt_parts.append("\nReturn JSON:")
        prompt_parts.append("""{"companies":[{
"name":"Company Name",
"confidence":"high",
"operates_in_country":true,
"business_type":"B2B",
"industry_category":"Industry",
"reasoning":"Brief reason",
"estimated_revenue":"50-100M",
"estimated_employees":"100-500",
"headquarters":{"city":"City"},
"csr_focus_areas":[]
}]}""")

        return "\n".join(prompt_parts)

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
            "search_segment": None
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


def determine_scale_strategy(
        num_models: int,
        target_count: int,
        criteria: SearchCriteria
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Determine the best strategy for LARGE SCALE searches (10K+ companies)
    Returns: (strategy_name, strategy_description, strategy_params)
    """

    # Analyze the search criteria
    has_multiple_locations = len(criteria.location.cities) > 1 or len(criteria.location.countries) > 1
    has_multiple_industries = len(criteria.industries) > 1
    num_locations = len(criteria.location.cities) if criteria.location.cities else len(criteria.location.countries)
    num_industries = len(criteria.industries)

    strategy_params = {
        'max_per_call': 50,  # GPT limit per call
        'calls_per_model': 1,  # Will be calculated
        'total_calls_needed': 1,  # Will be calculated
    }

    # Calculate calls needed
    strategy_params['total_calls_needed'] = (target_count + 49) // 50  # Round up
    strategy_params['calls_per_model'] = (strategy_params['total_calls_needed'] + num_models - 1) // num_models

    # Decision tree for LARGE SCALE searches
    if target_count >= 5000:
        # For massive searches, use hybrid approach
        return (
            "HYBRID_SCALE",
            f"Hybrid strategy for {target_count} companies using alphabet + secondary dimensions",
            strategy_params
        )

    elif target_count >= 1000:
        # For large searches, use alphabet with sub-segmentation
        return (
            "ALPHABET_PRIMARY",
            f"Alphabet segmentation with size/age sub-segments for {target_count} companies",
            strategy_params
        )

    elif has_multiple_locations and num_locations >= num_models * 2:
        # Geographic with rank offset for each location
        return (
            "GEOGRAPHIC_PRIMARY",
            f"Geographic segmentation across {num_locations} locations with rank offsets",
            strategy_params
        )

    elif has_multiple_industries and num_industries >= num_models * 2:
        # Industry with rank offset for each industry
        return (
            "INDUSTRY_PRIMARY",
            f"Industry segmentation across {num_industries} industries with rank offsets",
            strategy_params
        )

    elif target_count >= 500:
        # Rank-based discovery for medium-large searches
        return (
            "RANK_DISCOVERY",
            f"Rank-based discovery across {target_count} companies in tiers",
            strategy_params
        )

    else:
        # Default to simple alphabet for smaller searches
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
    """
    Generate unique instructions for LARGE SCALE searches
    Handles multiple calls per model for 10K+ searches
    """

    instructions = {
        'agent_number': model_index + 1,
        'total_agents': total_models,
        'strategy': strategy,
        'segment_id': f"agent_{model_index + 1}_call_{call_number}",
        'call_number': call_number,
        'total_calls': strategy_params['calls_per_model']
    }

    if strategy == "ALPHABET_PRIMARY":
        # For large scale, each model gets one or more letters
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        if target_per_model <= 500:
            # Simple letter assignment
            letters_per_model = len(alphabet) / total_models
            start_idx = int(model_index * letters_per_model)
            end_idx = int((model_index + 1) * letters_per_model) - 1
            if model_index == total_models - 1:
                end_idx = len(alphabet) - 1

            instructions['letter'] = alphabet[
                start_idx] if start_idx == end_idx else f"{alphabet[start_idx]}-{alphabet[end_idx]}"
            instructions['sub_segment'] = 'all'

        else:
            # For very large searches, use letter + sub-segment
            # Each model gets fewer letters but searches deeper
            letters_per_model = max(1, len(alphabet) // (total_models * 2))
            letter_index = (model_index * letters_per_model + (call_number - 1)) % len(alphabet)
            instructions['letter'] = alphabet[letter_index]

            # Rotate through sub-segments for different calls
            sub_segments = ['large', 'medium', 'small', 'established', 'emerging']
            instructions['sub_segment'] = sub_segments[(call_number - 1) % len(sub_segments)]

        instructions[
            'segment_description'] = f"Find companies starting with {instructions['letter']} ({instructions['sub_segment']})"

    elif strategy == "HYBRID_SCALE":
        # For 5K-10K searches, combine multiple dimensions
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # Primary: Alphabet ranges
        letters_per_model = 5  # Each model gets 5-6 letters
        start_idx = model_index * letters_per_model
        end_idx = min(start_idx + letters_per_model - 1, len(alphabet) - 1)

        instructions['primary_dimension'] = 'alphabet'
        instructions['letter_range'] = f"{alphabet[start_idx]}-{alphabet[end_idx]}"

        # Secondary: Rotate through other dimensions based on call number
        secondary_options = []

        if criteria.industries and len(criteria.industries) > 1:
            secondary_options.append(('industry_focus', criteria.industries))

        if criteria.location.cities and len(criteria.location.cities) > 1:
            secondary_options.append(('location_focus', criteria.location.cities))

        secondary_options.append(('size_focus', ['large', 'medium', 'small']))

        if secondary_options:
            secondary_idx = (call_number - 1) % len(secondary_options)
            secondary_type, secondary_values = secondary_options[secondary_idx]
            instructions['secondary_dimension'] = secondary_type

            if secondary_type == 'industry_focus':
                # Assign subset of industries
                ind_per_call = max(1, len(secondary_values) // strategy_params['calls_per_model'])
                start = (call_number - 1) * ind_per_call
                end = min(start + ind_per_call, len(secondary_values))
                instructions['industry_subset'] = [ind['name'] for ind in secondary_values[start:end]]

            elif secondary_type == 'location_focus':
                # Assign subset of locations
                loc_per_call = max(1, len(secondary_values) // strategy_params['calls_per_model'])
                start = (call_number - 1) * loc_per_call
                end = min(start + loc_per_call, len(secondary_values))
                instructions['location_subset'] = secondary_values[start:end]

            elif secondary_type == 'size_focus':
                instructions['size_subset'] = secondary_values[(call_number - 1) % len(secondary_values)]

        instructions[
            'segment_description'] = f"Companies {instructions['letter_range']} with {instructions.get('secondary_dimension', 'all')} focus"

    elif strategy == "RANK_DISCOVERY":
        # For rank-based discovery at scale
        total_ranks = target_per_model * total_models
        ranks_per_segment = total_ranks // (total_models * strategy_params['calls_per_model'])

        base_rank = model_index * target_per_model
        call_offset = (call_number - 1) * ranks_per_segment

        instructions['rank_start'] = base_rank + call_offset + 1
        instructions['rank_end'] = base_rank + call_offset + ranks_per_segment

        # Define tier descriptions
        tier_descriptions = [
            "industry leaders and well-known companies",
            "established mid-market players",
            "growing companies with regional presence",
            "emerging businesses and startups",
            "niche players and specialized firms",
            "local businesses and lesser-known entities"
        ]

        tier_index = min((instructions['rank_start'] - 1) // 500, len(tier_descriptions) - 1)
        instructions['tier_description'] = tier_descriptions[tier_index]
        instructions[
            'segment_description'] = f"Companies ranked {instructions['rank_start']}-{instructions['rank_end']} ({instructions['tier_description']})"

    elif strategy == "GEOGRAPHIC_PRIMARY":
        # Geographic with rank offsets for scale
        if criteria.location.cities:
            locations = criteria.location.cities
        else:
            locations = criteria.location.countries

        locs_per_model = max(1, len(locations) // total_models)
        start_idx = model_index * locs_per_model
        end_idx = start_idx + locs_per_model

        if model_index == total_models - 1:
            end_idx = len(locations)

        instructions['assigned_locations'] = locations[start_idx:end_idx]
        instructions['rank_offset'] = (call_number - 1) * 100  # Each call goes 100 ranks deeper
        instructions[
            'segment_description'] = f"Companies in {', '.join(instructions['assigned_locations'][:3])} (rank offset: {instructions['rank_offset']})"

    elif strategy == "INDUSTRY_PRIMARY":
        # Industry with rank offsets for scale
        if criteria.industries:
            industries = [ind['name'] for ind in criteria.industries]

            ind_per_model = max(1, len(industries) // total_models)
            start_idx = model_index * ind_per_model
            end_idx = start_idx + ind_per_model

            if model_index == total_models - 1:
                end_idx = len(industries)

            instructions['assigned_industries'] = industries[start_idx:end_idx]
            instructions['rank_offset'] = (call_number - 1) * 100  # Each call goes 100 ranks deeper
            instructions[
                'segment_description'] = f"Companies in {', '.join(instructions['assigned_industries'][:3])} (rank offset: {instructions['rank_offset']})"

    return instructions


async def execute_parallel_search(
        models: List[str],
        criteria: SearchCriteria,
        target_count: int
) -> Dict[str, Any]:
    """Execute search across multiple models optimized for LARGE SCALE (10K+)"""

    print(f"\n{'=' * 60}")
    print(f"LARGE SCALE PARALLEL SEARCH")
    print(f"{'=' * 60}")
    print(f"Target: {target_count} companies across {len(models)} models")

    # Determine strategy for large scale
    strategy, strategy_description, strategy_params = determine_scale_strategy(
        len(models),
        target_count,
        criteria
    )

    print(f"Strategy: {strategy}")
    print(f"Description: {strategy_description}")
    print(f"Total API calls needed: {strategy_params['total_calls_needed']}")
    print(f"Calls per model: {strategy_params['calls_per_model']}")
    print(f"{'=' * 60}\n")

    # Calculate targets
    max_per_call = strategy_params['max_per_call']  # 50 companies per GPT call
    calls_per_model = strategy_params['calls_per_model']
    base_per_model = target_count // len(models)

    # Execute searches
    all_companies = []
    model_stats = {}

    for model_idx, model in enumerate(models):
        model_companies = []
        model_target = base_per_model + (1 if model_idx < (target_count % len(models)) else 0)

        print(f"\nAgent {model_idx + 1} ({model}):")
        print(f"  Total target: {model_target} companies")
        print(f"  Will make {calls_per_model} calls")

        # Make multiple calls per model for large searches
        for call_num in range(1, min(calls_per_model + 1, (model_target + max_per_call - 1) // max_per_call + 1)):
            # Calculate this call's target
            remaining = model_target - len(model_companies)
            call_target = min(max_per_call, remaining)

            if call_target <= 0:
                break

            # Generate unique instructions for this call
            unique_instructions = generate_scale_instructions(
                model_idx,
                len(models),
                criteria,
                strategy,
                strategy_params,
                model_target,
                call_num
            )

            print(f"  Call {call_num}: Requesting {call_target} companies")
            print(f"    Segment: {unique_instructions.get('segment_description', 'Not specified')}")

            # Create agent and execute
            agent = EnhancedSearchStrategistAgent(deployment_name=model)

            try:
                result = await agent.generate_enhanced_strategy(
                    criteria,
                    target_count=call_target,
                    unique_instructions=unique_instructions
                )

                companies = result.get('companies', [])
                model_companies.extend(companies)
                print(f"    Found: {len(companies)} companies")

                # Rate limiting between calls
                if call_num < calls_per_model:
                    await asyncio.sleep(0.5)

            except Exception as e:
                print(f"    Error: {str(e)}")

        all_companies.extend(model_companies)
        model_stats[model] = {
            'target': model_target,
            'found': len(model_companies),
            'calls_made': call_num,
            'success_rate': (len(model_companies) / model_target * 100) if model_target > 0 else 0
        }

    # Deduplication
    print(f"\n{'=' * 60}")
    print("Deduplication Phase")
    print(f"{'=' * 60}")

    seen_names = set()
    seen_name_cores = set()
    unique_companies = []

    for company in all_companies:
        company_name = company.name.lower().strip()

        # Create core name for deduplication
        name_core = company_name
        for suffix in ['pty ltd', 'limited', 'ltd', 'inc', 'corporation', 'corp', 'llc', 'plc',
                       '& co', 'and company', 'group', 'holdings', 'international', 'global']:
            name_core = name_core.replace(suffix, '').strip()

        if name_core not in seen_name_cores and company_name not in seen_names:
            seen_names.add(company_name)
            seen_name_cores.add(name_core)
            unique_companies.append(company)

    # Trim to exact target if over
    if len(unique_companies) > target_count:
        unique_companies = unique_companies[:target_count]

    # Calculate final statistics
    total_found = len(all_companies)
    total_unique = len(unique_companies)
    duplicates_removed = total_found - total_unique
    dedup_rate = (duplicates_removed / total_found * 100) if total_found > 0 else 0
    achievement_rate = (total_unique / target_count * 100) if target_count > 0 else 0

    print(f"Total found: {total_found}")
    print(f"Unique companies: {total_unique}")
    print(f"Duplicates removed: {duplicates_removed} ({dedup_rate:.1f}%)")
    print(f"Target achievement: {total_unique}/{target_count} ({achievement_rate:.1f}%)")

    print(f"\n{'=' * 60}")
    print("Model Performance Summary")
    print(f"{'=' * 60}")
    for model, stats in model_stats.items():
        print(f"{model}:")
        print(f"  Target: {stats['target']}")
        print(f"  Found: {stats['found']}")
        print(f"  Calls: {stats['calls_made']}")
        print(f"  Success: {stats['success_rate']:.1f}%")
    print(f"{'=' * 60}\n")

    return {
        'companies': unique_companies,
        'metadata': {
            'parallel_execution': True,
            'scale_strategy': strategy,
            'strategy_description': strategy_description,
            'models_used': models,
            'model_stats': model_stats,
            'total_companies_before_dedup': total_found,
            'total_companies_after_dedup': total_unique,
            'duplicates_removed': duplicates_removed,
            'deduplication_rate': dedup_rate,
            'target_count': target_count,
            'target_achievement_rate': achievement_rate,
            'total_api_calls': sum(s['calls_made'] for s in model_stats.values()),
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S")
        }
    }