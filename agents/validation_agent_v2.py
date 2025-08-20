# validation_agent_v2.py
"""
Enhanced Validation Agent V2
Orchestrates the validation process using strategies and analyzers
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

# Use absolute imports when this is used as a module
try:
    # Try relative imports first (when used as part of package)
    from core.serper_client import EnhancedSerperClient, SerperEndpoint
    from core.validation_strategies import (
        ValidationStrategyBuilder,
        ValidationCriteria,
        ValidationStrategy,
        ValidationTier,
        ValidationQueryOptimizer
    )
    from core.validation_analyzer import (
        ValidationAnalyzer,
        ComprehensiveValidation,
        LocationValidation,
        FinancialValidation,
        CSRValidation,
        ReputationValidation
    )
except ImportError:
    # Fall back to absolute imports (when running directly)
    from core.serper_client import EnhancedSerperClient, SerperEndpoint
    from core.validation_strategies import (
        ValidationStrategyBuilder,
        ValidationCriteria,
        ValidationStrategy,
        ValidationTier,
        ValidationQueryOptimizer
    )
    from core.validation_analyzer import (
        ValidationAnalyzer,
        ComprehensiveValidation,
        LocationValidation,
        FinancialValidation,
        CSRValidation,
        ReputationValidation
    )

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation process"""
    serper_api_key: str
    max_parallel_queries: int = 5
    rate_limit_per_second: float = 10
    timeout: int = 30
    max_retries: int = 3
    cache_duration_hours: int = 24

    # Validation thresholds
    min_confidence_for_quick: float = 0.8
    min_score_for_approval: float = 60

    # Cost controls
    max_cost_per_company: float = 0.01  # $0.01
    max_total_cost: float = 10.0  # $10

    # Feature flags
    enable_caching: bool = True
    enable_batch_optimization: bool = True
    enable_smart_routing: bool = True


class EnhancedValidationAgent:
    """
    Enhanced validation agent that orchestrates the validation process
    using strategies, Serper client, and analyzers
    """

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.serper_client = None
        self.strategy_builder = ValidationStrategyBuilder()
        self.analyzer = ValidationAnalyzer()
        self.query_optimizer = ValidationQueryOptimizer()

        # Tracking
        self.total_validations = 0
        self.total_cost = 0.0
        self.validation_cache = {}
        self.start_time = datetime.now()

    async def __aenter__(self):
        """Async context manager entry"""
        self.serper_client = EnhancedSerperClient(
            api_key=self.config.serper_api_key,
            rate_limit=self.config.rate_limit_per_second,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
        await self.serper_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.serper_client:
            await self.serper_client.__aexit__(exc_type, exc_val, exc_tb)

    def _check_cache(self, company_name: str, location: str) -> Optional[ComprehensiveValidation]:
        """Check if we have cached validation for this company"""
        if not self.config.enable_caching:
            return None

        cache_key = f"{company_name.lower()}:{location.lower()}"
        if cache_key in self.validation_cache:
            cached = self.validation_cache[cache_key]
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cached['timestamp'])
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600

            if age_hours < self.config.cache_duration_hours:
                logger.info(f"Using cached validation for {company_name}")
                return cached['validation']

        return None

    def _update_cache(self, company_name: str, location: str, validation: ComprehensiveValidation):
        """Update validation cache"""
        if not self.config.enable_caching:
            return

        cache_key = f"{company_name.lower()}:{location.lower()}"
        self.validation_cache[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'validation': validation
        }

    async def validate_company(
            self,
            company_data: Dict[str, Any],
            criteria: ValidationCriteria,
            location: str,
            country: str,
            force_tier: Optional[ValidationTier] = None
    ) -> ComprehensiveValidation:
        """
        Validate a single company

        Args:
            company_data: Company information dictionary
            criteria: Validation criteria
            location: Target location
            country: Target country
            force_tier: Force a specific validation tier

        Returns:
            ComprehensiveValidation object with results
        """
        company_name = company_data.get("name", "")

        # Check cache first
        cached = self._check_cache(company_name, location)
        if cached:
            return cached

        start_time = time.time()

        # Build validation strategy
        strategy = self.strategy_builder.build_strategy(
            company_data,
            criteria,
            location,
            country
        )

        # Override tier if requested
        if force_tier:
            strategy.tier = force_tier
            strategy.queries = self.strategy_builder._build_queries_for_tier(
                strategy, company_data, criteria, location, country
            )
            strategy.calculate_cost()

        # Check cost limits
        if strategy.estimated_cost > self.config.max_cost_per_company:
            logger.warning(f"Validation cost ${strategy.estimated_cost} exceeds limit for {company_name}")
            # Downgrade to cheaper tier
            strategy.tier = ValidationTier.STANDARD
            strategy.queries = self.strategy_builder._build_queries_for_tier(
                strategy, company_data, criteria, location, country
            )[:3]  # Limit to 3 queries

        # Execute queries
        responses = await self.serper_client.batch_search(
            strategy.queries,
            parallel_limit=self.config.max_parallel_queries
        )

        # Initialize validation result
        validation = ComprehensiveValidation(
            company_name=company_name,
            validation_timestamp=datetime.now().isoformat(),
            location=LocationValidation(),
            financial=FinancialValidation(),
            csr=CSRValidation(),
            reputation=ReputationValidation(),
            serper_queries_used=len(responses)
        )

        # Analyze responses
        for response in responses:
            if response.endpoint == SerperEndpoint.PLACES:
                validation.location = self.analyzer.analyze_places_response(
                    response, company_name, location
                )

            elif response.endpoint == SerperEndpoint.MAPS:
                # Additional location data
                location_data = self.analyzer.analyze_places_response(
                    response, company_name, location
                )
                if location_data.offices:
                    validation.location.offices.extend(location_data.offices)

            elif response.endpoint == SerperEndpoint.SEARCH:
                # Check if this is financial or CSR search
                if any(term in response.query.lower() for term in ["revenue", "employee", "financial"]):
                    financial, _ = self.analyzer.analyze_web_response(response, company_name)
                    validation.financial = financial
                elif any(term in response.query.lower() for term in ["csr", "sustainability", "social"]):
                    _, csr = self.analyzer.analyze_web_response(response, company_name)
                    validation.csr = csr
                else:
                    # General search - extract both
                    financial, csr = self.analyzer.analyze_web_response(response, company_name)
                    if financial.evidence:
                        validation.financial = financial
                    if csr.evidence:
                        validation.csr = csr

            elif response.endpoint == SerperEndpoint.NEWS:
                validation.reputation = self.analyzer.analyze_news_response(
                    response, company_name
                )

        # Count total evidence
        validation.total_evidence_count = (
                len(validation.location.evidence) +
                len(validation.financial.evidence) +
                len(validation.csr.evidence) +
                len(validation.reputation.evidence)
        )

        # Calculate overall score
        validation.overall_score = self.analyzer.calculate_overall_score(validation)

        # Determine status
        validation.validation_status, validation.confidence_level = (
            self.analyzer.determine_validation_status(
                validation.overall_score,
                validation.total_evidence_count
            )
        )

        # Update tracking
        self.total_validations += 1
        self.total_cost += len(responses) * 0.001

        # Update cache
        self._update_cache(company_name, location, validation)

        # Log results
        elapsed = time.time() - start_time
        logger.info(
            f"Validated {company_name}: {validation.validation_status} "
            f"(score: {validation.overall_score:.1f}, evidence: {validation.total_evidence_count}, "
            f"time: {elapsed:.2f}s, cost: ${len(responses) * 0.001:.3f})"
        )

        return validation

    async def validate_batch(
            self,
            companies: List[Dict[str, Any]],
            criteria: ValidationCriteria,
            location: str,
            country: str,
            prioritize: bool = True
    ) -> List[ComprehensiveValidation]:
        """
        Validate a batch of companies

        Args:
            companies: List of company data dictionaries
            criteria: Validation criteria
            location: Target location
            country: Target country
            prioritize: Whether to prioritize by ICP tier

        Returns:
            List of ComprehensiveValidation objects
        """
        # Build strategies for all companies
        strategies = self.strategy_builder.build_batch_strategies(
            companies, criteria, location, country
        )

        # Optimize queries if enabled
        if self.config.enable_batch_optimization:
            all_queries = []
            for strategy in strategies:
                all_queries.extend(strategy.queries)

            optimized_queries = self.query_optimizer.optimize_queries(strategies)

            # Could further combine similar queries here
            if len(optimized_queries) < len(all_queries):
                logger.info(
                    f"Query optimization: {len(all_queries)} -> {len(optimized_queries)} queries"
                )

        # Validate companies
        validations = []

        for i, (company, strategy) in enumerate(zip(companies, strategies)):
            # Check cost limit
            if self.total_cost >= self.config.max_total_cost:
                logger.warning(f"Total cost limit reached (${self.total_cost:.2f})")
                break

            # Validate company
            validation = await self.validate_company(
                company,
                criteria,
                location,
                country
            )
            validations.append(validation)

            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Validated {i + 1}/{len(companies)} companies")

        return validations

    async def smart_validate(
            self,
            company_data: Dict[str, Any],
            criteria: ValidationCriteria,
            location: str,
            country: str
    ) -> ComprehensiveValidation:
        """
        Smart validation that adapts based on initial results

        Starts with quick validation and progressively deepens if needed
        """
        # Start with quick validation
        validation = await self.validate_company(
            company_data,
            criteria,
            location,
            country,
            force_tier=ValidationTier.QUICK
        )

        # Check if we need deeper validation
        if validation.overall_score < 60 or validation.confidence_level == "low":
            logger.info(f"Upgrading validation for {company_data['name']} to STANDARD")

            # Upgrade to standard validation
            validation = await self.validate_company(
                company_data,
                criteria,
                location,
                country,
                force_tier=ValidationTier.STANDARD
            )

            # Check if we need even deeper validation
            if validation.overall_score < 40 and company_data.get("icp_tier") in ["A", "B"]:
                logger.info(f"Upgrading validation for {company_data['name']} to COMPREHENSIVE")

                # Upgrade to comprehensive validation
                validation = await self.validate_company(
                    company_data,
                    criteria,
                    location,
                    country,
                    force_tier=ValidationTier.COMPREHENSIVE
                )

        return validation

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation activities"""
        runtime = (datetime.now() - self.start_time).total_seconds()

        return {
            "total_validations": self.total_validations,
            "total_cost": self.total_cost,
            "average_cost_per_validation": self.total_cost / max(1, self.total_validations),
            "runtime_seconds": runtime,
            "validations_per_minute": (self.total_validations / runtime) * 60 if runtime > 0 else 0,
            "cache_size": len(self.validation_cache),
            "serper_usage": self.serper_client.get_usage_stats() if self.serper_client else {}
        }

    async def validate_with_fallback(
            self,
            company_data: Dict[str, Any],
            criteria: ValidationCriteria,
            locations: List[str],
            country: str
    ) -> ComprehensiveValidation:
        """
        Validate with fallback to alternative locations

        Useful when a company might be listed under different city names
        """
        best_validation = None
        best_score = 0

        for location in locations:
            validation = await self.validate_company(
                company_data,
                criteria,
                location,
                country
            )

            if validation.overall_score > best_score:
                best_validation = validation
                best_score = validation.overall_score

                # If we found a good match, stop searching
                if validation.validation_status == "verified":
                    break

        return best_validation or validation

    def export_validation_report(
            self,
            validations: List[ComprehensiveValidation],
            format: str = "dict"
    ) -> Any:
        """
        Export validation results in various formats

        Args:
            validations: List of validation results
            format: Output format ('dict', 'csv', 'summary')

        Returns:
            Formatted validation report
        """
        if format == "dict":
            return [asdict(v) for v in validations]

        elif format == "summary":
            summary = {
                "total_companies": len(validations),
                "verified": sum(1 for v in validations if v.validation_status == "verified"),
                "partial": sum(1 for v in validations if v.validation_status == "partial"),
                "unverified": sum(1 for v in validations if v.validation_status == "unverified"),
                "rejected": sum(1 for v in validations if v.validation_status == "rejected"),
                "average_score": sum(v.overall_score for v in validations) / max(1, len(validations)),
                "total_evidence": sum(v.total_evidence_count for v in validations),
                "total_queries": sum(v.serper_queries_used for v in validations)
            }

            # Add tier breakdown
            tier_counts = {}
            for v in validations:
                status = v.validation_status
                if status not in tier_counts:
                    tier_counts[status] = []
                tier_counts[status].append({
                    "company": v.company_name,
                    "score": v.overall_score,
                    "evidence": v.total_evidence_count
                })

            summary["by_status"] = tier_counts
            return summary

        elif format == "csv":
            # Create CSV-friendly format
            csv_data = []
            for v in validations:
                csv_data.append({
                    "company_name": v.company_name,
                    "validation_status": v.validation_status,
                    "confidence_level": v.confidence_level,
                    "overall_score": v.overall_score,
                    "location_verified": v.location.verified,
                    "revenue_verified": v.financial.revenue_verified,
                    "revenue_range": v.financial.revenue_range,
                    "employees_verified": v.financial.employees_verified,
                    "employee_range": v.financial.employee_range,
                    "has_csr_programs": v.csr.has_csr_programs,
                    "csr_focus_areas": ", ".join(v.csr.focus_areas),
                    "certifications": ", ".join(v.csr.certifications),
                    "reputation_score": v.reputation.reputation_score,
                    "negative_signals": ", ".join(v.reputation.negative_signals),
                    "total_evidence": v.total_evidence_count,
                    "queries_used": v.serper_queries_used,
                    "timestamp": v.validation_timestamp
                })
            return csv_data

        else:
            raise ValueError(f"Unknown format: {format}")


class ValidationOrchestrator:
    """
    High-level orchestrator for complex validation workflows
    """

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.agent = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.agent = EnhancedValidationAgent(self.config)
        await self.agent.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.agent:
            await self.agent.__aexit__(exc_type, exc_val, exc_tb)

    async def validate_tiered_companies(
            self,
            companies_by_tier: Dict[str, List[Dict[str, Any]]],
            criteria: ValidationCriteria,
            location: str,
            country: str
    ) -> Dict[str, List[ComprehensiveValidation]]:
        """
        Validate companies grouped by tier with appropriate strategies

        Args:
            companies_by_tier: Dictionary with tier keys (A, B, C) and company lists
            criteria: Validation criteria
            location: Target location
            country: Target country

        Returns:
            Dictionary of tier -> validation results
        """
        results = {}

        # Define tier strategies
        tier_strategies = {
            "A": ValidationTier.COMPREHENSIVE,
            "B": ValidationTier.STANDARD,
            "C": ValidationTier.QUICK
        }

        for tier, companies in companies_by_tier.items():
            logger.info(f"Validating {len(companies)} Tier {tier} companies")

            # Determine validation depth for this tier
            validation_tier = tier_strategies.get(tier, ValidationTier.STANDARD)

            # Validate all companies in this tier
            tier_results = []
            for company in companies:
                validation = await self.agent.validate_company(
                    company,
                    criteria,
                    location,
                    country,
                    force_tier=validation_tier
                )
                tier_results.append(validation)

            results[tier] = tier_results

        return results

    async def progressive_validation(
            self,
            companies: List[Dict[str, Any]],
            criteria: ValidationCriteria,
            location: str,
            country: str,
            target_verified: int = 10
    ) -> List[ComprehensiveValidation]:
        """
        Progressively validate companies until target number verified

        Useful when you need a specific number of verified companies
        """
        validations = []
        verified_count = 0

        for company in companies:
            # Use smart validation
            validation = await self.agent.smart_validate(
                company,
                criteria,
                location,
                country
            )

            validations.append(validation)

            if validation.validation_status == "verified":
                verified_count += 1

                if verified_count >= target_verified:
                    logger.info(f"Reached target of {target_verified} verified companies")
                    break

        return validations

    async def validate_with_enrichment(
            self,
            companies: List[Dict[str, Any]],
            criteria: ValidationCriteria,
            location: str,
            country: str
    ) -> List[Dict[str, Any]]:
        """
        Validate and enrich company data with validation findings

        Returns original company data enriched with validation results
        """
        enriched_companies = []

        for company in companies:
            # Validate
            validation = await self.agent.validate_company(
                company,
                criteria,
                location,
                country
            )

            # Enrich original data
            enriched = company.copy()

            # Add validation results
            enriched["validation"] = {
                "status": validation.validation_status,
                "score": validation.overall_score,
                "confidence": validation.confidence_level,
                "timestamp": validation.validation_timestamp
            }

            # Add discovered data
            if validation.location.headquarters:
                enriched["verified_headquarters"] = validation.location.headquarters

            if validation.financial.revenue_range:
                enriched["verified_revenue"] = validation.financial.revenue_range

            if validation.financial.employee_range:
                enriched["verified_employees"] = validation.financial.employee_range

            if validation.csr.focus_areas:
                enriched["verified_csr_areas"] = validation.csr.focus_areas

            if validation.csr.certifications:
                enriched["verified_certifications"] = validation.csr.certifications

            if validation.reputation.negative_signals:
                enriched["risk_signals"] = validation.reputation.negative_signals

            enriched_companies.append(enriched)

        return enriched_companies