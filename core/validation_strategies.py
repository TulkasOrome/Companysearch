# validation_strategies.py
"""
Validation strategies for different company types and use cases
Implements tiered validation approaches based on confidence and requirements
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

from .serper_client import SerperQuery, SerperEndpoint, SerperResponse


class ValidationTier(Enum):
    """Validation depth tiers"""
    QUICK = "quick"  # 1 query - high confidence only
    STANDARD = "standard"  # 2-3 queries - medium confidence
    COMPREHENSIVE = "comprehensive"  # 4-6 queries - low confidence or critical
    DEEP = "deep"  # 7+ queries - full investigation


class ValidationPriority(Enum):
    """Priority levels for validation"""
    CRITICAL = "critical"  # Tier A prospects
    HIGH = "high"  # Tier B prospects
    MEDIUM = "medium"  # Tier C prospects
    LOW = "low"  # Others


@dataclass
class ValidationStrategy:
    """Defines a validation strategy for a company"""
    company_name: str
    tier: ValidationTier
    priority: ValidationPriority
    required_checks: List[str]
    optional_checks: List[str]
    exclusion_checks: List[str]
    queries: List[SerperQuery] = field(default_factory=list)
    estimated_cost: float = 0.0

    def calculate_cost(self):
        """Calculate estimated cost of validation"""
        self.estimated_cost = len(self.queries) * 0.001
        return self.estimated_cost


@dataclass
class ValidationCriteria:
    """Criteria for validation"""
    # Location requirements
    must_be_in_locations: List[str] = field(default_factory=list)
    must_be_within_radius: Optional[Tuple[str, float]] = None  # (location, radius_km)

    # Financial requirements
    min_revenue: Optional[float] = None
    max_revenue: Optional[float] = None
    revenue_currency: str = "USD"
    min_employees: Optional[int] = None
    max_employees: Optional[int] = None

    # CSR requirements
    required_csr_areas: List[str] = field(default_factory=list)
    required_certifications: List[str] = field(default_factory=list)

    # Event triggers
    required_events: List[str] = field(default_factory=list)

    # Exclusions
    excluded_industries: List[str] = field(default_factory=list)
    excluded_keywords: List[str] = field(default_factory=list)
    negative_signals: List[str] = field(default_factory=list)


class ValidationStrategyBuilder:
    """Builds validation strategies based on company and criteria"""

    # Signal keywords for different aspects
    CSR_KEYWORDS = [
        "corporate social responsibility", "csr", "sustainability", "esg",
        "community involvement", "charity", "donation", "sponsorship",
        "volunteer", "giving back", "social impact", "philanthropy",
        "environmental", "diversity", "inclusion", "ethics"
    ]

    FINANCIAL_KEYWORDS = [
        "revenue", "turnover", "sales", "income", "profit", "funding",
        "valuation", "investment", "capital", "financial results",
        "annual report", "investor relations", "earnings"
    ]

    NEGATIVE_KEYWORDS = [
        "bankruptcy", "lawsuit", "scandal", "fraud", "investigation",
        "violation", "fine", "penalty", "controversy", "complaint",
        "layoff", "closure", "restructuring", "misconduct"
    ]

    GROWTH_KEYWORDS = [
        "expansion", "growth", "acquisition", "merger", "new office",
        "hiring", "investment", "partnership", "launch", "opening",
        "record revenue", "milestone", "award", "recognition"
    ]

    def __init__(self):
        self.strategies = []

    def determine_validation_tier(
            self,
            confidence: str,
            icp_tier: Optional[str],
            icp_score: Optional[float]
    ) -> ValidationTier:
        """Determine appropriate validation tier based on confidence and ICP"""

        # High confidence + Tier A = Quick validation
        if confidence in ["absolute", "high"] and icp_tier == "A":
            return ValidationTier.QUICK

        # High confidence or Tier A/B = Standard validation
        elif confidence in ["absolute", "high"] or icp_tier in ["A", "B"]:
            return ValidationTier.STANDARD

        # Medium confidence or Tier C = Comprehensive validation
        elif confidence == "medium" or icp_tier == "C":
            return ValidationTier.COMPREHENSIVE

        # Low confidence or unknown = Deep validation
        else:
            return ValidationTier.DEEP

    def determine_priority(
            self,
            icp_tier: Optional[str],
            icp_score: Optional[float],
            revenue: Optional[str]
    ) -> ValidationPriority:
        """Determine validation priority"""

        if icp_tier == "A" or (icp_score and icp_score >= 80):
            return ValidationPriority.CRITICAL
        elif icp_tier == "B" or (icp_score and icp_score >= 60):
            return ValidationPriority.HIGH
        elif icp_tier == "C" or (icp_score and icp_score >= 40):
            return ValidationPriority.MEDIUM
        else:
            return ValidationPriority.LOW

    def build_strategy(
            self,
            company_data: Dict[str, Any],
            criteria: ValidationCriteria,
            location: str,
            country: str
    ) -> ValidationStrategy:
        """Build a validation strategy for a company"""

        # Extract company info
        company_name = company_data.get("name", "")
        confidence = company_data.get("confidence", "low")
        icp_tier = company_data.get("icp_tier")
        icp_score = company_data.get("icp_score")
        revenue = company_data.get("estimated_revenue")

        # Determine tier and priority
        tier = self.determine_validation_tier(confidence, icp_tier, icp_score)
        priority = self.determine_priority(icp_tier, icp_score, revenue)

        # Create strategy
        strategy = ValidationStrategy(
            company_name=company_name,
            tier=tier,
            priority=priority,
            required_checks=[],
            optional_checks=[],
            exclusion_checks=[]
        )

        # Build queries based on tier
        queries = self._build_queries_for_tier(
            strategy,
            company_data,
            criteria,
            location,
            country
        )
        strategy.queries = queries
        strategy.calculate_cost()

        return strategy

    def _build_queries_for_tier(
            self,
            strategy: ValidationStrategy,
            company_data: Dict[str, Any],
            criteria: ValidationCriteria,
            location: str,
            country: str
    ) -> List[SerperQuery]:
        """Build queries based on validation tier"""

        queries = []
        company_name = company_data["name"]

        if strategy.tier == ValidationTier.QUICK:
            # Just location verification
            queries.append(SerperQuery(
                query=f"{company_name} {location}",
                endpoint=SerperEndpoint.PLACES,
                location=location,
                country=country
            ))
            strategy.required_checks.append("location")

        elif strategy.tier == ValidationTier.STANDARD:
            # Location + basic web search
            queries.append(SerperQuery(
                query=f"{company_name} {location}",
                endpoint=SerperEndpoint.PLACES,
                location=location,
                country=country
            ))

            queries.append(SerperQuery(
                query=f'"{company_name}" revenue employees {country}',
                endpoint=SerperEndpoint.SEARCH,
                country=country
            ))

            strategy.required_checks.extend(["location", "financials"])

            # Add CSR if required
            if criteria.required_csr_areas:
                queries.append(SerperQuery(
                    query=f'"{company_name}" CSR "corporate social responsibility" {" ".join(criteria.required_csr_areas)}',
                    endpoint=SerperEndpoint.SEARCH,
                    country=country
                ))
                strategy.required_checks.append("csr")

        elif strategy.tier == ValidationTier.COMPREHENSIVE:
            # Full validation suite
            # 1. Location
            queries.append(SerperQuery(
                query=f"{company_name} {location}",
                endpoint=SerperEndpoint.PLACES,
                location=location,
                country=country
            ))

            # 2. Maps for additional locations
            queries.append(SerperQuery(
                query=f"{company_name} offices",
                endpoint=SerperEndpoint.MAPS,
                location=location,
                country=country
            ))

            # 3. Financial data
            queries.append(SerperQuery(
                query=f'"{company_name}" annual revenue {criteria.revenue_currency}',
                endpoint=SerperEndpoint.SEARCH,
                country=country
            ))

            # 4. Recent news
            queries.append(SerperQuery(
                query=company_name,
                endpoint=SerperEndpoint.NEWS,
                country=country,
                time_range="month"
            ))

            strategy.required_checks.extend([
                "location", "all_offices", "financials", "recent_news"
            ])

            # 5. CSR if required
            if criteria.required_csr_areas:
                queries.append(SerperQuery(
                    query=f'"{company_name}" sustainability CSR community',
                    endpoint=SerperEndpoint.SEARCH,
                    country=country
                ))
                strategy.required_checks.append("csr")

            # 6. Negative signals
            queries.append(SerperQuery(
                query=f'"{company_name}" scandal lawsuit investigation',
                endpoint=SerperEndpoint.NEWS,
                country=country,
                time_range="year"
            ))
            strategy.exclusion_checks.append("negative_signals")

        else:  # DEEP validation
            # Everything in comprehensive plus more
            queries.extend(self._build_queries_for_tier(
                strategy, company_data, criteria, location, country
            ))

            # Additional deep checks
            # 7. Shopping presence (for B2C)
            if company_data.get("business_type") in ["B2C", "D2C"]:
                queries.append(SerperQuery(
                    query=company_name,
                    endpoint=SerperEndpoint.SHOPPING,
                    location=location,
                    country=country
                ))
                strategy.optional_checks.append("shopping_presence")

            # 8. Patents (for tech companies)
            if "tech" in company_data.get("industry_category", "").lower():
                queries.append(SerperQuery(
                    query=f"{company_name} patent",
                    endpoint=SerperEndpoint.PATENTS,
                    country=country
                ))
                strategy.optional_checks.append("patents")

            # 9. Videos (for brand presence)
            queries.append(SerperQuery(
                query=f"{company_name} corporate",
                endpoint=SerperEndpoint.VIDEOS,
                country=country,
                num_results=5
            ))
            strategy.optional_checks.append("video_presence")

        return queries

    def build_batch_strategies(
            self,
            companies: List[Dict[str, Any]],
            criteria: ValidationCriteria,
            location: str,
            country: str
    ) -> List[ValidationStrategy]:
        """Build strategies for a batch of companies"""

        strategies = []
        for company in companies:
            strategy = self.build_strategy(company, criteria, location, country)
            strategies.append(strategy)

        # Sort by priority for execution order
        strategies.sort(key=lambda s: (s.priority.value, s.tier.value))

        return strategies


class ValidationQueryOptimizer:
    """Optimizes validation queries for efficiency"""

    def __init__(self):
        self.cache = {}
        self.batch_size = 5

    def optimize_queries(
            self,
            strategies: List[ValidationStrategy]
    ) -> List[SerperQuery]:
        """Optimize queries across multiple strategies"""

        # Collect all queries
        all_queries = []
        for strategy in strategies:
            all_queries.extend(strategy.queries)

        # Remove duplicates
        seen = set()
        unique_queries = []
        for query in all_queries:
            key = f"{query.endpoint.value}:{query.query}:{query.location}"
            if key not in seen:
                seen.add(key)
                unique_queries.append(query)

        # Group by endpoint for batch processing
        grouped = {}
        for query in unique_queries:
            endpoint = query.endpoint
            if endpoint not in grouped:
                grouped[endpoint] = []
            grouped[endpoint].append(query)

        # Order for optimal execution
        # Places first (most specific), then news (time-sensitive), then web
        endpoint_order = [
            SerperEndpoint.PLACES,
            SerperEndpoint.MAPS,
            SerperEndpoint.NEWS,
            SerperEndpoint.SEARCH,
            SerperEndpoint.SHOPPING,
            SerperEndpoint.VIDEOS,
            SerperEndpoint.IMAGES,
            SerperEndpoint.PATENTS,
            SerperEndpoint.SCHOLAR,
            SerperEndpoint.AUTOCOMPLETE
        ]

        optimized = []
        for endpoint in endpoint_order:
            if endpoint in grouped:
                optimized.extend(grouped[endpoint])

        return optimized

    def combine_similar_queries(
            self,
            queries: List[SerperQuery]
    ) -> List[SerperQuery]:
        """Combine similar queries for efficiency"""

        combined = []
        processed = set()

        for i, query in enumerate(queries):
            if i in processed:
                continue

            # Find similar queries
            similar = []
            for j, other in enumerate(queries[i + 1:], i + 1):
                if (query.endpoint == other.endpoint and
                        query.location == other.location and
                        query.country == other.country):
                    similar.append(j)
                    processed.add(j)

            if similar:
                # Combine query strings
                combined_query = query.query
                for idx in similar:
                    combined_query += f" OR {queries[idx].query}"

                # Create combined query
                new_query = SerperQuery(
                    query=combined_query,
                    endpoint=query.endpoint,
                    location=query.location,
                    country=query.country,
                    num_results=max(query.num_results, 20)
                )
                combined.append(new_query)
            else:
                combined.append(query)

        return combined