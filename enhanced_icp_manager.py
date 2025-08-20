#!/usr/bin/env python3
"""
enhanced_icp_manager.py - Manages ICP profiles for RMH Sydney and Guide Dogs Victoria
Provides structured criteria management and validation
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

from core.data_models import (
    SearchCriteria,
    LocationCriteria,
    FinancialCriteria,
    OrganizationalCriteria,
    BehavioralSignals
)


class ICPTier(Enum):
    """ICP Tier levels"""
    TIER_A = "A"  # Strategic/Perfect match
    TIER_B = "B"  # Exploratory/Good match
    TIER_C = "C"  # Potential/Acceptable match
    UNTIERED = "Untiered"


@dataclass
class ICPProfile:
    """Represents a complete ICP profile"""
    name: str
    organization: str
    description: str
    tiers: Dict[str, SearchCriteria]
    validation_rules: Dict[str, Any]
    exclusions: List[str]
    scoring_weights: Dict[str, float]


class ICPManager:
    """Manages ICP profiles and criteria generation"""

    def __init__(self):
        self.profiles = {}
        self._initialize_profiles()

    def _initialize_profiles(self):
        """Initialize RMH Sydney and Guide Dogs Victoria profiles"""

        # RMH Sydney Profile
        self.profiles['rmh_sydney'] = self._create_rmh_profile()

        # Guide Dogs Victoria Profile
        self.profiles['guide_dogs_victoria'] = self._create_guide_dogs_profile()

    def _create_rmh_profile(self) -> ICPProfile:
        """Create RMH Sydney ICP profile"""

        # Main criteria (applies to all tiers)
        base_criteria = SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                cities=["Sydney"],
                regions=["Greater Western Sydney", "Western Sydney", "Sydney CBD"]
            ),
            financial=FinancialCriteria(
                revenue_min=5_000_000,
                revenue_max=100_000_000,
                revenue_currency="AUD",
                giving_capacity_min=20_000
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=50,
                office_types=["Headquarters", "Major Office"]
            ),
            behavioral=BehavioralSignals(
                csr_focus_areas=["children", "community", "families", "health"],
                recent_events=["Office Move", "CSR Launch", "Expansion", "Anniversary"]
            ),
            business_types=["B2B", "B2C", "B2B2C"],
            industries=[
                {"name": "Construction/Trades", "priority": 1},
                {"name": "Property/Real Estate", "priority": 2},
                {"name": "Hospitality/Food Services", "priority": 3},
                {"name": "Professional Services", "priority": 4},
                {"name": "Manufacturing", "priority": 5},
                {"name": "Retail", "priority": 6}
            ],
            excluded_industries=["Fast Food", "Gambling", "Tobacco"],
            excluded_companies=["McDonald's", "KFC", "Burger King", "Subway", "Pizza Hut"],
            custom_prompt="Find companies with strong local community presence in Western Sydney"
        )

        # Tier variations
        tiers = {
            "A": base_criteria,  # Perfect match - use base criteria
            "B": SearchCriteria(  # Good match - relaxed revenue
                location=base_criteria.location,
                financial=FinancialCriteria(
                    revenue_min=2_000_000,
                    revenue_max=200_000_000,
                    revenue_currency="AUD"
                ),
                organizational=OrganizationalCriteria(
                    employee_count_min=20
                ),
                behavioral=BehavioralSignals(
                    csr_focus_areas=["community"]
                ),
                business_types=base_criteria.business_types,
                industries=base_criteria.industries,
                excluded_industries=base_criteria.excluded_industries
            ),
            "C": SearchCriteria(  # Potential match - very relaxed
                location=LocationCriteria(
                    countries=["Australia"],
                    states=["New South Wales"]
                ),
                financial=FinancialCriteria(
                    revenue_min=1_000_000,
                    revenue_currency="AUD"
                ),
                organizational=OrganizationalCriteria(),
                behavioral=BehavioralSignals(),
                business_types=["B2B", "B2C"],
                industries=[],
                excluded_industries=["Fast Food", "Gambling"]
            )
        }

        validation_rules = {
            "must_have": {
                "location": "Sydney or Greater Western Sydney",
                "revenue": "AUD 5-100M",
                "csr_focus": "Children or Community support",
                "giving_capacity": "≥AUD 20k annually"
            },
            "nice_to_have": {
                "employees": "50+ employees",
                "recent_events": "Office move, CSR launch, expansion",
                "industry": "Construction, Property, Hospitality"
            },
            "disqualifiers": {
                "competitors": "Fast food competitors to McDonald's",
                "misconduct": "Recent scandals or investigations",
                "industries": "Gambling, Tobacco"
            }
        }

        return ICPProfile(
            name="RMH Sydney",
            organization="Ronald McDonald House Greater Western Sydney",
            description="Target companies for room sponsorship and community partnership",
            tiers=tiers,
            validation_rules=validation_rules,
            exclusions=base_criteria.excluded_industries + base_criteria.excluded_companies,
            scoring_weights={
                "location": 0.35,
                "financial": 0.25,
                "csr": 0.25,
                "industry": 0.15
            }
        )

    def _create_guide_dogs_profile(self) -> ICPProfile:
        """Create Guide Dogs Victoria ICP profile"""

        # Tier A - Strategic Partners
        tier_a = SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                states=["Victoria"],
                cities=["Melbourne", "Geelong", "Ballarat", "Bendigo"]
            ),
            financial=FinancialCriteria(
                revenue_min=500_000_000,
                revenue_currency="AUD"
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=500,
                employee_count_by_location={"Victoria": 150}
            ),
            behavioral=BehavioralSignals(
                certifications=["B-Corp", "ISO 26000", "ISO 14001"],
                csr_focus_areas=["disability", "inclusion", "health", "accessibility"],
                esg_maturity="Mature"
            ),
            business_types=["B2B", "B2C", "Enterprise"],
            industries=[
                {"name": "Health/Life Sciences", "priority": 1},
                {"name": "Financial Services", "priority": 2},
                {"name": "Legal Services", "priority": 3},
                {"name": "Technology", "priority": 4},
                {"name": "FMCG", "priority": 5},
                {"name": "Property/Construction", "priority": 6}
            ],
            excluded_industries=["Gambling", "Tobacco", "Racing", "Animal Testing"],
            custom_prompt="Focus on companies with established CSR programs and disability inclusion initiatives"
        )

        # Tier B - Exploratory Partners
        tier_b = SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                states=["Victoria"]
            ),
            financial=FinancialCriteria(
                revenue_min=50_000_000,
                revenue_max=500_000_000,
                revenue_currency="AUD"
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=100,
                employee_count_max=500
            ),
            behavioral=BehavioralSignals(
                csr_focus_areas=["community", "health", "wellbeing", "education"]
            ),
            business_types=["B2B", "B2C"],
            industries=[
                {"name": "Manufacturing", "priority": 1},
                {"name": "Logistics", "priority": 2},
                {"name": "Universities/Education", "priority": 3},
                {"name": "Retail", "priority": 4},
                {"name": "Professional Services", "priority": 5}
            ],
            excluded_industries=["Gambling", "Tobacco", "Racing"]
        )

        # Tier C - Potential Partners
        tier_c = SearchCriteria(
            location=LocationCriteria(
                countries=["Australia"],
                states=["Victoria", "New South Wales"]
            ),
            financial=FinancialCriteria(
                revenue_min=10_000_000,
                revenue_currency="AUD"
            ),
            organizational=OrganizationalCriteria(
                employee_count_min=50
            ),
            behavioral=BehavioralSignals(),
            business_types=["B2B", "B2C"],
            industries=[],
            excluded_industries=["Gambling", "Tobacco", "Racing"]
        )

        validation_rules = {
            "tier_a_must_have": {
                "location": "HQ or 150+ employee office in Victoria",
                "revenue": "≥AUD 500M",
                "employees": "≥500 in Victoria",
                "csr_maturity": "B-Corp, ISO 26000, or published CSR report"
            },
            "tier_b_must_have": {
                "location": "Victoria presence",
                "revenue": "AUD 50-500M",
                "employees": "100-500",
                "csr": "CSR statement or DEI programs"
            },
            "requirements": {
                "giving_history": "≥AUD 25k gifts or donations",
                "event_hosting": "Can host 10-700 staff events",
                "partnerships": "No Vision Australia partnerships"
            },
            "disqualifiers": {
                "industries": "Gambling, tobacco, racing",
                "concerns": "Animal welfare issues",
                "partnerships": "Existing Vision Australia partnership"
            }
        }

        return ICPProfile(
            name="Guide Dogs Victoria",
            organization="Guide Dogs Victoria",
            description="Corporate partnership prospects for disability inclusion",
            tiers={"A": tier_a, "B": tier_b, "C": tier_c},
            validation_rules=validation_rules,
            exclusions=tier_a.excluded_industries,
            scoring_weights={
                "location": 0.25,
                "financial": 0.20,
                "csr": 0.35,
                "industry": 0.20
            }
        )

    def get_profile(self, profile_name: str) -> Optional[ICPProfile]:
        """Get a specific ICP profile"""
        return self.profiles.get(profile_name.lower().replace(' ', '_'))

    def get_criteria(self, profile_name: str, tier: str = "A") -> Optional[SearchCriteria]:
        """Get search criteria for a specific profile and tier"""
        profile = self.get_profile(profile_name)
        if profile:
            return profile.tiers.get(tier.upper())
        return None

    def list_profiles(self) -> List[str]:
        """List all available profiles"""
        return list(self.profiles.keys())

    def validate_company_against_icp(
            self,
            company: Dict[str, Any],
            profile_name: str,
            tier: str = "A"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a company against an ICP profile

        Returns:
            Tuple of (passes_validation, validation_details)
        """
        profile = self.get_profile(profile_name)
        if not profile:
            return False, {"error": "Profile not found"}

        validation_details = {
            "profile": profile_name,
            "tier": tier,
            "passed_checks": [],
            "failed_checks": [],
            "warnings": [],
            "score": 0
        }

        # Get the appropriate tier rules
        if profile_name == "rmh_sydney":
            validation_details = self._validate_rmh_company(company, profile, validation_details)
        elif profile_name == "guide_dogs_victoria":
            validation_details = self._validate_guide_dogs_company(company, profile, tier, validation_details)

        # Calculate overall pass/fail
        passes = len(validation_details["failed_checks"]) == 0

        return passes, validation_details

    def _validate_rmh_company(
            self,
            company: Dict[str, Any],
            profile: ICPProfile,
            details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate company against RMH Sydney criteria"""

        # Location check
        if company.get("operates_in_country") and any(
                loc in str(company.get("headquarters", "")).lower() + " ".join(
                    company.get("office_locations", [])).lower()
                for loc in ["sydney", "western sydney", "parramatta", "penrith"]
        ):
            details["passed_checks"].append("Location - Sydney/GWS")
            details["score"] += 35
        else:
            details["failed_checks"].append("Location - Not in Sydney/GWS")

        # Revenue check
        revenue = company.get("estimated_revenue", "")
        if revenue and any(x in str(revenue) for x in ["5", "10", "20", "50", "100"]):
            details["passed_checks"].append("Revenue - Within range")
            details["score"] += 25
        else:
            details["warnings"].append("Revenue - Cannot verify")

        # CSR check
        csr_areas = company.get("csr_focus_areas", [])
        if any(area in csr_areas for area in ["children", "community", "families"]):
            details["passed_checks"].append("CSR - Children/Community focus")
            details["score"] += 25
        else:
            details["failed_checks"].append("CSR - No children/community focus")

        # Industry check
        industry = company.get("industry_category", "").lower()
        if any(ind in industry for ind in ["construction", "property", "hospitality"]):
            details["passed_checks"].append("Industry - Priority sector")
            details["score"] += 15

        # Exclusion check
        if any(exc in company.get("name", "").lower() or exc in industry
               for exc in ["mcdonald", "kfc", "burger", "gambling"]):
            details["failed_checks"].append("Exclusion - Competitor or excluded industry")
            details["score"] = 0  # Automatic fail

        return details

    def _validate_guide_dogs_company(
            self,
            company: Dict[str, Any],
            profile: ICPProfile,
            tier: str,
            details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate company against Guide Dogs Victoria criteria"""

        # Location check (Victoria)
        if company.get("operates_in_country") and any(
                loc in str(company.get("headquarters", "")).lower() + " ".join(
                    company.get("office_locations", [])).lower()
                for loc in ["melbourne", "victoria", "geelong", "ballarat", "bendigo"]
        ):
            details["passed_checks"].append("Location - Victoria")
            details["score"] += 25
        else:
            details["failed_checks"].append("Location - Not in Victoria")

        # Revenue check (tier-specific)
        revenue = company.get("estimated_revenue", "")
        if tier == "A":
            if "billion" in str(revenue).lower() or any(x in str(revenue) for x in ["500", "600", "700", "800", "900"]):
                details["passed_checks"].append("Revenue - Tier A (≥500M)")
                details["score"] += 20
            else:
                details["failed_checks"].append("Revenue - Below Tier A threshold")
        else:  # Tier B
            if any(x in str(revenue) for x in ["50", "100", "200", "300", "400"]):
                details["passed_checks"].append("Revenue - Tier B (50-500M)")
                details["score"] += 20

        # CSR/Certification check
        certifications = company.get("certifications", [])
        csr_areas = company.get("csr_focus_areas", [])

        if tier == "A":
            if any(cert in certifications for cert in ["B-Corp", "ISO 26000", "ISO 14001"]):
                details["passed_checks"].append("Certifications - Tier A requirement")
                details["score"] += 35
            elif any(area in csr_areas for area in ["disability", "inclusion", "accessibility"]):
                details["passed_checks"].append("CSR - Disability/inclusion focus")
                details["score"] += 25
            else:
                details["warnings"].append("CSR - No certifications or disability focus")
        else:
            if csr_areas:
                details["passed_checks"].append("CSR - Has CSR programs")
                details["score"] += 20

        # Industry check
        industry = company.get("industry_category", "").lower()
        priority_industries = ["health", "finance", "legal", "technology", "fmcg"] if tier == "A" else ["manufacturing",
                                                                                                        "logistics",
                                                                                                        "education"]

        if any(ind in industry for ind in priority_industries):
            details["passed_checks"].append(f"Industry - Priority for Tier {tier}")
            details["score"] += 20

        # Exclusion check
        if any(exc in industry for exc in ["gambling", "tobacco", "racing"]):
            details["failed_checks"].append("Exclusion - Prohibited industry")
            details["score"] = 0  # Automatic fail

        return details

    def export_profile(self, profile_name: str, format: str = "json") -> str:
        """Export a profile configuration"""
        profile = self.get_profile(profile_name)
        if not profile:
            return None

        if format == "json":
            export_data = {
                "name": profile.name,
                "organization": profile.organization,
                "description": profile.description,
                "tiers": {
                    tier: {
                        "location": criteria.location.countries + criteria.location.cities,
                        "revenue_range": f"{criteria.financial.revenue_min}-{criteria.financial.revenue_max}",
                        "industries": [ind["name"] for ind in criteria.industries],
                        "exclusions": criteria.excluded_industries
                    }
                    for tier, criteria in profile.tiers.items()
                },
                "validation_rules": profile.validation_rules,
                "scoring_weights": profile.scoring_weights
            }
            return json.dumps(export_data, indent=2)

        return str(profile)


# Convenience functions for quick access
def get_rmh_criteria(tier: str = "A") -> SearchCriteria:
    """Get RMH Sydney search criteria"""
    manager = ICPManager()
    return manager.get_criteria("rmh_sydney", tier)


def get_guide_dogs_criteria(tier: str = "A") -> SearchCriteria:
    """Get Guide Dogs Victoria search criteria"""
    manager = ICPManager()
    return manager.get_criteria("guide_dogs_victoria", tier)


def validate_for_rmh(company: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Quick validation for RMH Sydney"""
    manager = ICPManager()
    return manager.validate_company_against_icp(company, "rmh_sydney")


def validate_for_guide_dogs(company: Dict[str, Any], tier: str = "A") -> Tuple[bool, Dict[str, Any]]:
    """Quick validation for Guide Dogs Victoria"""
    manager = ICPManager()
    return manager.validate_company_against_icp(company, "guide_dogs_victoria", tier)