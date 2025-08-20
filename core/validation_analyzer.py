# validation_analyzer.py
"""
Analyzes Serper responses to extract validation signals
Provides scoring and evidence extraction for company validation
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt

from .serper_client import SerperResponse, SerperEndpoint


@dataclass
class ValidationEvidence:
    """Evidence collected during validation"""
    source: str
    type: str
    content: str
    confidence: float
    timestamp: Optional[str] = None
    url: Optional[str] = None


@dataclass
class LocationValidation:
    """Location validation results"""
    verified: bool = False
    confidence: float = 0.0
    headquarters: Optional[Dict[str, Any]] = None
    offices: List[Dict[str, Any]] = field(default_factory=list)
    coordinates: Optional[Tuple[float, float]] = None
    proximity_match: bool = False
    distance_km: Optional[float] = None
    evidence: List[ValidationEvidence] = field(default_factory=list)


@dataclass
class FinancialValidation:
    """Financial validation results"""
    revenue_verified: bool = False
    revenue_confidence: float = 0.0
    revenue_range: Optional[str] = None
    revenue_currency: Optional[str] = None
    employees_verified: bool = False
    employees_confidence: float = 0.0
    employee_range: Optional[str] = None
    growth_indicators: List[str] = field(default_factory=list)
    financial_health: str = "unknown"
    evidence: List[ValidationEvidence] = field(default_factory=list)


@dataclass
class CSRValidation:
    """CSR and ESG validation results"""
    has_csr_programs: bool = False
    csr_confidence: float = 0.0
    focus_areas: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    giving_history: List[Dict[str, Any]] = field(default_factory=list)
    esg_maturity: str = "unknown"
    evidence: List[ValidationEvidence] = field(default_factory=list)


@dataclass
class ReputationValidation:
    """Reputation and recent events validation"""
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    leadership_changes: List[Dict[str, Any]] = field(default_factory=list)
    awards_recognition: List[str] = field(default_factory=list)
    negative_signals: List[str] = field(default_factory=list)
    controversies: List[Dict[str, Any]] = field(default_factory=list)
    reputation_score: float = 100.0  # Start at 100, deduct for issues
    evidence: List[ValidationEvidence] = field(default_factory=list)


@dataclass
class ComprehensiveValidation:
    """Complete validation results for a company"""
    company_name: str
    validation_timestamp: str
    location: LocationValidation
    financial: FinancialValidation
    csr: CSRValidation
    reputation: ReputationValidation
    overall_score: float = 0.0
    confidence_level: str = "low"
    validation_status: str = "unverified"
    total_evidence_count: int = 0
    serper_queries_used: int = 0


class ValidationAnalyzer:
    """Analyzes Serper responses to extract validation signals"""

    # Financial patterns
    REVENUE_PATTERNS = [
        r'\$?([\d,]+\.?\d*)\s*(million|billion|m|b)\s*(revenue|turnover|sales)',
        r'(revenue|turnover|sales)\s*of\s*\$?([\d,]+\.?\d*)\s*(million|billion|m|b)',
        r'annual\s*(revenue|turnover|sales).*?\$?([\d,]+\.?\d*)\s*(million|billion|m|b)',
        r'([\d,]+\.?\d*)\s*(million|billion|m|b)\s*in\s*(revenue|sales|turnover)'
    ]

    EMPLOYEE_PATTERNS = [
        r'([\d,]+)\s*employees',
        r'([\d,]+)\s*staff',
        r'team\s*of\s*([\d,]+)',
        r'workforce\s*of\s*([\d,]+)',
        r'employs\s*([\d,]+)',
        r'([\d,]+)\s*people'
    ]

    # CSR keywords mapped to focus areas
    CSR_FOCUS_MAP = {
        "children": ["children", "kids", "youth", "school", "education", "students"],
        "environment": ["environment", "sustainability", "green", "carbon", "climate", "eco"],
        "community": ["community", "local", "neighborhood", "volunteer", "civic"],
        "health": ["health", "medical", "hospital", "wellness", "mental health", "healthcare"],
        "diversity": ["diversity", "inclusion", "equity", "women", "minorities", "DEI"],
        "education": ["education", "scholarship", "learning", "literacy", "STEM"],
        "elderly": ["elderly", "seniors", "aged care", "retirement"],
        "disability": ["disability", "accessible", "inclusion", "special needs"],
        "veterans": ["veterans", "military", "armed forces", "service members"],
        "arts": ["arts", "culture", "music", "theater", "museum"]
    }

    # Certification patterns
    CERTIFICATION_PATTERNS = {
        "B-Corp": ["b-corp", "b corp", "certified b corporation"],
        "ISO 26000": ["iso 26000", "iso26000"],
        "ISO 14001": ["iso 14001", "iso14001"],
        "LEED": ["leed certified", "leed gold", "leed platinum"],
        "Fair Trade": ["fair trade", "fairtrade"],
        "Carbon Neutral": ["carbon neutral", "net zero", "carbon negative"]
    }

    # Negative signal keywords
    NEGATIVE_SIGNALS = [
        "bankruptcy", "lawsuit", "scandal", "fraud", "investigation",
        "violation", "fine", "penalty", "controversy", "complaint",
        "layoff", "closure", "restructuring", "misconduct", "breach",
        "recall", "accident", "injury", "death", "discrimination"
    ]

    # Growth indicators
    GROWTH_INDICATORS = [
        "expansion", "growth", "acquisition", "merger", "new office",
        "hiring", "investment", "partnership", "launch", "opening",
        "record revenue", "milestone", "award", "recognition", "ipo"
    ]

    def __init__(self):
        self.validation_cache = {}

    def analyze_places_response(
            self,
            response: SerperResponse,
            company_name: str,
            target_location: Optional[str] = None
    ) -> LocationValidation:
        """Analyze places/maps response for location validation"""

        validation = LocationValidation()

        if not response.success or not response.results:
            return validation

        company_lower = company_name.lower()

        for place in response.results[:5]:  # Check top 5 results
            place_name = place.get("title", "").lower()

            # Check for name match
            name_match = self._calculate_name_similarity(company_lower, place_name)

            if name_match > 0.7:  # 70% similarity threshold
                validation.verified = True
                validation.confidence = min(name_match * 100, 100)

                # Extract headquarters info
                validation.headquarters = {
                    "name": place.get("title"),
                    "address": place.get("address"),
                    "phone": place.get("phoneNumber"),
                    "website": place.get("website"),
                    "type": place.get("type", []),
                    "rating": place.get("rating"),
                    "reviews": place.get("reviews")
                }

                # Extract coordinates
                if "latitude" in place and "longitude" in place:
                    validation.coordinates = (place["latitude"], place["longitude"])
                elif "gps_coordinates" in place:
                    coords = place["gps_coordinates"]
                    validation.coordinates = (coords.get("latitude"), coords.get("longitude"))

                # Add evidence
                validation.evidence.append(ValidationEvidence(
                    source="places",
                    type="location",
                    content=f"Found {place.get('title')} at {place.get('address')}",
                    confidence=validation.confidence,
                    url=place.get("website")
                ))

                # Check proximity if needed
                if target_location and validation.coordinates:
                    # This would require geocoding the target location
                    # For now, we'll mark as matched if in same city
                    if target_location.lower() in place.get("address", "").lower():
                        validation.proximity_match = True

                break

        return validation

    def analyze_web_response(
            self,
            response: SerperResponse,
            company_name: str
    ) -> Tuple[FinancialValidation, CSRValidation]:
        """Analyze web search response for financial and CSR data"""

        financial = FinancialValidation()
        csr = CSRValidation()

        if not response.success or not response.results:
            return financial, csr

        for result in response.results:
            content = f"{result.get('title', '')} {result.get('snippet', '')}".lower()

            # Extract financial data
            self._extract_financial_data(content, result, financial)

            # Extract CSR data
            self._extract_csr_data(content, result, csr)

        # Calculate confidence scores
        if financial.evidence:
            financial.revenue_confidence = min(len(financial.evidence) * 30, 90)
            financial.employees_confidence = min(len([e for e in financial.evidence if "employee" in e.type]) * 30, 90)

        if csr.evidence:
            csr.csr_confidence = min(len(csr.evidence) * 25, 95)
            csr.has_csr_programs = len(csr.focus_areas) > 0

        return financial, csr

    def analyze_news_response(
            self,
            response: SerperResponse,
            company_name: str
    ) -> ReputationValidation:
        """Analyze news response for recent events and reputation"""

        reputation = ReputationValidation()

        if not response.success or not response.results:
            return reputation

        for news_item in response.results:
            title = news_item.get("title", "").lower()
            snippet = news_item.get("snippet", "").lower()
            content = f"{title} {snippet}"
            date = news_item.get("date")

            # Check for negative signals
            negative_found = []
            for signal in self.NEGATIVE_SIGNALS:
                if signal in content:
                    negative_found.append(signal)

            if negative_found:
                reputation.negative_signals.extend(negative_found)
                reputation.controversies.append({
                    "title": news_item.get("title"),
                    "date": date,
                    "source": news_item.get("source"),
                    "signals": negative_found,
                    "link": news_item.get("link")
                })
                reputation.reputation_score -= 10 * len(negative_found)

                reputation.evidence.append(ValidationEvidence(
                    source="news",
                    type="negative",
                    content=f"Negative signals found: {', '.join(negative_found)}",
                    confidence=90,
                    timestamp=date,
                    url=news_item.get("link")
                ))

            # Check for growth indicators
            growth_found = []
            for indicator in self.GROWTH_INDICATORS:
                if indicator in content:
                    growth_found.append(indicator)

            if growth_found:
                reputation.recent_events.append({
                    "title": news_item.get("title"),
                    "date": date,
                    "type": "growth",
                    "indicators": growth_found,
                    "source": news_item.get("source")
                })

                reputation.evidence.append(ValidationEvidence(
                    source="news",
                    type="positive",
                    content=f"Growth indicators: {', '.join(growth_found)}",
                    confidence=80,
                    timestamp=date,
                    url=news_item.get("link")
                ))

            # Check for leadership changes
            if any(term in content for term in ["ceo", "president", "executive", "appoint", "hire", "promote"]):
                reputation.leadership_changes.append({
                    "title": news_item.get("title"),
                    "date": date,
                    "source": news_item.get("source")
                })

            # Check for awards
            if any(term in content for term in ["award", "recognition", "winner", "best", "top"]):
                reputation.awards_recognition.append(news_item.get("title"))

        # Ensure score doesn't go below 0
        reputation.reputation_score = max(0, reputation.reputation_score)

        return reputation

    def _extract_financial_data(
            self,
            content: str,
            result: Dict[str, Any],
            financial: FinancialValidation
    ):
        """Extract financial information from content"""

        # Look for revenue
        for pattern in self.REVENUE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                for match in matches:
                    # Extract amount and unit
                    if isinstance(match, tuple):
                        amount = match[0] if match[0] else match[1]
                        unit = match[1] if match[1] in ['million', 'billion', 'm', 'b'] else match[2]
                    else:
                        amount = match
                        unit = "unknown"

                    financial.revenue_verified = True
                    financial.revenue_range = f"{amount} {unit}"

                    financial.evidence.append(ValidationEvidence(
                        source="web",
                        type="revenue",
                        content=f"Revenue: {amount} {unit}",
                        confidence=75,
                        url=result.get("link")
                    ))
                break

        # Look for employees
        for pattern in self.EMPLOYEE_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                for match in matches:
                    employee_count = match[0] if isinstance(match, tuple) else match
                    financial.employees_verified = True
                    financial.employee_range = employee_count

                    financial.evidence.append(ValidationEvidence(
                        source="web",
                        type="employees",
                        content=f"Employees: {employee_count}",
                        confidence=70,
                        url=result.get("link")
                    ))
                break

        # Assess financial health
        positive_terms = ["growth", "profit", "expansion", "record", "increase"]
        negative_terms = ["loss", "decline", "layoff", "restructuring", "bankruptcy"]

        positive_count = sum(1 for term in positive_terms if term in content)
        negative_count = sum(1 for term in negative_terms if term in content)

        if positive_count > negative_count * 2:
            financial.financial_health = "positive"
        elif negative_count > positive_count * 2:
            financial.financial_health = "concerning"
        else:
            financial.financial_health = "stable"

    def _extract_csr_data(
            self,
            content: str,
            result: Dict[str, Any],
            csr: CSRValidation
    ):
        """Extract CSR information from content"""

        # Check for CSR focus areas
        for area, keywords in self.CSR_FOCUS_MAP.items():
            if any(keyword in content for keyword in keywords):
                if area not in csr.focus_areas:
                    csr.focus_areas.append(area)

                    csr.evidence.append(ValidationEvidence(
                        source="web",
                        type="csr_focus",
                        content=f"CSR focus area: {area}",
                        confidence=60,
                        url=result.get("link")
                    ))

        # Check for certifications
        for cert_name, cert_patterns in self.CERTIFICATION_PATTERNS.items():
            if any(pattern in content for pattern in cert_patterns):
                if cert_name not in csr.certifications:
                    csr.certifications.append(cert_name)

                    csr.evidence.append(ValidationEvidence(
                        source="web",
                        type="certification",
                        content=f"Certification: {cert_name}",
                        confidence=85,
                        url=result.get("link")
                    ))

        # Determine ESG maturity
        if len(csr.certifications) >= 2:
            csr.esg_maturity = "Leading"
        elif len(csr.certifications) >= 1 or len(csr.focus_areas) >= 3:
            csr.esg_maturity = "Mature"
        elif len(csr.focus_areas) >= 2:
            csr.esg_maturity = "Developing"
        elif len(csr.focus_areas) >= 1:
            csr.esg_maturity = "Basic"
        else:
            csr.esg_maturity = "Unknown"

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two company names"""

        # Simple similarity based on common words
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def calculate_overall_score(
            self,
            validation: ComprehensiveValidation,
            weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate overall validation score"""

        if not weights:
            weights = {
                "location": 0.30,
                "financial": 0.25,
                "csr": 0.25,
                "reputation": 0.20
            }

        score = 0.0

        # Location score
        if validation.location.verified:
            score += weights["location"] * validation.location.confidence

        # Financial score
        financial_score = 0
        if validation.financial.revenue_verified:
            financial_score += 50
        if validation.financial.employees_verified:
            financial_score += 30
        if validation.financial.financial_health == "positive":
            financial_score += 20
        score += weights["financial"] * financial_score

        # CSR score
        csr_score = 0
        if validation.csr.has_csr_programs:
            csr_score += 40
        csr_score += len(validation.csr.certifications) * 20
        csr_score += len(validation.csr.focus_areas) * 10
        csr_score = min(csr_score, 100)
        score += weights["csr"] * csr_score

        # Reputation score
        score += weights["reputation"] * validation.reputation.reputation_score

        return min(score, 100)

    def determine_validation_status(
            self,
            score: float,
            evidence_count: int
    ) -> Tuple[str, str]:
        """Determine validation status and confidence level"""

        if score >= 80 and evidence_count >= 10:
            return "verified", "high"
        elif score >= 60 and evidence_count >= 5:
            return "partial", "medium"
        elif score >= 40:
            return "unverified", "low"
        else:
            return "rejected", "low"