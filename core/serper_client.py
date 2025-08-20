# serper_client.py
"""
Enhanced Serper Client with all API endpoints
Handles Places, Maps, News, Shopping, and Web searches
"""

import aiohttp
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class SerperEndpoint(Enum):
    """Available Serper API endpoints"""
    SEARCH = "search"
    PLACES = "places"
    MAPS = "maps"
    NEWS = "news"
    SHOPPING = "shopping"
    IMAGES = "images"
    VIDEOS = "videos"
    SCHOLAR = "scholar"
    PATENTS = "patents"
    AUTOCOMPLETE = "autocomplete"


@dataclass
class SerperQuery:
    """Structured query for Serper API"""
    query: str
    endpoint: SerperEndpoint
    location: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    num_results: int = 10
    time_range: Optional[str] = None  # for news: "day", "week", "month", "year"
    device: str = "desktop"  # "desktop" or "mobile"
    page: int = 1


@dataclass
class SerperResponse:
    """Structured response from Serper API"""
    endpoint: SerperEndpoint
    query: str
    results: List[Dict[str, Any]]
    total_results: Optional[int] = None
    answer_box: Optional[Dict[str, Any]] = None
    knowledge_graph: Optional[Dict[str, Any]] = None
    related_searches: Optional[List[str]] = None
    raw_response: Optional[Dict[str, Any]] = None
    query_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


class SerperRateLimiter:
    """Rate limiter for Serper API calls"""

    def __init__(self, max_per_second: float = 10):
        self.max_per_second = max_per_second
        self.min_interval = 1.0 / max_per_second
        self.last_request_time = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        async with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self.last_request_time = time.time()


class EnhancedSerperClient:
    """Enhanced Serper client with all API endpoints and smart features"""

    # Country configurations for localization
    COUNTRY_CONFIGS = {
        "United States": {"gl": "us", "hl": "en"},
        "United Kingdom": {"gl": "uk", "hl": "en"},
        "Canada": {"gl": "ca", "hl": "en"},
        "Australia": {"gl": "au", "hl": "en"},
        "Germany": {"gl": "de", "hl": "de"},
        "France": {"gl": "fr", "hl": "fr"},
        "Spain": {"gl": "es", "hl": "es"},
        "Italy": {"gl": "it", "hl": "it"},
        "Japan": {"gl": "jp", "hl": "ja"},
        "Brazil": {"gl": "br", "hl": "pt"},
        "India": {"gl": "in", "hl": "en"},
        "Netherlands": {"gl": "nl", "hl": "nl"},
        "Sweden": {"gl": "se", "hl": "sv"},
        "Singapore": {"gl": "sg", "hl": "en"},
        "Mexico": {"gl": "mx", "hl": "es"},
        "New Zealand": {"gl": "nz", "hl": "en"},
        "Ireland": {"gl": "ie", "hl": "en"},
    }

    def __init__(
            self,
            api_key: str,
            rate_limit: float = 10,
            timeout: int = 30,
            max_retries: int = 3
    ):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev"
        self.session = None
        self.rate_limiter = SerperRateLimiter(max_per_second=rate_limit)
        self.timeout = timeout
        self.max_retries = max_retries
        self.total_queries = 0
        self.total_cost = 0.0

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _get_country_params(self, country: str) -> Dict[str, str]:
        """Get country-specific parameters"""
        return self.COUNTRY_CONFIGS.get(
            country,
            {"gl": "us", "hl": "en"}
        )

    def _build_request_data(self, query: SerperQuery) -> Dict[str, Any]:
        """Build request data from query object"""
        data = {"q": query.query}

        # Add location parameters
        if query.location:
            data["location"] = query.location

        # Add country parameters
        if query.country:
            country_params = self._get_country_params(query.country)
            data.update(country_params)

        # Add language if specified
        if query.language:
            data["hl"] = query.language

        # Add number of results
        if query.num_results != 10:
            data["num"] = query.num_results

        # Add time range for news
        if query.time_range and query.endpoint == SerperEndpoint.NEWS:
            data["time"] = query.time_range

        # Add device type
        if query.device != "desktop":
            data["device"] = query.device

        # Add pagination
        if query.page > 1:
            data["page"] = query.page

        return data

    async def _make_request(
            self,
            endpoint: str,
            data: Dict[str, Any],
            retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Make HTTP request with retry logic"""
        await self.rate_limiter.acquire()

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

        url = f"{self.base_url}/{endpoint}"

        try:
            async with self.session.post(url, json=data, headers=headers) as response:
                self.total_queries += 1
                self.total_cost += 0.001  # $0.001 per query

                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limited
                    if retry_count < self.max_retries:
                        wait_time = (2 ** retry_count) * 2  # Exponential backoff
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        return await self._make_request(endpoint, data, retry_count + 1)
                    else:
                        logger.error("Max retries exceeded for rate limiting")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    return None

        except asyncio.TimeoutError:
            logger.error(f"Request timeout for endpoint {endpoint}")
            if retry_count < self.max_retries:
                return await self._make_request(endpoint, data, retry_count + 1)
            return None
        except Exception as e:
            logger.error(f"Request exception: {e}")
            return None

    async def execute_query(self, query: SerperQuery) -> SerperResponse:
        """Execute a single query"""
        start_time = time.time()

        # Build request data
        request_data = self._build_request_data(query)

        # Make request
        raw_response = await self._make_request(query.endpoint.value, request_data)

        query_time = time.time() - start_time

        if raw_response:
            # Parse response based on endpoint
            return self._parse_response(query, raw_response, query_time)
        else:
            return SerperResponse(
                endpoint=query.endpoint,
                query=query.query,
                results=[],
                success=False,
                error="Failed to get response from API",
                query_time=query_time
            )

    def _parse_response(
            self,
            query: SerperQuery,
            raw_response: Dict[str, Any],
            query_time: float
    ) -> SerperResponse:
        """Parse response based on endpoint type"""
        response = SerperResponse(
            endpoint=query.endpoint,
            query=query.query,
            results=[],
            raw_response=raw_response,
            query_time=query_time
        )

        # Extract common elements
        response.answer_box = raw_response.get("answerBox")
        response.knowledge_graph = raw_response.get("knowledgeGraph")
        response.related_searches = raw_response.get("relatedSearches", [])

        # Extract results based on endpoint
        if query.endpoint == SerperEndpoint.SEARCH:
            response.results = raw_response.get("organic", [])
            response.total_results = raw_response.get("searchInformation", {}).get("totalResults")

        elif query.endpoint == SerperEndpoint.PLACES:
            response.results = raw_response.get("places", [])

        elif query.endpoint == SerperEndpoint.MAPS:
            response.results = raw_response.get("places", [])

        elif query.endpoint == SerperEndpoint.NEWS:
            response.results = raw_response.get("news", [])

        elif query.endpoint == SerperEndpoint.SHOPPING:
            response.results = raw_response.get("shopping", [])

        elif query.endpoint == SerperEndpoint.IMAGES:
            response.results = raw_response.get("images", [])

        elif query.endpoint == SerperEndpoint.VIDEOS:
            response.results = raw_response.get("videos", [])

        elif query.endpoint == SerperEndpoint.SCHOLAR:
            response.results = raw_response.get("organic", [])

        elif query.endpoint == SerperEndpoint.PATENTS:
            response.results = raw_response.get("organic", [])

        elif query.endpoint == SerperEndpoint.AUTOCOMPLETE:
            response.results = raw_response.get("suggestions", [])

        return response

    async def batch_search(
            self,
            queries: List[SerperQuery],
            parallel_limit: int = 5
    ) -> List[SerperResponse]:
        """Execute multiple queries with controlled parallelism"""
        results = []

        for i in range(0, len(queries), parallel_limit):
            batch = queries[i:i + parallel_limit]

            # Execute batch in parallel
            tasks = [self.execute_query(query) for query in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Progress update
            logger.info(f"Processed {min(i + parallel_limit, len(queries))}/{len(queries)} queries")

        return results

    async def places_search(
            self,
            company_name: str,
            location: Optional[str] = None,
            country: Optional[str] = None
    ) -> SerperResponse:
        """Convenience method for places search"""
        query = SerperQuery(
            query=f"{company_name} {location or ''}".strip(),
            endpoint=SerperEndpoint.PLACES,
            location=location,
            country=country
        )
        return await self.execute_query(query)

    async def news_search(
            self,
            company_name: str,
            topics: List[str] = None,
            time_range: str = "month",
            country: Optional[str] = None
    ) -> SerperResponse:
        """Convenience method for news search"""
        query_str = company_name
        if topics:
            query_str += " " + " OR ".join(topics)

        query = SerperQuery(
            query=query_str,
            endpoint=SerperEndpoint.NEWS,
            time_range=time_range,
            country=country
        )
        return await self.execute_query(query)

    async def web_search(
            self,
            query_str: str,
            country: Optional[str] = None,
            num_results: int = 10
    ) -> SerperResponse:
        """Convenience method for web search"""
        query = SerperQuery(
            query=query_str,
            endpoint=SerperEndpoint.SEARCH,
            country=country,
            num_results=num_results
        )
        return await self.execute_query(query)

    async def shopping_search(
            self,
            product: str,
            location: Optional[str] = None,
            country: Optional[str] = None
    ) -> SerperResponse:
        """Convenience method for shopping search"""
        query = SerperQuery(
            query=product,
            endpoint=SerperEndpoint.SHOPPING,
            location=location,
            country=country
        )
        return await self.execute_query(query)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_queries": self.total_queries,
            "total_cost": self.total_cost,
            "cost_per_query": 0.001
        }