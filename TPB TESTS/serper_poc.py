#!/usr/bin/env python3
"""
tpb_gpt_serper_poc.py - POC for getting Tax Practitioner data using GPT-4.1 + Serper
Tests what contact information GPT-4.1 knows about Australian tax practitioners
Then enriches with Serper API for additional contact details
"""

import asyncio
import json
import http.client
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
import os
from openai import AsyncOpenAI


class TPBDataPOC:
    """POC for extracting Tax Practitioner data using GPT-4.1 and Serper"""

    def __init__(self, openai_api_key: str, serper_api_key: str):
        """Initialize with API keys"""
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.serper_api_key = serper_api_key
        self.results = []
        self.serper_credits_used = 0

    async def query_gpt_for_practitioners(self, location: str = "Sydney", count: int = 10) -> List[Dict[str, Any]]:
        """
        Query GPT-4.1 for tax practitioners in a location
        Test what data it actually knows
        """
        print(f"\n🤖 Querying GPT-4.1 for {count} tax practitioners in {location}...")
        print("=" * 60)

        prompt = f"""
        List {count} real tax agents, BAS agents, or tax financial advisers registered in {location}, Australia.

        For each practitioner, provide AS MUCH of the following information as you know:

        REQUIRED FIELDS (from TPB register):
        1. Business/Trading Name
        2. Registration Number (if known)
        3. Business Address
        4. Registration Date (if known)
        5. Registration Expiry Date (if known)
        6. Sufficient Number Individuals (ID only if known)
        7. Registration Status (leave blank if Active/Registered)
        8. Service Type (Tax Agent, BAS Agent, or Tax Financial Adviser)

        ADDITIONAL CONTACT INFO (if known):
        9. Phone Number
        10. Email Address
        11. Website
        12. Contact Person Name

        Return as a JSON array with these exact field names.
        Only include real practitioners you have knowledge of.
        If you don't know a field, use null.

        Focus on well-known firms that likely have public information available.
        """

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant with knowledge of Australian tax practitioners and the Tax Practitioners Board register."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3  # Lower temperature for factual data
            )

            content = response.choices[0].message.content
            data = json.loads(content)

            # Extract practitioners array from response
            if isinstance(data, dict):
                # Handle various possible response structures
                practitioners = data.get('practitioners', data.get('results', data.get('data', [])))
            else:
                practitioners = data

            print(f"✅ GPT-4.1 returned {len(practitioners)} practitioners")

            # Analyze what fields GPT actually knew
            self.analyze_gpt_knowledge(practitioners)

            return practitioners

        except Exception as e:
            print(f"❌ Error querying GPT-4.1: {e}")
            return []

    def analyze_gpt_knowledge(self, practitioners: List[Dict[str, Any]]):
        """Analyze what information GPT-4.1 actually provided"""
        print("\n📊 Analyzing GPT-4.1's knowledge:")

        field_counts = {
            "Business/Trading Name": 0,
            "Registration Number": 0,
            "Business Address": 0,
            "Registration Date": 0,
            "Registration Expiry Date": 0,
            "Sufficient Number Individuals": 0,
            "Registration Status": 0,
            "Service Type": 0,
            "Phone Number": 0,
            "Email Address": 0,
            "Website": 0,
            "Contact Person Name": 0
        }

        for practitioner in practitioners:
            for field in field_counts:
                # Check various possible field name formats
                field_variants = [
                    field,
                    field.lower().replace(" ", "_"),
                    field.replace(" ", ""),
                    field.replace("/", "_").replace(" ", "_")
                ]

                for variant in field_variants:
                    if practitioner.get(variant) and practitioner[variant] != "null":
                        field_counts[field] += 1
                        break

        print("\nField availability from GPT-4.1:")
        for field, count in field_counts.items():
            percentage = (count / len(practitioners) * 100) if practitioners else 0
            symbol = "✅" if percentage > 50 else "⚠️" if percentage > 0 else "❌"
            print(f"  {symbol} {field}: {count}/{len(practitioners)} ({percentage:.0f}%)")

    def search_serper(self, query: str, endpoint: str = "search") -> Dict[str, Any]:
        """
        Search using Serper API
        Endpoints: search, places, news
        """
        try:
            conn = http.client.HTTPSConnection("google.serper.dev")

            payload = json.dumps({
                "q": query,
                "gl": "au",
                "location": "Sydney, Australia" if endpoint == "places" else None
            })

            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }

            url = f"/{endpoint}" if endpoint != "search" else "/search"
            conn.request("POST", url, payload, headers)
            res = conn.getresponse()
            data = res.read()

            result = json.loads(data.decode("utf-8"))
            self.serper_credits_used += result.get('credits', 1)

            return result

        except Exception as e:
            print(f"    ❌ Serper {endpoint} error: {e}")
            return {}

    async def enrich_with_serper(self, practitioner: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich practitioner data using Serper API
        """
        name = practitioner.get('Business/Trading Name') or practitioner.get('business_name', 'Unknown')
        print(f"\n🔍 Enriching: {name}")

        enriched = practitioner.copy()
        enriched['serper_data'] = {}

        # 1. Web Search for general info
        print("  📝 Web search...")
        web_results = self.search_serper(f"{name} tax agent Australia")
        enriched['serper_data']['web'] = web_results

        # Extract website from organic results
        if web_results.get('organic'):
            for result in web_results['organic'][:3]:
                link = result.get('link', '')
                # Check if it's likely the company's website
                if name.lower().replace(' ', '') in link.lower().replace('-', ''):
                    enriched['Website'] = enriched.get('Website') or link
                    print(f"    🌐 Found website: {link}")
                    break

        # 2. Places Search for contact info
        print("  📍 Places search...")
        places_results = self.search_serper(f"{name}")
        enriched['serper_data']['places'] = places_results

        if places_results.get('places'):
            place = places_results['places'][0]

            # Extract contact information
            if place.get('phoneNumber'):
                enriched['Phone Number'] = enriched.get('Phone Number') or place['phoneNumber']
                print(f"    📞 Found phone: {place['phoneNumber']}")

            if place.get('address'):
                enriched['Business Address'] = enriched.get('Business Address') or place['address']
                print(f"    📍 Found address: {place['address']}")

            if place.get('website'):
                enriched['Website'] = enriched.get('Website') or place['website']
                print(f"    🌐 Found website: {place['website']}")

        # 3. Try to find registration number if missing
        if not enriched.get('Registration Number'):
            print("  🔢 Searching for registration number...")
            tpb_search = self.search_serper(f'"{name}" "registration number" site:tpb.gov.au')

            # Try to extract registration number from snippets
            if tpb_search.get('organic'):
                for result in tpb_search['organic']:
                    snippet = result.get('snippet', '')
                    # Look for 8-digit numbers (typical TPB format)
                    reg_numbers = re.findall(r'\b\d{8}\b', snippet)
                    if reg_numbers:
                        enriched['Registration Number'] = reg_numbers[0]
                        print(f"    🔢 Found registration: {reg_numbers[0]}")
                        break

        # Calculate completeness
        required_fields = [
            'Business/Trading Name', 'Registration Number', 'Business Address',
            'Registration Date', 'Registration Expiry Date', 'Service Type'
        ]

        filled_fields = sum(1 for field in required_fields if enriched.get(field))
        enriched['completeness'] = f"{filled_fields}/{len(required_fields)}"

        return enriched

    async def process_batch(self, practitioners: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of practitioners with Serper enrichment
        """
        print("\n" + "=" * 60)
        print("ENRICHING WITH SERPER API")
        print("=" * 60)

        enriched_results = []

        for i, practitioner in enumerate(practitioners, 1):
            print(f"\n[{i}/{len(practitioners)}]" + "-" * 40)

            try:
                enriched = await self.enrich_with_serper(practitioner)
                enriched_results.append(enriched)

                # Rate limiting
                if i < len(practitioners):
                    await asyncio.sleep(1)  # Be nice to the API

            except Exception as e:
                print(f"❌ Error enriching practitioner: {e}")
                enriched_results.append(practitioner)

        return enriched_results

    def generate_csv_output(self, practitioners: List[Dict[str, Any]], filename: str = None):
        """
        Generate CSV output matching TPB requirements
        """
        import csv

        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'tpb_poc_results_{timestamp}.csv'

        # Required field names for CSV
        fieldnames = [
            'Business Trading Name',
            'Registration Number',
            'Business Address',
            'Registration Date',
            'Registration Expiry Date',
            'Sufficient Number Individuals (ID(s))',
            'Registration Status',
            'Service Type',
            # Additional fields for analysis
            'Phone Number',
            'Email Address',
            'Website',
            'Data Source',
            'Completeness'
        ]

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for practitioner in practitioners:
                row = {}

                # Map fields (handle various naming conventions)
                row['Business Trading Name'] = (
                        practitioner.get('Business/Trading Name') or
                        practitioner.get('business_name') or
                        practitioner.get('name', '')
                )

                row['Registration Number'] = practitioner.get('Registration Number', '')
                row['Business Address'] = practitioner.get('Business Address', '')
                row['Registration Date'] = practitioner.get('Registration Date', '')
                row['Registration Expiry Date'] = practitioner.get('Registration Expiry Date', '')
                row['Sufficient Number Individuals (ID(s))'] = practitioner.get('Sufficient Number Individuals', '')
                row['Registration Status'] = practitioner.get('Registration Status', '')
                row['Service Type'] = practitioner.get('Service Type', 'Tax Agent')  # Default to Tax Agent

                # Additional fields
                row['Phone Number'] = practitioner.get('Phone Number', '')
                row['Email Address'] = practitioner.get('Email Address', '')
                row['Website'] = practitioner.get('Website', '')
                row['Data Source'] = 'GPT-4.1' if not practitioner.get('serper_data') else 'GPT-4.1 + Serper'
                row['Completeness'] = practitioner.get('completeness', '0/6')

                writer.writerow(row)

        print(f"\n💾 CSV saved to: {filename}")
        return filename

    def generate_report(self, practitioners: List[Dict[str, Any]]):
        """
        Generate summary report
        """
        print("\n" + "=" * 60)
        print("FINAL REPORT")
        print("=" * 60)

        total = len(practitioners)

        # Calculate statistics
        stats = {
            'with_reg_number': sum(1 for p in practitioners if p.get('Registration Number')),
            'with_address': sum(1 for p in practitioners if p.get('Business Address')),
            'with_phone': sum(1 for p in practitioners if p.get('Phone Number')),
            'with_email': sum(1 for p in practitioners if p.get('Email Address')),
            'with_website': sum(1 for p in practitioners if p.get('Website')),
            'with_dates': sum(1 for p in practitioners if p.get('Registration Date'))
        }

        print(f"\n📊 Data Completeness:")
        print(f"  Total Practitioners: {total}")
        print(
            f"  ✅ With Registration Number: {stats['with_reg_number']} ({stats['with_reg_number'] / total * 100:.0f}%)")
        print(f"  ✅ With Business Address: {stats['with_address']} ({stats['with_address'] / total * 100:.0f}%)")
        print(f"  ✅ With Phone Number: {stats['with_phone']} ({stats['with_phone'] / total * 100:.0f}%)")
        print(f"  ✅ With Email Address: {stats['with_email']} ({stats['with_email'] / total * 100:.0f}%)")
        print(f"  ✅ With Website: {stats['with_website']} ({stats['with_website'] / total * 100:.0f}%)")
        print(f"  ✅ With Registration Dates: {stats['with_dates']} ({stats['with_dates'] / total * 100:.0f}%)")

        print(f"\n💳 API Usage:")
        print(f"  Serper Credits Used: {self.serper_credits_used}")
        print(f"  Estimated cost: ${self.serper_credits_used * 0.0003:.2f}")

        print(f"\n📝 Sample Results:")
        for i, p in enumerate(practitioners[:3], 1):
            name = p.get('Business/Trading Name') or p.get('business_name', 'Unknown')
            print(f"\n  {i}. {name}")
            if p.get('Registration Number'):
                print(f"     Reg #: {p['Registration Number']}")
            if p.get('Phone Number'):
                print(f"     Phone: {p['Phone Number']}")
            if p.get('Website'):
                print(f"     Website: {p['Website']}")
            print(f"     Completeness: {p.get('completeness', 'N/A')}")

    async def run_poc(self, location: str = "Sydney", count: int = 10):
        """
        Run the full POC test
        """
        print("=" * 60)
        print("TAX PRACTITIONER DATA POC")
        print(f"Testing GPT-4.1 + Serper API")
        print(f"Location: {location}, Count: {count}")
        print("=" * 60)

        # Step 1: Query GPT-4.1
        practitioners = await self.query_gpt_for_practitioners(location, count)

        if not practitioners:
            print("❌ No practitioners returned from GPT-4.1")
            return []

        # Save GPT-only results for comparison
        gpt_only_file = f'tpb_gpt_only_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(gpt_only_file, 'w') as f:
            json.dump(practitioners, f, indent=2)
        print(f"\n💾 GPT-only results saved to: {gpt_only_file}")

        # Step 2: Enrich with Serper
        enriched_practitioners = await self.process_batch(practitioners)

        # Step 3: Generate outputs
        csv_file = self.generate_csv_output(enriched_practitioners)

        # Save enriched JSON
        enriched_file = f'tpb_enriched_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(enriched_file, 'w') as f:
            json.dump(enriched_practitioners, f, indent=2, default=str)
        print(f"💾 Enriched results saved to: {enriched_file}")

        # Step 4: Generate report
        self.generate_report(enriched_practitioners)

        return enriched_practitioners


async def main():
    """
    Main POC runner
    """
    # API Keys - replace with your actual keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY", "99c44b79892f5f7499accf2d7c26d93313880937")

    # Create POC instance
    poc = TPBDataPOC(OPENAI_API_KEY, SERPER_API_KEY)

    # Run POC test
    # Start with small batch to test
    results = await poc.run_poc(location="Sydney", count=10)

    print("\n" + "=" * 60)
    print("POC COMPLETE")
    print("=" * 60)

    print("\n🎯 Key Findings:")
    print("1. GPT-4.1 Knowledge: Check what fields it actually knows")
    print("2. Serper Enrichment: See how much additional data we can get")
    print("3. Cost Analysis: Calculate actual API costs for full dataset")
    print("4. Data Quality: Assess if this approach meets requirements")

    print("\n📈 Scaling to 80,000 records:")
    if poc.serper_credits_used > 0:
        credits_per_record = poc.serper_credits_used / len(results) if results else 0
        total_credits = credits_per_record * 80000
        total_cost = total_credits * 0.0003  # $0.30 per 1000
        print(f"  Estimated Serper credits: {total_credits:,.0f}")
        print(f"  Estimated Serper cost: ${total_cost:,.2f}")
        print(f"  GPT-4.1 cost: Would depend on token usage")

    return results


if __name__ == "__main__":
    # Run the POC
    asyncio.run(main()

    print("\n💡 Next Steps:")
    print("1. Review GPT-4.1's actual knowledge of tax practitioners")
    print("2. Assess if Serper enrichment provides enough value")
    print("3. Compare with direct TPB scraping approach")
    print("4. Decide on most cost-effective solution")