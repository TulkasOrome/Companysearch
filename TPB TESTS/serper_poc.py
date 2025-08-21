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
from openai import AzureOpenAI
from dotenv import load_dotenv
import csv

# Load environment variables
load_dotenv()


class TPBDataPOC:
    """POC for extracting Tax Practitioner data using GPT-4.1 and Serper"""

    def __init__(self, azure_key: str, azure_endpoint: str, azure_version: str, serper_api_key: str,
                 deployment_name: str = "gpt-4.1"):
        """Initialize with API keys"""
        self.client = AzureOpenAI(
            api_key=azure_key,
            api_version=azure_version,
            azure_endpoint=azure_endpoint
        )
        self.deployment_name = deployment_name
        self.serper_api_key = serper_api_key
        self.results = []
        self.serper_credits_used = 0

    def query_gpt_for_practitioners(self, location: str = "Sydney", count: int = 10) -> List[Dict[str, Any]]:
        """
        Query GPT-4.1 for tax practitioners in a location
        Test what data it actually knows
        """
        print(f"\n🤖 Querying Azure GPT-4 for {count} tax practitioners in {location}...")
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

        Return as a JSON array. Only include real practitioners you have knowledge of.
        If you don't know a field, use null.

        Format the response as a JSON object with a "practitioners" array:
        {{
          "practitioners": [
            {{
              "Business/Trading Name": "Example Accounting",
              "Registration Number": null,
              "Business Address": "123 Street, Sydney NSW 2000",
              "Service Type": "Tax Agent",
              ...
            }}
          ]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,  # Use deployment name from init
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant with knowledge of Australian tax practitioners."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for factual data
                max_tokens=2000
            )

            content = response.choices[0].message.content

            # Clean up response - remove markdown if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            # Parse JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from the text
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    data = json.loads(content[json_start:json_end])
                else:
                    print("❌ Could not parse JSON from response")
                    return []

            # Extract practitioners
            if isinstance(data, dict):
                practitioners = data.get('practitioners', data.get('results', []))
            elif isinstance(data, list):
                practitioners = data
            else:
                practitioners = []

            print(f"✅ GPT returned {len(practitioners)} practitioners")

            # Analyze what fields GPT actually knew
            self.analyze_gpt_knowledge(practitioners)

            return practitioners

        except Exception as e:
            print(f"❌ Error querying GPT: {e}")
            import traceback
            traceback.print_exc()
            return []

    def analyze_gpt_knowledge(self, practitioners: List[Dict[str, Any]]):
        """Analyze what information GPT-4.1 actually provided"""
        if not practitioners:
            print("No practitioners to analyze")
            return

        print("\n📊 Analyzing GPT's knowledge:")

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
                value = practitioner.get(field)
                if value and value != "null" and value != "":
                    field_counts[field] += 1

        print("\nField availability from GPT:")
        for field, count in field_counts.items():
            percentage = (count / len(practitioners) * 100) if practitioners else 0
            symbol = "✅" if percentage > 50 else "⚠️" if percentage > 0 else "❌"
            print(f"  {symbol} {field}: {count}/{len(practitioners)} ({percentage:.0f}%)")

    def search_serper(self, query: str, endpoint: str = "search") -> Dict[str, Any]:
        """Search using Serper API"""
        try:
            conn = http.client.HTTPSConnection("google.serper.dev")

            payload_dict = {
                "q": query,
                "gl": "au"
            }

            if endpoint == "places":
                payload_dict["location"] = "Sydney, Australia"

            payload = json.dumps(payload_dict)

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

    def enrich_with_serper(self, practitioner: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich practitioner data using Serper API"""
        name = practitioner.get('Business/Trading Name', 'Unknown')
        print(f"\n🔍 Enriching: {name}")

        enriched = practitioner.copy()
        enriched['serper_data'] = {}

        # 1. Web Search
        print("  📝 Web search...")
        web_results = self.search_serper(f"{name} tax agent Australia")
        enriched['serper_data']['web'] = web_results

        # Extract website from results
        if web_results.get('organic'):
            for result in web_results['organic'][:3]:
                link = result.get('link', '')
                if name.lower().replace(' ', '') in link.lower().replace('-', ''):
                    enriched['Website'] = enriched.get('Website') or link
                    print(f"    🌐 Found website: {link}")
                    break

        # 2. Places Search
        print("  📍 Places search...")
        places_results = self.search_serper(f"{name}", "places")
        enriched['serper_data']['places'] = places_results

        if places_results.get('places'):
            place = places_results['places'][0]

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

            if tpb_search.get('organic'):
                for result in tpb_search['organic']:
                    snippet = result.get('snippet', '')
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

    def process_batch(self, practitioners: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of practitioners with Serper enrichment"""
        print("\n" + "=" * 60)
        print("ENRICHING WITH SERPER API")
        print("=" * 60)

        enriched_results = []

        for i, practitioner in enumerate(practitioners, 1):
            print(f"\n[{i}/{len(practitioners)}]" + "-" * 40)

            try:
                enriched = self.enrich_with_serper(practitioner)
                enriched_results.append(enriched)

            except Exception as e:
                print(f"❌ Error enriching practitioner: {e}")
                enriched_results.append(practitioner)

        return enriched_results

    def generate_csv_output(self, practitioners: List[Dict[str, Any]], filename: str = None):
        """Generate CSV output matching TPB requirements"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'tpb_poc_results_{timestamp}.csv'

        fieldnames = [
            'Business Trading Name',
            'Registration Number',
            'Business Address',
            'Registration Date',
            'Registration Expiry Date',
            'Sufficient Number Individuals (ID(s))',
            'Registration Status',
            'Service Type',
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
                row = {
                    'Business Trading Name': practitioner.get('Business/Trading Name', ''),
                    'Registration Number': practitioner.get('Registration Number', ''),
                    'Business Address': practitioner.get('Business Address', ''),
                    'Registration Date': practitioner.get('Registration Date', ''),
                    'Registration Expiry Date': practitioner.get('Registration Expiry Date', ''),
                    'Sufficient Number Individuals (ID(s))': practitioner.get('Sufficient Number Individuals', ''),
                    'Registration Status': practitioner.get('Registration Status', ''),
                    'Service Type': practitioner.get('Service Type', 'Tax Agent'),
                    'Phone Number': practitioner.get('Phone Number', ''),
                    'Email Address': practitioner.get('Email Address', ''),
                    'Website': practitioner.get('Website', ''),
                    'Data Source': 'GPT + Serper' if practitioner.get('serper_data') else 'GPT Only',
                    'Completeness': practitioner.get('completeness', '0/6')
                }
                writer.writerow(row)

        print(f"\n💾 CSV saved to: {filename}")
        return filename

    def generate_report(self, practitioners: List[Dict[str, Any]]):
        """Generate summary report"""
        print("\n" + "=" * 60)
        print("FINAL REPORT")
        print("=" * 60)

        if not practitioners:
            print("No practitioners to report on")
            return

        total = len(practitioners)

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
        for key, label in [
            ('with_reg_number', 'Registration Number'),
            ('with_address', 'Business Address'),
            ('with_phone', 'Phone Number'),
            ('with_email', 'Email Address'),
            ('with_website', 'Website'),
            ('with_dates', 'Registration Dates')
        ]:
            count = stats[key]
            pct = (count / total * 100) if total > 0 else 0
            print(f"  ✅ With {label}: {count} ({pct:.0f}%)")

        print(f"\n💳 API Usage:")
        print(f"  Serper Credits Used: {self.serper_credits_used}")
        print(f"  Estimated cost: ${self.serper_credits_used * 0.0003:.2f}")

        print(f"\n📝 Sample Results:")
        for i, p in enumerate(practitioners[:3], 1):
            name = p.get('Business/Trading Name', 'Unknown')
            print(f"\n  {i}. {name}")
            if p.get('Registration Number'):
                print(f"     Reg #: {p['Registration Number']}")
            if p.get('Phone Number'):
                print(f"     Phone: {p['Phone Number']}")
            if p.get('Website'):
                print(f"     Website: {p['Website']}")

    def run_poc(self, location: str = "Sydney", count: int = 10):
        """Run the full POC test"""
        print("=" * 60)
        print("TAX PRACTITIONER DATA POC")
        print(f"Testing Azure GPT-4 + Serper API")
        print(f"Location: {location}, Count: {count}")
        print("=" * 60)

        # Step 1: Query GPT
        practitioners = self.query_gpt_for_practitioners(location, count)

        if not practitioners:
            print("❌ No practitioners returned from GPT")
            return []

        # Save GPT-only results
        gpt_file = f'tpb_gpt_only_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(gpt_file, 'w') as f:
            json.dump(practitioners, f, indent=2)
        print(f"\n💾 GPT-only results saved to: {gpt_file}")

        # Step 2: Enrich with Serper
        enriched_practitioners = self.process_batch(practitioners)

        # Step 3: Generate outputs
        self.generate_csv_output(enriched_practitioners)

        # Save enriched JSON
        enriched_file = f'tpb_enriched_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(enriched_file, 'w') as f:
            json.dump(enriched_practitioners, f, indent=2, default=str)
        print(f"💾 Enriched results saved to: {enriched_file}")

        # Step 4: Generate report
        self.generate_report(enriched_practitioners)

        return enriched_practitioners


def main():
    """Main POC runner"""
    # Load credentials from .env
    AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")

    # Deployment name - try common names
    DEPLOYMENT_NAME = "gpt-4.1"  # Change this to your actual deployment name

    if not AZURE_KEY or not AZURE_ENDPOINT:
        print("❌ Missing Azure OpenAI credentials in .env file")
        return

    if not SERPER_API_KEY:
        print("❌ Missing Serper API key in .env file")
        return

    print("✅ Loaded credentials from .env file")
    print(f"📍 Using Azure endpoint: {AZURE_ENDPOINT}")
    print(f"🤖 Using deployment: {DEPLOYMENT_NAME}")

    # Try to list available deployments (for debugging)
    try:
        from azure.core.credentials import AzureKeyCredential
        print("\n🔍 Attempting to connect to Azure OpenAI...")
    except:
        pass

    # Create POC instance
    poc = TPBDataPOC(AZURE_KEY, AZURE_ENDPOINT, AZURE_VERSION, SERPER_API_KEY, DEPLOYMENT_NAME)

    # Run POC test
    results = poc.run_poc(location="Sydney", count=5)  # Start with just 5 for testing

    print("\n" + "=" * 60)
    print("POC COMPLETE")
    print("=" * 60)

    print("\n🎯 Key Findings:")
    print("1. GPT Knowledge: Check what fields it actually knows")
    print("2. Serper Enrichment: See how much additional data we can get")
    print("3. Cost Analysis: Calculate actual API costs")

    if poc.serper_credits_used > 0 and results:
        credits_per_record = poc.serper_credits_used / len(results)
        total_credits = credits_per_record * 80000
        total_cost = total_credits * 0.0003
        print(f"\n📈 Scaling to 80,000 records:")
        print(f"  Estimated Serper credits: {total_credits:,.0f}")
        print(f"  Estimated Serper cost: ${total_cost:,.2f}")

    return results


if __name__ == "__main__":
    # Run the POC
    results = main()

    print("\n💡 Next Steps:")
    print("1. Review GPT's actual knowledge of tax practitioners")
    print("2. Assess if Serper enrichment provides enough value")
    print("3. Compare with direct TPB scraping approach")
    print("4. Decide on most cost-effective solution")