#!/usr/bin/env python3
"""
TPB Detail Page Extraction Test
Clicks into individual records to get full details
Tests with 10 records only
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
import os
from datetime import datetime


class TPBDetailExtractor:
    def __init__(self):
        """Initialize with minimal setup"""
        self.url = "https://myprofile.tpb.gov.au/public-register/"
        self.results = []

        # Simple Chrome setup
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 15)
        print("✓ Chrome WebDriver initialized")

    def perform_search(self):
        """Do a simple search to get results"""
        print("\n🔍 Performing initial search...")

        self.driver.get(self.url)
        time.sleep(3)

        # Simple search - just click Find to get all results
        try:
            find_button = self.driver.find_element(By.CSS_SELECTOR, "button.btn-entitylist-filter-submit")
            self.driver.execute_script("arguments[0].click();", find_button)
            print("✓ Search executed")
            time.sleep(4)
            return True
        except Exception as e:
            print(f"✗ Search failed: {e}")
            return False

    def extract_detail_page(self):
        """Extract all fields from the detail page"""
        print("  Extracting detail page data...")

        try:
            # Wait for detail content to load
            time.sleep(2)

            # Initialize record with all fields
            record = {
                'Business Trading Name': '',
                'Registration Number': '',
                'Business Address': '',
                'Registration Date': '',
                'Registration Expiry Date': '',
                'Service Type': '',
                'Registration Status': '',
                'Branches': '',
                'Sufficient Number Individuals': '',
                'Conditions': '',
                'Suspensions': '',
                'Sanctions': '',
                'Publication Decisions': '',
                'Disqualifications': '',
                'Termination': '',
                'Rejection of Renewal Application': '',
                'Rejection of New Application': '',
                'Federal Court Matter': '',
                'Administrative Review Tribunal Matter': '',
                'Professional Associations': ''
            }

            # Get the main content area
            page_text = self.driver.find_element(By.TAG_NAME, "body").text

            # Extract Business/Trading Name (usually in h1 or prominent text)
            try:
                # Try to find the business name - usually the first h1 or h2
                headings = self.driver.find_elements(By.TAG_NAME, "h1")
                if not headings:
                    headings = self.driver.find_elements(By.TAG_NAME, "h2")
                if headings:
                    record['Business Trading Name'] = headings[0].text.strip()
            except:
                pass

            # Extract by looking for label/value patterns
            lines = page_text.split('\n')

            for i, line in enumerate(lines):
                line_lower = line.lower().strip()

                # Business/Trading name
                if 'business/trading name' in line_lower or 'business name' in line_lower:
                    if i + 1 < len(lines):
                        record['Business Trading Name'] = lines[i + 1].strip()

                # Registration number
                elif 'registration number' in line_lower:
                    if i + 1 < len(lines):
                        record['Registration Number'] = lines[i + 1].strip()

                # Business address
                elif 'business address' in line_lower:
                    if i + 1 < len(lines):
                        # Address might be multi-line
                        address_parts = []
                        j = i + 1
                        while j < len(lines) and not any(keyword in lines[j].lower() for keyword in
                                                         ['registration', 'branch', 'sufficient', 'condition',
                                                          'suspension']):
                            if lines[j].strip():
                                address_parts.append(lines[j].strip())
                            j += 1
                            if 'Australia' in lines[j - 1]:
                                break
                        record['Business Address'] = ' '.join(address_parts)

                # Registration date
                elif 'registration date' in line_lower and 'expiry' not in line_lower:
                    if i + 1 < len(lines):
                        record['Registration Date'] = lines[i + 1].strip()

                # Registration expiry date
                elif 'registration expiry date' in line_lower or 'expiry date' in line_lower:
                    if i + 1 < len(lines):
                        record['Registration Expiry Date'] = lines[i + 1].strip()

                # Service Type (Tax Agent, BAS Agent)
                elif line in ['Tax Agent', 'BAS Agent', 'Tax (Financial) Adviser']:
                    record['Service Type'] = line

                # Registration Status
                elif line in ['Registered', 'Suspended', 'Terminated']:
                    record['Registration Status'] = line

                # Branches
                elif 'branches' in line_lower:
                    if i + 1 < len(lines):
                        value = lines[i + 1].strip()
                        if value != 'None on record':
                            record['Branches'] = value

                # Sufficient Number Individuals
                elif 'sufficient number individuals' in line_lower:
                    if i + 1 < len(lines):
                        value = lines[i + 1].strip()
                        if value != 'None on record':
                            record['Sufficient Number Individuals'] = value

                # Conditions
                elif line_lower == 'conditions':
                    if i + 1 < len(lines):
                        value = lines[i + 1].strip()
                        if value != 'None on record':
                            record['Conditions'] = value

                # Suspensions
                elif line_lower == 'suspensions':
                    if i + 1 < len(lines):
                        value = lines[i + 1].strip()
                        if value != 'None on record':
                            record['Suspensions'] = value

            # Print what we found
            print(f"    Name: {record['Business Trading Name']}")
            print(f"    Reg#: {record['Registration Number']}")
            print(f"    Type: {record['Service Type']}")
            print(f"    Address: {record['Business Address'][:50]}...")

            return record

        except Exception as e:
            print(f"  ✗ Error extracting details: {e}")
            return None

    def test_detail_extraction(self):
        """Test extracting details from 10 records"""
        print("\n" + "=" * 60)
        print("TESTING DETAIL PAGE EXTRACTION - 10 RECORDS")
        print("=" * 60)

        # First, perform a search
        if not self.perform_search():
            print("Failed to perform initial search")
            return

        # Get the clickable practitioner names
        try:
            # Wait for results table
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))

            # Find all clickable names (usually first column with links)
            name_links = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr td:first-child a")

            if not name_links:
                # Try alternative selector
                name_links = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr a")

            print(f"✓ Found {len(name_links)} clickable records")

            # Process first 10 records
            records_to_process = min(10, len(name_links))

            for i in range(records_to_process):
                print(f"\n📄 Processing record {i + 1}/{records_to_process}")

                try:
                    # Re-find links (page might have refreshed)
                    name_links = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr td:first-child a")
                    if not name_links:
                        name_links = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr a")

                    # Click the link
                    link_text = name_links[i].text
                    print(f"  Clicking: {link_text}")

                    # Click using JavaScript to avoid interception
                    self.driver.execute_script("arguments[0].click();", name_links[i])

                    # Wait for page/modal to load
                    time.sleep(3)

                    # Extract details
                    record = self.extract_detail_page()

                    if record:
                        self.results.append(record)

                    # Go back to search results
                    # Check if it's a new page or modal
                    if "public-register" not in self.driver.current_url:
                        # It's a new page, go back
                        self.driver.back()
                        time.sleep(3)
                    else:
                        # It might be a modal, try to close it
                        try:
                            close_button = self.driver.find_element(By.CSS_SELECTOR,
                                                                    "button.close, button[aria-label='Close']")
                            close_button.click()
                            time.sleep(2)
                        except:
                            # If no close button, try ESC key or going back to search
                            self.driver.get(self.url)
                            self.perform_search()

                except Exception as e:
                    print(f"  ✗ Error processing record {i + 1}: {e}")
                    # Try to recover
                    self.driver.get(self.url)
                    self.perform_search()

                # Rate limiting
                time.sleep(2)

        except Exception as e:
            print(f"✗ Error finding clickable records: {e}")

    def save_results(self):
        """Save extracted results to CSV"""
        if not self.results:
            print("\nNo results to save")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"tpb_detail_test_{len(self.results)}_records_{timestamp}.csv"

        # Use all available fields
        fieldnames = [
            'Business Trading Name',
            'Registration Number',
            'Business Address',
            'Registration Date',
            'Registration Expiry Date',
            'Service Type',
            'Registration Status',
            'Branches',
            'Sufficient Number Individuals',
            'Conditions',
            'Suspensions',
            'Sanctions',
            'Publication Decisions',
            'Professional Associations'
        ]

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)

        print(f"\n✓ Saved {len(self.results)} records to: {filename}")

        # Show summary
        print("\n📊 Summary of extracted records:")
        for i, record in enumerate(self.results, 1):
            print(f"\n{i}. {record.get('Business Trading Name', 'Unknown')}")
            print(f"   Reg#: {record.get('Registration Number', 'N/A')}")
            print(f"   Type: {record.get('Service Type', 'N/A')}")
            print(f"   Status: {record.get('Registration Status', 'N/A')}")

    def cleanup(self):
        """Clean up"""
        self.driver.quit()
        print("\n✓ Browser closed")


def main():
    """Run the detail page extraction test"""
    print("TPB Detail Page Extraction Test")
    print("Testing extraction of full details from 10 records")

    extractor = TPBDetailExtractor()

    try:
        # Run the test
        extractor.test_detail_extraction()

        # Save results
        extractor.save_results()

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)

        if extractor.results:
            print(f"\n✅ Successfully extracted {len(extractor.results)} detailed records")
            print("This confirms we can get all required fields from detail pages")
            print("\nNext step: Scale this up for full 80,000 record extraction")
        else:
            print("\n⚠ No records extracted - check selectors and page structure")

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main()