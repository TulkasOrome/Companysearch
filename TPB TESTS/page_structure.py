#!/usr/bin/env python3
"""
TPB Production Scraper - 50 Records with Complete Details
Extracts all 8 required fields from list and detail pages
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time
import re
import csv
import json
from datetime import datetime
import os
import random


class TPBProductionScraper50:
    def __init__(self, headless=False, target_records=50):
        """Initialize scraper"""
        self.url = "https://myprofile.tpb.gov.au/public-register/"
        self.results = []
        self.target_records = target_records
        self.total_scraped = 0
        self.session_start = datetime.now()

        # Rate limiting
        self.min_delay = 1.5
        self.max_delay = 3.0

        # Setup Chrome
        chrome_options = webdriver.ChromeOptions()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 15)

        print(f"✓ TPB Production Scraper initialized - Target: {target_records} records")

    def rate_limit(self):
        """Apply rate limiting between requests"""
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)

    def extract_list_data_with_status(self):
        """Extract data from search results including Registration Status"""
        list_data = []

        try:
            # Wait for table to load
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
            time.sleep(2)

            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")

                    if cells:
                        row_data = {
                            'list_name': '',
                            'registration_status': '',
                            'detail_url': None,
                            'registration_number': ''
                        }

                        # Get the full row text to check for status
                        row_text = row.text

                        # Extract name from first cell
                        if cells[0].text:
                            row_data['list_name'] = cells[0].text.strip()

                        # Check for registration status in the row
                        # Look for status indicators - these might be in any cell
                        status_found = False
                        for cell in cells:
                            cell_text = cell.text
                            if 'Suspended' in cell_text:
                                row_data['registration_status'] = 'Suspended'
                                status_found = True
                                break
                            elif 'Terminated' in cell_text:
                                row_data['registration_status'] = 'Terminated'
                                status_found = True
                                break
                            elif 'Under' in cell_text and 'review' in cell_text:
                                row_data['registration_status'] = 'Under ART review'
                                status_found = True
                                break
                            elif 'Federal Court' in cell_text:
                                row_data['registration_status'] = 'Federal Court Matter'
                                status_found = True
                                break

                        # If no special status found, it's Active/Registered (store as empty)
                        if not status_found:
                            row_data['registration_status'] = ''

                        # Try to get detail page link
                        try:
                            link = row.find_element(By.TAG_NAME, "a")
                            href = link.get_attribute('href')
                            if href and 'practitioner' in href:
                                row_data['detail_url'] = href
                                # Extract registration number from URL
                                ran_match = re.search(r'ran=(\d+)', href)
                                if ran_match:
                                    row_data['registration_number'] = ran_match.group(1)
                        except:
                            # No link found, try clicking the row
                            row_data['detail_url'] = 'click_required'

                        list_data.append(row_data)

                except Exception as e:
                    continue

        except Exception as e:
            print(f"Error extracting list data: {e}")

        return list_data

    def extract_detail_page_complete(self):
        """Extract all fields from detail page"""

        data = {
            'Business Trading Name': '',
            'Registration Number': '',
            'Business Address': '',
            'Registration Date': '',
            'Registration Expiry Date': '',
            'Sufficient Number Individuals (ID(s))': '',
            'Registration Status': '',
            'Service Type': ''
        }

        try:
            # Wait for page to load
            time.sleep(2)
            page_text = self.driver.find_element(By.TAG_NAME, "body").text

            # 1. Business Name - from heading
            try:
                for selector in ["h1", "h2", ".entity-title"]:
                    try:
                        heading = self.driver.find_element(By.CSS_SELECTOR, selector)
                        text = heading.text.strip()
                        if text and not any(
                                x in text.lower() for x in ['search', 'register', 'tpb', 'practitioner details']):
                            data['Business Trading Name'] = text
                            break
                    except:
                        continue
            except:
                pass

            # 2. Service Type
            service_types = ['Tax Agent', 'BAS Agent', 'Tax (Financial) Adviser']
            for service_type in service_types:
                if service_type in page_text:
                    data['Service Type'] = service_type
                    break

            # 3. Parse the page line by line for labeled fields
            lines = page_text.split('\n')

            for i, line in enumerate(lines):
                line_lower = line.lower().strip()

                # Registration Number
                if 'registration number' in line_lower and i < len(lines) - 1:
                    next_line = lines[i + 1].strip()
                    num_match = re.search(r'\d{5,}', next_line)
                    if num_match:
                        data['Registration Number'] = num_match.group()

                # Business Address - may span multiple lines
                elif 'business address' in line_lower and i < len(lines) - 1:
                    address_parts = []
                    for j in range(1, min(4, len(lines) - i)):
                        next_line = lines[i + j].strip()
                        if next_line and not any(
                                x in next_line.lower() for x in ['registration', 'sufficient', 'branches']):
                            address_parts.append(next_line)
                            # Stop when we hit a state (end of address)
                            if any(state in next_line for state in
                                   ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']):
                                break
                    if address_parts:
                        data['Business Address'] = ' '.join(address_parts)

                # Registration Date
                elif 'registration date' in line_lower and 'expiry' not in line_lower and i < len(lines) - 1:
                    date_match = re.search(r'\d{1,2}/\d{1,2}/\d{4}', lines[i + 1])
                    if date_match:
                        data['Registration Date'] = date_match.group()

                # Registration Expiry Date
                elif 'registration expiry' in line_lower and i < len(lines) - 1:
                    date_match = re.search(r'\d{1,2}/\d{1,2}/\d{4}', lines[i + 1])
                    if date_match:
                        data['Registration Expiry Date'] = date_match.group()

                # Sufficient Number Individuals
                elif 'sufficient number individuals' in line_lower and i < len(lines) - 1:
                    next_line = lines[i + 1].strip()
                    if 'None on record' in next_line:
                        data['Sufficient Number Individuals (ID(s))'] = ''
                    else:
                        # Extract ID numbers if present
                        id_matches = re.findall(r'\b\d{5,}\b', next_line)
                        if id_matches:
                            data['Sufficient Number Individuals (ID(s))'] = ', '.join(id_matches)
                        elif next_line and next_line not in ['Conditions', 'Branches', 'Suspensions']:
                            # Store the actual text if it's not a section header
                            data['Sufficient Number Individuals (ID(s))'] = next_line

            # 4. Fallback: Get registration number from URL
            if not data['Registration Number']:
                current_url = self.driver.current_url
                if 'ran=' in current_url:
                    ran_match = re.search(r'ran=(\d+)', current_url)
                    if ran_match:
                        data['Registration Number'] = ran_match.group(1)

            # 5. Fallback: Get dates using regex if not found
            if not data['Registration Date'] or not data['Registration Expiry Date']:
                dates = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', page_text)
                if dates:
                    if not data['Registration Date'] and len(dates) >= 1:
                        data['Registration Date'] = dates[0]
                    if not data['Registration Expiry Date'] and len(dates) >= 2:
                        data['Registration Expiry Date'] = dates[1]

        except Exception as e:
            print(f"Error extracting detail page: {e}")

        return data

    def scrape_practitioners(self):
        """Main scraping function"""
        print("\n" + "=" * 60)
        print(f"STARTING SCRAPE - TARGET: {self.target_records} RECORDS")
        print("=" * 60)

        page_number = 1

        while self.total_scraped < self.target_records:
            print(f"\n--- Processing Page {page_number} ---")

            # Navigate to search page if not there
            if page_number == 1:
                self.driver.get(self.url)
                time.sleep(3)

                # Click Find to get all records
                find_button = self.driver.find_element(By.CSS_SELECTOR, "button.btn-entitylist-filter-submit")
                self.driver.execute_script("arguments[0].click();", find_button)
                print("Searching for practitioners...")
                time.sleep(5)

            # Get current search results URL for returning
            search_url = self.driver.current_url

            # Extract list data with status
            list_data = self.extract_list_data_with_status()

            if not list_data:
                print("No more practitioners found")
                break

            print(f"Found {len(list_data)} practitioners on this page")

            # Process each practitioner
            for i, list_info in enumerate(list_data):
                if self.total_scraped >= self.target_records:
                    break

                print(
                    f"\nProcessing {self.total_scraped + 1}/{self.target_records}: {list_info.get('list_name', 'Unknown')}")

                try:
                    # Navigate to detail page
                    if list_info.get('detail_url') and list_info['detail_url'] != 'click_required':
                        self.driver.get(list_info['detail_url'])
                        self.rate_limit()
                    else:
                        # Try clicking on the row
                        rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
                        if i < len(rows):
                            self.driver.execute_script("arguments[0].click();", rows[i])
                            self.rate_limit()

                    # Extract detail page data
                    detail_data = self.extract_detail_page_complete()

                    # Merge list and detail data
                    complete_data = detail_data.copy()

                    # Add registration status from list
                    if list_info.get('registration_status') is not None:
                        complete_data['Registration Status'] = list_info['registration_status']

                    # Use registration number from list if not found in detail
                    if not complete_data['Registration Number'] and list_info.get('registration_number'):
                        complete_data['Registration Number'] = list_info['registration_number']

                    # Add to results
                    self.results.append(complete_data)
                    self.total_scraped += 1

                    # Progress update
                    if self.total_scraped % 10 == 0:
                        print(f"\n✓ Progress: {self.total_scraped}/{self.target_records} records scraped")
                        self.save_checkpoint()

                    # Return to search results for next practitioner
                    if self.total_scraped < self.target_records:
                        self.driver.get(search_url)
                        time.sleep(2)

                except Exception as e:
                    print(f"  Error processing practitioner: {e}")
                    # Return to search results
                    self.driver.get(search_url)
                    time.sleep(2)

            # Try to go to next page
            if self.total_scraped < self.target_records:
                if not self.navigate_to_next_page():
                    print("No more pages available")
                    break
                page_number += 1
                time.sleep(3)

        # Final save
        self.save_final_results()

    def navigate_to_next_page(self):
        """Navigate to the next page of results"""
        try:
            # Look for Next button
            next_xpath = "//a[contains(text(), 'Next') or contains(text(), '>')]"
            next_elements = self.driver.find_elements(By.XPATH, next_xpath)

            for element in next_elements:
                try:
                    # Check if not disabled
                    parent = element.find_element(By.XPATH, "..")
                    if 'disabled' not in parent.get_attribute('class'):
                        self.driver.execute_script("arguments[0].click();", element)
                        print("  → Moving to next page")
                        return True
                except:
                    continue

            # Alternative: Look for page numbers
            try:
                pagination = self.driver.find_elements(By.CSS_SELECTOR, ".pagination a")
                for page_link in pagination:
                    if page_link.text.isdigit():
                        current_page = int(page_link.text)
                        if current_page > 1:  # Click on next page number
                            self.driver.execute_script("arguments[0].click();", page_link)
                            return True
            except:
                pass

            return False

        except Exception as e:
            return False

    def save_checkpoint(self):
        """Save progress checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = f"tpb_checkpoint_{self.total_scraped}_records_{timestamp}.json"

        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'total_scraped': self.total_scraped,
            'results_count': len(self.results)
        }

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"  💾 Checkpoint saved: {checkpoint_file}")

    def save_final_results(self):
        """Save final results to CSV and JSON"""
        print("\n" + "=" * 60)
        print("SAVING RESULTS")
        print("=" * 60)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save to CSV
        csv_filename = f"tpb_results_{self.total_scraped}_records_{timestamp}.csv"

        fieldnames = [
            'Business Trading Name',
            'Registration Number',
            'Business Address',
            'Registration Date',
            'Registration Expiry Date',
            'Sufficient Number Individuals (ID(s))',
            'Registration Status',
            'Service Type'
        ]

        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {field: result.get(field, '') for field in fieldnames}
                writer.writerow(row)

        print(f"✓ CSV saved: {csv_filename}")
        print(f"  Total records: {len(self.results)}")

        # Save to JSON with statistics
        json_filename = f"tpb_results_{self.total_scraped}_records_{timestamp}.json"

        # Calculate field success rates
        field_rates = {}
        for field in fieldnames:
            found = sum(1 for r in self.results if r.get(field))
            field_rates[field] = {
                'found': found,
                'percentage': (found / len(self.results) * 100) if self.results else 0
            }

        # Calculate statistics
        elapsed = (datetime.now() - self.session_start).total_seconds()

        output = {
            'timestamp': datetime.now().isoformat(),
            'session_duration': f"{elapsed / 60:.1f} minutes",
            'total_records': len(self.results),
            'records_per_minute': (len(self.results) / (elapsed / 60)) if elapsed > 0 else 0,
            'field_success_rates': field_rates,
            'results': self.results
        }

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✓ JSON saved: {json_filename}")

        # Print statistics
        print("\n📊 FINAL STATISTICS:")
        print(f"  Total records scraped: {self.total_scraped}")
        print(f"  Time taken: {elapsed / 60:.1f} minutes")
        print(f"  Rate: {(len(self.results) / (elapsed / 60)):.1f} records/minute")

        print("\n📈 FIELD SUCCESS RATES:")
        for field, stats in field_rates.items():
            status = "✓" if stats['percentage'] >= 80 else "⚠" if stats['percentage'] >= 50 else "✗"
            print(f"  {status} {field}: {stats['percentage']:.0f}% ({stats['found']}/{len(self.results)})")

        # Show sample records
        if self.results:
            print("\n📝 SAMPLE RECORDS (first 2):")
            for i, record in enumerate(self.results[:2], 1):
                print(f"\nRecord {i}:")
                for field, value in record.items():
                    if value:
                        print(f"  {field}: {value[:50]}...")

    def cleanup(self):
        """Clean up resources"""
        try:
            self.driver.quit()
            print("\n✓ Browser closed")
        except:
            pass


def main():
    """Main execution"""
    print("TPB Production Scraper - 50 Records with Complete Details")
    print("=" * 60)

    scraper = None
    try:
        # Initialize scraper
        scraper = TPBProductionScraper50(headless=False, target_records=50)

        # Run scraping
        scraper.scrape_practitioners()

        print("\n✅ SCRAPING COMPLETE!")
        print("Check the CSV file for properly formatted results.")
        print("\n💡 For 80,000 records:")

        if scraper.total_scraped > 0:
            elapsed = (datetime.now() - scraper.session_start).total_seconds()
            rate = scraper.total_scraped / (elapsed / 60)  # records per minute
            estimated_hours = (80000 / rate) / 60

            print(f"  Estimated time: {estimated_hours:.1f} hours")
            print(f"  With 5 parallel scrapers: {estimated_hours / 5:.1f} hours")
            print(f"  With 10 parallel scrapers: {estimated_hours / 10:.1f} hours")

    except KeyboardInterrupt:
        print("\n\n⚠ Scraping interrupted by user")
        if scraper and scraper.results:
            scraper.save_final_results()
            print("✓ Progress saved before exit")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        if scraper and scraper.results:
            scraper.save_final_results()
    finally:
        if scraper:
            scraper.cleanup()

    print("\n✅ Complete!")


if __name__ == "__main__":
    main()