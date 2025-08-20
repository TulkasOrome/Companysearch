#!/usr/bin/env python3
"""
TPB Production Scraper - 1000 Records Test
Formats data according to requirements and saves CSV properly
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import time
import csv
import json
from datetime import datetime
import random
import os
import re


class TPB1000RecordsScraper:
    def __init__(self, headless=False, output_dir=None):
        """Initialize scraper with production settings"""
        self.url = "https://myprofile.tpb.gov.au/public-register/"
        self.results = []
        self.total_scraped = 0
        self.target_records = 1000
        self.session_start = datetime.now()
        self.request_count = 0
        self.errors = []
        self.current_page = 1
        self.current_search_params = {}

        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.dirname(os.path.abspath(__file__))

        print(f"📁 Output directory: {self.output_dir}")

        # Rate limiting settings
        self.min_delay = 1.5  # Reduced for testing
        self.max_delay = 3  # Reduced for testing
        self.pause_after_n_requests = 30
        self.pause_duration = 5

        # Setup Chrome
        chrome_options = webdriver.ChromeOptions()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 15)

        print("✓ TPB 1000 Records Scraper initialized")
        print(f"  Target: {self.target_records} records")
        print(f"  Rate limiting: {self.min_delay}-{self.max_delay}s between requests")

    def extract_row_data(self, row):
        """Extract data according to requirements:
        1. Business / Trading Name
        2. Registration Number
        3. Business Address
        4. Registration Date
        5. Registration Expiry Date
        6. Sufficient Number of Individuals (ID only)
        7. Registration Status
        8. Service Type
        """
        try:
            cells = row.find_elements(By.TAG_NAME, "td")

            # Debug: print number of cells
            if self.total_scraped == 0:
                print(f"  Debug: Row has {len(cells)} cells")

            # Extract data based on cell positions
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

            # Map cells to fields (adjust based on actual table structure)
            if len(cells) >= 4:
                # Cell 0: Usually the business/practitioner name
                data['Business Trading Name'] = cells[0].text.strip() if cells[0].text else ""

                # Cell 1: Trading name (if different)
                trading_name = cells[1].text.strip() if len(cells) > 1 and cells[1].text else ""
                if trading_name and trading_name != data['Business Trading Name']:
                    data['Business Trading Name'] = f"{data['Business Trading Name']} / {trading_name}"

                # Look for registration number (usually numeric, 5+ digits)
                for i, cell in enumerate(cells):
                    text = cell.text.strip()
                    if re.match(r'^\d{5,}$', text):
                        data['Registration Number'] = text
                        break

                # Service Type (Tax Agent, BAS Agent, etc.)
                for i, cell in enumerate(cells):
                    text = cell.text.strip()
                    if any(term in text for term in ['Tax Agent', 'BAS Agent', 'Tax (Financial) Adviser']):
                        data['Service Type'] = text
                        break

                # Registration Status
                for i, cell in enumerate(cells):
                    text = cell.text.strip()
                    if any(term in text for term in ['Registered', 'Suspended', 'Terminated', 'Active']):
                        # Leave blank if Active (as per requirements)
                        if text != 'Active' and text != 'Registered':
                            data['Registration Status'] = text
                        break

                # Look for dates (DD/MM/YYYY format)
                dates_found = []
                for cell in cells:
                    text = cell.text.strip()
                    date_matches = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text)
                    dates_found.extend(date_matches)

                if len(dates_found) >= 1:
                    data['Registration Date'] = dates_found[0]
                if len(dates_found) >= 2:
                    data['Registration Expiry Date'] = dates_found[1]

                # Business Address (look for suburb/state pattern)
                for cell in cells:
                    text = cell.text.strip()
                    if any(state in text for state in ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']):
                        data['Business Address'] = text
                        break

            return data

        except Exception as e:
            self.errors.append(f"Row extraction error: {str(e)}")
            return None

    def rate_limit(self):
        """Implement rate limiting"""
        self.request_count += 1

        # Random delay
        delay = random.uniform(self.min_delay, self.max_delay)

        # Extended pause after N requests
        if self.request_count % self.pause_after_n_requests == 0:
            print(f"  ⏸ Pausing {self.pause_duration}s after {self.request_count} requests...")
            time.sleep(self.pause_duration)
        else:
            time.sleep(delay)

    def scrape_current_page(self):
        """Scrape all results from current page"""
        page_results = []

        try:
            # Wait for table
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
            time.sleep(1)  # Let table fully load

            # Get all rows
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
            print(f"  Found {len(rows)} rows on page {self.current_page}")

            for i, row in enumerate(rows):
                if self.total_scraped >= self.target_records:
                    break

                data = self.extract_row_data(row)
                if data and data['Business Trading Name']:
                    page_results.append(data)
                    self.total_scraped += 1

                    # Progress update
                    if self.total_scraped % 25 == 0:
                        elapsed = (datetime.now() - self.session_start).total_seconds()
                        rate = self.total_scraped / elapsed if elapsed > 0 else 0
                        eta = (self.target_records - self.total_scraped) / rate if rate > 0 else 0
                        print(f"  ✓ Progress: {self.total_scraped}/{self.target_records} records")
                        print(f"    Rate: {rate:.2f} records/sec | ETA: {eta / 60:.1f} minutes")

            return page_results

        except TimeoutException:
            print("  ⚠ Timeout waiting for results")
            return []
        except Exception as e:
            print(f"  ✗ Page scraping error: {e}")
            self.errors.append(str(e))
            return []

    def navigate_to_next_page(self):
        """Navigate to next page"""
        try:
            # Look for Next button/link
            next_xpath = "//a[contains(text(), 'Next') or contains(text(), '>')]"
            next_elements = self.driver.find_elements(By.XPATH, next_xpath)

            for element in next_elements:
                try:
                    # Check if not disabled
                    parent = element.find_element(By.XPATH, "..")
                    if 'disabled' not in parent.get_attribute('class'):
                        self.driver.execute_script("arguments[0].click();", element)
                        self.current_page += 1
                        print(f"  → Moving to page {self.current_page}")
                        self.rate_limit()
                        time.sleep(2)  # Wait for page load
                        return True
                except:
                    continue

            return False

        except Exception as e:
            return False

    def perform_search(self, search_params):
        """Perform a search"""
        self.current_search_params = search_params
        self.current_page = 1

        print(f"\n🔍 New search: {search_params if search_params else 'All records'}")

        try:
            # Navigate to page
            if "public-register" not in self.driver.current_url:
                self.driver.get(self.url)
                time.sleep(3)

            # Reset filters
            try:
                reset_btn = self.driver.find_element(By.ID, "resetfilters")
                self.driver.execute_script("arguments[0].click();", reset_btn)
                time.sleep(2)
            except:
                pass

            # Apply search parameters
            if 'state' in search_params:
                # Find state dropdown
                selects = self.driver.find_elements(By.TAG_NAME, "select")
                for select_elem in selects:
                    try:
                        select = Select(select_elem)
                        options = [opt.text for opt in select.options]
                        if search_params['state'] in options:
                            select.select_by_visible_text(search_params['state'])
                            print(f"  Selected state: {search_params['state']}")
                            break
                    except:
                        continue

            if 'practitioner_type' in search_params:
                selects = self.driver.find_elements(By.TAG_NAME, "select")
                for select_elem in selects:
                    try:
                        select = Select(select_elem)
                        options = [opt.text for opt in select.options]
                        if search_params['practitioner_type'] in options:
                            select.select_by_visible_text(search_params['practitioner_type'])
                            print(f"  Selected type: {search_params['practitioner_type']}")
                            break
                    except:
                        continue

            # Click Find
            find_button = self.driver.find_element(By.CSS_SELECTOR, "button.btn-entitylist-filter-submit")
            self.driver.execute_script("arguments[0].click();", find_button)
            print("  ✓ Search executed")

            time.sleep(4)
            self.rate_limit()
            return True

        except Exception as e:
            print(f"  ✗ Search error: {e}")
            self.errors.append(str(e))
            return False

    def save_checkpoint(self):
        """Save checkpoint to JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = os.path.join(self.output_dir, f"tpb_checkpoint_{timestamp}.json")

        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'total_scraped': self.total_scraped,
            'current_page': self.current_page,
            'current_search': self.current_search_params,
            'errors_count': len(self.errors),
            'last_5_errors': self.errors[-5:] if self.errors else [],
            'session_duration': str(datetime.now() - self.session_start)
        }

        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            print(f"  💾 Checkpoint saved: {checkpoint_file}")
            return checkpoint_file
        except Exception as e:
            print(f"  ✗ Failed to save checkpoint: {e}")
            return None

    def save_results_to_csv(self, filename=None):
        """Save results to CSV with proper formatting"""
        if not self.results:
            print("No results to save")
            return None

        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tpb_results_{self.total_scraped}_records_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)

        # Define fieldnames in the required order
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

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in self.results:
                    # Ensure all fields exist
                    row = {}
                    for field in fieldnames:
                        row[field] = result.get(field, '')
                    writer.writerow(row)

            print(f"  ✓ CSV saved: {filepath}")
            print(f"    Records: {len(self.results)}")

            # Verify file was created
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / 1024  # KB
                print(f"    File size: {file_size:.1f} KB")
                return filepath
            else:
                print("  ✗ File was not created!")
                return None

        except Exception as e:
            print(f"  ✗ Failed to save CSV: {e}")
            return None

    def check_session_health(self):
        """Check if session is still valid"""
        try:
            self.driver.find_element(By.CSS_SELECTOR, "button.btn-entitylist-filter-submit")
            return True
        except:
            print("  ⚠ Session check failed, refreshing...")
            self.driver.refresh()
            time.sleep(5)
            try:
                self.driver.find_element(By.CSS_SELECTOR, "button.btn-entitylist-filter-submit")
                print("  ✓ Session restored")
                return True
            except:
                print("  ✗ Session lost")
                return False

    def run_scraping(self):
        """Main scraping loop for 1000 records"""
        print("\n" + "=" * 60)
        print(f"STARTING TPB SCRAPING - TARGET: {self.target_records} RECORDS")
        print("=" * 60)

        # Search strategies - rotate through states and types
        search_strategies = [
            {'state': 'New South Wales', 'practitioner_type': 'Tax Agent'},
            {'state': 'Victoria', 'practitioner_type': 'Tax Agent'},
            {'state': 'Queensland', 'practitioner_type': 'Tax Agent'},
            {'state': 'New South Wales', 'practitioner_type': 'BAS Agent'},
            {'state': 'Victoria', 'practitioner_type': 'BAS Agent'},
            {'state': 'Western Australia', 'practitioner_type': 'Tax Agent'},
            {'state': 'South Australia', 'practitioner_type': 'Tax Agent'},
            {'practitioner_type': 'Tax Agent'},  # All states
            {'practitioner_type': 'BAS Agent'},  # All states
        ]

        strategy_index = 0

        while self.total_scraped < self.target_records:
            # Session health check every 100 records
            if self.total_scraped > 0 and self.total_scraped % 100 == 0:
                if not self.check_session_health():
                    print("  Attempting to restore session...")
                    self.driver.get(self.url)
                    time.sleep(5)

            # Get next search strategy
            if strategy_index >= len(search_strategies):
                print("\n⚠ Exhausted all search strategies, using random selections...")
                strategy_index = 0

            current_strategy = search_strategies[strategy_index]
            strategy_index += 1

            # Perform search
            if not self.perform_search(current_strategy):
                continue

            # Scrape with pagination
            pages_without_results = 0
            max_pages_per_search = 20
            pages_scraped = 0

            while (self.total_scraped < self.target_records and
                   pages_scraped < max_pages_per_search and
                   pages_without_results < 3):

                # Scrape current page
                page_results = self.scrape_current_page()

                if page_results:
                    self.results.extend(page_results)
                    pages_without_results = 0
                else:
                    pages_without_results += 1

                pages_scraped += 1

                # Save checkpoint every 50 records
                if self.total_scraped % 50 == 0 and self.total_scraped > 0:
                    self.save_checkpoint()
                    self.save_results_to_csv()  # Save progress

                # Try next page
                if self.total_scraped < self.target_records:
                    if not self.navigate_to_next_page():
                        print("  No more pages for this search")
                        break

        # Final save
        print("\n" + "=" * 60)
        print("SCRAPING COMPLETE")
        print("=" * 60)

        # Statistics
        elapsed = (datetime.now() - self.session_start).total_seconds()

        print(f"\n📊 Final Statistics:")
        print(f"  Total records scraped: {self.total_scraped}")
        print(f"  Total time: {elapsed / 60:.1f} minutes ({elapsed:.0f} seconds)")
        print(f"  Average rate: {self.total_scraped / elapsed:.2f} records/second")
        print(f"  Total requests: {self.request_count}")
        print(f"  Errors encountered: {len(self.errors)}")

        # Save final results
        csv_file = self.save_results_to_csv()
        json_file = self.save_checkpoint()

        print(f"\n📁 Output Files:")
        if csv_file:
            print(f"  CSV: {csv_file}")
        if json_file:
            print(f"  JSON: {json_file}")

        # Projection for 80,000 records
        if self.total_scraped > 0:
            time_per_record = elapsed / self.total_scraped
            estimated_hours = (80000 * time_per_record) / 3600
            print(f"\n📈 Projection for 80,000 records:")
            print(f"  Estimated time: {estimated_hours:.1f} hours")
            print(f"  With 5 parallel scrapers: {estimated_hours / 5:.1f} hours")
            print(f"  With 10 parallel scrapers: {estimated_hours / 10:.1f} hours")

        # Show sample results
        if self.results:
            print(f"\n📝 Sample Results (first 3):")
            for i, result in enumerate(self.results[:3], 1):
                print(f"\nRecord {i}:")
                for key, value in result.items():
                    if value:  # Only show non-empty fields
                        print(f"  {key}: {value}")

    def cleanup(self):
        """Clean up resources"""
        try:
            self.driver.quit()
            print("\n✓ Browser closed")
        except:
            pass


def main():
    """Main execution"""
    print("TPB Production Scraper - 1000 Records Test")
    print("=" * 60)

    try:
        # Initialize scraper
        scraper = TPB1000RecordsScraper(headless=False)

        # Run scraping
        scraper.run_scraping()

    except KeyboardInterrupt:
        print("\n\n⚠ Scraping interrupted by user")
        if 'scraper' in locals():
            scraper.save_checkpoint()
            scraper.save_results_to_csv()
            print("✓ Progress saved before exit")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if 'scraper' in locals():
            scraper.save_results_to_csv()
    finally:
        if 'scraper' in locals():
            scraper.cleanup()


if __name__ == "__main__":
    main()