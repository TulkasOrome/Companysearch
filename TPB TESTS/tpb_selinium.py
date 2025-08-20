#!/usr/bin/env python3
"""
TPB Public Register Working Scraper
Optimized for React-based interface with proper element waiting and interaction
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotInteractableException
import time
import csv
import json
from datetime import datetime
import os


class TPBScraper:
    def __init__(self, headless=False):
        """Initialize the TPB Scraper with optimized settings"""
        self.url = "https://myprofile.tpb.gov.au/public-register/"
        self.results = []
        self.total_scraped = 0

        # Setup Chrome options
        chrome_options = webdriver.ChromeOptions()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # For better performance
        chrome_options.add_argument('--disable-images')
        chrome_options.add_argument('--disable-gpu')

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, 15)
            self.short_wait = WebDriverWait(self.driver, 5)
            print("✓ Chrome WebDriver initialized")
        except Exception as e:
            print(f"✗ Error initializing Chrome WebDriver: {e}")
            raise

    def wait_for_react(self):
        """Wait for React to finish rendering"""
        time.sleep(2)  # Initial wait for React
        # Wait for any loading indicators to disappear
        try:
            self.wait.until(lambda driver: driver.execute_script("return document.readyState") == "complete")
        except:
            pass

    def safe_click(self, element):
        """Safely click an element using multiple methods"""
        try:
            # Method 1: Regular click
            element.click()
        except ElementNotInteractableException:
            try:
                # Method 2: JavaScript click
                self.driver.execute_script("arguments[0].click();", element)
            except:
                # Method 3: Action chains
                ActionChains(self.driver).move_to_element(element).click().perform()

    def safe_send_keys(self, element, text):
        """Safely send keys to an element"""
        try:
            element.clear()
            element.send_keys(text)
        except ElementNotInteractableException:
            # Use JavaScript as fallback
            self.driver.execute_script(f"arguments[0].value = '{text}';", element)
            # Trigger change event for React
            self.driver.execute_script("arguments[0].dispatchEvent(new Event('change', { bubbles: true }));", element)

    def perform_search(self, search_params):
        """Perform a search with given parameters"""
        print(f"\nSearching with params: {search_params}")

        try:
            # Navigate to the page
            self.driver.get(self.url)
            self.wait_for_react()

            # Handle Practitioner Type dropdown (select with ID=2)
            if 'practitioner_type' in search_params:
                try:
                    type_select = self.wait.until(EC.presence_of_element_located((By.ID, "2")))
                    select = Select(type_select)
                    select.select_by_visible_text(search_params['practitioner_type'])
                    print(f"  ✓ Selected practitioner type: {search_params['practitioner_type']}")
                    time.sleep(1)
                except Exception as e:
                    print(f"  ⚠ Could not select practitioner type: {e}")

            # Handle State dropdown (select with ID=3)
            if 'state' in search_params:
                try:
                    state_select = self.wait.until(EC.presence_of_element_located((By.ID, "3")))
                    select = Select(state_select)
                    select.select_by_visible_text(search_params['state'])
                    print(f"  ✓ Selected state: {search_params['state']}")
                    time.sleep(1)
                except Exception as e:
                    print(f"  ⚠ Could not select state: {e}")

            # Handle Name/Business input (input with ID=0)
            if 'name' in search_params:
                try:
                    # Wait for the element to be clickable
                    name_input = self.wait.until(EC.element_to_be_clickable((By.ID, "0")))
                    # Click to focus
                    self.safe_click(name_input)
                    time.sleep(0.5)
                    # Clear and type
                    self.safe_send_keys(name_input, search_params['name'])
                    print(f"  ✓ Entered name: {search_params['name']}")
                    time.sleep(1)
                except Exception as e:
                    print(f"  ⚠ Could not enter name: {e}")

            # Handle Registration Number input (input with ID=1)
            if 'registration_number' in search_params:
                try:
                    reg_input = self.wait.until(EC.element_to_be_clickable((By.ID, "1")))
                    self.safe_click(reg_input)
                    time.sleep(0.5)
                    self.safe_send_keys(reg_input, search_params['registration_number'])
                    print(f"  ✓ Entered registration number: {search_params['registration_number']}")
                    time.sleep(1)
                except Exception as e:
                    print(f"  ⚠ Could not enter registration number: {e}")

            # Handle Suburb input (input with ID=4)
            if 'suburb' in search_params:
                try:
                    suburb_input = self.wait.until(EC.element_to_be_clickable((By.ID, "4")))
                    self.safe_click(suburb_input)
                    time.sleep(0.5)
                    self.safe_send_keys(suburb_input, search_params['suburb'])
                    print(f"  ✓ Entered suburb: {search_params['suburb']}")
                    time.sleep(1)
                except Exception as e:
                    print(f"  ⚠ Could not enter suburb: {e}")

            # Click the Find button
            try:
                # Find button by text
                find_button = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//button[text()='Find']")))
                self.safe_click(find_button)
                print("  ✓ Clicked Find button")

                # Wait for results to load
                time.sleep(3)
                self.wait_for_react()

                return True

            except Exception as e:
                print(f"  ✗ Could not click Find button: {e}")
                return False

        except Exception as e:
            print(f"✗ Search failed: {e}")
            return False

    def extract_results(self):
        """Extract results from the current page"""
        results_on_page = []

        try:
            # Wait a bit for results to render
            time.sleep(2)

            # Check if there are results
            page_text = self.driver.find_element(By.TAG_NAME, "body").text

            if "No results found" in page_text or "0 results" in page_text:
                print("  No results found")
                return []

            # Try multiple selectors for result containers
            result_selectors = [
                "div.MuiPaper-root",  # Material-UI cards
                "div[class*='card']",
                "div[class*='result']",
                "table tbody tr",
                "div[class*='practitioner']",
                "div[class*='item']"
            ]

            result_elements = []
            for selector in result_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    result_elements = elements
                    print(f"  Found {len(elements)} results using selector: {selector}")
                    break

            # Extract data from each result
            for i, element in enumerate(result_elements[:10], 1):  # Limit to first 10 for testing
                try:
                    text = element.text
                    if text and len(text) > 20:  # Filter out empty or minimal elements
                        # Parse the text to extract structured data
                        data = self.parse_result_text(text)
                        if data:
                            results_on_page.append(data)
                            print(f"    Result {i}: {data.get('name', 'Unknown')[:50]}")
                except Exception as e:
                    print(f"    Error extracting result {i}: {e}")

            return results_on_page

        except Exception as e:
            print(f"  Error extracting results: {e}")
            return []

    def parse_result_text(self, text):
        """Parse the raw text from a result element into structured data"""
        lines = text.split('\n')
        data = {
            'raw_text': text,
            'extracted_date': datetime.now().isoformat()
        }

        # Common patterns in TPB data
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for registration number (usually numeric)
            if line.isdigit() and len(line) >= 5:
                data['registration_number'] = line

            # Look for registration type
            elif 'Tax Agent' in line or 'BAS Agent' in line or 'Tax (Financial) Adviser' in line:
                data['service_type'] = line

            # Look for status
            elif 'Registered' in line or 'Suspended' in line or 'Terminated' in line:
                data['registration_status'] = line

            # Look for dates (DD/MM/YYYY format)
            elif '/' in line and len(line) == 10:
                if 'registration_date' not in data:
                    data['registration_date'] = line
                else:
                    data['expiry_date'] = line

            # Business name (usually the first substantial line)
            elif len(line) > 5 and 'name' not in data:
                data['name'] = line

        return data if 'name' in data or 'registration_number' in data else None

    def handle_pagination(self):
        """Handle pagination to get all results"""
        all_results = []
        page = 1

        while True:
            print(f"\n  Processing page {page}...")

            # Extract results from current page
            results = self.extract_results()
            all_results.extend(results)

            # Check for next page
            try:
                # Look for Next button or pagination controls
                next_selectors = [
                    "//button[contains(text(), 'Next')]",
                    "//a[contains(text(), 'Next')]",
                    "//button[@aria-label='Next page']",
                    "//button[contains(@class, 'next')]"
                ]

                next_button = None
                for selector in next_selectors:
                    try:
                        elements = self.driver.find_elements(By.XPATH, selector)
                        if elements and elements[0].is_enabled():
                            next_button = elements[0]
                            break
                    except:
                        continue

                if next_button:
                    self.safe_click(next_button)
                    print(f"  ✓ Moving to page {page + 1}")
                    time.sleep(2)
                    self.wait_for_react()
                    page += 1

                    # Safety limit for testing
                    if page > 5:
                        print("  Reached page limit for testing")
                        break
                else:
                    print("  No more pages")
                    break

            except Exception as e:
                print(f"  No next page found: {e}")
                break

        return all_results

    def save_results(self, results, filename=None):
        """Save results to CSV file"""
        if not results:
            print("No results to save")
            return

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tpb_results_{timestamp}.csv"

        # Get all unique keys from results
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())

        # Write to CSV
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = sorted(list(all_keys))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"\n✓ Saved {len(results)} results to {filename}")

    def run_test_scrape(self):
        """Run a test scrape with various search parameters"""
        print("\n" + "=" * 60)
        print("RUNNING TEST SCRAPE")
        print("=" * 60)

        # Test different search combinations
        test_searches = [
            {'state': 'New South Wales', 'practitioner_type': 'Tax Agent'},
            {'suburb': 'Sydney'},
            {'name': 'Smith'},
        ]

        all_results = []

        for search_params in test_searches:
            if self.perform_search(search_params):
                results = self.handle_pagination()
                all_results.extend(results)
                print(f"  Collected {len(results)} results")

            # Be respectful with delays
            time.sleep(2)

        # Save all results
        self.save_results(all_results)

        print(f"\n✓ Total results collected: {len(all_results)}")

        return all_results

    def cleanup(self):
        """Clean up the WebDriver"""
        try:
            self.driver.quit()
            print("\n✓ WebDriver closed")
        except:
            pass


def main():
    """Main execution function"""
    print("TPB Register Scraper - Optimized for React")
    print("=" * 60)

    try:
        # Initialize scraper (set headless=True for background operation)
        scraper = TPBScraper(headless=False)

        # Run test scrape
        results = scraper.run_test_scrape()

        # Print sample results
        if results:
            print("\nSample Results:")
            for i, result in enumerate(results[:3], 1):
                print(f"\nResult {i}:")
                for key, value in result.items():
                    if key != 'raw_text':  # Skip raw text in display
                        print(f"  {key}: {value}")

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if 'scraper' in locals():
            scraper.cleanup()


if __name__ == "__main__":
    main()