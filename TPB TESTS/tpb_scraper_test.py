#!/usr/bin/env python3
"""
TPB Basic Test Script - Simple and Focused
Based on diagnostic findings
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time
import json


class TPBBasicTest:
    def __init__(self):
        """Initialize with minimal setup"""
        self.url = "https://myprofile.tpb.gov.au/public-register/"

        # Simple Chrome setup
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 15)
        print("✓ Chrome WebDriver initialized")

    def test_basic_navigation(self):
        """Test 1: Can we load the page?"""
        print("\n" + "=" * 60)
        print("TEST 1: BASIC PAGE LOAD")
        print("=" * 60)

        self.driver.get(self.url)
        time.sleep(3)

        title = self.driver.title
        print(f"✓ Page loaded: {title}")

        # Check if key elements exist
        try:
            # Check for the Find button
            find_button = self.driver.find_element(By.CSS_SELECTOR, "button.btn-entitylist-filter-submit")
            print(f"✓ Find button located: {find_button.is_displayed()}")

            # Check for input fields
            name_input = self.driver.find_element(By.ID, "0")
            print(f"✓ Name input field located (ID=0)")

            reg_input = self.driver.find_element(By.ID, "1")
            print(f"✓ Registration number input located (ID=1)")

            suburb_input = self.driver.find_element(By.ID, "4")
            print(f"✓ Suburb input located (ID=4)")

        except Exception as e:
            print(f"✗ Error locating elements: {e}")

    def test_search_by_name(self):
        """Test 2: Can we search by name?"""
        print("\n" + "=" * 60)
        print("TEST 2: SEARCH BY NAME")
        print("=" * 60)

        try:
            # Get the name input field
            name_input = self.driver.find_element(By.ID, "0")

            # Clear and enter a name
            name_input.clear()
            name_input.send_keys("Smith")
            print("✓ Entered 'Smith' in name field")

            # Get current value to confirm
            value = name_input.get_attribute('value')
            print(f"✓ Field value confirmed: '{value}'")

            # Find and click the Find button
            find_button = self.driver.find_element(By.CSS_SELECTOR, "button.btn-entitylist-filter-submit")

            # Try JavaScript click since regular click might not work
            self.driver.execute_script("arguments[0].click();", find_button)
            print("✓ Clicked Find button (via JavaScript)")

            # Wait for results
            print("⏳ Waiting for results to load...")
            time.sleep(5)

            # Check if results appeared
            self.check_for_results()

        except Exception as e:
            print(f"✗ Search test failed: {e}")

    def test_search_by_suburb(self):
        """Test 3: Can we search by suburb?"""
        print("\n" + "=" * 60)
        print("TEST 3: SEARCH BY SUBURB")
        print("=" * 60)

        try:
            # First, reset the form
            reset_button = self.driver.find_element(By.ID, "resetfilters")
            self.driver.execute_script("arguments[0].click();", reset_button)
            print("✓ Reset filters")
            time.sleep(3)

            # Get the suburb input
            suburb_input = self.driver.find_element(By.ID, "4")
            suburb_input.clear()
            suburb_input.send_keys("Sydney")
            print("✓ Entered 'Sydney' in suburb field")

            # Click Find
            find_button = self.driver.find_element(By.CSS_SELECTOR, "button.btn-entitylist-filter-submit")
            self.driver.execute_script("arguments[0].click();", find_button)
            print("✓ Clicked Find button")

            # Wait for results
            print("⏳ Waiting for results...")
            time.sleep(5)

            # Check results
            self.check_for_results()

        except Exception as e:
            print(f"✗ Suburb search failed: {e}")

    def test_dropdown_selection(self):
        """Test 4: Can we use the dropdowns?"""
        print("\n" + "=" * 60)
        print("TEST 4: DROPDOWN SELECTION")
        print("=" * 60)

        try:
            # Reset first
            reset_button = self.driver.find_element(By.ID, "resetfilters")
            self.driver.execute_script("arguments[0].click();", reset_button)
            time.sleep(3)

            # Find all select elements
            selects = self.driver.find_elements(By.TAG_NAME, "select")
            print(f"✓ Found {len(selects)} dropdown selects")

            # Try to select from each
            for i, select_elem in enumerate(selects):
                try:
                    select = Select(select_elem)
                    options = select.options

                    if len(options) > 1:  # Has actual options
                        print(f"\nDropdown {i + 1}:")
                        print(f"  Options: {[opt.text for opt in options[:5]]}")  # First 5 options

                        # Try to select "Tax Agent" if available
                        for opt in options:
                            if "Tax Agent" in opt.text:
                                select.select_by_visible_text(opt.text)
                                print(f"  ✓ Selected: {opt.text}")
                                break
                except Exception as e:
                    print(f"  ⚠ Could not interact with dropdown {i + 1}")

            # Click Find after selection
            find_button = self.driver.find_element(By.CSS_SELECTOR, "button.btn-entitylist-filter-submit")
            self.driver.execute_script("arguments[0].click();", find_button)
            print("\n✓ Clicked Find button after dropdown selection")

            time.sleep(5)
            self.check_for_results()

        except Exception as e:
            print(f"✗ Dropdown test failed: {e}")

    def check_for_results(self):
        """Check what results we got"""
        print("\n📊 Checking for results...")

        try:
            # Method 1: Look for table rows
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
            if rows:
                print(f"✓ Found {len(rows)} table rows")

                # Sample first row
                if rows:
                    first_row = rows[0]
                    cells = first_row.find_elements(By.TAG_NAME, "td")
                    if cells:
                        print(f"  First row has {len(cells)} cells")
                        for j, cell in enumerate(cells[:5]):  # First 5 cells
                            text = cell.text[:50] if cell.text else "(empty)"
                            print(f"    Cell {j + 1}: {text}")

            # Method 2: Look for result count
            page_text = self.driver.find_element(By.TAG_NAME, "body").text

            # Check for "X results" or "X items"
            import re
            results_pattern = r'(\d+)\s*(results?|items?|records?)'
            match = re.search(results_pattern, page_text, re.IGNORECASE)
            if match:
                print(f"✓ Page shows: {match.group(0)}")

            # Check for "No results"
            if "No results" in page_text or "No records" in page_text:
                print("⚠ No results found")

            # Method 3: Check entity grid
            try:
                entity_grid = self.driver.find_element(By.ID, "EntityList13dc1fcc-d5e4-ee11-904c-6045bde7144a")
                print("✓ Entity grid is present")

                # Check for data in the grid
                grid_rows = entity_grid.find_elements(By.CSS_SELECTOR, "tr[data-id]")
                if grid_rows:
                    print(f"  ✓ Grid contains {len(grid_rows)} data rows")
            except:
                pass

        except Exception as e:
            print(f"⚠ Error checking results: {e}")

    def test_pagination(self):
        """Test 5: Check if pagination exists"""
        print("\n" + "=" * 60)
        print("TEST 5: PAGINATION CHECK")
        print("=" * 60)

        try:
            # Look for pagination elements
            pagination_elements = self.driver.find_elements(By.CSS_SELECTOR, ".pagination")
            if pagination_elements:
                print(f"✓ Found pagination controls")

                # Look for page links
                page_links = self.driver.find_elements(By.CSS_SELECTOR, ".pagination a")
                print(f"  Found {len(page_links)} page links")

                # Check for Next button
                next_buttons = self.driver.find_elements(By.XPATH, "//a[contains(text(), 'Next')]")
                if next_buttons:
                    print("  ✓ Next button found")
                    if next_buttons[0].is_enabled():
                        print("    Button is enabled - multiple pages exist")
                    else:
                        print("    Button is disabled - on last page")
            else:
                print("⚠ No pagination controls found")

        except Exception as e:
            print(f"✗ Pagination check failed: {e}")

    def capture_network_request(self):
        """Test 6: Check what happens when we search (network activity)"""
        print("\n" + "=" * 60)
        print("TEST 6: NETWORK ACTIVITY CHECK")
        print("=" * 60)

        try:
            # Execute JavaScript to monitor AJAX calls
            self.driver.execute_script("""
                window.capturedRequests = [];
                const originalFetch = window.fetch;
                window.fetch = function(...args) {
                    window.capturedRequests.push({url: args[0], method: 'FETCH'});
                    return originalFetch.apply(this, args);
                };

                const originalXHR = window.XMLHttpRequest.prototype.open;
                window.XMLHttpRequest.prototype.open = function(method, url) {
                    window.capturedRequests.push({url: url, method: method});
                    return originalXHR.apply(this, arguments);
                };
            """)
            print("✓ Network monitoring enabled")

            # Perform a search
            name_input = self.driver.find_element(By.ID, "0")
            name_input.clear()
            name_input.send_keys("Test")

            find_button = self.driver.find_element(By.CSS_SELECTOR, "button.btn-entitylist-filter-submit")
            self.driver.execute_script("arguments[0].click();", find_button)

            time.sleep(3)

            # Get captured requests
            captured = self.driver.execute_script("return window.capturedRequests;")
            if captured:
                print(f"✓ Captured {len(captured)} network requests:")
                for req in captured:
                    print(f"  {req['method']}: {req['url'][:80]}...")

        except Exception as e:
            print(f"✗ Network capture failed: {e}")

    def run_all_tests(self):
        """Run all tests in sequence"""
        print("\n" + "=" * 60)
        print("TPB BASIC FUNCTIONALITY TEST SUITE")
        print("=" * 60)

        self.test_basic_navigation()
        self.test_search_by_name()
        self.test_search_by_suburb()
        self.test_dropdown_selection()
        self.test_pagination()
        self.capture_network_request()

        print("\n" + "=" * 60)
        print("TEST SUITE COMPLETE")
        print("=" * 60)

        # Summary
        print("\n📋 SUMMARY:")
        print("If all tests passed, the site is scrapeable using:")
        print("  - Input IDs: 0 (name), 1 (reg number), 4 (suburb)")
        print("  - Button selector: button.btn-entitylist-filter-submit")
        print("  - Results in: table tbody tr OR entity grid")
        print("  - Use JavaScript clicks for buttons")
        print("\n💡 Next step: Build full scraper with pagination and data extraction")

    def cleanup(self):
        """Clean up"""
        input("\n🔍 Press Enter to close browser and exit...")
        self.driver.quit()
        print("✓ Browser closed")


if __name__ == "__main__":
    print("Starting TPB Basic Test Suite...")
    print("This will test all the basic functionality we need to scrape")

    tester = TPBBasicTest()
    try:
        tester.run_all_tests()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        tester.cleanup()