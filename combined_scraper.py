import requests
from bs4 import BeautifulSoup
import json
import time
from typing import Dict, List, Optional

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def scrape_outer_table(start_val: int, table_type: int) -> List[Dict]:
    """
    Scrapes table data for the given table_type and start value.
    Collects: test_id, name, url, remote_testing, adaptive_irt
    """
    url = f"{CATALOG_URL}?start={start_val}&type={table_type}"
    print(f"🔄 Scraping outer page: {url} (Table {table_type}, start={start_val})")

    try:
        res = requests.get(url, headers=HEADERS)
        if res.status_code != 200:
            print(f"⚠️ Failed with status code {res.status_code}")
            return []

        soup = BeautifulSoup(res.text, "html.parser")
        table_wrapper = soup.find("div", class_="custom__table-wrapper")
        if not table_wrapper:
            print(f"⚠️ Table {table_type} not found on page start={start_val}")
            return []

        rows = table_wrapper.find_all("tr")[1:]  # Skip header
        page_data = []

        for row in rows:
            a_tag = row.find("a")
            if not a_tag:
                continue

            test_id = row.get("data-course-id", None)
            name = a_tag.get_text(strip=True)
            url = BASE_URL + a_tag["href"]

            columns = row.find_all("td")
            if len(columns) < 4:
                continue

            remote_testing = "Yes" if columns[1].find("span", class_="-yes") else "No"
            adaptive_irt = "Yes" if columns[2].find("span", class_="-yes") else "No"

            page_data.append({
                "test_id": test_id,
                "name": name,
                "url": url,
                "remote_testing": remote_testing,
                "adaptive_irt": adaptive_irt
            })

        print(f"✅ Found {len(page_data)} entries from Table {table_type}")
        return page_data

    except Exception as e:
        print(f"❌ Error in outer scraping: {e}")
        return []

def scrape_inner_page(url: str) -> Optional[Dict]:
    """
    Scrapes additional details from the inner page of each test.
    """
    print(f"🔍 Scraping inner page: {url}")
    
    try:
        res = requests.get(url, headers=HEADERS)
        if res.status_code != 200:
            print(f"⚠️ Failed to fetch inner page with status code {res.status_code}")
            return None

        soup = BeautifulSoup(res.text, "html.parser")
        
        # Initialize details dictionary
        details = {}
        
        # Get description
        description_div = soup.find("div", class_="product-description")
        if description_div:
            details["Description"] = description_div.get_text(strip=True)
        
        # Get job levels
        job_levels_div = soup.find("div", class_="job-levels")
        if job_levels_div:
            details["Job levels"] = job_levels_div.get_text(strip=True)
        
        # Get languages
        languages_div = soup.find("div", class_="languages")
        if languages_div:
            details["Languages"] = languages_div.get_text(strip=True)
        
        # Get assessment length
        length_div = soup.find("div", class_="assessment-length")
        if length_div:
            details["Assessment length"] = length_div.get_text(strip=True)
        
        # Get test type
        test_type_div = soup.find("div", class_="test-type")
        if test_type_div:
            test_types = [t.strip() for t in test_type_div.get_text().split(",")]
            details["Test Type"] = test_types
        
        # Get remote testing from details
        remote_testing_div = soup.find("div", class_="remote-testing")
        if remote_testing_div:
            details["Remote Testing (from details)"] = remote_testing_div.get_text(strip=True)
        
        # Get category
        category_div = soup.find("div", class_="category")
        if category_div:
            details["category"] = category_div.get_text(strip=True)
        
        return details

    except Exception as e:
        print(f"❌ Error in inner scraping: {e}")
        return None

def remove_duplicates(data: List[Dict]) -> List[Dict]:
    """
    Removes duplicate entries based on test_id
    """
    seen = set()
    unique_data = []
    
    for item in data:
        test_id = item.get("test_id")
        if test_id and test_id not in seen:
            seen.add(test_id)
            unique_data.append(item)
    
    return unique_data

def main():
    # Initialize data storage
    all_data = []
    
    # Scrape Table 1 (type=1): 32 pages = start 0 to 372
    print("\n📊 Starting Table 1 scraping...")
    for start_val in range(0, 373, 12):
        page_data = scrape_outer_table(start_val, table_type=1)
        
        # For each entry, scrape inner page details
        for entry in page_data:
            inner_details = scrape_inner_page(entry["url"])
            if inner_details:
                entry.update(inner_details)
            time.sleep(1)  # Be nice to the server
        
        all_data.extend(page_data)
        time.sleep(2)  # Be nice to the server
    
    # Remove duplicates
    unique_data = remove_duplicates(all_data)
    
    # Save the data
    with open("combined_test_solutions.json", "w", encoding="utf-8") as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Done scraping! Total unique entries: {len(unique_data)}")
    
    # Scrape Table 2 (type=2): 12 pages = start 0 to 132

if __name__ == "__main__":
    main() 