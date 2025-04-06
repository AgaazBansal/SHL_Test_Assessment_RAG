import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def scrape_table(start_val, table_type):
    """
    Scrapes table data for the given table_type and start value.
    Collects: test_id, name, url, remote_testing, adaptive_irt
    """
    url = f"{CATALOG_URL}?start={start_val}&type={table_type}"
    print(f"🔄 Scraping: {url} (Table {table_type}, start={start_val})")

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
        print(f"❌ Error: {e}")
        return []

def scrape_full_table(table_type, max_start):
    """
    Loops over all pages for the given table type.
    """
    all_data = []

    for start_val in range(0, max_start + 1, 12):
        data = scrape_table(start_val, table_type)
        all_data.extend(data)
        time.sleep(2)  # Respectful crawling

    return all_data

def main():
    # Scrape Table 1 (type=1): 32 pages = start 0 to 372
    table1_data = scrape_full_table(table_type=1, max_start=372)
    with open("table1_individual_test_solutions.json", "w", encoding="utf-8") as f:
        json.dump(table1_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done scraping Table 1! Total entries: {len(table1_data)}")

    # Scrape Table 2 (type=2): 12 pages = start 0 to 132
    #table2_data = scrape_full_table(table_type=2, max_start=132)
    #with open("table2_job_solutions.json", "w", encoding="utf-8") as f:
        #json.dump(table2_data, f, indent=2, ensure_ascii=False)

    #print(f"\n✅ Done scraping Table 2! Total entries: {len(table2_data)}")

if __name__ == "__main__":
    main()
