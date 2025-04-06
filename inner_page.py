import json
import time
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def extract_test_details(url):
    """Scrapes the test detail page and returns structured information"""
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Locate the main product-catalogue block
        main_block = soup.find("div", class_="product-catalogue module")
        if not main_block:
            print(f"⚠️ No product-catalogue block found at {url}")
            return None

        # Extract title
        title = main_block.find("h1").get_text(strip=True) if main_block.find("h1") else "N/A"

        # Extract test_id from data-course-id
        test_id = None
        course_div = main_block.find("div", attrs={"data-course-id": True})
        if course_div:
            test_id = course_div["data-course-id"]

        # Extract all rows within the module
        rows = main_block.select("div.product-catalogue-training-calendar__row")
        extracted_data = {}

        for row in rows:
            heading = row.find("h4")
            content = row.find("p")
            if heading and content:
                extracted_data[heading.get_text(strip=True)] = content.get_text(strip=True)

            # Handle "Downloads" section
            if heading and heading.get_text(strip=True) == "Downloads":
                links = row.select("a")
                downloads = []
                for link in links:
                    downloads.append({
                        "text": link.get_text(strip=True),
                        "href": link["href"]
                    })
                extracted_data["Downloads"] = downloads

        # Extract test type and remote testing (within same parent div)
        length_row = main_block.find("h4", string="Assessment length")
        test_types = []
        remote_testing_status = "N/A"

        if length_row:
            parent_div = length_row.find_parent("div", class_="product-catalogue-training-calendar__row")
            if parent_div:
                type_spans = parent_div.select("span.product-catalogue__key")
                test_types = [span.get_text(strip=True) for span in type_spans]

                # Remote Testing Indicator
                remote_p = parent_div.find("p", string=lambda text: "Remote Testing" in text if text else False)
                if remote_p:
                    if remote_p.find("span", class_="-yes"):
                        remote_testing_status = "Yes"
                    elif remote_p.find("span", class_="-no"):
                        remote_testing_status = "No"

        extracted_data["Test Type"] = test_types
        extracted_data["Remote Testing (from details)"] = remote_testing_status

        return {
            "test_id": test_id,
            "title": title,
            "url": url,
            "details": extracted_data
        }

    except Exception as e:
        print(f"❌ Failed to scrape {url}: {str(e)}")
        return None

def scrape_all_details(input_json_path, output_json_path):
    """Reads list of tests, scrapes inner pages, and writes to output"""
    with open(input_json_path, 'r', encoding='utf-8') as f:
        tests = json.load(f)

    all_data = []
    for i, test in enumerate(tests):
        print(f"🔍 Scraping {i+1}/{len(tests)}: {test['name']}")
        detail = extract_test_details(test['url'])
        if detail:
            all_data.append(detail)
        time.sleep(1.5)  # polite scraping delay

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Scraped details of {len(all_data)} tests to {output_json_path}")

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    # Choose input and output files
    scrape_all_details("table1_individual_test_solutions.json", "details.json")
    # scrape_all_details("job_solutions.json", "job_test_details.json")
