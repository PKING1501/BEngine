import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "https://stockanalysis.com/stocks/?page={}"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def scrape_stockanalysis():
    data = []

    for page in range(1, 13):  # pages 2 to 12
        print(f"Scraping page {page}...")
        url = BASE_URL.format(page)
        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            print(f"Failed to fetch page {page}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        rows = soup.find_all("tr", class_="svelte-1ro3niy")

        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue

            try:
                symbol = cols[0].text.strip()
                company = cols[1].text.strip()
                industry = cols[2].text.strip()

                data.append({
                    "Symbol": symbol,
                    "Company Name": company,
                    "Industry": industry
                })
            except Exception as e:
                continue

        time.sleep(1)  # be polite to server

    return pd.DataFrame(data)


def main():
    # Step 1: Scrape
    scraped_df = scrape_stockanalysis()
    scraped_df.to_csv("scraped_industries.csv", index=False)
    print("Scraped data saved.")

    # Step 2: Load your existing CSV
    market_df = pd.read_csv("./V3/market_cap_ranked.csv")

    # Step 3: Merge on Symbol
    merged_df = market_df.merge(
        scraped_df[["Symbol", "Industry"]],
        on="Symbol",
        how="left"
    )

    # Step 4: Save updated CSV
    merged_df.to_csv("market_cap_ranked.csv", index=False)

    print("Final merged file saved as market_cap_ranked.csv")


if __name__ == "__main__":
    main()