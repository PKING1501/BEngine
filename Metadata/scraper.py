# import requests
# from bs4 import BeautifulSoup
# import csv

# # URL of the page
# url = "https://stockanalysis.com/list/mega-cap-stocks/"  # <-- replace with actual URL

# headers = {
#     "User-Agent": "Mozilla/5.0"
# }

# response = requests.get(url, headers=headers)
# soup = BeautifulSoup(response.text, "html.parser")
# time.sleep(10)  # be polite and avoid overwhelming the server
# # ---- Extract headers ----
# table_head = soup.find("thead")
# header_cells = table_head.find_all("th")

# headers = []
# for th in header_cells:
#     text = th.get_text(strip=True)
#     headers.append(text)

# print("Headers:", headers)

# # ---- Extract rows ----
# rows = []
# table_body = soup.find_all("tr", class_="svelte-1ro3niy")

# for tr in table_body:
#     cols = tr.find_all("td")
    
#     if len(cols) == 0:
#         continue  # skip header row
    
#     row = []

#     # No.
#     row.append(cols[0].get_text(strip=True))
    
#     # Symbol (inside <a>)
#     symbol = cols[1].find("a").get_text(strip=True)
#     row.append(symbol)
    
#     # Market Cap
#     row.append(cols[2].get_text(strip=True))
    
#     # Industry
#     row.append(cols[3].get_text(strip=True))
    
#     # Sector
#     row.append(cols[4].get_text(strip=True))

#     rows.append(row)

# # ---- Write to CSV ----
# with open("stocks.csv", "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(headers)
#     writer.writerows(rows)

# print("stocks.csv")

import csv
import time
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------------- CONFIG ----------------
URL = "https://stockanalysis.com/list/nano-cap-stocks/?page=2"
OUTPUT_FILE = "stocks.csv"

# ---------------- SETUP ----------------
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 15)

driver.get(URL)

print("\n👉 Please manually do the following in the browser:")
print("1. Click 'Indicators'")
print("2. Enable 'Industry' and 'Sector'")
print("3. Wait for table to update\n")

input("⏳ Press ENTER here AFTER you have selected Industry & Sector...")

# ---------------- SCRAPE ----------------
rows = []

table_rows = wait.until(EC.presence_of_all_elements_located(
    (By.CSS_SELECTOR, "tbody tr")
))

for tr in table_rows:
    cols = tr.find_elements(By.TAG_NAME, "td")

    if len(cols) < 5:
        continue

    try:
        symbol = cols[1].find_element(By.TAG_NAME, "a").text.strip()
        industry = cols[3].text.strip()
        sector = cols[4].text.strip()

        rows.append([symbol, industry, sector])
    except:
        continue

# ---------------- SAVE (APPEND MODE) ----------------
file_exists = os.path.isfile(OUTPUT_FILE)

with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # Write header only if file is new
    if not file_exists:
        writer.writerow(["Symbol", "Industry", "Sector"])

    writer.writerows(rows)

driver.quit()

print(f"\n✅ Added {len(rows)} rows to {OUTPUT_FILE}")