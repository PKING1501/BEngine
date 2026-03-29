import csv

# -------- FILE PATHS --------
INPUT_MAIN = "./V3/market_cap_ranked.csv"
INPUT_LOOKUP = "./industry_sector_info.csv"
OUTPUT_FILE = "./V3/market_cap_ranked_with_sector.csv"

# -------- LOAD LOOKUP (Symbol -> Sector) --------
symbol_to_sector = {}

with open(INPUT_LOOKUP, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        symbol = row["Symbol"].strip()
        sector = row["Sector"].strip()
        symbol_to_sector[symbol] = sector

print(f"Loaded {len(symbol_to_sector)} symbols from industry_sector_info.csv")

# -------- PROCESS MAIN FILE --------
updated_rows = []
missing_symbols = []
used_symbols = set()

with open(INPUT_MAIN, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames + ["Sector"]  # add new column

    for row in reader:
        symbol = row["Symbol"].strip()

        if symbol in symbol_to_sector:
            row["Sector"] = symbol_to_sector[symbol]
            used_symbols.add(symbol)
        else:
            row["Sector"] = ""
            missing_symbols.append(symbol)

        updated_rows.append(row)

# -------- FIND UNUSED SYMBOLS (in lookup but not in main) --------
unused_symbols = set(symbol_to_sector.keys()) - used_symbols

# -------- WRITE OUTPUT --------
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(updated_rows)

# -------- LOGGING --------
print("\n✅ Merge Complete!")
print(f"Output saved to: {OUTPUT_FILE}")

print(f"\n❌ Symbols NOT FOUND in lookup ({len(missing_symbols)}):")
for s in missing_symbols[:20]:  # print first 20
    print(s)
if len(missing_symbols) > 20:
    print("...")

print(f"\n⚠️ Symbols in lookup but NOT USED ({len(unused_symbols)}):")
for s in list(unused_symbols)[:20]:
    print(s)
if len(unused_symbols) > 20:
    print("...")