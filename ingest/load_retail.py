"""Part 2: turn real retail transactions into searchable invoice vectors.

Source: UCI "Online Retail II" (id 502), CC BY 4.0 - real transactions from a
UK non-store online retailer, Dec 2009 to Dec 2011, ~1.07M rows.

Pipeline:  load -> clean -> roll up to invoice -> narrative -> embed -> DynamoDB

Usage:
    python ingest/load_retail.py --dry-run --limit 5          # inspect output
    python ingest/load_retail.py --source csv --path x.csv    # local file
    python ingest/load_retail.py --limit 5000                 # embed and load
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

REGION = os.environ.get("AWS_REGION", "us-east-1")
TABLE = os.environ.get("TABLE_NAME", "RetailTransactions")

# Bump when you change the narrative template - it invalidates every embedding.
TEMPLATE_VERSION = "v1"

# StockCodes that are not products. Leave these in and your semantic search
# cheerfully returns postage and bank charges.
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "m", "D", "C2", "S", "BANK CHARGES", "AMAZONFEE",
    "CRUK", "PADS", "B", "gift_0001",
}

# Inline filters are equality-only (no >, no BETWEEN), so every filterable
# numeric has to become a bucketed string.
VALUE_BANDS = [(50, "under-50"), (250, "50-250"), (1000, "250-1000")]
SIZE_BANDS = [(3, "small-1-2"), (11, "medium-3-10"), (31, "large-11-30")]


def load_raw(source: str, path: str | None) -> pd.DataFrame:
    if source == "csv":
        return pd.read_csv(path)
    from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo

    return fetch_ucirepo(id=502).data.features


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"UnitPrice": "Price", "CustomerID": "Customer ID",
                            "InvoiceNo": "Invoice"})
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()

    before = len(df)
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)]
    df = df[df["Description"].notna() & (df["Description"] != "") & (df["Description"] != "nan")]
    df = df[df["Quantity"] > 0]          # negatives are returns; keep separately if you want them
    df = df[df["Price"] > 0]             # zero-price rows are adjustments, not sales
    df = df[~df["Invoice"].str.upper().str.startswith("C")]  # cancellations
    print(f"  cleaned: {before} -> {len(df)} rows ({before - len(df)} dropped)")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["LineTotal"] = df["Quantity"] * df["Price"]
    return df


def band(value: float, bands: list, top: str) -> str:
    for threshold, label in bands:
        if value < threshold:
            return label
    return top


def rollup(df: pd.DataFrame) -> pd.DataFrame:
    """One row per invoice. The invoice - the basket - is the embeddable unit.
    A single line item is just a product name; the basket has structure."""
    grouped = df.groupby("Invoice").agg(
        date=("InvoiceDate", "first"),
        country=("Country", "first"),
        customer=("Customer ID", "first"),
        lineCount=("StockCode", "nunique"),
        totalUnits=("Quantity", "sum"),
        totalValue=("LineTotal", "sum"),
        items=("Description", lambda s: list(dict.fromkeys(s))),
    ).reset_index()
    return grouped[grouped["lineCount"] > 0]


def narrative(row) -> str:
    """The template IS the model. Everything the agent can search on must
    appear here in natural language - the embedding sees nothing else."""
    items = ", ".join(d.lower() for d in row["items"][:15])
    more = f" and {len(row['items']) - 15} further items" if len(row["items"]) > 15 else ""
    scale = band(row["lineCount"], SIZE_BANDS, "bulk-31-plus").split("-")[0]
    return (
        f"A {scale} order of {int(row['lineCount'])} distinct products "
        f"({int(row['totalUnits'])} units) totalling GBP {row['totalValue']:.2f}, "
        f"placed by a customer in {row['country']} "
        f"in {row['date'].strftime('%B %Y')}. "
        f"The basket contained: {items}{more}."
    )


def build_item(row) -> dict:
    from embeddings import to_ddb_list  # local import so --dry-run needs no AWS

    text = narrative(row)
    return {
        "pk": {"S": f"INV#{row['Invoice']}"},
        "sk": {"S": "SUMMARY"},
        # Vector index HASH. yearMonth, not country: this dataset is ~90% UK,
        # so partitioning by country builds one huge hot partition.
        "yearMonth": {"S": row["date"].strftime("%Y-%m")},
        # Inline filters - equality only.
        "country": {"S": row["country"]},
        "valueBand": {"S": band(row["totalValue"], VALUE_BANDS, "over-1000")},
        "basketSize": {"S": band(row["lineCount"], SIZE_BANDS, "bulk-31-plus")},
        # Raw values for display; you cannot filter on these at search time.
        "totalValue": {"N": f"{row['totalValue']:.2f}"},
        "lineCount": {"N": str(int(row["lineCount"]))},
        "customerId": {"S": str(row["customer"]) if pd.notna(row["customer"]) else "unknown"},
        "invoiceDate": {"S": row["date"].isoformat()},
        "docTitle": {"S": f"Invoice {row['Invoice']} - {row['country']}"},
        "chunkText": {"S": text},
        "templateVersion": {"S": TEMPLATE_VERSION},
        "textHash": {"S": hashlib.sha256(text.encode()).hexdigest()[:16]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["uci", "csv"], default="csv")
    ap.add_argument("--path", default=str(Path(__file__).parent / "sample_retail.csv"))
    ap.add_argument("--limit", type=int, default=5000, help="invoices to process")
    ap.add_argument("--dry-run", action="store_true", help="no Bedrock, no writes")
    args = ap.parse_args()

    print(f"Loading from {args.source}...")
    df = clean(load_raw(args.source, args.path))
    invoices = rollup(df).head(args.limit)
    print(f"  rolled up to {len(invoices)} invoices\n")

    if args.dry_run:
        for _, row in invoices.head(5).iterrows():
            print(f"--- {row['Invoice']} ---")
            print(narrative(row))
            print(f"  yearMonth={row['date'].strftime('%Y-%m')}  "
                  f"country={row['country']}  "
                  f"valueBand={band(row['totalValue'], VALUE_BANDS, 'over-1000')}  "
                  f"basketSize={band(row['lineCount'], SIZE_BANDS, 'bulk-31-plus')}\n")
        print(f"[dry run] would embed and write {len(invoices)} invoices")
        return

    import boto3
    from embeddings import embed, to_ddb_list

    ddb = boto3.client("dynamodb", region_name=REGION)
    items = []
    for n, (_, row) in enumerate(invoices.iterrows(), 1):
        item = build_item(row)
        item["embedding"] = to_ddb_list(embed(item["chunkText"]["S"]))
        items.append(item)
        if n % 100 == 0:
            print(f"  embedded {n}/{len(invoices)}")

    for start in range(0, len(items), 25):  # BatchWriteItem caps at 25
        pending = {TABLE: [{"PutRequest": {"Item": i}} for i in items[start:start + 25]]}
        for _ in range(5):
            resp = ddb.batch_write_item(RequestItems=pending)
            pending = resp.get("UnprocessedItems") or {}
            if not pending:
                break
    print(f"Loaded {len(items)} invoices into {TABLE}.")


if __name__ == "__main__":
    main()
