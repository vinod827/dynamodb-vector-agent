"""Embed product descriptions with Bedrock and load transactions into DynamoDB.

    python ingest/embed_and_load.py --input data/transactions.ndjson --dry-run
    python ingest/embed_and_load.py --input data/transactions.ndjson

THE COST INSIGHT: transactional data references the same products over and over.
A million order lines drawn from 2,000 products contains 2,000 distinct
descriptions. Embedding per-row would make 1,000,000 Bedrock calls to produce
2,000 distinct vectors. This script embeds once per productId and reuses the
result, which is a ~500x reduction in both cost and wall-clock time.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

REGION = os.environ.get("AWS_REGION", "us-east-1")
TABLE = os.environ.get("TABLE_NAME", "RetailTransactions")
EMBED_WORKERS = int(os.environ.get("EMBED_WORKERS", "8"))


def read_records(path: Path):
    """Yield records from either a JSON array or NDJSON. Detected from the first
    non-space character, so --format either way works without a flag."""
    with path.open() as fh:
        first = fh.read(1)
        while first.isspace():
            first = fh.read(1)
        fh.seek(0)
        if first == "[":
            yield from json.load(fh)   # whole file in memory - fine below ~100k
        else:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def collect_products(path: Path) -> dict:
    """First pass: unique productId -> description."""
    products = {}
    rows = 0
    for rec in read_records(path):
        rows += 1
        products.setdefault(rec["productId"], rec["productDescription"])
    print(f"  {rows:,} rows -> {len(products):,} distinct products "
          f"({rows / max(len(products), 1):.0f}x reuse)")
    return products


def embed_catalog(products: dict) -> dict:
    """Second pass: one embedding per distinct product, in parallel."""
    from embeddings import embed

    ids = list(products)
    vectors, done, t0 = {}, 0, time.time()

    def one(pid):
        return pid, embed(products[pid])

    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        for pid, vec in pool.map(one, ids):
            vectors[pid] = vec
            done += 1
            if done % 250 == 0:
                rate = done / (time.time() - t0)
                print(f"  embedded {done:,}/{len(ids):,} ({rate:.0f}/s)")
    return vectors


def to_ddb_list(vector) -> dict:
    """Storage shape for a vector attribute: an L of N."""
    return {"L": [{"N": repr(float(v))} for v in vector]}


def to_item(rec: dict, vector) -> dict:
    """Every transactional attribute is kept. The embedding is one more column."""
    s = rec["shipment"]
    return {
        "pk": {"S": f"ORDER#{rec['orderId']}"},
        "sk": {"S": f"LINE#{rec['orderLineId']:03d}"},
        # --- vector index HASH: each SearchVectors call is scoped to one value ---
        "marketplace": {"S": rec["marketplace"]},
        # --- inline filters: must be TOP-LEVEL scalars, equality only ---
        "category": {"S": rec["category"]},
        "priceBand": {"S": rec["priceBand"]},
        "discountBand": {"S": rec["discountBand"]},
        "orderStatus": {"S": rec["orderStatus"]},
        # --- operational attributes, returned with search results ---
        "orderDate": {"S": rec["orderDate"]},
        "customerId": {"S": rec["customerId"]},
        "productId": {"S": rec["productId"]},
        "productName": {"S": rec["productName"]},
        "productDescription": {"S": rec["productDescription"]},
        "subCategory": {"S": rec["subCategory"]},
        "brand": {"S": rec["brand"]},
        "countryOfOrigin": {"S": rec["countryOfOrigin"]},
        "currency": {"S": rec["currency"]},
        "unitPrice": {"N": str(rec["unitPrice"])},
        "quantity": {"N": str(rec["quantity"])},
        "grossAmount": {"N": str(rec["grossAmount"])},
        "discountPercent": {"N": str(rec["discountPercent"])},
        "discountAmount": {"N": str(rec["discountAmount"])},
        "netAmount": {"N": str(rec["netAmount"])},
        "paymentMethod": {"S": rec["paymentMethod"]},
        # nested map is fine for storage - just not filterable
        "shipment": {
            "M": {
                "carrier": {"S": s["carrier"]},
                "service": {"S": s["service"]},
                "shippedFrom": {"S": s["shippedFrom"]},
                "shipToCountry": {"S": s["shipToCountry"]},
                "shippedDate": {"S": s["shippedDate"]},
                "estimatedDays": {"N": str(s["estimatedDays"])},
                "trackingId": {"S": s["trackingId"]},
            }
        },
        # --- the vector column ---
        "descriptionEmbedding": to_ddb_list(vector),
    }


def write_all(path: Path, vectors: dict) -> int:
    import boto3

    ddb = boto3.client("dynamodb", region_name=REGION)
    batch, written = [], 0

    def flush(items):
        pending = {TABLE: [{"PutRequest": {"Item": i}} for i in items]}
        for attempt in range(6):
            resp = ddb.batch_write_item(RequestItems=pending)
            pending = resp.get("UnprocessedItems") or {}
            if not pending:
                return
            time.sleep(2 ** attempt * 0.1)
        raise RuntimeError("items unprocessed after 6 attempts")

    for rec in read_records(path):
        batch.append(to_item(rec, vectors[rec["productId"]]))
        if len(batch) == 25:  # BatchWriteItem hard limit
            flush(batch)
            written += 25
            batch = []
            if written % 10000 == 0:
                print(f"  written {written:,}")
    if batch:
        flush(batch)
        written += len(batch)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/transactions_10k.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    path = Path(a.input)

    print("Pass 1: collecting distinct products...")
    products = collect_products(path)

    if a.dry_run:
        sample = next(iter(products.items()))
        print(f"\n[dry run] would make {len(products):,} Bedrock calls, "
              f"not one per row")
        print(f"  example: {sample[0]} -> {sample[1][:90]}...")
        rec = next(read_records(path))
        item = to_item(rec, [0.0] * 4)
        print(f"  each item carries {len(item)} attributes, "
              f"incl. descriptionEmbedding")
        print(f"  index HASH=marketplace  inline filters="
              f"category, priceBand, discountBand, orderStatus")
        return

    print("Pass 2: embedding distinct products...")
    vectors = embed_catalog(products)

    print("Pass 3: writing transactions...")
    n = write_all(path, vectors)
    print(f"Loaded {n:,} transaction lines into {TABLE}.")


if __name__ == "__main__":
    main()
