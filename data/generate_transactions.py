"""Generate transactional retail data as JSON Lines.

    python data/generate_transactions.py --count 1000000 --out data/transactions.ndjson

Why NDJSON and not a single JSON array: a 1M-element array cannot be parsed
incrementally - json.load() pulls the whole file into memory. One object per line
streams in constant memory and is what DynamoDB's S3 import expects anyway.

Sizing: ~700 bytes/record, so 1M records is roughly 700 MB. Do not commit it.
"""
import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from catalog import build_catalog  # noqa: E402

MARKETPLACES = ["US", "UK", "DE", "JP", "IN", "CA", "AU"]
SHIP_FROM = {
    "US": ["Dallas TX", "Newark NJ", "Reno NV"], "UK": ["Rugeley", "Doncaster"],
    "DE": ["Bad Hersfeld", "Leipzig"], "JP": ["Chiba", "Osaka"],
    "IN": ["Hyderabad", "Gurugram"], "CA": ["Brampton ON", "Delta BC"],
    "AU": ["Sydney NSW", "Melbourne VIC"],
}
CARRIERS = ["UPS", "DHL", "FedEx", "Royal Mail", "Japan Post", "Blue Dart", "Australia Post"]
SERVICES = ["standard", "expedited", "next-day", "economy"]
PAYMENTS = ["credit_card", "debit_card", "gift_card", "wallet", "bank_transfer"]
STATUSES = ["delivered", "delivered", "delivered", "shipped", "processing", "returned", "cancelled"]

PRICE_BANDS = [(20, "under-20"), (50, "20-50"), (150, "50-150"), (500, "150-500")]
DISCOUNT_BANDS = [(1, "none"), (11, "1-10pct"), (26, "11-25pct")]


def band(v, bands, top):
    for threshold, label in bands:
        if v < threshold:
            return label
    return top


def generate(count: int, out_path: Path, n_products: int, seed: int,
             fmt: str = "ndjson") -> None:
    rng = random.Random(seed)
    catalog = build_catalog(n_products, seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # ndjson streams in constant memory and is what DynamoDB S3 import expects.
    # json emits a single array - readable, but only sane below ~100k records.
    with out_path.open("w") as fh:
        if fmt == "json":
            fh.write("[\n")
        for i in range(count):
            p = catalog[rng.randrange(len(catalog))]
            mkt = rng.choice(MARKETPLACES)
            qty = rng.choices([1, 2, 3, 5, 10], weights=[60, 20, 10, 7, 3])[0]
            disc = rng.choices([0, 5, 10, 15, 20, 30, 50], weights=[50, 12, 12, 10, 8, 6, 2])[0]
            unit = round(p["listPrice"] * rng.uniform(0.9, 1.15), 2)
            gross = round(unit * qty, 2)
            disc_amt = round(gross * disc / 100, 2)
            ordered = start + timedelta(minutes=rng.randrange(0, 60 * 24 * 550))
            shipped = ordered + timedelta(hours=rng.randrange(4, 72))
            eta = rng.randrange(1, 9)

            rec = {
                "orderId": f"ORD-{i // 3:08d}",
                "orderLineId": (i % 3) + 1,
                "orderDate": ordered.isoformat(),
                "customerId": f"CUST-{rng.randrange(0, count // 8 + 2):07d}",
                "marketplace": mkt,
                # --- product attributes (description is what gets embedded) ---
                "productId": p["productId"],
                "productName": p["productName"],
                "productDescription": p["productDescription"],
                "category": p["category"],
                "subCategory": p["subCategory"],
                "brand": p["brand"],
                "countryOfOrigin": p["countryOfOrigin"],
                # --- money ---
                "currency": {"US": "USD", "UK": "GBP", "DE": "EUR", "JP": "JPY",
                             "IN": "INR", "CA": "CAD", "AU": "AUD"}[mkt],
                "unitPrice": unit,
                "quantity": qty,
                "grossAmount": gross,
                "discountPercent": disc,
                "discountAmount": disc_amt,
                "netAmount": round(gross - disc_amt, 2),
                "paymentMethod": rng.choice(PAYMENTS),
                # --- fulfilment ---
                "orderStatus": rng.choice(STATUSES),
                "shipment": {
                    "carrier": rng.choice(CARRIERS),
                    "service": rng.choice(SERVICES),
                    "shippedFrom": rng.choice(SHIP_FROM[mkt]),
                    "shipToCountry": mkt,
                    "shippedDate": shipped.isoformat(),
                    "estimatedDays": eta,
                    "trackingId": f"TRK{rng.randrange(10**11, 10**12)}",
                },
                # --- derived bands: inline filters are equality-only, so any
                #     numeric you want to filter on must be bucketed here ---
                "priceBand": band(unit, PRICE_BANDS, "over-500"),
                "discountBand": band(disc, DISCOUNT_BANDS, "over-25pct"),
            }
            if fmt == "json":
                fh.write(("  " if i == 0 else ",\n  ") + json.dumps(rec))
            else:
                fh.write(json.dumps(rec) + "\n")

            if count > 50000 and (i + 1) % 100000 == 0:
                print(f"  {i + 1:,} / {count:,}")

        if fmt == "json":
            fh.write("\n]\n")

    mb = out_path.stat().st_size / 1e6
    print(f"Wrote {count:,} records to {out_path} ({mb:.1f} MB, "
          f"{len(catalog):,} distinct products)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1000, help="transaction lines")
    ap.add_argument("--products", type=int, default=2000, help="distinct products")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/transactions.ndjson")
    ap.add_argument("--format", choices=["ndjson", "json"], default="ndjson")
    a = ap.parse_args()
    generate(a.count, Path(a.out), a.products, a.seed, a.format)
