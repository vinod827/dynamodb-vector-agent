"""Domain config. Adding a corpus = one entry here + an ingest run.

`partition` is the value written to the vector index HASH attribute, so each
domain is a separate index partition and therefore a separate SearchVectors call.
"""

DOMAINS = {
    "catalog": {
        "partition": "catalog",
        "doc_type": "kb",
        "hint": "product descriptions, materials, use cases, sizing",
    },
    "policy": {
        "partition": "policy",
        "doc_type": "faq",
        "hint": "returns, refunds, shipping times, warranty terms",
    },
    "support": {
        "partition": "support",
        "doc_type": "kb",
        "hint": "troubleshooting, care instructions, common complaints",
    },
}

def describe() -> str:
    return "\n".join(f"- {k}: {v['hint']}" for k, v in DOMAINS.items())
