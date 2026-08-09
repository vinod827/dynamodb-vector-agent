"""Query the vector index without deploying anything.

    python scripts/query_local.py "wireless headphones for commuting"
    python scripts/query_local.py "waterproof jacket" UK electronics
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from retriever import semantic_search  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: query_local.py "text" [marketplace] [category]')
    query = sys.argv[1]
    marketplace = sys.argv[2] if len(sys.argv) > 2 else "US"
    category = sys.argv[3] if len(sys.argv) > 3 else None

    hits = semantic_search(
        query=query,
        partition=marketplace,   # vector index HASH - mandatory, exactly one value
        doc_type=category,       # inline filter - optional, equality only
        top_k=5,
    )
    print(f"{len(hits)} hits for {query!r} in marketplace={marketplace}"
          f"{f' category={category}' if category else ''}\n")
    print(json.dumps(hits, indent=2))