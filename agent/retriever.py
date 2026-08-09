"""SearchVectors wrapper. One call = one index partition."""
import logging
import os
from typing import List, Optional

import boto3

from embeddings import embed, to_search_vector

log = logging.getLogger(__name__)

REGION = os.environ.get("AWS_REGION", "us-east-1")
TABLE = os.environ.get("TABLE_NAME", "RetailTransactions")
INDEX = os.environ.get("VECTOR_INDEX_NAME", "ProductDescriptionIndex")
# The attribute used as the vector index HASH. Every search must supply exactly
# one value for it - there is no cross-partition search, no IN, no ranges.
PARTITION_ATTR = os.environ.get("PARTITION_ATTR", "marketplace")

# COSINE returns a distance: lower == more similar. TopK always returns K rows
# even when nothing matches, so threshold rather than trusting the count.
MAX_COSINE_DISTANCE = float(os.environ.get("MAX_COSINE_DISTANCE", "0.85"))

# Do NOT set endpoint_url - SearchVectors resolves to its own search endpoint.
_ddb = boto3.client("dynamodb", region_name=REGION)


def semantic_search(
    query: str,
    partition: str,
    doc_type: Optional[str] = None,   # maps to the `category` inline filter
    top_k: int = 5,
) -> List[dict]:
    top_k = max(1, min(int(top_k), 100))  # API ceiling is 100

    condition = f"{PARTITION_ATTR} = :p"
    values = {":p": {"S": partition}}
    if doc_type:
        # Inline filters are optional and equality-only.
        condition += " AND category = :d"
        values[":d"] = {"S": doc_type}

    resp = _ddb.search_vectors(
        TableName=TABLE,
        IndexName=INDEX,
        SearchVector=to_search_vector(embed(query)),
        TopK=top_k,
        SearchConditionExpression=condition,
        ExpressionAttributeValues=values,
        # Only attributes projected into the index can be returned. The
        # embedding itself is excluded by default, which is what we want.
        ProjectionExpression=(
            "pk, sk, productId, productName, productDescription, brand, "
            "category, subCategory, countryOfOrigin, marketplace, currency, "
            "unitPrice, quantity, netAmount, discountPercent, orderStatus, orderDate"
        ),
        ReturnConsumedCapacity="TOTAL",
    )

    consumed = resp.get("ConsumedCapacity", {}).get("VectorSearchRequestBytes")
    log.info("SearchVectors partition=%s k=%s bytes=%s", partition, top_k, consumed)

    hits = []
    for row in resp.get("SearchResults", []):
        if row["Score"] > MAX_COSINE_DISTANCE:
            continue
        it = row["Item"]

        def s_(k):
            return it.get(k, {}).get("S", "")

        def n_(k):
            return it.get(k, {}).get("N", "")

        hits.append(
            {
                "orderId": s_("pk"),
                "line": s_("sk"),
                "productId": s_("productId"),
                "productName": s_("productName"),
                "description": s_("productDescription"),
                "brand": s_("brand"),
                "category": s_("category"),
                "subCategory": s_("subCategory"),
                "countryOfOrigin": s_("countryOfOrigin"),
                "marketplace": s_("marketplace"),
                "price": f"{n_('unitPrice')} {s_('currency')}",
                "quantity": n_("quantity"),
                "netAmount": n_("netAmount"),
                "discountPercent": n_("discountPercent"),
                "orderStatus": s_("orderStatus"),
                "orderDate": s_("orderDate"),
                # COSINE distance: lower is more similar, 0 is identical.
                "score": round(row["Score"], 4),
                "searchBytes": consumed,
            }
        )
    return hits