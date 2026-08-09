"""Titan Text Embeddings V2 wrapper.

The query vector and the stored vectors MUST come from the same model with the
same dimension count, or results are meaningless (or rejected outright).
"""
import json
import os
from typing import List

import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "us-east-1")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
DIMENSIONS = int(os.environ.get("EMBED_DIMENSIONS", "1024"))  # 256 | 512 | 1024

_bedrock = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    # InvokeModel is rate-limited; let botocore retry rather than failing the request.
    config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
)


def embed(text: str) -> List[float]:
    resp = _bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {
                "inputText": text,
                "dimensions": DIMENSIONS,
                # Unit-length vectors: recommended for COSINE, required for DOT_PRODUCT.
                "normalize": True,
            }
        ),
    )
    vector = json.loads(resp["body"].read())["embedding"]
    if len(vector) != DIMENSIONS:
        raise ValueError(f"expected {DIMENSIONS} dims, model returned {len(vector)}")
    return vector


def to_ddb_list(vector: List[float]) -> dict:
    """Shape for STORING in an item attribute: an L of N."""
    return {"L": [{"N": repr(float(v))} for v in vector]}


def to_search_vector(vector: List[float]) -> list:
    """Shape for QUERYING via SearchVectors: a plain array, no L wrapper."""
    return [{"N": repr(float(v))} for v in vector]
