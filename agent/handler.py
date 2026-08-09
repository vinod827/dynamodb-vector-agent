"""Single-Lambda retrieval agent with in-process parallel fan-out.

One function, one Bedrock tool. The model emits a list of per-domain sub-queries
in a single tool call; the Lambda runs those SearchVectors calls concurrently on
threads and fuses the results with RRF before handing them back.

Why fan out at all: each SearchVectors call is scoped to exactly one vector index
partition. A question spanning domains cannot be answered by one search - the
data model forces the parallelism.
"""
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import boto3
from botocore.config import Config

from domains import DOMAINS, describe
from fuse import reciprocal_rank_fusion
from retriever import semantic_search

log = logging.getLogger()
log.setLevel(logging.INFO)

REGION = os.environ.get("AWS_REGION", "us-east-1")
# Set from `aws bedrock list-inference-profiles` - do not hardcode.
AGENT_MODEL_ID = os.environ["AGENT_MODEL_ID"]
MAX_TOOL_TURNS = int(os.environ.get("MAX_TOOL_TURNS", "3"))
FANOUT_WIDTH = int(os.environ.get("FANOUT_WIDTH", "3"))
SEARCH_TIMEOUT_S = int(os.environ.get("SEARCH_TIMEOUT_S", "15"))

_bedrock = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    # Parallel embed calls are exactly the shape that trips Bedrock throttling.
    config=Config(read_timeout=120, retries={"max_attempts": 5, "mode": "adaptive"}),
)

SYSTEM_PROMPT = f"""You are a retrieval assistant over a partitioned knowledge base.

Available domains:
{describe()}

Call search_domains ONCE with one entry per domain that could plausibly hold part
of the answer. Write a DIFFERENT sub-query for each domain, phrased in that
domain's own vocabulary - the index is semantic, so the wording you choose
determines what comes back. Do not simply repeat the user's question three times.

Searches run in parallel and results are rank-fused across domains, so querying a
domain that turns out to be irrelevant is cheap. Missing one is not.

Ground every claim in the returned chunks and cite titles. If the evidence is
thin, say so instead of filling the gap yourself."""

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "search_domains",
                "description": (
                    "Run semantic searches across knowledge-base domains in "
                    "parallel and return rank-fused results."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "searches": {
                                "type": "array",
                                "maxItems": FANOUT_WIDTH,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "domain": {
                                            "type": "string",
                                            "enum": list(DOMAINS.keys()),
                                        },
                                        "query": {
                                            "type": "string",
                                            "description": "Sub-query in this domain's vocabulary.",
                                        },
                                    },
                                    "required": ["domain", "query"],
                                },
                            },
                            "top_k": {"type": "integer", "description": "Per domain, 1-10."},
                        },
                        "required": ["searches"],
                    }
                },
            }
        }
    ]
}


def _search_one(domain: str, query: str, top_k: int) -> List[dict]:
    cfg = DOMAINS[domain]
    return semantic_search(
        query=query,
        partition=cfg["partition"],
        doc_type=cfg.get("doc_type"),
        top_k=top_k,
    )


def _fan_out(searches: List[dict], top_k: int) -> Dict[str, Any]:
    """Run one SearchVectors call per domain, concurrently. A domain that fails
    or times out yields no hits rather than failing the whole request."""
    seen, planned = set(), []
    for s in searches[:FANOUT_WIDTH]:
        d = s.get("domain")
        if d in DOMAINS and d not in seen:
            seen.add(d)
            planned.append((d, s["query"]))

    result_sets: Dict[str, List[dict]] = {}
    errors: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(planned) or 1) as pool:
        futures = {
            pool.submit(_search_one, d, q, top_k): (d, q) for d, q in planned
        }
        for fut in as_completed(futures, timeout=SEARCH_TIMEOUT_S):
            domain, _ = futures[fut]
            try:
                result_sets[domain] = fut.result()
            except Exception as exc:
                log.exception("domain %s failed", domain)
                errors[domain] = str(exc)

    return {
        "results": reciprocal_rank_fusion(result_sets, top_n=8),
        "searched": {d: q for d, q in planned},
        "hitsPerDomain": {d: len(h) for d, h in result_sets.items()},
        "failedDomains": errors,
    }


def lambda_handler(event, _context):
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"error": "body must be JSON"})

    question = (body.get("question") or "").strip()
    if not question:
        return _response(400, {"error": "question is required"})

    messages = [{"role": "user", "content": [{"text": question}]}]
    trace: List[dict] = []

    for _ in range(MAX_TOOL_TURNS):
        resp = _bedrock.converse(
            modelId=AGENT_MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": 1500, "temperature": 0.2},
        )
        assistant_msg = resp["output"]["message"]
        messages.append(assistant_msg)

        if resp["stopReason"] != "tool_use":
            break

        tool_results = []
        for block in assistant_msg["content"]:
            if "toolUse" not in block:
                continue
            tu = block["toolUse"]
            try:
                payload = _fan_out(
                    tu["input"].get("searches", []),
                    min(int(tu["input"].get("top_k", 5)), 10),
                )
                trace.append(
                    {
                        "searched": payload["searched"],
                        "hitsPerDomain": payload["hitsPerDomain"],
                        "failedDomains": payload["failedDomains"],
                    }
                )
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tu["toolUseId"],
                            "content": [{"json": payload}],
                        }
                    }
                )
            except Exception as exc:  # report to the model, don't 500
                log.exception("fan-out failed")
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tu["toolUseId"],
                            "content": [{"text": f"search error: {exc}"}],
                            "status": "error",
                        }
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    answer = "".join(
        b.get("text", "") for b in messages[-1].get("content", []) if isinstance(b, dict)
    )
    return _response(200, {"answer": answer, "trace": trace})


def _response(status: int, payload: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(payload),
    }
