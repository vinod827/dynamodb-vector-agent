"""Lambda handler: an agent that searches retail transactions semantically.

One function, one tool. The model decides what to search for and how to phrase
it, which is what makes this an agent rather than a search endpoint.

Each SearchVectors call is scoped to exactly one marketplace (the vector index
partition key), so the tool takes marketplace as a required argument.
"""
import json
import logging
import os
from typing import Any, Dict, List

import boto3
from botocore.config import Config

from retriever import semantic_search

log = logging.getLogger()
log.setLevel(logging.INFO)

REGION = os.environ.get("AWS_REGION", "us-east-1")
# Set from `aws bedrock list-inference-profiles` - do not hardcode a model id.
AGENT_MODEL_ID = os.environ["AGENT_MODEL_ID"]
MAX_TOOL_TURNS = int(os.environ.get("MAX_TOOL_TURNS", "4"))

MARKETPLACES = ["US", "UK", "DE", "JP", "IN", "CA", "AU"]
CATEGORIES = ["electronics", "kitchen", "outdoor", "apparel", "home", "beauty"]

_bedrock = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    config=Config(read_timeout=120, retries={"max_attempts": 5, "mode": "adaptive"}),
)

SYSTEM_PROMPT = f"""You are a retail analyst assistant with access to a
transaction database covering these marketplaces: {", ".join(MARKETPLACES)}.

Use the semantic_search tool to find relevant orders. Search with a descriptive
phrase of what the shopper wants, not the user's literal words - the index
matches on meaning, so "something warm for winter running" will find thermal
base layers even with no shared vocabulary.

IMPORTANT LIMITATION: this tool returns a small sample of matching orders ranked
by similarity. It does NOT return totals, counts, averages or any aggregate over
the whole dataset. If asked "what was total revenue" or "how many orders", say
plainly that you can only show examples, and offer to show representative orders
instead. Never add up the rows you retrieved and present the result as a total.

Ground every claim in the rows returned. Cite product names, and mention price
and marketplace where relevant. If nothing relevant comes back, say so."""

TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "semantic_search",
                "description": (
                    "Find retail transactions whose product description is "
                    "semantically similar to a query. Returns a ranked SAMPLE of "
                    "matching orders with full product and order details. "
                    "Ranked by cosine distance - lower scores are closer matches. "
                    "This returns examples only, never aggregates or totals."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Descriptive phrase of the product or need, "
                                    "phrased the way a product description would be."
                                ),
                            },
                            "marketplace": {
                                "type": "string",
                                "enum": MARKETPLACES,
                                "description": "Which marketplace to search. Required.",
                            },
                            "category": {
                                "type": "string",
                                "enum": CATEGORIES,
                                "description": "Optional category filter.",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "How many results, 1-20. Default 5.",
                            },
                        },
                        "required": ["query", "marketplace"],
                    }
                },
            }
        }
    ]
}


def _run_tool(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    hits = semantic_search(
        query=tool_input["query"],
        partition=tool_input["marketplace"],
        doc_type=tool_input.get("category"),
        top_k=min(int(tool_input.get("top_k", 5)), 20),
    )
    return {
        "results": hits,
        "note": "A ranked sample, not a complete or aggregate result set.",
    }


def lambda_handler(event, _context):
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"error": "body must be JSON"})

    question = (body.get("question") or "").strip()
    if not question:
        return _response(400, {"error": "question is required"})

    messages: List[dict] = [{"role": "user", "content": [{"text": question}]}]
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
                payload = _run_tool(tu["input"])
                trace.append({"searched": tu["input"],
                              "hits": len(payload["results"])})
                tool_results.append({"toolResult": {
                    "toolUseId": tu["toolUseId"],
                    "content": [{"json": payload}],
                }})
            except Exception as exc:  # report to the model, don't 500
                log.exception("search failed")
                tool_results.append({"toolResult": {
                    "toolUseId": tu["toolUseId"],
                    "content": [{"text": f"search error: {exc}"}],
                    "status": "error",
                }})
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