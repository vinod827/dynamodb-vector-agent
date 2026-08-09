"""Run the agent locally, no deploy needed.

    export AGENT_MODEL_ID=<from: aws bedrock list-inference-profiles>
    python scripts/ask_local.py "what do people buy for their morning commute?"
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from handler import lambda_handler  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python scripts/ask_local.py "your question"')

    event = {"body": json.dumps({"question": " ".join(sys.argv[1:])})}
    resp = lambda_handler(event, None)
    payload = json.loads(resp["body"])

    print(payload.get("answer", payload))
    if payload.get("trace"):
        print("\n--- searches the agent ran ---")
        for t in payload["trace"]:
            print(f"  {t['searched']}  -> {t['hits']} hits")