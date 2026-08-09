"""Reciprocal Rank Fusion.

Cosine distances produced from DIFFERENT query vectors are not comparable - a
0.42 from the catalog search and a 0.42 from the policy search mean nothing
relative to each other. So we fuse on rank position only, never on raw score.
"""
from typing import Dict, List

RRF_K = 60  # standard damping constant; larger = flatter weighting


def reciprocal_rank_fusion(
    result_sets: Dict[str, List[dict]], top_n: int = 8
) -> List[dict]:
    """result_sets maps domain -> that domain's hits, already in rank order."""
    fused: Dict[str, dict] = {}

    for domain, hits in result_sets.items():
        for rank, hit in enumerate(hits):
            key = f"{hit['docId']}#{hit['chunkId']}"
            entry = fused.setdefault(
                key,
                {**hit, "rrfScore": 0.0, "foundBy": [], "domainScores": {}},
            )
            entry["rrfScore"] += 1.0 / (RRF_K + rank + 1)
            entry["foundBy"].append(domain)
            # Keep the raw distance for display/debugging only.
            entry["domainScores"][domain] = hit["score"]

    ranked = sorted(fused.values(), key=lambda h: h["rrfScore"], reverse=True)
    for h in ranked:
        h["rrfScore"] = round(h["rrfScore"], 5)
    return ranked[:top_n]
