"""
Ranking Module for Grok Recruiter
Simple signal-based ranking (no PageRank/scipy)
"""

import math
from typing import Dict, List
import csv
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .graph_builder import Node, Edge


def compute_rankings(
    nodes: Dict[str, Node],
    edges: List[Edge],
) -> Dict[str, Node]:
    """
    Compute simple signal-based scores for all nodes.

    Signal score = weighted sum of interactions from root accounts
    Underratedness = signal_score / log(1 + followers)

    Args:
        nodes: Dictionary of user_id -> Node
        edges: List of Edge objects

    Returns:
        Updated nodes dictionary with scores
    """
    if not nodes or not edges:
        print("[ranking] No data to rank")
        return nodes

    print("\n" + "=" * 60)
    print("COMPUTING RANKINGS")
    print("=" * 60)

    # Get root node IDs
    root_ids = {uid for uid, node in nodes.items() if node.is_root}
    print(f"  Root nodes: {len(root_ids)}")

    # Compute signal scores based on edge weights
    # Signal = sum of weighted interactions involving root accounts
    signal_scores: Dict[str, float] = defaultdict(float)

    for edge in edges:
        # Interaction FROM root (root liked/retweeted candidate's content)
        if edge.src_user_id in root_ids and edge.dst_user_id not in root_ids:
            signal_scores[edge.dst_user_id] += edge.weight * 2  # Outbound from root is strong signal

        # Interaction TO root (candidate engaged with root)
        if edge.dst_user_id in root_ids and edge.src_user_id not in root_ids:
            signal_scores[edge.src_user_id] += edge.weight

    # Normalize scores
    max_signal = max(signal_scores.values()) if signal_scores else 1.0

    # Update nodes with scores
    for uid, node in nodes.items():
        raw_signal = signal_scores.get(uid, 0.0)
        node.pagerank_score = raw_signal / max_signal if max_signal > 0 else 0.0

        # Underratedness = signal / log(1 + followers)
        if node.followers_count > 0:
            node.underratedness_score = node.pagerank_score / math.log(1 + node.followers_count)
        else:
            node.underratedness_score = node.pagerank_score

    # Print top candidates by underratedness
    print("\n  Top 20 candidates by underratedness score:")
    print("  " + "-" * 60)

    candidates = [n for n in nodes.values() if n.is_candidate]
    top_candidates = sorted(candidates, key=lambda n: n.underratedness_score, reverse=True)[:20]

    for i, node in enumerate(top_candidates, 1):
        print(
            f"  {i:2}. @{node.handle:<20} | "
            f"followers: {node.followers_count:>7,} | "
            f"score: {node.underratedness_score:.6f}"
        )

    print("  " + "-" * 60)
    print(f"\n  Total candidates: {len(candidates)}")

    return nodes


def export_ranked_nodes(
    nodes: Dict[str, Node],
    output_path: str,
    sort_by: str = "underratedness_score",
):
    """
    Export nodes sorted by ranking score.

    Args:
        nodes: Dictionary of user_id -> Node
        output_path: Output CSV path
        sort_by: Field to sort by (default: underratedness_score)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Sort nodes
    sorted_nodes = sorted(
        nodes.values(),
        key=lambda n: getattr(n, sort_by, 0),
        reverse=True,
    )

    # Write to CSV
    if sorted_nodes:
        fields = list(asdict(sorted_nodes[0]).keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for node in sorted_nodes:
                writer.writerow(asdict(node))

        print(f"  Exported {len(sorted_nodes)} ranked nodes to {output_path}")
