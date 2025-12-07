"""
Ranking Module for Grok Recruiter
Supports both simple signal-based ranking and NetworkX PageRank
"""

import math
from typing import Dict, List, Optional, Set
import csv
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import networkx as nx

from .graph_builder import Node, Edge


def compute_pagerank(
    graph_or_nodes,
    edges_or_seed_ids=None,
    seed_ids: Optional[Set[str]] = None,
    alpha: float = 0.85,
    use_weights: bool = True,
) -> Dict[str, float]:
    """
    Compute Personalized PageRank with seeds as personalization.

    Can be called in two ways:
    1. compute_pagerank(nodes_dict, edges_list, seed_ids=...)
    2. compute_pagerank(nx_graph, seed_ids_list)

    Args:
        graph_or_nodes: Either a Dict[str, Node] or a NetworkX DiGraph
        edges_or_seed_ids: Either List[Edge] or list of seed IDs
        seed_ids: Set of seed/root user IDs for personalization (for dict mode)
        alpha: Damping factor (default 0.85)
        use_weights: Whether to use edge weights

    Returns:
        Dictionary of user_id -> pagerank score
    """
    # Check if first arg is a NetworkX graph
    if isinstance(graph_or_nodes, nx.DiGraph):
        G = graph_or_nodes
        # Second arg is the seed IDs list
        if edges_or_seed_ids is not None:
            seed_ids = set(edges_or_seed_ids) if not isinstance(edges_or_seed_ids, set) else edges_or_seed_ids
    else:
        # Original mode: nodes dict + edges list
        nodes = graph_or_nodes
        edges = edges_or_seed_ids or []

        # Build NetworkX graph
        G = nx.DiGraph()

        # Add all nodes
        for uid in nodes.keys():
            G.add_node(uid)

        # Add edges
        for edge in edges:
            if use_weights:
                G.add_edge(edge.src_user_id, edge.dst_user_id, weight=edge.weight)
            else:
                G.add_edge(edge.src_user_id, edge.dst_user_id)

        # If no seed_ids provided, find roots from nodes
        if seed_ids is None:
            seed_ids = {uid for uid, node in nodes.items() if node.is_root}

    # Build personalization vector (seeds get equal weight)
    personalization = None
    if seed_ids:
        # Only include seeds that are in the graph
        valid_seeds = set(seed_ids) & set(G.nodes())
        if valid_seeds:
            personalization = {uid: 1.0 / len(valid_seeds) for uid in valid_seeds}

    # Compute PageRank
    weight_attr = "weight" if use_weights else None
    pr = nx.pagerank(G, alpha=alpha, personalization=personalization, weight=weight_attr)

    return pr


def compute_rankings(
    nodes: Dict[str, Node],
    edges: List[Edge],
    use_pagerank: bool = True,
) -> Dict[str, Node]:
    """
    Compute ranking scores for all nodes.

    If use_pagerank=True, uses Personalized PageRank with seeds.
    Otherwise, uses simple signal-based scoring.

    Underratedness = score / log(1 + followers)

    Args:
        nodes: Dictionary of user_id -> Node
        edges: List of Edge objects
        use_pagerank: Whether to use PageRank (default True)

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
    print(f"  Root nodes (seeds): {len(root_ids)}")

    if use_pagerank:
        print("  Method: Personalized PageRank")

        # Compute PageRank
        pr_scores = compute_pagerank(nodes, edges, seed_ids=root_ids)

        # Update nodes with PageRank scores
        for uid, node in nodes.items():
            node.pagerank_score = pr_scores.get(uid, 0.0)

    else:
        print("  Method: Signal-based scoring")

        # Compute signal scores based on edge weights
        signal_scores: Dict[str, float] = defaultdict(float)

        for edge in edges:
            # Interaction FROM root (root liked/retweeted candidate's content)
            if edge.src_user_id in root_ids and edge.dst_user_id not in root_ids:
                signal_scores[edge.dst_user_id] += edge.weight * 2

            # Interaction TO root (candidate engaged with root)
            if edge.dst_user_id in root_ids and edge.src_user_id not in root_ids:
                signal_scores[edge.src_user_id] += edge.weight

        # Normalize scores
        max_signal = max(signal_scores.values()) if signal_scores else 1.0

        # Update nodes with scores
        for uid, node in nodes.items():
            raw_signal = signal_scores.get(uid, 0.0)
            node.pagerank_score = raw_signal / max_signal if max_signal > 0 else 0.0

    # Compute underratedness for all nodes
    # Underratedness = pagerank / log(1 + followers)
    # This normalizes by popularity to find "hidden gems"
    for uid, node in nodes.items():
        if node.followers_count > 0:
            node.underratedness_score = node.pagerank_score / math.log(1 + node.followers_count)
        else:
            node.underratedness_score = node.pagerank_score

    # Print top candidates by underratedness
    print("\n  Top 20 candidates by UNDERRATEDNESS score:")
    print("  (High PageRank + Low Followers = Hidden Gem)")
    print("  " + "-" * 70)

    candidates = [n for n in nodes.values() if n.is_candidate]
    top_underrated = sorted(candidates, key=lambda n: n.underratedness_score, reverse=True)[:20]

    for i, node in enumerate(top_underrated, 1):
        print(
            f"  {i:2}. @{node.handle:<20} | "
            f"followers: {node.followers_count:>7,} | "
            f"ppr: {node.pagerank_score:.6f} | "
            f"underrated: {node.underratedness_score:.6f}"
        )

    # Also show top by raw PageRank
    print("\n  Top 20 candidates by raw PAGERANK score:")
    print("  " + "-" * 70)

    top_pagerank = sorted(candidates, key=lambda n: n.pagerank_score, reverse=True)[:20]

    for i, node in enumerate(top_pagerank, 1):
        print(
            f"  {i:2}. @{node.handle:<20} | "
            f"followers: {node.followers_count:>7,} | "
            f"ppr: {node.pagerank_score:.6f}"
        )

    print("  " + "-" * 70)
    print(f"\n  Total candidates: {len(candidates)}")

    return nodes


def get_top_candidates(
    nodes: Dict[str, Node],
    by: str = "underratedness_score",
    limit: int = 200,
    min_followers: int = 50,
    max_followers: int = 10000,
) -> List[Node]:
    """
    Get top candidates by a ranking metric.

    Args:
        nodes: Dictionary of user_id -> Node
        by: Ranking field (underratedness_score or pagerank_score)
        limit: Maximum candidates to return
        min_followers: Minimum follower filter
        max_followers: Maximum follower filter (for "underrated")

    Returns:
        List of top Node objects
    """
    candidates = [
        n for n in nodes.values()
        if n.is_candidate
        and min_followers <= n.followers_count <= max_followers
    ]

    return sorted(candidates, key=lambda n: getattr(n, by, 0), reverse=True)[:limit]


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
