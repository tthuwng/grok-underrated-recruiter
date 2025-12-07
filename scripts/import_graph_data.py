#!/usr/bin/env python3
"""Import graph data from CSV files to the deployed API with screening data merged."""

import csv
import json
import requests
import sys
from pathlib import Path

# Configuration
API_URL = "https://grok-recruiter-api.fly.dev"
DATA_DIR = Path(__file__).parent.parent / "data"
NODES_CSV = DATA_DIR / "processed" / "nodes.csv"
EDGES_CSV = DATA_DIR / "processed" / "edges.csv"
FAST_SCREEN_DIR = DATA_DIR / "evaluations" / "fast_screen"
ENRICHED_DIR = DATA_DIR / "enriched"
BATCH_SIZE = 500  # Import in batches to avoid timeouts


def load_fast_screen_data():
    """Load fast screen results from JSON files."""
    fast_screen = {}
    if not FAST_SCREEN_DIR.exists():
        return fast_screen

    for f in FAST_SCREEN_DIR.glob("*.json"):
        try:
            # Extract handle from filename: fast_<handle>.json or fast___<handle>.json
            name = f.stem  # e.g., "fast___ghostfail" or "fast__asrinivas_"
            if name.startswith("fast_"):
                handle = name[5:].lstrip("_")  # Remove "fast_" and leading underscores
            else:
                handle = name

            data = json.loads(f.read_text())
            fast_screen[handle.lower()] = {
                "grok_relevant": data.get("pass_filter", False),
                "grok_role": data.get("potential_role", ""),
            }
        except Exception as e:
            print(f"  Warning: Could not load {f}: {e}")

    return fast_screen


def load_deep_eval_data():
    """Load deep evaluation results from JSON files."""
    deep_evals = {}
    if not ENRICHED_DIR.exists():
        return deep_evals

    # Load individual deep_*.json files
    for f in ENRICHED_DIR.glob("deep_*.json"):
        try:
            data = json.loads(f.read_text())
            handle = data.get("handle", "").lower()
            if handle:
                deep_evals[handle] = {
                    "grok_evaluated": True,
                    "final_score": data.get("final_score", 0),
                }
        except Exception as e:
            print(f"  Warning: Could not load {f}: {e}")

    # Also load candidates_deep_eval.json if exists
    candidates_file = ENRICHED_DIR / "candidates_deep_eval.json"
    if candidates_file.exists():
        try:
            candidates = json.loads(candidates_file.read_text())
            for c in candidates:
                handle = c.get("handle", "").lower()
                if handle:
                    deep_evals[handle] = {
                        "grok_evaluated": True,
                        "final_score": c.get("final_score", 0),
                    }
        except Exception as e:
            print(f"  Warning: Could not load candidates_deep_eval.json: {e}")

    return deep_evals


def load_nodes_from_csv(path: Path, fast_screen: dict, deep_evals: dict):
    """Load nodes from CSV file and merge with screening data."""
    nodes = []
    merged_fast = 0
    merged_deep = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            handle = row.get("handle", "").lower()

            node = {
                "user_id": row.get("user_id", ""),
                "handle": row.get("handle", ""),
                "name": row.get("name", ""),
                "bio": row.get("bio", "")[:500] if row.get("bio") else "",
                "followers_count": int(row.get("followers_count", 0) or 0),
                "following_count": int(row.get("following_count", 0) or 0),
                "is_seed": row.get("is_root", "").lower() == "true",
                "is_candidate": row.get("is_candidate", "").lower() == "true",
                "discovered_via": row.get("discovered_via", ""),
                "pagerank_score": float(row.get("pagerank_score", 0) or 0),
                "underratedness_score": float(row.get("underratedness_score", 0) or 0),
                "grok_relevant": None,
                "grok_role": None,
                "grok_evaluated": False,
                "final_score": None,
            }

            # Merge fast screen data
            if handle in fast_screen:
                node["grok_relevant"] = fast_screen[handle]["grok_relevant"]
                node["grok_role"] = fast_screen[handle]["grok_role"]
                merged_fast += 1

            # Merge deep eval data
            if handle in deep_evals:
                node["grok_evaluated"] = deep_evals[handle]["grok_evaluated"]
                node["final_score"] = deep_evals[handle]["final_score"]
                merged_deep += 1

            nodes.append(node)

    print(f"  Merged {merged_fast} fast screen results")
    print(f"  Merged {merged_deep} deep eval results")
    return nodes


def load_edges_from_csv(path: Path):
    """Load edges from CSV file."""
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append({
                "src_user_id": row.get("src_user_id", ""),
                "dst_user_id": row.get("dst_user_id", ""),
                "interaction_type": row.get("interaction_type", "follow"),
                "weight": float(row.get("weight", 1.0) or 1.0),
            })
    return edges


def import_batch(nodes: list, edges: list):
    """Import a batch of nodes and edges."""
    response = requests.post(
        f"{API_URL}/graph/import",
        json={"nodes": nodes, "edges": edges},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def main():
    print("=" * 60)
    print("IMPORTING GRAPH DATA TO FLY.IO")
    print("=" * 60)

    # Load screening data first
    print("\nLoading fast screen data...")
    fast_screen = load_fast_screen_data()
    print(f"  Loaded {len(fast_screen)} fast screen results")

    print("\nLoading deep eval data...")
    deep_evals = load_deep_eval_data()
    print(f"  Loaded {len(deep_evals)} deep eval results")

    # Load nodes and merge screening data
    print(f"\nLoading nodes from {NODES_CSV}...")
    nodes = load_nodes_from_csv(NODES_CSV, fast_screen, deep_evals)
    print(f"  Loaded {len(nodes)} nodes total")

    print(f"\nLoading edges from {EDGES_CSV}...")
    edges = load_edges_from_csv(EDGES_CSV)
    print(f"  Loaded {len(edges)} edges")

    # First clear the graph
    print("\nClearing existing graph...")
    try:
        resp = requests.delete(f"{API_URL}/graph", timeout=30)
        print(f"  {resp.json()}")
    except Exception as e:
        print(f"  Warning: {e}")

    # Import nodes in batches
    print(f"\nImporting nodes in batches of {BATCH_SIZE}...")
    total_nodes = 0
    failed_batches = 0
    for i in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[i:i + BATCH_SIZE]
        try:
            result = import_batch(batch, [])
            total_nodes += result.get("nodes_added", 0)
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(nodes) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  Batch {batch_num}/{total_batches}: +{result.get('nodes_added', 0)} nodes (total: {total_nodes})")
        except Exception as e:
            failed_batches += 1
            print(f"  Batch {i // BATCH_SIZE + 1} failed: {e}")

    # Import edges in batches
    print(f"\nImporting edges in batches of {BATCH_SIZE}...")
    total_edges = 0
    for i in range(0, len(edges), BATCH_SIZE):
        batch = edges[i:i + BATCH_SIZE]
        try:
            result = import_batch([], batch)
            total_edges += result.get("edges_added", 0)
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(edges) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  Batch {batch_num}/{total_batches}: +{result.get('edges_added', 0)} edges (total: {total_edges})")
        except Exception as e:
            failed_batches += 1
            print(f"  Batch {i // BATCH_SIZE + 1} failed: {e}")

    # Verify
    print("\nVerifying import...")
    try:
        status = requests.get(f"{API_URL}/status", timeout=30).json()
        print(f"  Graph status: {status}")
    except Exception as e:
        print(f"  Error: {e}")

    # Check stats
    print("\nChecking stats...")
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=30).json()
        print(f"  Stats: {stats}")
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n{'=' * 60}")
    print(f"IMPORT COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Nodes imported: {total_nodes}")
    print(f"  Edges imported: {total_edges}")
    if failed_batches:
        print(f"  Failed batches: {failed_batches}")


if __name__ == "__main__":
    main()
