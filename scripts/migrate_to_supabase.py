#!/usr/bin/env python3
"""
Migrate data from local CSV/JSON files to Supabase PostgreSQL.

Usage:
    python scripts/migrate_to_supabase.py

Environment variables required:
    SUPABASE_URL - Your Supabase project URL
    SUPABASE_SERVICE_KEY - Service role key (not anon key)
"""

import os
import csv
import json
from pathlib import Path
from datetime import datetime

try:
    from supabase import create_client, Client
except ImportError:
    print("Please install supabase: pip install supabase")
    exit(1)

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
NODES_CSV = DATA_DIR / "processed" / "nodes.csv"
ENRICHED_DIR = DATA_DIR / "enriched"
SEEDS_YAML = Path(__file__).parent.parent / "seeds.yaml"

def get_supabase_client() -> Client:
    """Initialize Supabase client with service role key."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError(
            "Missing environment variables. Set:\n"
            "  SUPABASE_URL=https://xxx.supabase.co\n"
            "  SUPABASE_SERVICE_KEY=eyJ..."
        )

    return create_client(url, key)


def get_or_create_base_graph(supabase: Client) -> str:
    """Get or create the base graph and return its ID."""
    # Check if base graph exists
    result = supabase.table("graphs").select("id").eq("is_base_graph", True).execute()

    if result.data:
        graph_id = result.data[0]["id"]
        print(f"Found existing base graph: {graph_id}")
        return graph_id

    # Create base graph
    result = supabase.table("graphs").insert({
        "name": "xAI Seeds Graph",
        "is_base_graph": True,
        "is_public": True
    }).execute()

    graph_id = result.data[0]["id"]
    print(f"Created base graph: {graph_id}")
    return graph_id


def migrate_nodes(supabase: Client, graph_id: str) -> dict:
    """Migrate nodes from CSV and return handle -> node_id mapping."""
    if not NODES_CSV.exists():
        print(f"Nodes CSV not found: {NODES_CSV}")
        return {}

    print(f"Reading nodes from {NODES_CSV}...")

    nodes = []
    with open(NODES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes.append({
                "graph_id": graph_id,
                "x_user_id": row.get("user_id", ""),
                "x_handle": row.get("handle", ""),
                "x_name": row.get("name", ""),
                "x_bio": row.get("bio", ""),
                "x_followers_count": int(row.get("followers_count", 0) or 0),
                "x_following_count": int(row.get("following_count", 0) or 0),
                "x_tweet_count": int(row.get("tweet_count", 0) or 0),
                "is_seed": row.get("is_root", "False").lower() == "true",
                "is_candidate": row.get("is_candidate", "False").lower() == "true",
                "discovered_via": row.get("discovered_via", ""),
                "pagerank_score": float(row.get("pagerank_score", 0) or 0),
                "underratedness_score": float(row.get("underratedness_score", 0) or 0),
                "fast_screened": row.get("grok_evaluated", "False").lower() == "true",
                "fast_screen_passed": row.get("grok_relevant", "False").lower() == "true",
                "deep_evaluated": False,  # Will be updated when migrating evaluations
            })

    print(f"Found {len(nodes)} nodes to migrate")

    # Batch insert (Supabase supports up to ~1000 rows per insert)
    handle_to_id = {}
    batch_size = 500

    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]
        try:
            result = supabase.table("nodes").upsert(
                batch,
                on_conflict="graph_id,x_user_id"
            ).execute()

            for node in result.data:
                handle_to_id[node["x_handle"].lower()] = node["id"]

            print(f"  Migrated nodes {i} to {i + len(batch)}")
        except Exception as e:
            print(f"  Error migrating batch {i}: {e}")

    print(f"Migrated {len(handle_to_id)} nodes")
    return handle_to_id


def migrate_evaluations(supabase: Client, handle_to_id: dict):
    """Migrate deep evaluations from JSON files."""
    if not ENRICHED_DIR.exists():
        print(f"Enriched directory not found: {ENRICHED_DIR}")
        return

    eval_files = list(ENRICHED_DIR.glob("deep_*.json"))
    print(f"Found {len(eval_files)} evaluation files")

    migrated = 0
    for json_file in eval_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            handle = eval_data.get("handle", "").lower()
            if handle not in handle_to_id:
                continue

            node_id = handle_to_id[handle]

            # Build evaluation record
            evaluation = {
                "node_id": node_id,
                "final_score": eval_data.get("final_score"),
                "summary": eval_data.get("summary"),
                "strengths": eval_data.get("strengths", []),
                "concerns": eval_data.get("concerns", []),
                "recommended_role": eval_data.get("recommended_role"),
                "github_url": eval_data.get("github_url"),
                "linkedin_url": eval_data.get("linkedin_url"),
                "top_repos": eval_data.get("top_repos"),
            }

            # Add score breakdowns
            for field in ["technical_depth", "project_evidence", "mission_alignment",
                         "exceptional_ability", "communication"]:
                if field in eval_data and eval_data[field]:
                    data = eval_data[field]
                    if isinstance(data, dict):
                        if field == "project_evidence":
                            evaluation["project_evidence_score"] = data.get("score")
                            evaluation["project_evidence_text"] = data.get("evidence")
                        else:
                            evaluation[f"{field}_score"] = data.get("score")
                            evaluation[f"{field}_evidence"] = data.get("evidence")

            # Insert evaluation
            supabase.table("evaluations").upsert(
                evaluation,
                on_conflict="node_id"
            ).execute()

            # Mark node as deep_evaluated
            supabase.table("nodes").update({
                "deep_evaluated": True
            }).eq("id", node_id).execute()

            migrated += 1

        except Exception as e:
            print(f"  Error migrating {json_file.name}: {e}")

    print(f"Migrated {migrated} evaluations")


def migrate_seeds(supabase: Client, graph_id: str):
    """Migrate seeds from seeds.yaml."""
    if not SEEDS_YAML.exists():
        print(f"Seeds YAML not found: {SEEDS_YAML}")
        return

    try:
        import yaml
    except ImportError:
        print("Please install pyyaml: pip install pyyaml")
        return

    with open(SEEDS_YAML, "r") as f:
        seeds_data = yaml.safe_load(f)

    handles = seeds_data.get("seed_accounts", [])
    print(f"Found {len(handles)} seeds in YAML")

    # Get node data for seeds
    seeds = []
    for handle in handles:
        handle_clean = handle.lower().lstrip("@")

        # Try to find existing node
        result = supabase.table("nodes").select("x_user_id, x_name, x_bio, x_followers_count").eq(
            "graph_id", graph_id
        ).eq("x_handle", handle_clean).execute()

        seed = {
            "graph_id": graph_id,
            "x_handle": handle_clean,
            "x_user_id": result.data[0]["x_user_id"] if result.data else "",
            "x_name": result.data[0].get("x_name", "") if result.data else "",
            "x_bio": result.data[0].get("x_bio", "") if result.data else "",
            "x_followers_count": result.data[0].get("x_followers_count", 0) if result.data else 0,
            "is_active": True,
        }
        seeds.append(seed)

    # Batch insert
    try:
        supabase.table("seeds").upsert(
            seeds,
            on_conflict="graph_id,x_user_id"
        ).execute()
        print(f"Migrated {len(seeds)} seeds")
    except Exception as e:
        print(f"Error migrating seeds: {e}")


def verify_migration(supabase: Client, graph_id: str):
    """Verify migration results."""
    print("\n=== Migration Verification ===")

    # Count nodes
    result = supabase.table("nodes").select("id", count="exact").eq("graph_id", graph_id).execute()
    print(f"Total nodes: {result.count}")

    # Count candidates
    result = supabase.table("nodes").select("id", count="exact").eq(
        "graph_id", graph_id
    ).eq("is_candidate", True).execute()
    print(f"Candidates: {result.count}")

    # Count deep evaluated
    result = supabase.table("nodes").select("id", count="exact").eq(
        "graph_id", graph_id
    ).eq("deep_evaluated", True).execute()
    print(f"Deep evaluated: {result.count}")

    # Count evaluations
    result = supabase.table("evaluations").select("id", count="exact").execute()
    print(f"Evaluations: {result.count}")

    # Count seeds
    result = supabase.table("seeds").select("id", count="exact").eq("graph_id", graph_id).execute()
    print(f"Seeds: {result.count}")

    # Sample top candidates
    print("\nTop 5 candidates by score:")
    result = supabase.table("nodes").select(
        "x_handle, pagerank_score, underratedness_score"
    ).eq("graph_id", graph_id).eq("deep_evaluated", True).order(
        "underratedness_score", desc=True
    ).limit(5).execute()

    for node in result.data:
        print(f"  @{node['x_handle']}: PR={node['pagerank_score']:.6f}, UR={node['underratedness_score']:.6f}")


def main():
    print("=== Grok Recruiter Data Migration ===\n")

    # Initialize Supabase client
    print("Connecting to Supabase...")
    supabase = get_supabase_client()
    print("Connected!\n")

    # Get or create base graph
    graph_id = get_or_create_base_graph(supabase)
    print()

    # Migrate nodes
    handle_to_id = migrate_nodes(supabase, graph_id)
    print()

    # Migrate evaluations
    migrate_evaluations(supabase, handle_to_id)
    print()

    # Migrate seeds
    migrate_seeds(supabase, graph_id)

    # Verify
    verify_migration(supabase, graph_id)

    print("\n=== Migration Complete ===")


if __name__ == "__main__":
    main()
