#!/usr/bin/env python3
"""
Grok Recruiter - Taste-Graph Talent Discovery
Builds a knowledge graph from X and uses Grok to evaluate candidates
"""

import argparse
import sys
from pathlib import Path

import yaml

from .evaluator import CandidateEvaluator, print_evaluation_summary
from .graph_builder import GraphBuilder
from .grok_client import GrokClient
from .ranking import compute_rankings, export_ranked_nodes
from .x_client import XClient


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Grok Recruiter - Discover underrated candidates from X"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="seeds.yaml",
        help="Path to configuration YAML file (default: seeds.yaml)",
    )
    parser.add_argument(
        "--criteria",
        type=str,
        default="criteria.yaml",
        help="Path to evaluation criteria YAML (default: criteria.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for CSV files (default: data/processed)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/raw",
        help="Cache directory for API responses (default: data/raw)",
    )
    parser.add_argument(
        "--eval-cache-dir",
        type=str,
        default="data/evaluations",
        help="Cache directory for Grok evaluations (default: data/evaluations)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable API response caching",
    )
    parser.add_argument(
        "--skip-grok",
        action="store_true",
        help="Skip Grok evaluation phase (just build graph)",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=50,
        help="Maximum candidates to evaluate with Grok (default: 50)",
    )

    args = parser.parse_args()

    # Load configuration
    print("=" * 70)
    print("GROK RECRUITER - TASTE-GRAPH TALENT DISCOVERY")
    print("=" * 70)

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration: {e}")
        sys.exit(1)

    roots = config.get("roots", [])
    settings = config.get("settings", {})

    print("\nConfiguration:")
    print(f"  Config file: {args.config}")
    print(f"  Criteria file: {args.criteria}")
    print(f"  Root accounts: {roots}")
    print(f"  Max followers: {settings.get('max_followers_candidate', 50000):,}")
    print(f"  Max Grok evaluations: {args.max_eval}")
    print(f"  Output directory: {args.output_dir}")

    if not roots:
        print("\nError: No root accounts specified in configuration")
        sys.exit(1)

    # Initialize X client
    cache_dir = None if args.no_cache else args.cache_dir
    try:
        x_client = XClient(cache_dir=cache_dir)
    except KeyError as e:
        print(f"\nError: Missing environment variable: {e}")
        print("Make sure you have set up your .env file with X API credentials")
        print("See .env.example for the required variables")
        sys.exit(1)

    # ========================================
    # PHASE 1-3: Build initial graph
    # ========================================
    builder = GraphBuilder(
        x_client=x_client,
        roots=roots,
        settings=settings,
    )

    try:
        builder.build_graph()
    except Exception as e:
        print(f"\nError during graph building: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # ========================================
    # PHASE 4: Grok Evaluation (NEW)
    # ========================================
    if not args.skip_grok:
        print("\n" + "=" * 70)
        print("PHASE 4: GROK EVALUATION")
        print("=" * 70)

        try:
            # Check for Grok API key
            grok_client = GrokClient(cache_dir=args.eval_cache_dir)

            # Initialize evaluator with seed handles for employee exclusion
            evaluator = CandidateEvaluator(
                x_client=x_client,
                grok_client=grok_client,
                criteria_path=args.criteria,
                min_followers=settings.get("min_followers_candidate", 100),
                max_followers=settings.get("max_followers_candidate", 100000),
                min_tweets=50,
                seed_handles=set(roots),  # Exclude known xAI employees
                use_fast_screen=True,  # Enable fast LLM screening
            )

            # Get candidates to evaluate (prioritize by discovery method)
            # Priority: followed_by_root > liked_by_root > retweeted_root > replied_to_root
            discovery_priority = {
                "followed_by_root": 0,
                "liked_by_root": 1,
                "retweeted_root": 2,
                "replied_to_root": 3,
                "liked_root_tweet": 4,
            }

            candidates = [
                {
                    "user_id": node.user_id,
                    "handle": node.handle,
                    "bio": node.bio,
                    "followers_count": node.followers_count,
                    "tweet_count": node.tweet_count,
                    "discovered_via": node.discovered_via,
                    # TODO: Add pinned_tweet and location when available
                }
                for node in builder.nodes.values()
                if node.is_candidate
            ]

            # Sort by discovery priority (followed_by_root first)
            candidates.sort(key=lambda c: discovery_priority.get(c.get("discovered_via", ""), 99))

            print(f"\n  Found {len(candidates)} candidates to evaluate")

            # Run evaluation
            results = evaluator.evaluate_batch(candidates, max_candidates=args.max_eval)

            # Update nodes with Grok evaluations
            for candidate, pre_filter, evaluation in results:
                user_id = candidate.get("user_id")
                if user_id in builder.nodes:
                    node = builder.nodes[user_id]
                    node.grok_evaluated = True

                    if evaluation:
                        node.grok_relevant = evaluation.relevant
                        node.grok_score = evaluation.score
                        node.grok_reasoning = evaluation.reasoning
                        node.grok_skills = ",".join(evaluation.detected_skills)
                        node.grok_role = evaluation.recommended_role
                        node.grok_exceptional_work = evaluation.exceptional_work
                        node.grok_red_flags = ",".join(evaluation.red_flags)

            # Print summary
            print_evaluation_summary(results)

        except ValueError as e:
            print(f"\n  [warn] Skipping Grok evaluation: {e}")
            print("  Make sure XAI_API_KEY is set in your .env file")

    # ========================================
    # PHASE 5: Compute Rankings
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 5: COMPUTING RANKINGS")
    print("=" * 70)

    nodes = compute_rankings(builder.nodes, builder.edges)

    # ========================================
    # PHASE 6: Export Results
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 6: EXPORTING RESULTS")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = output_dir / "nodes.csv"
    edges_path = output_dir / "edges.csv"

    export_ranked_nodes(nodes, str(nodes_path))
    builder.export_to_csv(str(nodes_path), str(edges_path))

    # Final summary
    print("\n" + "=" * 70)
    print("DISCOVERY COMPLETE")
    print("=" * 70)

    total_nodes = len(builder.nodes)
    total_edges = len(builder.edges)
    candidates = sum(1 for n in builder.nodes.values() if n.is_candidate)
    grok_evaluated = sum(1 for n in builder.nodes.values() if n.grok_evaluated)
    grok_relevant = sum(1 for n in builder.nodes.values() if n.grok_relevant)

    print(f"""
Summary:
  Total accounts discovered: {total_nodes:,}
  Total interactions mapped: {total_edges:,}
  Candidate accounts (follower filter): {candidates:,}
  Grok evaluated: {grok_evaluated}
  Grok relevant (high-quality): {grok_relevant}

Output files:
  Nodes: {nodes_path}
  Edges: {edges_path}

Key columns in nodes.csv:
  - grok_relevant: True if Grok thinks they're xAI-caliber
  - grok_score: 0.0-1.0 quality score
  - grok_role: recommended role (research/engineering/infrastructure)
  - grok_reasoning: Why Grok thinks they're relevant
  - underratedness_score: High signal + low followers = hidden gem
""")


if __name__ == "__main__":
    main()
