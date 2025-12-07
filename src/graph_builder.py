"""
Graph Builder for Grok Recruiter
Constructs a knowledge graph from X API data
"""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

from .x_client import XClient


@dataclass
class Node:
    """Represents an X account in the graph."""

    user_id: str
    handle: str
    name: str
    bio: str = ""
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    is_root: bool = False
    is_candidate: bool = False
    discovered_via: str = ""

    # Scores from ranking module
    pagerank_score: float = 0.0
    underratedness_score: float = 0.0

    # Grok evaluation fields (NEW)
    grok_evaluated: bool = False
    grok_relevant: bool = False
    grok_score: float = 0.0
    grok_reasoning: str = ""
    grok_skills: str = ""  # Comma-separated
    grok_role: str = ""  # research/engineering/infrastructure/none
    grok_exceptional_work: str = ""
    grok_red_flags: str = ""  # Comma-separated


@dataclass
class Edge:
    """Represents an interaction between two accounts."""

    src_user_id: str
    dst_user_id: str
    interaction_type: str  # like, reply, retweet, quote
    tweet_id: str
    created_at: str
    weight: float
    depth: int


class GraphBuilder:
    """Builds a knowledge graph from X interactions."""

    # Edge weights by interaction type (following is strongest signal)
    WEIGHT_MAP = {
        "follow": 5.0,  # Strongest - seed chose to follow
        "retweet": 3.0,  # Strong - seed amplified their content
        "reply": 2.5,  # Strong - seed engaged in conversation
        "quote": 2.0,  # Moderate - seed commented on their content
        "like": 1.0,  # Weak - passive engagement
    }

    def __init__(
        self,
        x_client: XClient,
        roots: List[str],
        settings: Dict[str, Any],
    ):
        """
        Initialize the graph builder.

        Args:
            x_client: X API client instance
            roots: List of root account handles
            settings: Configuration settings
        """
        self.x = x_client
        self.root_handles = roots
        self.settings = settings

        # Graph data
        self.nodes: Dict[str, Node] = {}  # user_id -> Node
        self.edges: List[Edge] = []

        # Track which users we need to hydrate
        self._pending_hydration: Set[str] = set()

        # Track filtered employees (for reporting)
        self.filtered_employees: List[str] = []  # List of handles filtered out

    def _add_or_update_node(
        self,
        user: Dict[str, Any],
        is_root: bool = False,
        discovered_via: str = "",
    ) -> Node:
        """
        Add or update a node in the graph.

        Args:
            user: User object from X API
            is_root: Whether this is a root/seed account
            discovered_via: How this user was discovered

        Returns:
            The created or updated Node
        """
        uid = user.get("id", "")
        if not uid:
            return None

        metrics = user.get("public_metrics", {})

        # Check if node exists
        existing = self.nodes.get(uid)
        if existing:
            # Update flags
            existing.is_root = existing.is_root or is_root
            if discovered_via and not existing.discovered_via:
                existing.discovered_via = discovered_via
            # Update metrics if we have them
            if metrics:
                existing.followers_count = metrics.get(
                    "followers_count", existing.followers_count
                )
                existing.following_count = metrics.get(
                    "following_count", existing.following_count
                )
                existing.tweet_count = metrics.get("tweet_count", existing.tweet_count)
            return existing

        # Extract user data
        handle = user.get("username", f"id_{uid}")
        name = user.get("name", "")
        bio = user.get("description", "")

        # NOTE: xAI/X employee and organization filtering is done by fast LLM
        # in grok_client.py fast_screen() - not programmatically here

        # Create new node
        node = Node(
            user_id=uid,
            handle=handle,
            name=name,
            bio=bio,
            followers_count=metrics.get("followers_count", 0),
            following_count=metrics.get("following_count", 0),
            tweet_count=metrics.get("tweet_count", 0),
            is_root=is_root,
            discovered_via=discovered_via,
        )

        # Apply candidate criteria
        max_followers = self.settings.get("max_followers_candidate", 50000)
        min_followers = self.settings.get("min_followers_candidate", 50)
        node.is_candidate = (
            not node.is_root and min_followers <= node.followers_count <= max_followers
        )

        self.nodes[uid] = node
        return node

    def _add_stub_node(self, user_id: str, discovered_via: str) -> Node:
        """
        Add a stub node that will be hydrated later.

        Args:
            user_id: X user ID
            discovered_via: How this user was discovered

        Returns:
            The stub Node
        """
        if user_id in self.nodes:
            return self.nodes[user_id]

        # Mark for hydration
        self._pending_hydration.add(user_id)

        node = Node(
            user_id=user_id,
            handle=f"id_{user_id}",
            name="",
            discovered_via=discovered_via,
        )
        self.nodes[user_id] = node
        return node

    def _add_edge(
        self,
        src_user_id: str,
        dst_user_id: str,
        interaction_type: str,
        tweet_id: str,
        created_at: str,
        depth: int,
    ):
        """
        Add an edge to the graph.

        Args:
            src_user_id: Source user ID
            dst_user_id: Destination user ID
            interaction_type: Type of interaction
            tweet_id: Associated tweet ID
            created_at: Timestamp
            depth: Hop distance from roots
        """
        weight = self.WEIGHT_MAP.get(interaction_type, 1.0)

        edge = Edge(
            src_user_id=src_user_id,
            dst_user_id=dst_user_id,
            interaction_type=interaction_type,
            tweet_id=tweet_id,
            created_at=created_at,
            weight=weight,
            depth=depth,
        )
        self.edges.append(edge)

    def _hydrate_pending_users(self):
        """Hydrate stub nodes with full user data."""
        if not self._pending_hydration:
            return

        print(
            f"\n[hydrate] Fetching details for {len(self._pending_hydration)} users..."
        )

        # Get user details in batches
        user_ids = list(self._pending_hydration)
        users = self.x.get_users_by_ids(user_ids)

        # Update nodes with full data
        for user in users:
            uid = user.get("id")
            if uid and uid in self.nodes:
                node = self.nodes[uid]
                node.handle = user.get("username", node.handle)
                node.name = user.get("name", node.name)
                node.bio = user.get("description", "")

                metrics = user.get("public_metrics", {})
                node.followers_count = metrics.get("followers_count", 0)
                node.following_count = metrics.get("following_count", 0)
                node.tweet_count = metrics.get("tweet_count", 0)

                # Re-evaluate candidate status
                max_followers = self.settings.get("max_followers_candidate", 50000)
                min_followers = self.settings.get("min_followers_candidate", 50)
                node.is_candidate = (
                    not node.is_root
                    and min_followers <= node.followers_count <= max_followers
                )

        self._pending_hydration.clear()
        print(f"[hydrate] Updated {len(users)} user profiles")

    def build_graph(self):
        """Build the full knowledge graph from root accounts."""
        print("\n" + "=" * 60)
        print("PHASE 1: Resolving root accounts")
        print("=" * 60)

        root_users: Dict[str, Dict[str, Any]] = {}

        # Resolve root handles to user objects
        for handle in self.root_handles:
            user = self.x.get_user_by_username(handle)
            if not user:
                print(f"  [warn] Could not resolve @{handle}")
                continue

            root_users[handle] = user
            self._add_or_update_node(user, is_root=True)
            print(
                f"  [root] @{handle} (ID: {user['id']}, followers: {user.get('public_metrics', {}).get('followers_count', 0):,})"
            )

        print(f"\n  Resolved {len(root_users)}/{len(self.root_handles)} root accounts")

        print("\n" + "=" * 60)
        print(
            "PHASE 2: Expanding from roots (priority: follow > retweet > reply > like)"
        )
        print("=" * 60)

        for handle, root_user in root_users.items():
            root_id = root_user["id"]
            print(f"\n[root] Expanding @{handle}")

            # 2a) WHO SEEDS FOLLOW (strongest signal - they chose to follow)
            self._expand_following(root_id)

            # 2b) Get root's tweets and find engagers (retweets, replies)
            # Note: Likes endpoints not available on Free/Basic X API tier
            self._expand_root_tweets(root_id)

        print("\n" + "=" * 60)
        print("PHASE 3: Hydrating discovered users")
        print("=" * 60)

        self._hydrate_pending_users()

        # Summary
        print("\n" + "=" * 60)
        print("DISCOVERY COMPLETE")
        print("=" * 60)
        print(f"  Total nodes: {len(self.nodes)}")
        print(f"  Total edges: {len(self.edges)}")
        print(f"  Root accounts: {sum(1 for n in self.nodes.values() if n.is_root)}")
        print(
            f"  Candidate accounts: {sum(1 for n in self.nodes.values() if n.is_candidate)}"
        )

        # Edge type breakdown
        edge_counts: Dict[str, int] = {}
        for edge in self.edges:
            edge_counts[edge.interaction_type] = (
                edge_counts.get(edge.interaction_type, 0) + 1
            )
        print(f"  Edge breakdown: {edge_counts}")

    def expand_depth_2(self, top_k: int = 50, max_following_per_node: int = 100):
        """
        Expand graph to depth 2 by following who top candidates follow.

        Strategy: Take top-k candidates from depth 1 (most followed by seeds)
        and see who THEY follow. This finds "friends of friends".

        Args:
            top_k: Number of top depth-1 candidates to expand from
            max_following_per_node: Max accounts to fetch per candidate
        """
        print("\n" + "=" * 60)
        print("DEPTH 2: Expanding from top candidates")
        print("=" * 60)

        # Find candidates with highest in-degree (followed by most seeds)
        candidate_in_degree: Dict[str, int] = {}
        for edge in self.edges:
            if edge.interaction_type == "follow" and edge.depth == 1:
                dst = edge.dst_user_id
                if dst in self.nodes and self.nodes[dst].is_candidate:
                    candidate_in_degree[dst] = candidate_in_degree.get(dst, 0) + 1

        # Sort by in-degree
        top_candidates = sorted(
            candidate_in_degree.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        print(
            f"  Expanding from top {len(top_candidates)} candidates (by seed follow count)"
        )

        depth2_discovered = 0

        for user_id, in_degree in top_candidates:
            node = self.nodes.get(user_id)
            if not node:
                continue

            print(f"\n  [depth2] @{node.handle} (followed by {in_degree} seeds)")

            # Get who this candidate follows
            following = self.x.get_user_following(
                user_id, max_results=max_following_per_node
            )

            for user in following:
                fol_id = user.get("id")
                if not fol_id or fol_id == user_id:
                    continue

                # Skip if already in graph (we want NEW discoveries)
                if fol_id in self.nodes:
                    continue

                # Skip seeds
                if any(n.user_id == fol_id and n.is_root for n in self.nodes.values()):
                    continue

                # Add new node
                new_node = self._add_or_update_node(
                    user, discovered_via="depth2_following"
                )
                if new_node:
                    depth2_discovered += 1

                    # Add edge: candidate -> new person (depth 2)
                    self._add_edge(
                        src_user_id=user_id,
                        dst_user_id=fol_id,
                        interaction_type="follow",
                        tweet_id="",
                        created_at="",
                        depth=2,
                    )

        # Hydrate new discoveries
        print(f"\n  Discovered {depth2_discovered} new accounts at depth 2")
        self._hydrate_pending_users()

        # Updated summary
        print("\n" + "=" * 60)
        print("DEPTH 2 COMPLETE")
        print("=" * 60)
        print(f"  Total nodes: {len(self.nodes)}")
        print(f"  Total edges: {len(self.edges)}")

        depth1_candidates = sum(
            1
            for n in self.nodes.values()
            if n.is_candidate and n.discovered_via != "depth2_following"
        )
        depth2_candidates = sum(
            1
            for n in self.nodes.values()
            if n.is_candidate and n.discovered_via == "depth2_following"
        )
        print(f"  Depth 1 candidates: {depth1_candidates}")
        print(f"  Depth 2 candidates: {depth2_candidates}")

    def _expand_following(self, root_id: str):
        """Expand graph via accounts the root follows (strongest signal)."""
        max_following = self.settings.get("max_following_per_root", 1000)
        following = self.x.get_user_following(root_id, max_results=max_following)

        print(f"  [following] Processing {len(following)} accounts followed by root")

        for user in following:
            user_id = user.get("id")
            if not user_id or user_id == root_id:
                continue

            # Skip if this is another root (xAI employee)
            # They'll be processed separately
            if user_id in self.nodes and self.nodes[user_id].is_root:
                continue

            # Add with full user data (following API returns user objects)
            self._add_or_update_node(user, discovered_via="followed_by_root")

            # Add edge: root -> user (root follows them)
            self._add_edge(
                src_user_id=root_id,
                dst_user_id=user_id,
                interaction_type="follow",
                tweet_id="",  # No tweet associated with follow
                created_at="",
                depth=1,
            )

    def _expand_liked_tweets(self, root_id: str):
        """Expand graph via tweets the root has liked."""
        max_liked = self.settings.get("max_liked_tweets", 100)
        liked_tweets = self.x.get_user_liked_tweets(root_id, max_results=max_liked)

        for tweet in liked_tweets:
            author_id = tweet.get("author_id")
            if not author_id or author_id == root_id:
                continue

            # Add author as discovered node
            self._add_stub_node(author_id, discovered_via="liked_by_root")

            # Add edge: root -> author (root liked their tweet)
            self._add_edge(
                src_user_id=root_id,
                dst_user_id=author_id,
                interaction_type="like",
                tweet_id=tweet.get("id", ""),
                created_at=tweet.get("created_at", ""),
                depth=1,
            )

    def _expand_root_tweets(self, root_id: str):
        """Expand graph via engagements on root's tweets.

        RATE LIMIT AWARE (Basic tier):
        - GET /2/tweets/:id/retweeted_by: 5 requests / 15 mins PER USER
        - GET /2/tweets/search/recent: 60 requests / 15 mins PER USER/APP

        Strategy: Only process top 3 tweets per root to stay under limits.
        With 90 seeds: 90 * 3 * 2 = 540 calls (spread over time with caching)
        """
        max_tweets = self.settings.get("max_root_tweets", 50)
        root_tweets = self.x.get_user_tweets(root_id, max_results=max_tweets)

        # RATE LIMIT: Only process top 3 tweets per root
        # This keeps us under Basic tier limits when running across all seeds
        max_tweets_to_process = min(len(root_tweets), 3)

        for tweet in root_tweets[:max_tweets_to_process]:
            tweet_id = tweet.get("id")
            created_at = tweet.get("created_at", "")
            conv_id = tweet.get("conversation_id", tweet_id)

            # Get users who retweeted (5 req/15min per user on Basic)
            max_retweeters = self.settings.get("max_retweeters_per_tweet", 100)
            retweeters = self.x.get_retweeting_users(
                tweet_id, max_results=max_retweeters
            )

            for user in retweeters:
                user_id = user.get("id")
                if not user_id or user_id == root_id:
                    continue

                self._add_or_update_node(user, discovered_via="retweeted_root")

                self._add_edge(
                    src_user_id=user_id,
                    dst_user_id=root_id,
                    interaction_type="retweet",
                    tweet_id=tweet_id,
                    created_at=created_at,
                    depth=1,
                )

            # Get replies (60 req/15min on Basic - more headroom)
            max_replies = self.settings.get("max_replies_per_conversation", 50)
            replies = self.x.search_replies(conv_id, max_results=max_replies)

            for reply in replies:
                author_id = reply.get("author_id")
                if not author_id or author_id == root_id:
                    continue

                self._add_stub_node(author_id, discovered_via="replied_to_root")

                self._add_edge(
                    src_user_id=author_id,
                    dst_user_id=root_id,
                    interaction_type="reply",
                    tweet_id=reply.get("id", ""),
                    created_at=reply.get("created_at", ""),
                    depth=1,
                )

    def export_to_csv(self, nodes_path: str, edges_path: str):
        """
        Export graph to CSV files.

        Args:
            nodes_path: Path for nodes CSV
            edges_path: Path for edges CSV
        """
        # Ensure directories exist
        Path(nodes_path).parent.mkdir(parents=True, exist_ok=True)
        Path(edges_path).parent.mkdir(parents=True, exist_ok=True)

        # Export nodes
        if self.nodes:
            node_fields = list(asdict(next(iter(self.nodes.values()))).keys())
            with open(nodes_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=node_fields)
                writer.writeheader()
                for node in self.nodes.values():
                    writer.writerow(asdict(node))
            print(f"  Exported {len(self.nodes)} nodes to {nodes_path}")

        # Export edges
        if self.edges:
            edge_fields = list(asdict(self.edges[0]).keys())
            with open(edges_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=edge_fields)
                writer.writeheader()
                for edge in self.edges:
                    writer.writerow(asdict(edge))
            print(f"  Exported {len(self.edges)} edges to {edges_path}")
