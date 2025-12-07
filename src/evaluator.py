"""
Candidate Evaluator for Grok Recruiter
Pre-filters candidates and orchestrates Grok evaluation
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .grok_client import (
    CandidateEvaluation,
    FastScreenResult,
    GrokClient,
    load_criteria,
)
from .x_client import XClient


@dataclass
class PreFilterResult:
    """Result of pre-filtering a candidate."""

    passed: bool
    reason: str
    fast_screen: Optional[FastScreenResult] = None


class CandidateEvaluator:
    """Evaluates candidates using pre-filtering and Grok."""

    # Red flag keywords in bio (case-insensitive)
    BIO_RED_FLAGS = [
        r"\bcrypto\b",
        r"\bnft\b",
        r"\bforex\b",
        r"\btrading\b",
        r"\bweb3\b",
        r"\bdefi\b",
        r"\b100x\b",
        r"\bget rich\b",
        r"\bpassive income\b",
        r"\bfollow for follow\b",
        r"\bf4f\b",
    ]

    # Tech keywords that override red flags
    TECH_OVERRIDES = [
        r"\bml\b",
        r"\bai\b",
        r"\bmachine learning\b",
        r"\bdeep learning\b",
        r"\bpython\b",
        r"\brust\b",
        r"\bc\+\+\b",
        r"\bcuda\b",
        r"\bgpu\b",
        r"\btransformer\b",
        r"\bllm\b",
        r"\bphd\b",
        r"\bresearch\b",
        r"\bengineer\b",
        r"\bdeveloper\b",
        r"\bsoftware\b",
    ]

    # Patterns to detect xAI/X employees (not candidates)
    EMPLOYEE_PATTERNS = [
        r"\b@xai\b",
        r"\bxai\b",
        r"\bx\.ai\b",
        r"\bworking at xai\b",
        r"\bengineer at xai\b",
        r"\b@x\b",
        r"\btwitter\b",
        r"\bworking at x\b",
        r"\bengineer at x\b",
        r"\bx corp\b",
    ]

    def __init__(
        self,
        x_client: XClient,
        grok_client: GrokClient,
        criteria_path: str = "criteria.yaml",
        min_followers: int = 100,
        max_followers: int = 100000,
        min_tweets: int = 50,
        seed_handles: Optional[Set[str]] = None,
        use_fast_screen: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            x_client: X API client
            grok_client: Grok API client
            criteria_path: Path to criteria YAML file
            min_followers: Minimum followers for candidates
            max_followers: Maximum followers (underrated threshold)
            min_tweets: Minimum tweet count
            seed_handles: Set of seed handles to exclude (xAI employees)
            use_fast_screen: Whether to use fast LLM screening before full eval
        """
        self.x = x_client
        self.grok = grok_client
        self.criteria = load_criteria(criteria_path)

        self.min_followers = min_followers
        self.max_followers = max_followers
        self.min_tweets = min_tweets
        self.seed_handles = {h.lower() for h in (seed_handles or set())}
        self.use_fast_screen = use_fast_screen

    def is_employee(self, handle: str, bio: str) -> bool:
        """Check if user is likely an xAI/X employee (not a candidate)."""
        handle_lower = handle.lower()

        # Check if in seed list (known xAI employees)
        if handle_lower in self.seed_handles:
            return True

        # Check bio for employee patterns
        bio_lower = bio.lower() if bio else ""
        for pattern in self.EMPLOYEE_PATTERNS:
            if re.search(pattern, bio_lower):
                return True

        return False

    def pre_filter(
        self,
        handle: str,
        bio: str,
        followers: int,
        tweet_count: int,
        pinned_tweet: Optional[str] = None,
        location: Optional[str] = None,
    ) -> PreFilterResult:
        """
        Pre-filter a candidate before expensive Grok evaluation.

        Args:
            handle: X username
            bio: Profile bio
            followers: Follower count
            tweet_count: Total tweet count
            pinned_tweet: Optional pinned tweet text
            location: Optional location

        Returns:
            PreFilterResult indicating if candidate should be evaluated
        """
        # Check if this is an xAI/X employee (not a candidate)
        if self.is_employee(handle, bio):
            return PreFilterResult(False, "xAI/X employee (not a candidate)")

        # Check follower count
        if followers < self.min_followers:
            return PreFilterResult(
                False, f"followers ({followers}) < {self.min_followers}"
            )

        if followers > self.max_followers:
            return PreFilterResult(
                False, f"followers ({followers}) > {self.max_followers}"
            )

        # Check tweet count
        if tweet_count < self.min_tweets:
            return PreFilterResult(False, f"tweets ({tweet_count}) < {self.min_tweets}")

        # Check for empty bio
        if not bio or len(bio.strip()) < 10:
            return PreFilterResult(False, "bio too short or empty")

        # Check bio for red flags (quick heuristic)
        bio_lower = bio.lower() if bio else ""

        # Check for tech keywords first (override red flags)
        has_tech_signal = any(
            re.search(pattern, bio_lower) for pattern in self.TECH_OVERRIDES
        )

        if not has_tech_signal:
            # Check for red flags only if no tech signal
            for pattern in self.BIO_RED_FLAGS:
                if re.search(pattern, bio_lower):
                    return PreFilterResult(False, f"bio red flag: {pattern}")

        # Fast LLM screening (if enabled)
        if self.use_fast_screen:
            fast_result = self.grok.fast_screen(
                handle=handle,
                bio=bio,
                pinned_tweet=pinned_tweet,
                location=location,
            )
            if not fast_result.pass_filter:
                return PreFilterResult(
                    False, f"fast_screen: {fast_result.reason}", fast_result
                )

            return PreFilterResult(
                True, f"passed (role: {fast_result.potential_role})", fast_result
            )

        return PreFilterResult(True, "passed pre-filter")

    def evaluate_candidate(
        self,
        user_id: str,
        handle: str,
        bio: str,
        followers: int,
        tweet_count: int,
        pinned_tweet: Optional[str] = None,
        location: Optional[str] = None,
    ) -> Tuple[PreFilterResult, Optional[CandidateEvaluation]]:
        """
        Evaluate a single candidate.

        Args:
            user_id: X user ID
            handle: X username
            bio: Profile bio
            followers: Follower count
            tweet_count: Total tweet count
            pinned_tweet: Optional pinned tweet text
            location: Optional location

        Returns:
            Tuple of (pre_filter_result, grok_evaluation or None)
        """
        # Pre-filter first (includes fast LLM screen)
        pre_filter = self.pre_filter(
            handle, bio, followers, tweet_count, pinned_tweet, location
        )
        if not pre_filter.passed:
            return pre_filter, None

        # Fetch candidate's tweets
        tweets = self.x.get_user_tweets(user_id, max_results=50)
        if not tweets:
            return PreFilterResult(False, "could not fetch tweets"), None

        # Extract tweet text
        tweet_texts = [t.get("text", "") for t in tweets if t.get("text")]

        if len(tweet_texts) < 10:
            return PreFilterResult(False, "not enough tweet content"), None

        # Evaluate with Grok (full evaluation)
        evaluation = self.grok.evaluate_candidate(
            handle=handle,
            bio=bio,
            tweets=tweet_texts,
            followers=followers,
            criteria=self.criteria,
        )

        return pre_filter, evaluation

    def evaluate_batch(
        self,
        candidates: List[Dict],
        max_candidates: int = 50,
    ) -> List[Tuple[Dict, PreFilterResult, Optional[CandidateEvaluation]]]:
        """
        Evaluate a batch of candidates.

        Args:
            candidates: List of candidate dicts with user_id, handle, bio, etc.
            max_candidates: Maximum candidates to evaluate (for testing)

        Returns:
            List of (candidate, pre_filter_result, evaluation) tuples
        """
        results = []
        evaluated = 0
        skipped = 0
        fast_screened = 0

        print(
            f"\n[evaluator] Processing {len(candidates)} candidates (max: {max_candidates})"
        )

        for candidate in candidates:
            if evaluated >= max_candidates:
                break

            user_id = candidate.get("user_id", "")
            handle = candidate.get("handle", "")
            bio = candidate.get("bio", "")
            followers = candidate.get("followers_count", 0)
            tweet_count = candidate.get("tweet_count", 0)
            pinned_tweet = candidate.get("pinned_tweet", None)
            location = candidate.get("location", None)

            # Pre-filter (includes fast LLM screen)
            pre_filter, evaluation = self.evaluate_candidate(
                user_id=user_id,
                handle=handle,
                bio=bio,
                followers=followers,
                tweet_count=tweet_count,
                pinned_tweet=pinned_tweet,
                location=location,
            )

            results.append((candidate, pre_filter, evaluation))

            if pre_filter.passed:
                evaluated += 1
                if evaluation and evaluation.relevant:
                    print(
                        f"  ✓ @{handle} - score: {evaluation.score:.2f} - {evaluation.reasoning[:60]}..."
                    )
                else:
                    print(f"  ✗ @{handle} - not relevant")
            else:
                skipped += 1
                if pre_filter.fast_screen:
                    fast_screened += 1

        print(
            f"\n[evaluator] Results: {evaluated} evaluated, {skipped} skipped by pre-filter ({fast_screened} by fast screen)"
        )

        # Summary of relevant candidates
        relevant = [(c, e) for c, pf, e in results if e and e.relevant]
        print(f"[evaluator] Found {len(relevant)} relevant candidates")

        return results


def print_evaluation_summary(
    results: List[Tuple[Dict, PreFilterResult, Optional[CandidateEvaluation]]],
):
    """Print a formatted summary of evaluation results."""
    relevant = [(c, e) for c, pf, e in results if e and e.relevant]

    if not relevant:
        print("\nNo relevant candidates found.")
        return

    # Sort by score
    relevant.sort(key=lambda x: x[1].score, reverse=True)

    print("\n" + "=" * 70)
    print("TOP CANDIDATES BY GROK SCORE")
    print("=" * 70)

    for i, (candidate, evaluation) in enumerate(relevant[:20], 1):
        handle = candidate.get("handle", "unknown")
        followers = candidate.get("followers_count", 0)

        print(f"\n{i:2}. @{handle} (score: {evaluation.score:.2f})")
        print(f"    Followers: {followers:,}")
        print(f"    Role: {evaluation.recommended_role}")
        print(f"    Skills: {', '.join(evaluation.detected_skills[:5])}")
        print(f"    Reasoning: {evaluation.reasoning}")
        if evaluation.exceptional_work:
            print(f"    Exceptional work: {evaluation.exceptional_work}")
        if evaluation.red_flags:
            print(f"    Red flags: {', '.join(evaluation.red_flags)}")

    print("\n" + "=" * 70)
