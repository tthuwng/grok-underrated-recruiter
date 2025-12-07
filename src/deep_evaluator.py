"""
Deep Evaluation Judge for Grok Recruiter
Uses xAI Search Tools (web_search + x_search) to deeply analyze candidates
"""

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from xai_sdk import Client
    from xai_sdk.chat import user
    from xai_sdk.tools import web_search, x_search

    XAI_SDK_AVAILABLE = True
except ImportError:
    XAI_SDK_AVAILABLE = False
    print("[warn] xai_sdk not installed. Run: pip install xai-sdk>=1.3.1")


@dataclass
class ScoreDetail:
    """Individual criterion score with evidence."""

    score: int  # 0-10
    evidence: str


@dataclass
class DeepEvaluation:
    """Complete deep evaluation result."""

    # Candidate info
    handle: str
    bio: str
    followers: int

    # Scores (0-10 scale)
    technical_depth: ScoreDetail
    project_evidence: ScoreDetail
    mission_alignment: ScoreDetail
    exceptional_ability: ScoreDetail
    communication: ScoreDetail

    # Composite score (0-100)
    final_score: float

    # Explanations
    summary: str
    strengths: List[str]
    concerns: List[str]
    recommended_role: str  # research/engineering/infrastructure/none

    # External links found
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    top_repos: Optional[List[str]] = None

    # Metadata
    citations: Optional[List[str]] = None
    raw_response: Optional[str] = None


class DeepEvaluator:
    """
    Deep evaluation using xAI Search Tools + Grok reasoning.

    Uses:
    - x_search: Find candidate's recent posts and discussions
    - web_search: Fetch GitHub and LinkedIn profiles
    - grok-4-1-fast-reasoning: Analyze and score on rubric
    """

    RUBRIC_WEIGHTS = {
        "technical_depth": 0.25,
        "project_evidence": 0.25,
        "mission_alignment": 0.20,
        "exceptional_ability": 0.20,
        "communication": 0.10,
    }

    def __init__(self, cache_dir: str = "data/enriched"):
        """Initialize the deep evaluator."""
        if not XAI_SDK_AVAILABLE:
            raise ImportError("xai_sdk required. Run: pip install xai-sdk>=1.3.1")

        self.api_key = os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY not found in environment")

        self.client = Client(api_key=self.api_key)
        self.model = "grok-4-1-fast-reasoning"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, handle: str) -> Path:
        """Get cache file path for a handle."""
        safe_handle = re.sub(r"[^a-zA-Z0-9_]", "_", handle.lower())
        return self.cache_dir / f"deep_{safe_handle}.json"

    def _load_cached(self, handle: str) -> Optional[DeepEvaluation]:
        """Load cached evaluation if exists."""
        cache_path = self._get_cache_path(handle)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                # Reconstruct dataclass
                return DeepEvaluation(
                    handle=data["handle"],
                    bio=data["bio"],
                    followers=data["followers"],
                    technical_depth=ScoreDetail(**data["technical_depth"]),
                    project_evidence=ScoreDetail(**data["project_evidence"]),
                    mission_alignment=ScoreDetail(**data["mission_alignment"]),
                    exceptional_ability=ScoreDetail(**data["exceptional_ability"]),
                    communication=ScoreDetail(**data["communication"]),
                    final_score=data["final_score"],
                    summary=data["summary"],
                    strengths=data["strengths"],
                    concerns=data["concerns"],
                    recommended_role=data["recommended_role"],
                    github_url=data.get("github_url"),
                    linkedin_url=data.get("linkedin_url"),
                    top_repos=data.get("top_repos"),
                    citations=data.get("citations"),
                )
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def _save_cache(self, evaluation: DeepEvaluation):
        """Save evaluation to cache."""
        cache_path = self._get_cache_path(evaluation.handle)

        # Convert to dict and handle non-JSON-serializable types
        data = asdict(evaluation)

        # Convert citations (RepeatedScalarContainer) to list
        if data.get("citations") is not None:
            data["citations"] = list(data["citations"])

        # Convert top_repos if needed
        if data.get("top_repos") is not None:
            data["top_repos"] = list(data["top_repos"])

        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

    def _build_prompt(self, handle: str, bio: str, followers: int) -> str:
        """Build the evaluation prompt."""
        return f"""You are evaluating a candidate for xAI engineering roles. Use your search tools to gather information, then score them on a rubric.

## Candidate
Handle: @{handle}
Bio: {bio}
Followers: {followers:,}

## Your Task
1. Use x_search to find their recent technical posts and discussions on X
2. Use web_search on github.com to find their GitHub profile and repositories
3. Use web_search on linkedin.com to find their professional background
4. Analyze all gathered information against the rubric below

## Scoring Rubric (0-10 for each)

### Technical Depth (25%)
- 10: Novel research contributions, deep systems knowledge, published papers
- 7: Strong understanding shown in posts, meaningful technical discussions
- 4: Basic technical knowledge, some good insights
- 1: Buzzwords only, no demonstrated depth

### Project Evidence (25%)
- 10: Popular open source projects (1k+ stars), shipped products at scale
- 7: Active GitHub with quality projects, meaningful contributions
- 4: Some side projects, learning in public
- 1: No visible project work

### Mission Alignment (20%)
- 10: Obsessed with AI/AGI/reasoning, posts frequently about frontier AI
- 7: Clearly interested in ML/AI, follows the space
- 4: Some AI interest among other topics
- 1: No relevant interest shown

### Exceptional Ability (20%)
- 10: Clear evidence of solving hard problems, unique insights, ownership
- 7: Good track record, competent engineer
- 4: Standard experience, nothing standout
- 1: No evidence of exceptional work

### Communication (10%)
- 10: Excellent technical writing, clear explanations, teaches well
- 7: Good communication, explains things clearly
- 4: Adequate communication
- 1: Hard to understand, poor explanations

## Response Format
After gathering information, respond with JSON only:
{{
  "technical_depth": {{"score": 0-10, "evidence": "specific examples found"}},
  "project_evidence": {{"score": 0-10, "evidence": "repos/projects found"}},
  "mission_alignment": {{"score": 0-10, "evidence": "relevant posts/interests"}},
  "exceptional_ability": {{"score": 0-10, "evidence": "hard problems solved"}},
  "communication": {{"score": 0-10, "evidence": "writing quality examples"}},
  "summary": "2-3 sentence overall assessment",
  "strengths": ["strength1", "strength2"],
  "concerns": ["concern1", "concern2"],
  "recommended_role": "research" | "engineering" | "infrastructure" | "none",
  "github_url": "url or null",
  "linkedin_url": "url or null",
  "top_repos": ["repo1", "repo2"]
}}"""

    def _parse_response(
        self, response_text: str, handle: str, bio: str, followers: int
    ) -> DeepEvaluation:
        """Parse JSON response from Grok."""
        # Find JSON in response
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response_text[:200]}")

        data = json.loads(json_match.group())

        # Calculate final score
        final_score = (
            sum(
                data[criterion]["score"] * weight
                for criterion, weight in self.RUBRIC_WEIGHTS.items()
            )
            * 10
        )  # Scale to 0-100

        return DeepEvaluation(
            handle=handle,
            bio=bio,
            followers=followers,
            technical_depth=ScoreDetail(
                score=data["technical_depth"]["score"],
                evidence=data["technical_depth"]["evidence"],
            ),
            project_evidence=ScoreDetail(
                score=data["project_evidence"]["score"],
                evidence=data["project_evidence"]["evidence"],
            ),
            mission_alignment=ScoreDetail(
                score=data["mission_alignment"]["score"],
                evidence=data["mission_alignment"]["evidence"],
            ),
            exceptional_ability=ScoreDetail(
                score=data["exceptional_ability"]["score"],
                evidence=data["exceptional_ability"]["evidence"],
            ),
            communication=ScoreDetail(
                score=data["communication"]["score"],
                evidence=data["communication"]["evidence"],
            ),
            final_score=final_score,
            summary=data.get("summary", ""),
            strengths=data.get("strengths", []),
            concerns=data.get("concerns", []),
            recommended_role=data.get("recommended_role", "none"),
            github_url=data.get("github_url"),
            linkedin_url=data.get("linkedin_url"),
            top_repos=data.get("top_repos"),
            raw_response=response_text,
        )

    def evaluate(
        self,
        handle: str,
        bio: str,
        followers: int,
        use_cache: bool = True,
        verbose: bool = True,
    ) -> DeepEvaluation:
        """
        Deep evaluate a candidate using xAI Search Tools.

        Args:
            handle: X handle (without @)
            bio: Profile bio
            followers: Follower count
            use_cache: Whether to use cached results
            verbose: Whether to print progress

        Returns:
            DeepEvaluation with scores and reasoning
        """
        # Check cache
        if use_cache:
            cached = self._load_cached(handle)
            if cached:
                if verbose:
                    print(f"  [cache] @{handle} - score: {cached.final_score:.1f}")
                return cached

        if verbose:
            print(f"  [eval] @{handle} - searching X, GitHub, LinkedIn...")

        # Create chat with search tools
        chat = self.client.chat.create(
            model=self.model,
            tools=[
                web_search(),  # For GitHub and LinkedIn
                x_search(),  # For X posts
            ],
        )

        # Build and send prompt
        prompt = self._build_prompt(handle, bio, followers)
        chat.append(user(prompt))

        # Stream response (to see tool calls)
        response_text = ""
        citations = []
        chunk_count = 0
        last_tool_call_time = None
        import time

        start_time = time.time()

        try:
            if verbose:
                print("    [debug] Starting stream...")

            # Use iterator with timeout detection
            stream_iterator = chat.stream()
            last_chunk_time = time.time()
            CHUNK_TIMEOUT = 120  # 2 minutes max between chunks

            while True:
                try:
                    # Check if we've been waiting too long
                    if time.time() - last_chunk_time > CHUNK_TIMEOUT:
                        raise TimeoutError(f"No response for {CHUNK_TIMEOUT}s - likely rate limited")

                    response, chunk = next(stream_iterator)
                    last_chunk_time = time.time()
                    chunk_count += 1
                    elapsed = time.time() - start_time

                    # Debug: show chunk info periodically
                    if verbose and chunk_count % 10 == 0:
                        print(f"    [debug] Chunk {chunk_count}, elapsed: {elapsed:.1f}s")

                    # Collect tool calls for verbose output
                    if chunk.tool_calls:
                        if verbose:
                            for tc in chunk.tool_calls:
                                print(
                                    f"    -> {tc.function.name}: {tc.function.arguments[:50]}..."
                                )

                    # Collect response content
                    if chunk.content:
                        response_text += chunk.content
                        if verbose and len(response_text) % 200 == 0:
                            print(
                                f"    [debug] Response length: {len(response_text)} chars"
                            )

                except StopIteration:
                    break

            if verbose:
                print(
                    f"    [debug] Stream complete. {chunk_count} chunks, {time.time() - start_time:.1f}s total"
                )

        except TimeoutError as timeout_error:
            if verbose:
                print(f"    [timeout] {timeout_error}")
            raise

        except Exception as stream_error:
            if verbose:
                print(
                    f"    [error] Stream failed after {chunk_count} chunks: {stream_error}"
                )
            raise

        # Get citations (convert from RepeatedScalarContainer to list)
        if response.citations:
            citations = list(response.citations)

        # Parse response
        try:
            evaluation = self._parse_response(response_text, handle, bio, followers)
            evaluation.citations = citations

            # Cache result
            self._save_cache(evaluation)

            if verbose:
                print(
                    f"  [done] @{handle} - score: {evaluation.final_score:.1f} ({evaluation.recommended_role})"
                )

            return evaluation

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            if verbose:
                print(f"  [error] @{handle} - failed to parse: {e}")
            # Return a failed evaluation
            return DeepEvaluation(
                handle=handle,
                bio=bio,
                followers=followers,
                technical_depth=ScoreDetail(score=0, evidence="parse_error"),
                project_evidence=ScoreDetail(score=0, evidence="parse_error"),
                mission_alignment=ScoreDetail(score=0, evidence="parse_error"),
                exceptional_ability=ScoreDetail(score=0, evidence="parse_error"),
                communication=ScoreDetail(score=0, evidence="parse_error"),
                final_score=0.0,
                summary=f"Evaluation failed: {e}",
                strengths=[],
                concerns=["evaluation_failed"],
                recommended_role="none",
                raw_response=response_text,
            )

    def evaluate_batch(
        self,
        candidates: List[Dict[str, Any]],
        max_candidates: int = 200,
        verbose: bool = True,
        delay_between: int = 30,  # Seconds between candidates to avoid rate limits
        max_retries: int = 3,
    ) -> List[DeepEvaluation]:
        """
        Evaluate a batch of candidates.

        Args:
            candidates: List of dicts with handle, bio, followers_count
            max_candidates: Maximum to evaluate
            verbose: Print progress
            delay_between: Seconds to wait between candidates (rate limit protection)
            max_retries: Max retries per candidate on failure

        Returns:
            List of DeepEvaluation results
        """
        import time

        results = []
        consecutive_failures = 0

        if verbose:
            print(
                f"\n[deep_eval] Evaluating {min(len(candidates), max_candidates)} candidates"
            )
            print(f"  Rate limit protection: {delay_between}s delay between candidates")
            print("=" * 60)

        for i, candidate in enumerate(candidates[:max_candidates]):
            handle = candidate.get("handle", "")
            bio = candidate.get("bio", "")
            followers = candidate.get("followers_count", 0)

            if verbose:
                print(f"\n[{i + 1}/{min(len(candidates), max_candidates)}] @{handle}")

            # Retry logic
            evaluation = None
            for retry in range(max_retries):
                try:
                    evaluation = self.evaluate(handle, bio, followers, verbose=verbose)
                    consecutive_failures = 0
                    break
                except Exception as e:
                    consecutive_failures += 1
                    if verbose:
                        print(f"  [retry {retry + 1}/{max_retries}] Failed: {e}")

                    # Exponential backoff
                    wait_time = delay_between * (2**retry)
                    if verbose:
                        print(f"  [wait] Sleeping {wait_time}s before retry...")
                    time.sleep(wait_time)

            if evaluation:
                results.append(evaluation)
            else:
                if verbose:
                    print(f"  [skip] Failed after {max_retries} retries")

            # Check for too many consecutive failures (likely rate limited)
            if consecutive_failures >= 3:
                if verbose:
                    print(
                        "\n[RATE LIMITED] 3 consecutive failures. Waiting 5 minutes..."
                    )
                time.sleep(300)  # 5 minute cooldown
                consecutive_failures = 0

            # Delay between candidates (skip if cached)
            if i < len(candidates[:max_candidates]) - 1:
                # Check if next candidate is cached
                next_handle = candidates[i + 1].get("handle", "")
                next_cached = self._load_cached(next_handle)

                if not next_cached and evaluation and evaluation.final_score > 0:
                    if verbose:
                        print(
                            f"  [rate limit] Waiting {delay_between}s before next candidate..."
                        )
                    time.sleep(delay_between)

        # Summary
        if verbose:
            print("\n" + "=" * 60)
            print("DEEP EVALUATION COMPLETE")
            print("=" * 60)

            avg_score = (
                sum(e.final_score for e in results) / len(results) if results else 0
            )
            high_scorers = [e for e in results if e.final_score >= 70]

            print(f"  Evaluated: {len(results)}")
            print(f"  Average score: {avg_score:.1f}")
            print(f"  High scorers (70+): {len(high_scorers)}")

            if high_scorers:
                print("\n  Top candidates:")
                for e in sorted(
                    high_scorers, key=lambda x: x.final_score, reverse=True
                )[:10]:
                    print(
                        f"    @{e.handle}: {e.final_score:.1f} ({e.recommended_role})"
                    )

        return results


def print_evaluation(evaluation: DeepEvaluation):
    """Pretty print a deep evaluation."""
    print(f"\n{'=' * 60}")
    print(f"@{evaluation.handle} (Score: {evaluation.final_score:.1f}/100)")
    print(f"{'=' * 60}")
    print(f"Bio: {evaluation.bio[:100]}...")
    print(f"Followers: {evaluation.followers:,}")
    print(f"Recommended Role: {evaluation.recommended_role}")
    print()

    print("Scores:")
    print(f"  Technical Depth:     {evaluation.technical_depth.score}/10")
    print(f"    {evaluation.technical_depth.evidence[:80]}...")
    print(f"  Project Evidence:    {evaluation.project_evidence.score}/10")
    print(f"    {evaluation.project_evidence.evidence[:80]}...")
    print(f"  Mission Alignment:   {evaluation.mission_alignment.score}/10")
    print(f"    {evaluation.mission_alignment.evidence[:80]}...")
    print(f"  Exceptional Ability: {evaluation.exceptional_ability.score}/10")
    print(f"    {evaluation.exceptional_ability.evidence[:80]}...")
    print(f"  Communication:       {evaluation.communication.score}/10")
    print(f"    {evaluation.communication.evidence[:80]}...")
    print()

    print(f"Summary: {evaluation.summary}")
    print()

    if evaluation.strengths:
        print("Strengths:")
        for s in evaluation.strengths:
            print(f"  + {s}")

    if evaluation.concerns:
        print("Concerns:")
        for c in evaluation.concerns:
            print(f"  - {c}")

    if evaluation.github_url:
        print(f"\nGitHub: {evaluation.github_url}")
    if evaluation.linkedin_url:
        print(f"LinkedIn: {evaluation.linkedin_url}")
    if evaluation.top_repos:
        print(f"Top repos: {', '.join(evaluation.top_repos[:3])}")


if __name__ == "__main__":
    # Test with a sample candidate
    from dotenv import load_dotenv

    load_dotenv()

    evaluator = DeepEvaluator()

    # Test evaluation
    test_candidate = {
        "handle": "karpathy",
        "bio": "AI @ Tesla, OpenAI. Building Andrej AI.",
        "followers_count": 900000,
    }

    result = evaluator.evaluate(
        handle=test_candidate["handle"],
        bio=test_candidate["bio"],
        followers=test_candidate["followers_count"],
        use_cache=False,
    )

    print_evaluation(result)
