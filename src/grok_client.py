"""
Grok API Client for Candidate Evaluation
Uses xAI's Grok API to evaluate candidates against hiring criteria
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class FastScreenResult:
    """Result from fast bio/pinned tweet screening."""

    pass_filter: bool
    reason: str
    potential_role: str  # research, engineering, infrastructure, unlikely


@dataclass
class CandidateEvaluation:
    """Structured evaluation result from Grok."""

    relevant: bool
    score: float  # 0.0 - 1.0
    reasoning: str
    detected_skills: List[str]
    red_flags: List[str]
    recommended_role: str  # research, engineering, infrastructure, none
    standout_tweets: List[int]  # Indices of impressive tweets
    exceptional_work: str  # Summary of their best work


class GrokClient:
    """Client for xAI Grok API with caching support."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize Grok client.

        Args:
            cache_dir: Directory to cache evaluation results
        """
        self.api_key = os.environ.get("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI_API_KEY not found in environment")

        self.base_url = "https://api.x.ai/v1"
        self.model = "grok-4-1-fast-reasoning"  # Full evaluation model (with reasoning)
        self.fast_model = (
            "grok-4-1-fast-non-reasoning"  # Fast screening model (no reasoning)
        )

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Separate cache for fast screens
            self.fast_cache_dir = self.cache_dir / "fast_screen"
            self.fast_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, handle: str, tweets_hash: str) -> str:
        """Generate cache key for an evaluation."""
        return f"eval_{handle}_{tweets_hash[:16]}"

    def _load_from_cache(self, cache_key: str) -> Optional[CandidateEvaluation]:
        """Load cached evaluation if available."""
        if not self.cache_dir:
            return None

        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                data = json.load(f)
                print(f"  [cache] Loaded evaluation for {cache_key}")
                return CandidateEvaluation(**data)
        return None

    def _save_to_cache(self, cache_key: str, evaluation: CandidateEvaluation):
        """Save evaluation to cache."""
        if not self.cache_dir:
            return

        cache_path = self.cache_dir / f"{cache_key}.json"
        with open(cache_path, "w") as f:
            json.dump(asdict(evaluation), f, indent=2)

    def _call_api(
        self, messages: List[Dict[str, str]], model: Optional[str] = None
    ) -> str:
        """Make API call to Grok."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": 0.3,  # Lower for more consistent evaluations
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            print(f"  [error] Grok API error: {response.status_code} - {response.text}")
            return ""

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def fast_screen(
        self,
        handle: str,
        bio: str,
        pinned_tweet: Optional[str] = None,
        location: Optional[str] = None,
    ) -> FastScreenResult:
        """
        Fast screening using minimal input (bio + pinned tweet).
        Uses grok-2-fast for speed and cost efficiency.

        Args:
            handle: X username
            bio: Profile bio
            pinned_tweet: Optional pinned tweet text
            location: Optional location string

        Returns:
            FastScreenResult with pass/fail and reason
        """
        # Check cache first
        cache_key = f"fast_{handle}"
        if self.cache_dir:
            cache_path = self.fast_cache_dir / f"{cache_key}.json"
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    return FastScreenResult(**data)

        # Build minimal prompt
        prompt = self._build_fast_screen_prompt(handle, bio, pinned_tweet, location)

        # Call fast model
        response = self._call_api(
            [
                {"role": "system", "content": self._get_fast_screen_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            model=self.fast_model,
        )

        if not response:
            return FastScreenResult(
                pass_filter=False, reason="api_error", potential_role="unlikely"
            )

        # Parse response
        result = self._parse_fast_screen_response(response)

        # Cache result
        if self.cache_dir:
            cache_path = self.fast_cache_dir / f"{cache_key}.json"
            with open(cache_path, "w") as f:
                json.dump(asdict(result), f)

        return result

    def _get_fast_screen_system_prompt(self) -> str:
        """System prompt for fast screening."""
        return """You are a fast pre-filter for xAI recruiting. Your job is to quickly decide if a candidate is worth deeper evaluation.

PASS if bio/content suggests:
- Software engineer, ML engineer, researcher, or systems engineer
- Works with: Python, Rust, C++, CUDA, ML, LLMs, distributed systems, compilers
- Has built real projects or works at strong tech companies
- PhD/research background in relevant fields (CS, ML, physics, math)

REJECT (these are CRITICAL - never pass these):
1. xAI/X EMPLOYEES: Anyone who works at, worked at, or is affiliated with xAI, X (Twitter), X Corp, or Grok. Look for:
   - Bio mentions: "xAI", "x.ai", "@xAI", "at X", "X Corp", "Twitter", "Grok team", "working on Grok"
   - Handle patterns suggesting xAI affiliation
   - Any mention of being an xAI/X employee, engineer, researcher, or team member

2. ORGANIZATIONS/COMPANIES: Not individual people. Look for:
   - Company accounts, brand accounts, product accounts
   - Bio reads like a company description ("We are...", "Our mission...", official accounts)
   - News outlets, research labs (as institutions, not individuals), DAOs, foundations
   - Handles that are clearly brand names

3. NON-TECHNICAL:
   - Crypto/NFT/forex focus without technical depth
   - Marketing, growth hacking, content creation focus
   - No technical indicators at all

Be fast and decisive. When in doubt about technical fit, PASS. But ALWAYS REJECT xAI/X employees and organizations.
Respond with JSON only."""

    def _build_fast_screen_prompt(
        self,
        handle: str,
        bio: str,
        pinned_tweet: Optional[str],
        location: Optional[str],
    ) -> str:
        """Build prompt for fast screening."""
        parts = [f"@{handle}"]
        if bio:
            parts.append(f"Bio: {bio[:300]}")
        if location:
            parts.append(f"Location: {location}")
        if pinned_tweet:
            parts.append(f"Pinned: {pinned_tweet[:400]}")

        return f"""{chr(10).join(parts)}

Quick filter:
1. Is this an xAI/X employee? (REJECT if yes - reason: "xai_employee" or "x_employee")
2. Is this an organization/company account? (REJECT if yes - reason: "organization")
3. Is this person potentially an xAI engineering candidate? (PASS if technical)

Respond with JSON only:
{{"pass": true/false, "reason": "one line reason", "role": "research"|"engineering"|"infrastructure"|"unlikely"}}"""

    def _parse_fast_screen_response(self, response: str) -> FastScreenResult:
        """Parse fast screen response."""
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()

            data = json.loads(response)
            return FastScreenResult(
                pass_filter=bool(data.get("pass", False)),
                reason=str(data.get("reason", "")),
                potential_role=str(data.get("role", "unlikely")),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            # Default to pass if parsing fails - let full evaluator decide
            return FastScreenResult(
                pass_filter=True,
                reason="parse_error_default_pass",
                potential_role="engineering",
            )

    def evaluate_candidate(
        self,
        handle: str,
        bio: str,
        tweets: List[str],
        followers: int,
        criteria: str,
    ) -> Optional[CandidateEvaluation]:
        """
        Evaluate a candidate using Grok.

        Args:
            handle: X username
            bio: Profile bio
            tweets: List of recent tweets
            followers: Follower count
            criteria: Evaluation criteria from criteria.yaml

        Returns:
            CandidateEvaluation or None if evaluation fails
        """
        # Check cache first
        tweets_hash = hashlib.md5("".join(tweets).encode()).hexdigest()
        cache_key = self._get_cache_key(handle, tweets_hash)

        cached = self._load_from_cache(cache_key)
        if cached:
            return cached

        # Build prompt
        prompt = self._build_evaluation_prompt(handle, bio, tweets, followers, criteria)

        # Call Grok
        print(f"  [grok] Evaluating @{handle}...")
        response = self._call_api(
            [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt},
            ]
        )

        if not response:
            return None

        # Parse response
        evaluation = self._parse_response(response)
        if evaluation:
            self._save_to_cache(cache_key, evaluation)

        return evaluation

    def _get_system_prompt(self) -> str:
        """System prompt for Grok evaluator."""
        return """You are an elite technical recruiter for xAI, evaluating candidates based on their X (Twitter) presence.

Your job is to identify exceptional engineers - the kind Elon Musk would personally want to meet.

Key principles:
1. "Evidence of exceptional ability" is the only non-negotiable
2. Look for people who BUILD things, not just talk about them
3. Deep technical understanding > buzzwords
4. Mission obsession > career optimization
5. The 800-follower engineer who ships > the 100k influencer who posts takes

Be rigorous and selective. Most people are NOT xAI-caliber.
Only mark relevant=true if there's clear evidence of exceptional technical ability.

Always respond with valid JSON only, no markdown formatting."""

    def _build_evaluation_prompt(
        self,
        handle: str,
        bio: str,
        tweets: List[str],
        followers: int,
        criteria: str,
    ) -> str:
        """Build the evaluation prompt."""
        # Number the tweets for reference
        numbered_tweets = "\n".join(
            f"[{i + 1}] {tweet[:500]}" for i, tweet in enumerate(tweets[:30])
        )

        return f"""## xAI Hiring Criteria
{criteria}

## Candidate Profile
Handle: @{handle}
Bio: {bio}
Followers: {followers:,}

## Their Recent Tweets (numbered for reference)
{numbered_tweets}

## Evaluation Task

Analyze this candidate for xAI engineering roles.

Consider:
1. EXCEPTIONAL WORK: Have they built something impressive? Shipped products? Open source?
2. TECHNICAL DEPTH: Real understanding with code/math/experiments, or just buzzwords?
3. MISSION FIT: Interested in LLMs, training, inference, reasoning, evals?
4. OWNERSHIP: Do they take initiative? Solve hard problems themselves?

Remember: The key xAI interview question is "What exceptional work have you done?"
Can you answer this based on their tweets?

Respond with JSON only (no markdown):
{{
  "relevant": true/false,
  "score": 0.0-1.0,
  "reasoning": "2-3 sentences explaining your assessment",
  "detected_skills": ["skill1", "skill2", ...],
  "red_flags": ["flag1", ...] or [],
  "recommended_role": "research" | "engineering" | "infrastructure" | "none",
  "standout_tweets": [1, 5, 12],
  "exceptional_work": "Brief description of their best work, or empty string if none evident"
}}"""

    def _parse_response(self, response: str) -> Optional[CandidateEvaluation]:
        """Parse Grok's JSON response into CandidateEvaluation."""
        try:
            # Clean up response (remove markdown if present)
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()

            data = json.loads(response)

            return CandidateEvaluation(
                relevant=bool(data.get("relevant", False)),
                score=float(data.get("score", 0.0)),
                reasoning=str(data.get("reasoning", "")),
                detected_skills=list(data.get("detected_skills", [])),
                red_flags=list(data.get("red_flags", [])),
                recommended_role=str(data.get("recommended_role", "none")),
                standout_tweets=list(data.get("standout_tweets", [])),
                exceptional_work=str(data.get("exceptional_work", "")),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  [error] Failed to parse Grok response: {e}")
            print(f"  [debug] Response was: {response[:500]}")
            return None


def load_criteria(criteria_path: str = "criteria.yaml") -> str:
    """Load and format criteria from YAML file."""
    import yaml

    with open(criteria_path, "r") as f:
        criteria = yaml.safe_load(f)

    # Format as string for the prompt
    parts = []

    parts.append(f"Company Context:\n{criteria.get('company_context', '')}")
    parts.append(f"\nKey Question: {criteria.get('key_question', '')}")

    if "evaluation_criteria" in criteria:
        parts.append("\nEvaluation Criteria:")
        for name, details in criteria["evaluation_criteria"].items():
            parts.append(f"\n{name.upper()} (weight: {details.get('weight', 0)}):")
            parts.append(f"  {details.get('description', '')}")

    if "tech_stack" in criteria:
        parts.append("\nRelevant Tech Stack:")
        for category, items in criteria["tech_stack"].items():
            parts.append(f"  {category}: {', '.join(items)}")

    if "red_flags" in criteria:
        parts.append(f"\nRed Flags: {', '.join(criteria['red_flags'])}")

    return "\n".join(parts)
