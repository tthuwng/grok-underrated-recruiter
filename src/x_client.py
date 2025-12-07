"""
X API Client for Grok Recruiter
Handles OAuth1 authentication, rate limiting, and pagination
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from requests_oauthlib import OAuth1

load_dotenv()

BASE_URL = "https://api.twitter.com/2"


class XClient:
    """X API v2 client with rate limiting and caching support."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the X API client.

        Args:
            cache_dir: Optional directory to cache API responses
        """
        self.auth = OAuth1(
            os.environ["X_API_KEY"],
            os.environ["X_API_KEY_SECRET"],
            os.environ["X_ACCESS_TOKEN"],
            os.environ["X_ACCESS_TOKEN_SECRET"],
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given key."""
        # Sanitize the cache key for filesystem
        safe_key = cache_key.replace("/", "_").replace("?", "_").replace("&", "_")
        return self.cache_dir / f"{safe_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load response from cache if available."""
        if not self.cache_dir:
            return None
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            with open(cache_path, "r") as f:
                print(f"  [cache] Loading from {cache_path.name}")
                return json.load(f)
        return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]):
        """Save response to cache."""
        if not self.cache_dir:
            return
        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, "w") as f:
            json.dump(data, f)

    def _get(
        self,
        path: str,
        params: Dict[str, Any],
        cache_key: Optional[str] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Make a GET request to the X API with rate limit handling.

        Args:
            path: API endpoint path
            params: Query parameters
            cache_key: Optional key for caching
            max_retries: Max retries on rate limit

        Returns:
            API response as dict
        """
        # Try cache first
        if cache_key:
            cached = self._load_from_cache(cache_key)
            if cached:
                return cached

        url = f"{BASE_URL}{path}"

        for attempt in range(max_retries):
            try:
                r = requests.get(url, auth=self.auth, params=params, timeout=30)

                if r.status_code == 429:
                    # Rate limited - wait and retry
                    reset_time = int(r.headers.get("x-rate-limit-reset", 0))
                    sleep_for = max(reset_time - int(time.time()), 60)
                    print(
                        f"  [rate limit] Sleeping {sleep_for}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(sleep_for)
                    continue

                if r.status_code == 403:
                    print(
                        f"  [forbidden] Endpoint {path} not available (likely API tier restriction)"
                    )
                    return {"data": [], "meta": {}, "error": "forbidden"}

                if r.status_code == 404:
                    print(f"  [not found] Resource not found: {path}")
                    return {"data": [], "meta": {}, "error": "not_found"}

                r.raise_for_status()
                data = r.json()

                # Cache successful response
                if cache_key:
                    self._save_to_cache(cache_key, data)

                return data

            except requests.exceptions.Timeout:
                print("  [timeout] Request timed out, retrying...")
                time.sleep(5)
            except requests.exceptions.RequestException as e:
                print(f"  [error] Request failed: {e}")
                if attempt == max_retries - 1:
                    return {"data": [], "meta": {}, "error": str(e)}
                time.sleep(5)

        return {"data": [], "meta": {}, "error": "max_retries_exceeded"}

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user details by username.

        Args:
            username: X username (without @)

        Returns:
            User object or None
        """
        print(f"  Fetching user: @{username}")
        data = self._get(
            f"/users/by/username/{username}",
            {
                "user.fields": "public_metrics,created_at,description,location,verified,profile_image_url"
            },
            cache_key=f"user_{username}",
        )
        return data.get("data")

    def get_users_by_ids(self, user_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Batch fetch users by IDs (max 100 per request).

        Args:
            user_ids: List of user IDs

        Returns:
            List of user objects
        """
        if not user_ids:
            return []

        all_users = []
        # Process in batches of 100
        for i in range(0, len(user_ids), 100):
            batch = user_ids[i : i + 100]
            ids_str = ",".join(batch)
            print(f"  Hydrating {len(batch)} users (batch {i // 100 + 1})")

            data = self._get(
                "/users",
                {
                    "ids": ids_str,
                    "user.fields": "public_metrics,created_at,description,location,verified,profile_image_url",
                },
                cache_key=f"users_batch_{i}_{hash(ids_str)}",
            )
            all_users.extend(data.get("data", []))

        return all_users

    def get_user_liked_tweets(
        self, user_id: str, max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get tweets liked by a user.

        Args:
            user_id: X user ID
            max_results: Max tweets to fetch

        Returns:
            List of tweet objects
        """
        print(f"  Fetching liked tweets for user {user_id}")
        tweets: List[Dict[str, Any]] = []
        params = {
            "max_results": min(max_results, 100),
            "tweet.fields": "author_id,created_at,public_metrics,conversation_id,text",
            "expansions": "author_id",
            "user.fields": "public_metrics,description",
        }
        next_token = None
        pages = 0
        max_pages = (max_results // 100) + 1

        while pages < max_pages:
            if next_token:
                params["pagination_token"] = next_token

            data = self._get(
                f"/users/{user_id}/liked_tweets",
                params,
                cache_key=f"liked_{user_id}_page{pages}",
            )

            if "error" in data:
                break

            tweets.extend(data.get("data", []))

            meta = data.get("meta", {})
            next_token = meta.get("next_token")
            pages += 1

            if not next_token or len(tweets) >= max_results:
                break

        print(f"    Found {len(tweets)} liked tweets")
        return tweets[:max_results]

    def get_user_tweets(
        self, user_id: str, max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get tweets posted by a user.

        Args:
            user_id: X user ID
            max_results: Max tweets to fetch

        Returns:
            List of tweet objects
        """
        print(f"  Fetching tweets for user {user_id}")
        params = {
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics,conversation_id,text",
            "exclude": "replies,retweets",
        }

        data = self._get(
            f"/users/{user_id}/tweets",
            params,
            cache_key=f"tweets_{user_id}",
        )

        tweets = data.get("data", [])
        print(f"    Found {len(tweets)} tweets")
        return tweets[:max_results]

    def get_liking_users(
        self, tweet_id: str, max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get users who liked a specific tweet.

        Args:
            tweet_id: Tweet ID
            max_results: Max users to fetch

        Returns:
            List of user objects
        """
        data = self._get(
            f"/tweets/{tweet_id}/liking_users",
            {
                "max_results": min(max_results, 100),
                "user.fields": "public_metrics,created_at,description",
            },
            cache_key=f"likers_{tweet_id}",
        )
        return data.get("data", [])

    def get_retweeting_users(
        self, tweet_id: str, max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get users who retweeted a specific tweet.

        Args:
            tweet_id: Tweet ID
            max_results: Max users to fetch

        Returns:
            List of user objects
        """
        data = self._get(
            f"/tweets/{tweet_id}/retweeted_by",
            {
                "max_results": min(max_results, 100),
                "user.fields": "public_metrics,created_at,description",
            },
            cache_key=f"retweeters_{tweet_id}",
        )
        return data.get("data", [])

    def search_replies(
        self, conversation_id: str, max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search for replies in a conversation.

        Args:
            conversation_id: Conversation ID (usually the original tweet ID)
            max_results: Max replies to fetch

        Returns:
            List of tweet objects
        """
        query = f"conversation_id:{conversation_id}"
        params = {
            "query": query,
            "max_results": min(max_results, 100),
            "tweet.fields": "author_id,created_at,public_metrics,conversation_id,text",
        }

        data = self._get(
            "/tweets/search/recent",
            params,
            cache_key=f"replies_{conversation_id}",
        )
        return data.get("data", [])

    def get_user_following(
        self, user_id: str, max_results: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get accounts that a user follows (strongest signal - they chose to follow).

        Args:
            user_id: X user ID
            max_results: Max accounts to fetch (paginated)

        Returns:
            List of user objects
        """
        print(f"  Fetching following for user {user_id}")
        following: List[Dict[str, Any]] = []
        params = {
            "max_results": min(max_results, 1000),
            "user.fields": "public_metrics,created_at,description,location,verified,pinned_tweet_id",
        }
        next_token = None
        pages = 0
        max_pages = (max_results // 1000) + 1

        while pages < max_pages:
            if next_token:
                params["pagination_token"] = next_token

            data = self._get(
                f"/users/{user_id}/following",
                params,
                cache_key=f"following_{user_id}_page{pages}",
            )

            if "error" in data:
                break

            following.extend(data.get("data", []))

            meta = data.get("meta", {})
            next_token = meta.get("next_token")
            pages += 1

            if not next_token or len(following) >= max_results:
                break

        print(f"    Found {len(following)} accounts followed")
        return following[:max_results]

    def get_pinned_tweet(self, user_id: str, pinned_tweet_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a user's pinned tweet by ID.

        Args:
            user_id: X user ID (for cache key)
            pinned_tweet_id: Tweet ID of the pinned tweet

        Returns:
            Tweet object or None
        """
        if not pinned_tweet_id:
            return None

        data = self._get(
            f"/tweets/{pinned_tweet_id}",
            {
                "tweet.fields": "created_at,public_metrics,text,author_id",
            },
            cache_key=f"pinned_{user_id}_{pinned_tweet_id}",
        )
        return data.get("data")
