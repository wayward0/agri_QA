#!/usr/bin/env python3
"""Run expanded Wikipedia fetch to build ~300MB knowledge base.

Uses Wikipedia Category API to discover real article titles,
then fetches content via REST API.
"""

import sys
import time
sys.path.insert(0, ".")

from rag.knowledge_builder import (
    fetch_titles_by_category, fetch_wikipedia_articles, save_articles,
    AGRICULTURAL_CATEGORIES,
)
from pathlib import Path


def main():
    print("=" * 60)
    print("Agricultural Knowledge Base Builder")
    print("=" * 60)
    print()

    # Step 1: Discover article titles from categories
    print("Step 1: Discovering article titles from Wikipedia categories...")
    titles = fetch_titles_by_category(include_subcategories=True)
    print(f"Found {len(titles)} unique article titles.")
    print()

    # Step 2: Fetch article content
    raw_path = "data/raw/articles.json"
    print(f"Step 2: Fetching article content (max 5000)...")
    articles = fetch_wikipedia_articles(
        titles,
        depth=0,  # No link following — categories give us enough titles
        max_articles=5000,
        save_path=raw_path,
        resume=True,
    )

    # Final stats
    total_chars = sum(len(a.get("content", "")) for a in articles)
    total_mb = total_chars / (1024 * 1024)
    print(f"\nFinal: {len(articles)} articles, {total_mb:.1f} MB")

    # Save final
    save_articles(articles, raw_path)
    print(f"Saved to {raw_path}")


if __name__ == "__main__":
    main()
