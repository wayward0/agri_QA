#!/usr/bin/env python3
"""Fetch agricultural articles from Wikipedia multistream dump files.

Downloads multistream XML dump files, decompresses and parses them
to extract articles matching agricultural keywords.
"""

import bz2
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

# --- Config ---
DUMPS_BASE = "https://dumps.wikimedia.org/enwiki/latest"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
DUMPS_CACHE = RAW_DIR / "dumps"
ARTICLES_PATH = RAW_DIR / "articles.json"

# Smaller multistream files to download (file_name, page_range_description)
DUMP_FILES = [
    "enwiki-latest-pages-articles-multistream1.xml-p1p41242.bz2",
    "enwiki-latest-pages-articles-multistream2.xml-p41243p151573.bz2",
    "enwiki-latest-pages-articles-multistream3.xml-p151574p311329.bz2",
    "enwiki-latest-pages-articles-multistream4.xml-p311330p558391.bz2",
    "enwiki-latest-pages-articles-multistream5.xml-p558392p958045.bz2",
    "enwiki-latest-pages-articles-multistream6.xml-p958046p1483661.bz2",
    "enwiki-latest-pages-articles-multistream7.xml-p1483662p2134111.bz2",
    "enwiki-latest-pages-articles-multistream8.xml-p2134112p2936260.bz2",
    "enwiki-latest-pages-articles-multistream9.xml-p2936261p4045402.bz2",
]

# Agricultural keywords for title matching
AG_TITLE_KEYWORDS = [
    "crop", "plant", "soil", "farm", "seed", "grain", "fruit", "vegetable",
    "fertiliz", "pesticid", "herbicid", "irrigat", "harvest", "cultivar",
    "livestock", "cattle", "poultry", "dairy", "wheat", "rice", "corn",
    "maize", "soybean", "cotton", "tobacco", "sugarcane", "potato",
    "tomato", "lettuce", "pepper", "bean", "pea", "carrot", "onion",
    "apple", "grape", "citrus", "berry", "almond", "walnut", "pecan",
    "sunflower", "canola", "rapeseed", "flax", "hemp", "jute",
    "rubber", "coffee", "tea", "cocoa", "cinnamon", "vanilla",
    "ginger", "turmeric", "garlic", "basil", "oregano", "mint",
    "coconut", "palm", "agriculture", "agronom", "horticultur",
    "forestry", "silvicultur", "pasture", "rangeland", "orchard",
    "vineyard", "nursery", "greenhouse", "hydroponic", "aquaponic",
    "compost", "mulch", "tillage", "plow", "plough",
    "seedling", "transplant", "germinat", "propagat", "graft",
    "pruning", "pollinat", "disease", "pest", "fungus", "bacteria",
    "virus", "nematode", "weed", "insecticide", "fungicide",
    "organic", "sustainable", "regenerative", "permaculture",
    "crop rotation", "cover crop", "intercrop", "monoculture",
    "irrigation", "drip", "sprinkler", "nitrogen", "phosphorus",
    "potassium", "manure", "biosolid", "humus", "mycorrhiza",
    "rhizobium", "aquaculture", "fisher", "mariculture",
    "beekeeping", "apiary", "pollinator", "honeybee",
    "food science", "food safety", "food processing", "preservation",
    "canning", "ferment", "dehydration", "nutrition", "vitamin",
    "protein", "agribusiness", "agroforestry", "agroecolog",
    "cereal", "legume", "tuber", "brassica", "solanum", "cucurbit",
    "allium", "drought", "erosion", "salinity", "soil science",
    "pedology", "clay", "silt", "sand", "loam", "peat",
    "tractor", "combine", "thresher", "harvester",
    "wheat", "barley", "oat", "rye", "sorghum", "millet",
    "quinoa", "amaranth", "buckwheat", "teff", "spelt",
    "chickpea", "lentil", "cowpea", "pigeon pea", "mung bean",
    "peanut", "groundnut", "cashew", "pistachio", "macadamia",
    "avocado", "mango", "papaya", "banana", "pineapple",
    "strawberry", "blueberry", "raspberry", "blackberry",
    "peach", "plum", "cherry", "apricot", "nectarine",
    "watermelon", "cantaloupe", "honeydew", "squash",
    "broccoli", "cauliflower", "cabbage", "kale", "spinach",
    "celery", "asparagus", "artichoke", "radish", "turnip",
    "beet", "parsnip", "rutabaga", "yam", "cassava", "taro",
    "chili", "jalapeño", "habanero",
    "basil", "cilantro", "parsley", "dill", "chive",
    "rosemary", "thyme", "sage", "lavender", "chamomile",
    "USDA", "FAO", "CIMMYT", "IRRI",
]

AG_CONTENT_KEYWORDS = [
    "agriculture", "farming", "crop", "cultivation", "harvest",
    "plant growth", "soil", "irrigation", "fertilizer", "pesticide",
    "livestock", "poultry", "dairy", "grain", "seed", "seedling",
    "organic farming", "sustainable agriculture", "food production",
]


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress display."""
    if dest.exists():
        # Check if it's complete by trying to read the end
        size = dest.stat().st_size
        if size > 10_000_000:  # > 10MB probably complete
            print(f"  Already downloaded: {dest.name} ({size / 1e6:.0f} MB)")
            return True

    print(f"  Downloading {desc or dest.name}...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    start = time.time()

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded % (50 * 1024 * 1024) == 0:
                elapsed = time.time() - start
                speed = downloaded / elapsed / 1e6
                pct = downloaded / total * 100 if total else 0
                print(f"    {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%) - {speed:.1f} MB/s")

    elapsed = time.time() - start
    print(f"    Done in {elapsed:.0f}s ({downloaded / 1e6:.0f} MB)")
    return True


def clean_wikitext(text: str) -> str:
    """Convert wikitext to clean plain text."""
    # Remove comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Remove ref tags and content
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/?\s*>', '', text)

    # Remove other HTML-like tags
    text = re.sub(r'<gallery[^>]*>.*?</gallery>', '', text, flags=re.DOTALL)
    text = re.sub(r'<math[^>]*>.*?</math>', '', text, flags=re.DOTALL)
    text = re.sub(r'<chem[^>]*>.*?</chem>', '', text, flags=re.DOTALL)
    text = re.sub(r'<source[^>]*>.*?</source>', '', text, flags=re.DOTALL)
    text = re.sub(r'<code[^>]*>.*?</code>', '', text, flags=re.DOTALL)
    text = re.sub(r'<pre[^>]*>.*?</pre>', '', text, flags=re.DOTALL)
    text = re.sub(r'<includeonly>.*?</includeonly>', '', text, flags=re.DOTALL)
    text = re.sub(r'<noinclude>.*?</noinclude>', '', text, flags=re.DOTALL)
    text = re.sub(r'<onlyinclude>.*?</onlyinclude>', '', text, flags=re.DOTALL)
    text = re.sub(r'<imagemap[^>]*>.*?</imagemap>', '', text, flags=re.DOTALL)
    text = re.sub(r'<timeline[^>]*>.*?</timeline>', '', text, flags=re.DOTALL)
    text = re.sub(r'<hiero[^>]*>.*?</hiero>', '', text, flags=re.DOTALL)
    text = re.sub(r'<mapframe[^>]*>.*?</mapframe>', '', text, flags=re.DOTALL)
    text = re.sub(r'<maplink[^>]*>.*?</maplink>', '', text, flags=re.DOTALL)

    # Remove templates {{ }} (handle nesting)
    while '{{' in text:
        old = text
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
        if text == old:
            break

    # Remove tables {| |}
    while '{|' in text:
        old = text
        text = re.sub(r'\{\|[^{}]*\|\}', '', text, flags=re.DOTALL)
        if text == old:
            break

    # Remove [[Category:...]], [[File:...]], [[Image:...]]
    text = re.sub(r'\[\[(?:Category|File|Image):[^\]]*\]\]', '', text, flags=re.IGNORECASE)

    # Convert [[link|display]] to display
    text = re.sub(r'\[\[[^\]|]*\|([^\]]*)\]\]', r'\1', text)
    # Convert [[link]] to link
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)

    # Remove external links [url text]
    text = re.sub(r'\[https?://[^\s\]]+\s*([^\]]*)\]', r'\1', text)

    # Remove bold/italic markup
    text = re.sub(r"'{2,3}", '', text)

    # Remove headings (keep text)
    text = re.sub(r'={2,6}\s*(.*?)\s*={2,6}', r'\1', text)

    # Remove HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)

    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'  +', ' ', text)

    return text.strip()


def extract_sections(content: str) -> List[Dict]:
    """Extract sections from article content."""
    sections = []
    lines = content.split('\n')
    current_section = {"title": "Introduction", "content": []}

    for line in lines:
        heading_match = re.match(r'^(={2,6})\s*(.*?)\s*\1\s*$', line)
        if heading_match:
            if current_section["content"]:
                text = '\n'.join(current_section["content"]).strip()
                if len(text) > 50:
                    sections.append({
                        "title": current_section["title"],
                        "content": text,
                    })
            current_section = {
                "title": heading_match.group(2).strip(),
                "content": [],
            }
        else:
            current_section["content"].append(line)

    if current_section["content"]:
        text = '\n'.join(current_section["content"]).strip()
        if len(text) > 50:
            sections.append({
                "title": current_section["title"],
                "content": text,
            })

    return sections


def is_agricultural_title(title: str) -> bool:
    """Check if a title matches agricultural keywords."""
    title_lower = title.lower()
    return any(kw in title_lower for kw in AG_TITLE_KEYWORDS)


def parse_dump_streaming(bz2_path: Path, target_titles: Optional[Set[str]] = None) -> List[Dict]:
    """Parse a multistream bz2 dump file using streaming decompression.

    Extracts articles matching agricultural keywords from their titles.
    """
    articles = []
    total_pages = 0
    matched_pages = 0

    print(f"  Parsing {bz2_path.name}...")

    with open(bz2_path, "rb") as f:
        decompressor = bz2.BZ2Decompressor()
        buffer = ""

        while True:
            chunk = f.read(1024 * 1024)  # Read 1MB at a time
            if not chunk:
                break

            try:
                text = decompressor.decompress(chunk).decode("utf-8", errors="replace")
            except EOFError:
                break
            except Exception:
                continue

            buffer += text

            # Process complete <page>...</page> elements
            while "<page>" in buffer and "</page>" in buffer:
                page_start = buffer.find("<page>")
                page_end = buffer.find("</page>") + len("</page>")

                if page_start >= 0 and page_end > page_start:
                    page_xml = buffer[page_start:page_end]
                    buffer = buffer[page_end:]

                    # Parse the page
                    article = parse_page(page_xml)
                    total_pages += 1

                    if article:
                        articles.append(article)
                        matched_pages += 1

                    if total_pages % 10000 == 0:
                        print(f"    Processed {total_pages} pages, {matched_pages} agricultural articles, {len(articles)} total")
                else:
                    break

    print(f"  Finished {bz2_path.name}: {total_pages} pages processed, {len(articles)} agricultural articles")
    return articles


def parse_page(page_xml: str) -> Optional[Dict]:
    """Parse a single <page> XML element into an article dict."""
    try:
        # Extract title
        title_match = re.search(r'<title>(.*?)</title>', page_xml)
        if not title_match:
            return None
        title = title_match.group(1).strip()

        # Check namespace (only main articles)
        ns_match = re.search(r'<ns>(\d+)</ns>', page_xml)
        if ns_match and int(ns_match.group(1)) != 0:
            return None

        # Check if agricultural
        if not is_agricultural_title(title):
            return None

        # Extract page ID
        id_match = re.search(r'<id>(\d+)</id>', page_xml)
        page_id = int(id_match.group(1)) if id_match else 0

        # Extract text content
        text_match = re.search(r'<text[^>]*>(.*?)</text>', page_xml, re.DOTALL)
        if not text_match:
            return None

        raw_text = text_match.group(1)

        # Decode XML entities
        raw_text = raw_text.replace('&lt;', '<').replace('&gt;', '>')
        raw_text = raw_text.replace('&amp;', '&').replace('&quot;', '"')
        raw_text = raw_text.replace('&apos;', "'")

        # Skip redirects and disambiguation
        if raw_text.startswith('#REDIRECT') or raw_text.startswith('#redirect'):
            return None
        if '{{disambig' in raw_text.lower()[:500]:
            return None
        if '{{redirect' in raw_text.lower()[:200]:
            return None

        # Clean to plain text
        content = clean_wikitext(raw_text)

        # Skip stubs
        if len(content) < 200:
            return None

        # Additional content relevance check
        content_lower = content.lower()
        if not any(kw in content_lower for kw in AG_CONTENT_KEYWORDS):
            # Title matched but content isn't agricultural - skip
            # (unless it's a very common agricultural term)
            common_ag = ["agriculture", "crop", "plant", "soil", "farm", "livestock"]
            if not any(kw in title.lower() for kw in common_ag):
                return None

        sections = extract_sections(content)

        return {
            "title": title,
            "content": content,
            "sections": sections,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "page_id": page_id,
        }

    except Exception:
        return None


def main():
    print("=" * 60)
    print("Wikipedia Agricultural Knowledge Base Builder (Dumps)")
    print("=" * 60)
    print()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DUMPS_CACHE.mkdir(parents=True, exist_ok=True)

    # Step 1: Download dump files
    print("Step 1: Downloading multistream dump files...")
    downloaded_files = []

    for filename in DUMP_FILES:
        url = f"{DUMPS_BASE}/{filename}"
        dest = DUMPS_CACHE / filename

        if download_file(url, dest, filename):
            downloaded_files.append(dest)
        else:
            print(f"  Failed to download {filename}")

    print(f"\nDownloaded {len(downloaded_files)} files.")
    print()

    # Step 2: Parse dump files for agricultural articles
    print("Step 2: Parsing dump files for agricultural articles...")
    all_articles = []
    seen_titles = set()

    # Resume if possible
    if ARTICLES_PATH.exists():
        try:
            with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
                all_articles = json.load(f)
            seen_titles = {a["title"] for a in all_articles}
            print(f"  Resumed {len(all_articles)} articles from cache")
        except Exception:
            pass

    for dump_path in downloaded_files:
        articles = parse_dump_streaming(dump_path)
        new_count = 0
        for article in articles:
            if article["title"] not in seen_titles:
                all_articles.append(article)
                seen_titles.add(article["title"])
                new_count += 1

        print(f"  Added {new_count} new articles from {dump_path.name}")

        # Incremental save
        with open(ARTICLES_PATH, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)

        total_chars = sum(len(a.get("content", "")) for a in all_articles)
        print(f"  Total so far: {len(all_articles)} articles, {total_chars / 1e6:.1f} MB")

    total_chars = sum(len(a.get("content", "")) for a in all_articles)
    total_mb = total_chars / (1024 * 1024)
    print(f"\nStep 2 complete: {len(all_articles)} articles, {total_mb:.1f} MB")

    if len(all_articles) == 0:
        print("No articles found! Check keyword matching.")
        return

    # Step 3: Build indices
    print("\nStep 3: Building FAISS + BM25 indices...")
    from rag.knowledge_builder import (
        hierarchical_chunk, build_faiss_index, build_bm25_index,
        save_metadata, save_passages_jsonl,
    )

    passages = hierarchical_chunk(all_articles)
    print(f"  Created {len(passages)} passages.")

    (DATA_DIR / "chunks").mkdir(parents=True, exist_ok=True)
    save_passages_jsonl(passages, str(DATA_DIR / "chunks" / "passages.jsonl"))

    (DATA_DIR / "index").mkdir(parents=True, exist_ok=True)
    print("  Building FAISS index...")
    build_faiss_index(passages, output_path=str(DATA_DIR / "index" / "faiss.index"))
    print("  Building BM25 index...")
    build_bm25_index(passages, output_path=str(DATA_DIR / "index" / "bm25.pkl"))
    save_metadata(passages, str(DATA_DIR / "index" / "metadata.json"))

    print(f"\n{'=' * 60}")
    print(f"Done! Knowledge base ready.")
    print(f"  Articles: {len(all_articles)}")
    print(f"  Passages: {len(passages)}")
    print(f"  Content: {total_mb:.1f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
