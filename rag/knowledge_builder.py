"""Offline knowledge base builder.

Fetches Wikipedia articles via REST API, chunks them hierarchically,
builds FAISS (dense) and BM25 (sparse) indices.

Run once to produce index files on disk.
"""

import json
import pickle
import random
import time
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
import requests
from rank_bm25 import BM25Okapi


def _jitter(base: float, ratio: float = 0.3) -> float:
    """Add random jitter to a delay. e.g. _jitter(6.0) → 4.2~7.8"""
    return base * (1 + random.uniform(-ratio, ratio))


SEED_TOPICS = {
    # === Plant & Seed Health ===
    "Plant and Seed Health": [
        "Plant_disease", "Seed_treatment", "Germination", "Plant_pathology",
        "Phytopathology", "Plant_virus", "Seed", "Sprouting",
        "Disease_resistance_in_plants", "Plant_nursery",
        "Grafting", "Pruning", "Pollination", "Seed_bank",
        "Plant_breeding", "Hybrid_(biology)",
        "Dormancy", "Stratification_(seeds)", "Scarification_(botany)",
        "Transplanting", "Propagation", "Cutting_(plant)",
        "Plant_nutrition", "Chlorosis", "Wilting",
        "Abscission", "Senescence", "Photoperiodism",
    ],
    # === Crop Management ===
    "Crop Management": [
        "Crop_rotation", "Tillage", "Cover_crop", "Companion_planting",
        "Harvest", "Intercropping", "Monoculture", "Polyculture",
        "No-till_farming", "Strip_tillage", "Contour_ploughing",
        "Double_cropping", "Relay_cropping", "Crop_yield",
        "Agriculture", "Sustainable_agriculture",
        "Precision_agriculture", "Organic_farming",
        "Regenerative_agriculture", "Agroforestry",
        "Raised-bed_gardening", "Succession_planting",
        "Crop_calendar", "Hardiness_zone",
        "Planting_depth", "Row_spacing", "Seed_rate",
        "Thinning", "Hilling", "Staking",
    ],
    # === Crop Inputs ===
    "Crop Inputs": [
        "Fertilizer", "Pesticide", "Herbicide", "Insecticide", "Fungicide",
        "Organic_fertilizer", "Slow-release_fertilizer",
        "Biopesticide", "Neem", "Pyrethrum",
        "Manure", "Compost", "Bone_meal",
        "Foliar_feed", "Soil_amendment",
        "Superphosphate", "Ammonium_nitrate", "Urea",
        "Potash", "Limestone", "Gypsum",
        "Seaweed_fertilizer", "Fish_emulsion",
        "Rhizobium_inoculant", "Mycorrhizal_inoculant",
    ],
    # === Post-Harvest ===
    "Abiotic Harvest": [
        "Post-harvest", "Food_preservation", "Grain_storage",
        "Cold_chain", "Food_spoilage", "Silage",
        "Drying_(food)", "Canning", "Fermentation_in_food_processing",
        "Modified_atmosphere", "Food_irradiation",
        "Root_cellar", "Hay", "Straw",
        "Ripening", "Bruise", "Shelf_life",
        "Vacuum_packing", "Pickling", "Dehydration",
    ],
    # === Soil ===
    "Abiotic Soil": [
        "Soil", "Soil_health", "Soil_pH", "Soil_fertility",
        "Soil_organic_matter", "Soil_structure",
        "Soil_moisture", "Soil_test", "Soil_horizon",
        "Sandy_loam", "Clay_soil", "Loam",
        "Soil_compaction", "Soil_conservation", "Soil_erosion",
        "Vermicompost", "Mycorrhiza", "Rhizobium",
        "Soil_microbiology", "Soil_food_web",
        "Soil_amendment", "Soil_contamination",
        "Soil_profile", "Soil_taxonomy",
        "Chernozem", "Podzol", "Laterite",
        "Soil_aeration", "Soil_warming",
        "Drainage", "Water_table",
    ],
    # === Weather & Water ===
    "Abiotic Weather": [
        "Drought", "Frost", "Agricultural_meteorology",
        "Irrigation", "Drip_irrigation", "Sprinkler_irrigation",
        "Flood_irrigation", "Water_use_in_agriculture",
        "Greenhouse", "Cold_frame", "Row_cover",
        "Windbreak", "Mulching", "Shade_cloth",
        "Climate_change_and_agriculture",
        "Hail", "Wind_damage", "Flooding",
        "Evapotranspiration", "Water_stress",
        "Microclimate", "Frost_protection",
        "Rainwater_harvesting", "Water_conservation",
        "Deficit_irrigation", "Fertigation",
    ],
    # === Cover Crops ===
    "Cover Crop": [
        "Cover_crop", "Green_manure", "Nitrogen_fixation",
        "Legume", "Nitrogen_cycling",
        "Crimson_clover", "Hairy_vetch", "Winter_rye",
        "Buckwheat", "Radish_(plant)", "Mustard_plant",
        "Crop_residue", "Living_mulch",
        "Alfalfa", "Red_clover", "White_clover",
        "Sweet_clover", "Cowpea", "Peanut",
        "Vetch", "Field_pea", "Sunn_hemp",
    ],
    # === Plant Diseases (detailed) ===
    "Biotic Diseases": [
        "Fungal_infection", "Bacterial_wilt", "Mildew",
        "Rust_(fungus)", "Botrytis_cinerea",
        "Fusarium", "Pythium", "Phytophthora",
        "Powdery_mildew", "Downy_mildew",
        "Root_rot", "Leaf_spot", "Anthracnose",
        "Verticillium_wilt", "Fire_blight",
        "Clubroot", "Damping_off", "Canker_(plant_disease)",
        "Scab_(plant_disease)", "Smut_(fungus)",
        "Sclerotinia", "Alternaria", "Cercospora",
        "Bacterial_spot", "Bacterial_speck",
        "Tobacco_mosaic_virus", "Cucumber_mosaic_virus",
        "Tomato_spotted_wilt_virus", "Barley_yellow_dwarf_virus",
        "Nematode", "Root-knot_nematode",
        "Crown_gall", "Gall", "Witches'_broom",
    ],
    # === Insect Pests (detailed) ===
    "Biotic Insects": [
        "Agricultural_pest", "Integrated_pest_management",
        "Beneficial_insect", "Pollinator", "Biological_pest_control",
        "Aphid", "Whitefly", "Thrips", "Spider_mite",
        "Colorado_potato_beetle", "Corn_borer",
        "Armyworm", "Cutworm", "Grasshopper",
        "Ladybug", "Lacewing", "Parasitoid_wasp",
        "Japanese_beetle", "Squash_bug", "Cucumber_beetle",
        "Flea_beetle", "Cabbage_looper", "Imported_cabbageworm",
        "Tomato_hornworm", "Corn_earworm",
        "Stink_bug", "Plant_bug", "Leafhopper",
        "Wireworm", "Root_maggot", "Seed_corn_maggot",
        "Codling_moth", "Apple_maggot", "Plum_curculio",
        "Scale_insect", "Mealybug", "Psyllid",
        "Nematode_(plant_parasitic)", "Slug", "Snail",
    ],
    # === Weeds (detailed) ===
    "Biotic Weeds": [
        "Weed", "Weed_control", "Allelopathy",
        "Herbicide_resistance",
        "Crabgrass", "Pigweed", "Lamb's-quarters",
        "Bindweed", "Dandelion", "Nutsedge",
        "Mechanical_weed_control", "Flame_weeding",
        "Foxtail_(grass)", "Chickweed", "Purslane",
        "Nightshade", "Horseweed", "Ragweed",
        "Thistle", "Dock_(plant)", "Quackgrass",
        "Johnsongrass", "Bermuda_grass",
        "Palmer_amaranth", "Waterhemp",
        "Integrated_weed_management", "Cover_crop#Weed_suppression",
    ],
    # === Common Crops (expanded) ===
    "Common Crops - Grains": [
        "Corn", "Wheat", "Rice", "Soybean", "Sorghum",
        "Barley", "Oat", "Rye", "Millet",
        "Triticale", "Quinoa", "Buckwheat",
        "Spelt", "Emmer", "Einkorn",
    ],
    "Common Crops - Vegetables": [
        "Tomato", "Potato", "Lettuce", "Pepper",
        "Squash", "Cucumber", "Bean", "Onion",
        "Carrot", "Broccoli", "Cabbage", "Cauliflower",
        "Spinach", "Kale", "Celery", "Asparagus",
        "Pea", "Beet", "Radish", "Turnip",
        "Eggplant", "Zucchini", "Pumpkin", "Melon",
        "Watermelon", "Sweet_corn", "Artichoke",
        "Rhubarb", "Horseradish", "Rutabaga",
    ],
    "Common Crops - Fruits": [
        "Strawberry", "Blueberry", "Apple", "Grape",
        "Peach", "Pear", "Cherry", "Plum",
        "Raspberry", "Blackberry", "Cranberry",
        "Orange", "Lemon", "Lime", "Grapefruit",
        "Avocado", "Mango", "Pineapple", "Banana",
        "Fig", "Pomegranate", "Kiwi_fruit",
        "Olive", "Date_palm", "Coconut",
        "Persimmon", "Papaya", "Guava",
    ],
    "Common Crops - Industrial": [
        "Cotton", "Tobacco", "Sunflower", "Canola",
        "Sugarcane", "Sugar_beet", "Hemp",
        "Flax", "Jute", "Sisal",
        "Rubber_tree", "Coffee", "Cocoa_bean",
        "Tea", "Hop", "Lavender",
    ],
    # === Soil Chemistry ===
    "Soil Chemistry": [
        "Nitrogen", "Phosphorus", "Potassium",
        "Calcium_in_biochemistry", "Magnesium",
        "Micronutrient", "Soil_acidification",
        "Liming", "Cation-exchange_capacity",
        "Buffering_capacity", "Nutrient_deficiency",
        "Zinc_deficiency_(plant_disorder)",
        "Iron_deficiency_(plant_disorder)",
        "Manganese_deficiency_(plant_disorder)",
        "Boron_deficiency_(plant_disorder)",
        "Copper_deficiency_(plant_disorder)",
        "Sulfur_deficiency_(plant_disorder)",
        "Nutrient_toxicity", "Fertilizer_burn",
        "Soil_salinity", "Sodicity",
    ],
    # === Farming Systems ===
    "Farming Systems": [
        "Permaculture", "Agroecology",
        "Hydroponics", "Aeroponics",
        "Aquaponics", "Vertical_farming",
        "Crop-livestock", "Silvopasture",
        "Shade-grown_coffee", "Biodynamic_farming",
        "Community-supported_agriculture",
        "Market_garden", "Kitchen_garden",
        "Plantation", "Ranch", "Farm",
    ],
}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "AgriQA/1.0 (agricultural research; non-commercial dataset building)"
})
# Connection pool settings to avoid stale connections
adapter = requests.adapters.HTTPAdapter(
    pool_connections=2,
    pool_maxsize=2,
    max_retries=0,  # We handle retries manually
)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)

# Keywords for title-based link filtering (depth expansion)
AG_KEYWORDS = [
    "crop", "plant", "soil", "farm", "seed", "grain", "fruit", "vegetable",
    "fertiliz", "pesticid", "herbicid", "irrigat", "harvest", "cultivar",
    "livestock", "cattle", "poultry", "dairy", "wheat", "rice", "corn",
    "maize", "soybean", "cotton", "tobacco", "sugarcane", "potato",
    "tomato", "lettuce", "pepper", "bean", "pea", "carrot", "onion",
    "apple", "grape", "citrus", "berry", "nut ", "almond", "walnut",
    "disease", "pest", "weed", "fungus", "nematode", "insect",
    "compost", "manure", "mulch", "tillage", "plough", "plow",
    "agricult", "horticult", "agronom", "silvicult",
    "greenhouse", "nursery", "orchard", "vineyard", "pasture",
    "organic", "sustainable", "regenerative", "permaculture",
    "drought", "flood", "frost", "climate",
    "bacteria", "virus", "pathogen", "symbiosis", "mycorrhiz",
    "pollinator", "bee", "butterfly", "bird",
    "garden", "landscape", "turf", "lawn",
    "food", "nutrition", "vitamin", "mineral",
    "ethanol", "biodiesel", "biofuel", "biomass",
]

# Stricter keywords for content-based relevance check
AG_CONTENT_KEYWORDS = [
    "crop", "plant", "soil", "farm", "seed", "grain", "agriculture",
    "harvest", "cultivar", "fertiliz", "pesticid", "irrigat", "tillage",
    "livestock", "cattle", "poultry", "dairy", "wheat", "rice", "corn",
    "soybean", "cotton", "disease resistance", "pest management",
    "weed control", "compost", "manure", "organic farming",
    "greenhouse", "orchard", "vineyard", "pasture", "nursery",
    "horticult", "agronom", "planting", "growing season",
    "nutrient deficiency", "soil pH", "soil fertility",
    "crop rotation", "cover crop", "intercropping",
    "food production", "food security", "food processing",
]


def _is_agriculturally_relevant(content: str, min_matches: int = 3) -> bool:
    """Check if article content is agriculturally relevant.

    Counts keyword occurrences in content. Requires min_matches to accept.
    """
    content_lower = content.lower()
    matches = sum(1 for kw in AG_CONTENT_KEYWORDS if kw in content_lower)
    return matches >= min_matches


# Wikipedia categories to collect agricultural articles
AGRICULTURAL_CATEGORIES = [
    # Crops & plants
    "Category:Crops", "Category:Cereals", "Category:Legumes",
    "Category:Root_vegetables", "Category:Leaf_vegetables",
    "Category:Fruit", "Category:Nuts", "Category:Spices",
    "Category:Herbs", "Category:Oilseeds", "Category:Fiber_crops",
    "Category:Forage_crops", "Category:Cover_crops",
    "Category:Cultivated_plants", "Category:Plant_varieties",
    # Crop subcategories
    "Category:Wheat", "Category:Rice", "Category:Maize",
    "Category:Potato_cultivars", "Category:Tomato_cultivars",
    "Category:Soybean", "Category:Cotton",
    # Diseases & pests
    "Category:Plant_diseases", "Category:Fungal_plant_diseases",
    "Category:Bacterial_plant_diseases", "Category:Viral_plant_diseases",
    "Category:Agricultural_pests", "Category:Insect_pests",
    "Category:Weeds", "Category:Plant_pathogens",
    # Soil & nutrients
    "Category:Soil", "Category:Soil_science", "Category:Fertilizers",
    "Category:Soil_amendments", "Category:Plant_nutrition",
    # Farming practices
    "Category:Agriculture", "Category:Farming",
    "Category:Irrigation", "Category:Tillage",
    "Category:Organic_farming", "Category:Sustainable_agriculture",
    "Category:Precision_agriculture", "Category:Agroforestry",
    "Category:Greenhouses", "Category:Hydroponics",
    # Post-harvest & food
    "Category:Food_preservation", "Category:Grain_storage",
    "Category:Food_processing", "Category:Agricultural_machinery",
    # Pest management
    "Category:Pesticides", "Category:Herbicides", "Category:Fungicides",
    "Category:Insecticides", "Category:Biological_pest_control",
    "Category:Integrated_pest_management",
    # Livestock (relevant to mixed farming)
    "Category:Livestock", "Category:Dairy_farming",
    "Category:Poultry_farming", "Category:Animal_feed",
    # Agricultural science
    "Category:Agronomy", "Category:Horticulture",
    "Category:Plant_breeding", "Category:Plant_science",
    "Category:Agricultural_science",
]


def _is_agricultural_category(cat_name: str) -> bool:
    """Check if a subcategory name is agriculturally relevant."""
    name_lower = cat_name.lower()
    # Exclude clearly non-agricultural categories
    exclude_keywords = [
        "fictional", "game", "video", "film", "music", "book",
        "character", "sample return", "space", "lunar", "martian",
        "earthworks", "engineering", "geotechnical",
        "by country", "by continent", "by region",
        "templates", "stubs", "lists",
    ]
    if any(kw in name_lower for kw in exclude_keywords):
        return False
    # Include if it has agricultural keywords
    include_keywords = [
        "crop", "plant", "soil", "farm", "agri", "pest", "disease",
        "fertiliz", "irrigat", "seed", "grain", "cereal", "vegetable",
        "fruit", "nut", "legume", "root", "leaf", "herb", "spice",
        "fiber", "forage", "cover", "livestock", "dairy", "poultry",
        "compost", "manure", "organic", "tillage", "harvest",
        "cultivar", "variety", "breed", "hybrid",
        "fungus", "bacteria", "virus", "nematode", "weed", "insect",
        "sand", "clay", "loam", "sediment", "contamination", "improver",
    ]
    return any(kw in name_lower for kw in include_keywords)


def fetch_titles_by_category(
    categories: Optional[List[str]] = None,
    max_per_category: int = 500,
    include_subcategories: bool = True,
    checkpoint_path: Optional[str] = None,
) -> List[str]:
    """Fetch article titles from Wikipedia agricultural categories.

    Uses MediaWiki Category API (separate rate limits from REST content API).
    Supports checkpoint/resume: saves progress after each category.

    Args:
        categories: List of Category:... names. If None, uses AGRICULTURAL_CATEGORIES.
        max_per_category: Max articles per category.
        include_subcategories: If True, also fetch from subcategories (1 level).
        checkpoint_path: If provided, save/load checkpoint to this path.

    Returns:
        Deduplicated list of Wikipedia article titles.
    """
    if categories is None:
        categories = AGRICULTURAL_CATEGORIES

    all_titles = set()
    visited_cats = set()

    # Resume from checkpoint
    if checkpoint_path:
        cp_file = Path(checkpoint_path)
        if cp_file.exists():
            try:
                with open(cp_file, "r", encoding="utf-8") as f:
                    cp = json.load(f)
                all_titles = set(cp.get("titles", []))
                visited_cats = set(cp.get("visited_categories", []))
                print(f"  Resumed from checkpoint: {len(all_titles)} titles, {len(visited_cats)} categories done")
            except Exception as e:
                print(f"  Could not load checkpoint: {e}")

    def _save_checkpoint():
        if checkpoint_path:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({
                    "titles": sorted(all_titles),
                    "visited_categories": sorted(visited_cats),
                }, f, ensure_ascii=False, indent=2)

    def _fetch_category(cat: str) -> tuple:
        """Fetch article titles from one category.

        Returns (titles, api_failed): api_failed is True if the request
        itself failed (network/403/429), False if it succeeded but returned
        no results (empty category).
        """
        titles = []
        cmcontinue = None
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": cat,
                "cmlimit": str(min(max_per_category, 500)),
                "cmtype": "page",
                "format": "json",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue

            data = _api_get(params)
            if "query" not in data:
                # Empty response could mean blocked (403) or actual API failure
                return titles, True

            pages = data["query"].get("categorymembers", [])
            for p in pages:
                titles.append(p["title"])

            if "continue" in data and len(titles) < max_per_category:
                cmcontinue = data["continue"]["cmcontinue"]
                time.sleep(_jitter(6.0))
            else:
                break

        return titles, False

    def _fetch_subcategories(cat: str) -> List[str]:
        """Fetch subcategory names from one category."""
        data = _api_get({
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat,
            "cmlimit": "500",
            "cmtype": "subcat",
            "format": "json",
        })
        if "query" not in data:
            return []
        return [p["title"] for p in data["query"].get("categorymembers", [])]

    api_failures = 0
    for cat in categories:
        if cat in visited_cats:
            continue
        visited_cats.add(cat)

        # If Wikipedia is blocking us, wait before retrying
        if api_failures >= 3:
            cooldown = _jitter(600)
            print(f"  {api_failures} consecutive API failures. Cooling down {cooldown:.0f}s...")
            time.sleep(cooldown)
            api_failures = 0

        # Fetch articles from this category
        titles, failed = _fetch_category(cat)
        if failed:
            api_failures += 1
        else:
            api_failures = 0
        all_titles.update(titles)
        print(f"  {cat}: {len(titles)} articles")
        time.sleep(_jitter(6.0))

        # Fetch subcategories
        if include_subcategories:
            subcats = _fetch_subcategories(cat)
            time.sleep(_jitter(6.0))
            for subcat in subcats:
                if subcat in visited_cats:
                    continue
                visited_cats.add(subcat)  # mark visited even if irrelevant
                if not _is_agricultural_category(subcat):
                    continue
                sub_titles, sub_failed = _fetch_category(subcat)
                if sub_failed:
                    api_failures += 1
                else:
                    api_failures = 0
                all_titles.update(sub_titles)
                print(f"    {subcat}: {len(sub_titles)} articles")
                time.sleep(_jitter(6.0))

        _save_checkpoint()

    result = sorted(all_titles)
    print(f"  Total unique titles: {len(result)}")
    return result


def _api_get(params: dict, retries: int = 3) -> dict:
    """Make a MediaWiki API request with retry and 429 backoff."""
    params["format"] = "json"
    for attempt in range(retries):
        try:
            r = SESSION.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                timeout=30,
            )
            if r.status_code == 429:
                wait = _jitter(180 * (attempt + 1))
                print(f"  429 rate limited. Waiting {wait:.0f}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "connection reset" in err_str or "connection aborted" in err_str
            if is_rate_limit:
                wait = _jitter(180 * (attempt + 1))
                print(f"  Rate limited (connection reset/429). Waiting {wait:.0f}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
            elif attempt < retries - 1:
                wait = _jitter(45 * (attempt + 1))
                print(f"  Request error: {e}. Retrying in {wait:.0f}s (attempt {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                print(f"  API error after {retries} attempts: {e}")
                return {}


def _fetch_page_content(title: str) -> Optional[Dict]:
    """Fetch a single Wikipedia page via REST API (not rate-limited).

    Uses /api/rest_v1/page/html/ endpoint which has separate rate limits
    from the MediaWiki API.

    Returns dict with title, content, sections, url, or None if not found.
    """
    import re
    from html.parser import HTMLParser

    url_title = title.replace(" ", "_")
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{url_title}"

    html = None
    for attempt in range(3):
        try:
            r = SESSION.get(api_url, timeout=30)
            if r.status_code == 200:
                html = r.text
                break
            elif r.status_code == 404:
                return None
            elif r.status_code == 429:
                wait = _jitter(180 * (attempt + 1))
                print(f"  REST 429 for {title}. Waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            else:
                time.sleep(_jitter(5.0))
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "connection reset" in err_str or "connection aborted" in err_str
            if is_rate_limit:
                wait = _jitter(180 * (attempt + 1))
                print(f"  Rate limited for {title}. Waiting {wait:.0f}s...")
                time.sleep(wait)
            elif attempt < 2:
                wait = _jitter(45 * (attempt + 1))
                print(f"  Request error for {title}: {e}. Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                try:
                    r = SESSION.get(api_url, timeout=30, verify=False)
                    if r.status_code == 200:
                        html = r.text
                except Exception:
                    pass

    if not html or len(html) < 500:
        return None

    if not html or len(html) < 500:
        return None

    # Parse HTML to extract sections
    class WikiHTMLParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.sections = []
            self.current_section = None
            self.current_text = []
            self.in_heading = False
            self.heading_level = 0
            self.skip = False
            self.skip_tags = {'script', 'style', 'sup', 'table'}
            self.tag_stack = []

        def handle_starttag(self, tag, attrs):
            self.tag_stack.append(tag)
            if tag in ('h2', 'h3'):
                self.in_heading = True
                self.heading_level = int(tag[1])
                # Save previous section
                text = ' '.join(''.join(self.current_text).split())
                if self.current_section and text:
                    self.sections.append({
                        "title": self.current_section,
                        "text": text,
                    })
                self.current_text = []
            elif tag in self.skip_tags:
                self.skip = True

        def handle_endtag(self, tag):
            if self.tag_stack and self.tag_stack[-1] == tag:
                self.tag_stack.pop()
            if tag in ('h2', 'h3'):
                self.in_heading = False
            elif tag in self.skip_tags:
                self.skip = False

        def handle_data(self, data):
            if self.in_heading:
                self.current_section = data.strip()
            elif not self.skip:
                self.current_text.append(data)

        def finish(self):
            text = ' '.join(''.join(self.current_text).split())
            if self.current_section and text:
                self.sections.append({
                    "title": self.current_section,
                    "text": text,
                })

    parser = WikiHTMLParser()
    try:
        parser.feed(html)
        parser.finish()
    except Exception:
        return None

    if not parser.sections:
        # Fallback: strip all tags
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) < 200:
            return None
        return {
            "title": title,
            "content": text,
            "sections": [{"title": "Main", "text": text}],
            "url": f"https://en.wikipedia.org/wiki/{url_title}",
        }

    # First section before any heading is the intro
    full_content = "\n\n".join(s["text"] for s in parser.sections)

    # Check for intro (content before first h2)
    # The parser starts with current_section=None, so first section text
    # goes into current_text until first heading. We need to handle this.
    # Re-parse: collect intro text separately
    intro_text = ""
    heading_sections = []
    for s in parser.sections:
        # Sections with common intro names or first unnamed
        if s["title"] in ("Introduction", "Overview", None):
            intro_text = s["text"]
        else:
            heading_sections.append(s)

    # Build final sections list
    final_sections = []
    if intro_text:
        final_sections.append({"title": "Introduction", "text": intro_text})
    final_sections.extend(heading_sections)

    if not final_sections:
        final_sections = parser.sections

    full_content = "\n\n".join(s["text"] for s in final_sections)

    return {
        "title": title,
        "content": full_content,
        "sections": final_sections,
        "url": f"https://en.wikipedia.org/wiki/{url_title}",
    }


def _clean_wikitext(text: str) -> str:
    """Remove wikitext markup, leaving clean prose."""
    import re
    # Remove templates {{ }}
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    # Remove references <ref>...</ref> and <ref .../>
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/?>", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove file/image links [[File:...]]
    text = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    # Convert [[Link|text]] to text, [[Link]] to Link
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # Remove external links [url text]
    text = re.sub(r"\[https?://[^\s]+\s*([^\]]*)\]", r"\1", text)
    # Remove bold/italic markup
    text = re.sub(r"'{2,}", "", text)
    # Remove remaining markup
    text = re.sub(r"__[A-Z]+__", "", text)
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def _get_links(title: str, limit: int = 50) -> List[str]:
    """Get internal links from a Wikipedia page via REST API HTML.

    Extracts href links from the HTML, avoiding MediaWiki API rate limits.
    """
    import re
    from urllib.parse import unquote

    url_title = title.replace(" ", "_")
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{url_title}"

    html = None
    for attempt in range(3):
        try:
            r = SESSION.get(api_url, timeout=30)
            if r.status_code == 200:
                html = r.text
                break
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)

    if not html:
        return []

    # Extract internal wiki links from HTML
    links = re.findall(r'href=["\'](?:\./|/wiki/)([^"\']+)', html)

    seen = set()
    result = []
    for link in links:
        # Skip special pages, sections, files
        if any(link.startswith(prefix) for prefix in [
            'Special:', 'File:', 'Category:', 'Template:',
            'Help:', 'Wikipedia:', 'Talk:', 'User:',
            'Portal:', '#', 'cite_note', 'mw-',
        ]):
            continue
        decoded = unquote(link).replace('_', ' ')
        if decoded not in seen and decoded != title:
            seen.add(decoded)
            result.append(decoded)
            if len(result) >= limit:
                break

    return result


def fetch_wikipedia_articles(
    topics: List[str],
    depth: int = 1,
    max_articles: int = 3000,
    save_path: Optional[str] = None,
    resume: bool = True,
) -> List[Dict]:
    """Fetch Wikipedia articles for given topics, optionally following links.

    Args:
        topics: List of Wikipedia page titles.
        depth: How many levels of linked articles to follow (0 = only seed topics).
        max_articles: Maximum total articles to fetch.
        save_path: If provided, save articles incrementally to this path.
        resume: If True and save_path exists, load existing articles first.

    Returns:
        List of dicts: [{title, content, sections, url}]
    """
    visited = set()
    articles = []

    # Resume from existing file
    if resume and save_path:
        save_file = Path(save_path)
        if save_file.exists():
            try:
                with open(save_file, "r", encoding="utf-8") as f:
                    articles = json.load(f)
                visited = {a["title"] for a in articles}
                print(f"  Resumed {len(articles)} articles from {save_path}")
            except Exception as e:
                print(f"  Could not resume: {e}")

    def _save_incremental():
        if save_path and articles:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)

    def _fetch(title: str) -> Optional[Dict]:
        title = title.strip()
        if not title or title in visited:
            return None
        visited.add(title)
        article = _fetch_page_content(title)
        if article:
            articles.append(article)
            print(f"  + {title} ({len(article['content'])} chars) [{len(articles)} total]")
        else:
            print(f"  x {title}")
        time.sleep(_jitter(5.0))
        return article

    # Fetch seed topics
    seed_count_before = len(articles)
    api_failures = 0
    for i, topic in enumerate(topics):
        if len(articles) >= max_articles:
            break
        prev_count = len(articles)
        article = _fetch(topic)
        # Only count actual API/network failures, not 404s (article not found)
        if article is None and topic in visited:
            # _fetch_page_content returned None — could be 404 or network error
            # We don't count this as an API failure since 404 is normal
            pass
        # If Wikipedia is blocking us (connection errors pile up), cool down
        # This is detected by the _api_get retry logic, not by missing articles
        # Incremental save every 50 articles
        if len(articles) % 50 == 0 and len(articles) > seed_count_before:
            _save_incremental()
            print(f"  [checkpoint] Saved {len(articles)} articles")

    _save_incremental()
    print(f"  Seed phase complete: {len(articles)} articles total.")

    # Follow links to depth with content-based filtering
    if depth > 0:
        for d in range(depth):
            if len(articles) >= max_articles:
                break
            current_titles = [a["title"] for a in articles]
            new_count = 0

            for title in current_titles:
                if len(articles) >= max_articles:
                    break
                links = _get_links(title)
                for link in links:
                    if len(articles) >= max_articles:
                        break
                    link_lower = link.lower()
                    # Title-based pre-filter
                    if not any(kw in link_lower for kw in AG_KEYWORDS):
                        continue
                    article = _fetch(link)
                    if not article:
                        continue
                    # Content-based relevance check
                    if not _is_agriculturally_relevant(article["content"]):
                        # Remove irrelevant article
                        articles.pop()
                        visited.discard(link)
                        continue
                    new_count += 1
                    # Incremental save every 50 new articles
                    if new_count % 50 == 0:
                        _save_incremental()
                        print(f"  [checkpoint] Depth {d+1}: saved {len(articles)} articles")

            _save_incremental()
            print(f"  Depth {d+1}: added {new_count} articles. Total: {len(articles)}")

    _save_incremental()
    print(f"  Total articles fetched: {len(articles)}")
    return articles


def hierarchical_chunk(
    articles: List[Dict],
    min_tokens: int = 100,
    max_tokens: int = 400,
) -> List[Dict]:
    """Chunk articles hierarchically: Article -> Section -> Passage.

    Args:
        articles: Output from fetch_wikipedia_articles.
        min_tokens: Minimum tokens per passage (short passages merge forward).
        max_tokens: Maximum tokens per passage.

    Returns:
        List of passage dicts: [{id, text, article_title, section_title, passage_index}]
    """
    passages = []
    passage_id = 0

    for article in articles:
        sections = article.get("sections", [])
        if not sections:
            sections = [{"title": "Main", "text": article["content"]}]

        for section in sections:
            section_text = section["text"].strip()
            if not section_text:
                continue

            # Split by paragraphs
            paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]

            # Merge short paragraphs
            merged = []
            buffer = ""
            for para in paragraphs:
                token_count = len(para.split())
                if buffer and len(buffer.split()) + token_count > max_tokens:
                    merged.append(buffer)
                    buffer = para
                else:
                    buffer = buffer + " " + para if buffer else para
            if buffer:
                merged.append(buffer)

            # Split long passages
            final_passages = []
            for text in merged:
                if len(text.split()) <= max_tokens:
                    final_passages.append(text)
                else:
                    sentences = text.replace(". ", ".\n").split("\n")
                    chunk = ""
                    for sent in sentences:
                        if chunk and len(chunk.split()) + len(sent.split()) > max_tokens:
                            final_passages.append(chunk)
                            chunk = sent
                        else:
                            chunk = chunk + " " + sent if chunk else sent
                    if chunk:
                        final_passages.append(chunk)

            for idx, text in enumerate(final_passages):
                if len(text.split()) >= min_tokens:
                    passages.append({
                        "id": f"passage_{passage_id}",
                        "text": text,
                        "article_title": article["title"],
                        "section_title": section["title"],
                        "passage_index": idx,
                        "url": article.get("url", ""),
                    })
                    passage_id += 1

    return passages


def build_faiss_index(
    passages: List[Dict],
    embedding_model=None,
    dim: int = 1024,
    output_path: Optional[str] = None,
) -> faiss.Index:
    """Embed passages and build FAISS IndexFlatIP.

    Args:
        passages: Output from hierarchical_chunk.
        embedding_model: EmbeddingClient or SentenceTransformer instance (must have .encode()).
        dim: Embedding dimension.
        output_path: If provided, save index to this path.

    Returns:
        FAISS index.
    """
    if embedding_model is None:
        raise ValueError("embedding_model is required")

    texts = [p["text"] for p in passages]
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    if output_path:
        faiss.write_index(index, output_path)

    return index


def build_bm25_index(passages: List[Dict], output_path: Optional[str] = None):
    """Build BM25 index from passages.

    Args:
        passages: Output from hierarchical_chunk.
        output_path: If provided, save model to this path via pickle.

    Returns:
        BM25Okapi model.
    """
    tokenized = [p["text"].lower().split() for p in passages]
    bm25 = BM25Okapi(tokenized)

    if output_path:
        with open(output_path, "wb") as f:
            pickle.dump(bm25, f)

    return bm25


def save_metadata(passages: List[Dict], output_path: str):
    """Save passage metadata for retrieval results."""
    metadata = []
    for p in passages:
        metadata.append({
            "id": p["id"],
            "article_title": p["article_title"],
            "section_title": p["section_title"],
            "passage_index": p["passage_index"],
            "url": p.get("url", ""),
            "text": p["text"],
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_passages_jsonl(passages: List[Dict], output_path: str):
    """Save passages as JSONL for reproducibility."""
    with open(output_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def save_articles(articles: List[Dict], output_path: str):
    """Save raw articles as JSON for reference."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def build_knowledge_base(
    topics: Optional[List[str]] = None,
    output_dir: str = "data",
    embedding_model=None,
    dim: int = 1024,
    depth: int = 0,
    max_articles: int = 5000,
    use_categories: bool = True,
):
    """Orchestrate the full knowledge base build.

    Args:
        topics: List of Wikipedia page titles. If None, uses categories or SEED_TOPICS.
        output_dir: Base output directory.
        embedding_model_name: Embedding model.
        dim: Embedding dimension.
        depth: Link-following depth (0 = no link following).
        max_articles: Maximum articles to fetch.
        use_categories: If True, discover titles via Wikipedia Category API.
    """
    if topics is None:
        if use_categories:
            print("Discovering titles via Wikipedia categories...")
            topics = fetch_titles_by_category()
        else:
            topics = []
            for category_topics in SEED_TOPICS.values():
                topics.extend(category_topics)
            seen = set()
            unique_topics = []
            for t in topics:
                if t not in seen:
                    seen.add(t)
                    unique_topics.append(t)
            topics = unique_topics

    out = Path(output_dir)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    (out / "index").mkdir(parents=True, exist_ok=True)

    raw_path = str(out / "raw" / "articles.json")
    print(f"Fetching {len(topics)} articles (depth={depth}, max={max_articles})...")
    articles = fetch_wikipedia_articles(
        topics, depth=depth, max_articles=max_articles,
        save_path=raw_path, resume=True,
    )
    print(f"Fetched {len(articles)} articles.")

    # Calculate total size
    total_chars = sum(len(a.get("content", "")) for a in articles)
    total_mb = total_chars / (1024 * 1024)
    print(f"Total content size: {total_mb:.1f} MB")

    print("Saving raw articles...")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    save_articles(articles, raw_path)

    print("Chunking articles...")
    passages = hierarchical_chunk(articles)
    print(f"Created {len(passages)} passages.")

    print("Saving passages...")
    save_passages_jsonl(passages, str(out / "chunks" / "passages.jsonl"))

    print("Building FAISS index...")
    build_faiss_index(passages, embedding_model, dim, str(out / "index" / "faiss.index"))

    print("Building BM25 index...")
    build_bm25_index(passages, str(out / "index" / "bm25.pkl"))

    print("Saving metadata...")
    save_metadata(passages, str(out / "index" / "metadata.json"))

    print("Knowledge base build complete.")
    return passages
