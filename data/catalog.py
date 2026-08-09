"""Deterministic product catalog.

Transactional data references a comparatively small set of products, so the
catalog is built once and sampled from. This matters enormously for cost: with
2,000 distinct products you make 2,000 Bedrock embedding calls no matter whether
you generate 100k transactions or 10 million.
"""
import random

CATEGORIES = {
    "electronics": {
        "subs": ["headphones", "chargers", "smartwatches", "keyboards", "monitors"],
        "brands": ["Anker", "Soundcore", "Logitech", "Belkin", "Aukey", "Razer"],
        "origins": ["China", "Vietnam", "Taiwan", "South Korea"],
        "traits": ["noise cancelling", "fast charging", "wireless", "backlit",
                   "USB-C", "Bluetooth 5.3", "low latency", "ultra-portable"],
        "uses": ["daily commuting", "home office work", "long flights",
                 "gaming sessions", "travel and field work"],
        "price": (15, 320),
    },
    "kitchen": {
        "subs": ["cookware", "small appliances", "storage", "cutlery", "bakeware"],
        "brands": ["Tefal", "OXO", "Cuisinart", "Joseph Joseph", "Le Creuset"],
        "origins": ["France", "Italy", "China", "Germany", "Portugal"],
        "traits": ["non-stick", "dishwasher safe", "stainless steel", "cast iron",
                   "BPA free", "induction compatible", "heat resistant"],
        "uses": ["weeknight cooking", "batch meal prep", "baking", "slow braising",
                 "everyday family meals"],
        "price": (8, 240),
    },
    "outdoor": {
        "subs": ["camping", "hiking", "cycling", "fishing", "climbing"],
        "brands": ["Coleman", "Osprey", "Black Diamond", "MSR", "Vango"],
        "origins": ["Vietnam", "China", "Bangladesh", "Sri Lanka"],
        "traits": ["waterproof", "ripstop nylon", "packable", "ultralight",
                   "reinforced seams", "insulated", "quick-drying"],
        "uses": ["multi-day treks", "weekend camping", "wet weather", "cold conditions",
                 "commuting by bike"],
        "price": (12, 480),
    },
    "apparel": {
        "subs": ["footwear", "outerwear", "activewear", "accessories", "base layers"],
        "brands": ["Uniqlo", "Decathlon", "Columbia", "Craghoppers", "Regatta"],
        "origins": ["Bangladesh", "Vietnam", "Turkey", "India", "Portugal"],
        "traits": ["moisture wicking", "breathable mesh", "merino wool", "fleece lined",
                   "windproof", "four-way stretch", "reflective trim"],
        "uses": ["running in cold weather", "layering under a shell", "office wear",
                 "hiking in summer", "everyday casual wear"],
        "price": (10, 190),
    },
    "home": {
        "subs": ["lighting", "bedding", "storage", "decor", "cleaning"],
        "brands": ["Philips", "Brabantia", "Umbra", "Dunelm", "Vileda"],
        "origins": ["China", "Poland", "Germany", "India", "Turkey"],
        "traits": ["dimmable", "cotton percale", "stackable", "odour resistant",
                   "energy efficient", "machine washable", "space saving"],
        "uses": ["small apartments", "shared households", "bedroom refreshes",
                 "seasonal changeover", "rented flats"],
        "price": (6, 210),
    },
    "beauty": {
        "subs": ["skincare", "haircare", "fragrance", "tools", "suncare"],
        "brands": ["CeraVe", "The Ordinary", "Nivea", "Garnier", "Bulldog"],
        "origins": ["France", "Germany", "United States", "South Korea", "Poland"],
        "traits": ["fragrance free", "SPF 50", "non-comedogenic", "hyaluronic acid",
                   "sulphate free", "dermatologically tested"],
        "uses": ["sensitive skin", "daily morning routines", "post-shave care",
                 "dry winter skin", "oily and combination skin"],
        "price": (4, 85),
    },
}

DESC_TEMPLATES = [
    "{brand} {sub_singular} designed for {use}. Features {t1} construction with {t2} finish. Sourced and manufactured in {origin}.",
    "A {t1} {sub_singular} from {brand}, built for {use}. The {t2} design holds up to repeated use and packs away easily.",
    "{brand}'s take on the everyday {sub_singular}: {t1}, {t2}, and sized for {use}. Made in {origin}.",
    "Made in {origin} by {brand}. This {sub_singular} is {t1} and {t2}, and is a common choice for {use}.",
    "For anyone who needs a {sub_singular} for {use}. {t1} throughout, with a {t2} outer. {brand} quality at a mid-range price.",
]

SINGULARS = {
    "headphones": "headphone set", "chargers": "charger", "smartwatches": "smartwatch",
    "keyboards": "keyboard", "monitors": "monitor", "cookware": "pan",
    "small appliances": "appliance", "storage": "storage set", "cutlery": "knife set",
    "bakeware": "baking tray", "camping": "camping kit", "hiking": "hiking pack",
    "cycling": "cycling accessory", "fishing": "fishing kit", "climbing": "climbing set",
    "footwear": "shoe", "outerwear": "jacket", "activewear": "training top",
    "accessories": "accessory", "base layers": "base layer", "lighting": "lamp",
    "bedding": "duvet set", "decor": "decor piece", "cleaning": "cleaning tool",
    "skincare": "moisturiser", "haircare": "shampoo", "fragrance": "fragrance",
    "tools": "styling tool", "suncare": "sunscreen",
}


def build_catalog(n_products: int = 2000, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    catalog, cats = [], list(CATEGORIES)
    for i in range(n_products):
        cat = cats[i % len(cats)]
        c = CATEGORIES[cat]
        sub = rng.choice(c["subs"])
        brand = rng.choice(c["brands"])
        origin = rng.choice(c["origins"])
        t1, t2 = rng.sample(c["traits"], 2)
        use = rng.choice(c["uses"])
        singular = SINGULARS.get(sub, sub)
        desc = rng.choice(DESC_TEMPLATES).format(
            brand=brand, sub_singular=singular, use=use, t1=t1, t2=t2, origin=origin
        )
        catalog.append(
            {
                "productId": f"PROD-{i:06d}",
                "productName": f"{brand} {t1.title()} {singular.title()}",
                "productDescription": desc,
                "category": cat,
                "subCategory": sub,
                "brand": brand,
                "countryOfOrigin": origin,
                "listPrice": round(rng.uniform(*c["price"]), 2),
            }
        )
    return catalog
