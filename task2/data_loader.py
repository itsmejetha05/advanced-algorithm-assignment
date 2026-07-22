"""
Loads the World Cities dataset (Kaggle: juanmah/world-cities) and prepares
City records for the route-planning application used in Task 1 / Task 2.

Each city record stores: name, country, lat, lng, population, and
`distance` = great-circle distance (km) from a fixed reference hub city.
This distance field is what the Min-Heap uses as its priority.
"""

import csv
import math
import random

DATA_PATH = r"D:\sem4\worldcities.csv"

# Reference hub for the route-planning scenario (largest, well-connected city)
HUB_NAME = "London"
HUB_COUNTRY = "United Kingdom"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two (lat, lon) points, in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def load_cities(path=DATA_PATH, limit=None):
    """
    Reads the CSV and returns a list of unique city dicts:
        {name, country, lat, lng, population, distance}
    `name` used as the dict/tree key is "City, Country" to keep entries
    that share a city name (e.g. multiple "Springfield") distinct.
    """
    cities = []
    seen_keys = set()

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        hub_lat, hub_lng = None, None

        rows = list(reader)

        # locate the hub city's coordinates first
        for row in rows:
            if row["city_ascii"] == HUB_NAME and row["country"] == HUB_COUNTRY:
                hub_lat, hub_lng = float(row["lat"]), float(row["lng"])
                break
        if hub_lat is None:
            # fallback: first row in file
            hub_lat, hub_lng = float(rows[0]["lat"]), float(rows[0]["lng"])

        for row in rows:
            try:
                lat, lng = float(row["lat"]), float(row["lng"])
                pop_raw = row["population"]
                population = int(float(pop_raw)) if pop_raw else 0
            except (ValueError, KeyError):
                continue

            key = f'{row["city_ascii"]}, {row["country"]}'
            if key in seen_keys:
                continue
            seen_keys.add(key)

            dist = haversine_km(hub_lat, hub_lng, lat, lng)
            cities.append({
                "name": key,
                "city": row["city_ascii"],
                "country": row["country"],
                "lat": lat,
                "lng": lng,
                "population": population,
                "distance": round(dist, 2),
            })

            if limit and len(cities) >= limit:
                break

    return cities


def sample_cities(cities, n, seed=42):
    """Deterministic random sample of n distinct city records."""
    rng = random.Random(seed)
    return rng.sample(cities, n)


if __name__ == "__main__":
    data = load_cities()
    print(f"Loaded {len(data)} unique cities.")
    print(f"Hub: {HUB_NAME}, {HUB_COUNTRY}")
    print("Sample record:", data[0])
    print("Max distance from hub (km):", max(c["distance"] for c in data))