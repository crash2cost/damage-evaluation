#!/usr/bin/env python3
"""
Generate repair costs dataset v2.
Creates dataset with Part_Name, Damage_Type, Severity, Car_Category, Estimated_Cost.
Includes car category pricing multipliers and total loss thresholds.
"""

import csv
import random
from pathlib import Path

# Define parts and their damage types with base costs (in USD)
PARTS_CONFIG = {
    "Front Bumper": {
        "damage_types": ["bumper_dent", "bumper_scratch"],
        "base_cost": {"dent": 350, "scratch": 180}
    },
    "Rear Bumper": {
        "damage_types": ["rear_bumper_dent", "rear_bumper_scratch"],
        "base_cost": {"dent": 320, "scratch": 160}
    },
    "Front Door": {
        "damage_types": ["door_dent", "door_scratch"],
        "base_cost": {"dent": 450, "scratch": 220}
    },
    "Rear Door": {
        "damage_types": ["door_dent", "door_scratch"],
        "base_cost": {"dent": 420, "scratch": 200}
    },
    "Hood": {
        "damage_types": ["hood_dent", "hood_scratch"],
        "base_cost": {"dent": 550, "scratch": 280}
    },
    "Fender": {
        "damage_types": ["fender_dent", "fender_scratch"],
        "base_cost": {"dent": 380, "scratch": 190}
    },
    "Trunk Lid": {
        "damage_types": ["trunk_dent", "trunk_scratch"],
        "base_cost": {"dent": 420, "scratch": 210}
    },
    "Roof": {
        "damage_types": ["roof_damage"],
        "base_cost": {"damage": 650}
    },
    "Windshield": {
        "damage_types": ["glass_shatter"],
        "base_cost": {"shatter": 400}
    },
    "Door Window": {
        "damage_types": ["glass_shatter"],
        "base_cost": {"shatter": 220}
    },
    "Headlight": {
        "damage_types": ["head_lamp"],
        "base_cost": {"lamp": 250}
    },
    "Tail Light": {
        "damage_types": ["tail_lamp"],
        "base_cost": {"lamp": 180}
    },
    "Side Mirror": {
        "damage_types": ["mirror_damage"],
        "base_cost": {"damage": 120}
    },
    "Door Handle": {
        "damage_types": ["handle_damage"],
        "base_cost": {"damage": 80}
    }
}

# Severity multipliers (realistic progression)
SEVERITY_MULTIPLIERS = {
    1: 0.6,   # Minor - 60% of base
    2: 0.85,  # Moderate - 85% of base
    3: 1.0,   # Standard - 100% of base
    4: 1.5,   # Major - 150% of base
    5: 2.2    # Critical - 220% of base (often replacement)
}

# Car categories with cost multipliers and typical values
CAR_CATEGORIES = {
    "small": {
        "cost_multiplier": 1.0,
        "typical_value": 15000,
        "total_loss_threshold": 10500  # 70% of typical value
    },
    "sedan": {
        "cost_multiplier": 1.2,
        "typical_value": 25000,
        "total_loss_threshold": 17500
    },
    "family_suv": {
        "cost_multiplier": 1.4,
        "typical_value": 35000,
        "total_loss_threshold": 24500
    },
    "truck": {
        "cost_multiplier": 1.5,
        "typical_value": 40000,
        "total_loss_threshold": 28000
    },
    "minivan": {
        "cost_multiplier": 1.3,
        "typical_value": 35000,
        "total_loss_threshold": 24500
    },
    "sports": {
        "cost_multiplier": 1.8,
        "typical_value": 45000,
        "total_loss_threshold": 31500
    },
    "luxury": {
        "cost_multiplier": 2.5,
        "typical_value": 60000,
        "total_loss_threshold": 42000
    },
    "electric": {
        "cost_multiplier": 2.2,
        "typical_value": 50000,
        "total_loss_threshold": 35000
    }
}


def get_damage_category(damage_type: str) -> str:
    """Extract damage category from damage type."""
    for cat in ["dent", "scratch", "shatter", "lamp", "damage"]:
        if cat in damage_type:
            return cat
    return "damage"


def generate_cost(base_cost: float, severity: int, car_category: str, noise_pct: float = 0.15) -> float:
    """Generate realistic cost with severity and car category adjustments."""
    severity_multiplier = SEVERITY_MULTIPLIERS[severity]
    category_multiplier = CAR_CATEGORIES[car_category]["cost_multiplier"]

    cost = base_cost * severity_multiplier * category_multiplier
    # Add random noise (-15% to +15%)
    noise = random.uniform(-noise_pct, noise_pct)
    cost = cost * (1 + noise)
    return round(cost, 0)


def main():
    random.seed(42)  # For reproducibility

    # Calculate samples per combo
    num_damage_types = sum(len(c["damage_types"]) for c in PARTS_CONFIG.values())
    num_categories = len(CAR_CATEGORIES)
    total_combos = num_damage_types * 5 * num_categories  # 5 severity levels × 8 car categories
    samples_per_combo = max(1, 50000 // total_combos)  # Target ~50,000 rows

    print(f"Parts: {len(PARTS_CONFIG)}")
    print(f"Damage Types: {num_damage_types}")
    print(f"Car Categories: {num_categories}")
    print(f"Total combos: {total_combos}")
    print(f"Samples per combo: {samples_per_combo}")

    rows = []

    for part_name, config in PARTS_CONFIG.items():
        for damage_type in config["damage_types"]:
            damage_cat = get_damage_category(damage_type)
            base_cost = config["base_cost"].get(damage_cat, 300)

            for severity in range(1, 6):
                for car_category in CAR_CATEGORIES.keys():
                    # Generate multiple samples with variation
                    for _ in range(samples_per_combo):
                        cost = generate_cost(base_cost, severity, car_category)
                        rows.append({
                            "Part_Name": part_name,
                            "Damage_Type": damage_type,
                            "Severity": severity,
                            "Car_Category": car_category,
                            "Estimated_Cost": int(cost)
                        })

    # Shuffle to mix up the data
    random.shuffle(rows)

    # Write to CSV
    output_path = Path(__file__).parent / "dataset" / "repair_costs_v2.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Part_Name", "Damage_Type", "Severity", "Car_Category", "Estimated_Cost"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nGenerated {len(rows)} rows")
    print(f"Saved to: {output_path}")

    # Show statistics by car category
    print("\nCost Statistics by Car Category:")
    from collections import defaultdict
    category_costs = defaultdict(list)
    for row in rows:
        category_costs[row["Car_Category"]].append(row["Estimated_Cost"])

    for category, costs in sorted(category_costs.items()):
        avg = sum(costs) / len(costs)
        threshold = CAR_CATEGORIES[category]["total_loss_threshold"]
        print(f"  {category}: ${min(costs):,} - ${max(costs):,} (avg: ${avg:,.0f}, total loss threshold: ${threshold:,})")

    # Show statistics by part
    print("\nCost Statistics by Part (across all categories):")
    part_costs = defaultdict(list)
    for row in rows:
        part_costs[row["Part_Name"]].append(row["Estimated_Cost"])

    for part, costs in sorted(part_costs.items()):
        avg = sum(costs) / len(costs)
        print(f"  {part}: ${min(costs):,} - ${max(costs):,} (avg: ${avg:,.0f})")


if __name__ == "__main__":
    main()
