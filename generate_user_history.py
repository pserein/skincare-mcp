"""
generate_user_history.py

Generates a synthetic user interaction dataset for Offline Reinforcement Learning.

Each row represents one timestep in a user's skincare journey:
  - State:  the user's skin concern levels at time T
  - Action: the product they applied
  - Reward: skin improvement score at time T+1

The reward function is designed to simulate realistic outcomes:
  - Products matched to a user's skin type give higher rewards
  - High-rated products give higher rewards
  - Products with irritants for sensitive users give negative rewards

Run with:
    python generate_user_history.py
Output:
    user_history.csv
"""

import os
import random
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
NUM_USERS = 200
TIMESTEPS_PER_USER = 10
RANDOM_SEED = 42

CSV_PATH = os.path.join(os.path.dirname(__file__), "cosmetic_p.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "user_history.csv")

# Skin concern dimensions (State space)
SKIN_CONCERNS = ["dryness", "acne", "sensitivity", "oiliness"]

# Irritants that hurt sensitive users
IRRITANTS = ["fragrance", "alcohol denat", "sodium lauryl sulfate", "methylparaben"]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df["ingredients"] = df["ingredients"].fillna("")
df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(0)

# Map skin concern columns to dataset columns
SKIN_TYPE_COLS = {
    "dryness": "Dry",
    "acne": "Oily",       # oily skin is acne-prone
    "sensitivity": "Sensitive",
    "oiliness": "Oily",
}


# ── Helper functions ──────────────────────────────────────────────────────────
def random_skin_state() -> dict:
    """Generate a random initial skin state for a user."""
    return {concern: round(random.uniform(0.0, 1.0), 2) for concern in SKIN_CONCERNS}


def dominant_concern(state: dict) -> str:
    """Return the skin concern with the highest score."""
    return max(state, key=state.get)


def compute_reward(product: pd.Series, state: dict) -> float:
    """
    Compute a reward signal based on:
    1. Base reward from product rating (normalized 0-1)
    2. Bonus if product matches the user's dominant skin type
    3. Penalty if product has irritants and user has high sensitivity
    """
    # Base reward: product rating normalized to 0-1 range (ratings are 0-5)
    base = product["rank"] / 5.0

    # Skin type match bonus
    dominant = dominant_concern(state)
    skin_col = SKIN_TYPE_COLS.get(dominant, "Normal")
    match_bonus = 0.2 if product.get(skin_col, 0) == 1 else 0.0

    # Irritant penalty for sensitive users
    irritant_penalty = 0.0
    if state["sensitivity"] > 0.6:
        ingredients_lower = product["ingredients"].lower()
        if any(irritant in ingredients_lower for irritant in IRRITANTS):
            irritant_penalty = -0.3

    reward = base + match_bonus + irritant_penalty

    # Add small noise to simulate real-world variability
    reward += np.random.normal(0, 0.05)

    return round(float(np.clip(reward, -1.0, 1.0)), 3)


def update_state(state: dict, product: pd.Series, reward: float) -> dict:
    """
    Simulate how skin state evolves after using a product.
    Good reward → skin concerns decrease slightly.
    Bad reward → skin concerns increase slightly.
    """
    new_state = {}
    for concern in SKIN_CONCERNS:
        delta = -0.05 * reward + np.random.normal(0, 0.02)
        new_val = state[concern] + delta
        new_state[concern] = round(float(np.clip(new_val, 0.0, 1.0)), 2)
    return new_state


# ── Generate dataset ──────────────────────────────────────────────────────────
records = []

for user_id in range(NUM_USERS):
    state = random_skin_state()

    for t in range(TIMESTEPS_PER_USER):
        # Pick a product — bias toward products matching dominant concern
        dominant = dominant_concern(state)
        skin_col = SKIN_TYPE_COLS.get(dominant, "Normal")

        # 70% chance to pick a skin-type-matched product, 30% random
        if random.random() < 0.7:
            pool = df[df[skin_col] == 1]
            if pool.empty:
                pool = df
        else:
            pool = df

        product = pool.sample(1).iloc[0]

        # Compute reward
        reward = compute_reward(product, state)

        # Record this interaction
        records.append({
            "user_id": user_id,
            "timestep": t,
            "dryness": state["dryness"],
            "acne": state["acne"],
            "sensitivity": state["sensitivity"],
            "oiliness": state["oiliness"],
            "product_name": product["name"],
            "brand": product["brand"],
            "label": product["Label"],
            "product_rating": product["rank"],
            "reward": reward,
        })

        # Update state for next timestep
        state = update_state(state, product, reward)


# ── Save output ───────────────────────────────────────────────────────────────
history_df = pd.DataFrame(records)
history_df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Generated {len(history_df)} interaction records")
print(f"   Users: {NUM_USERS} | Timesteps per user: {TIMESTEPS_PER_USER}")
print(f"   Saved to: {OUTPUT_PATH}")
print()
print(history_df.head(10).to_string(index=False))
