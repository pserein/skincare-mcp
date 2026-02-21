# Skincare MCP — Ingredient-Based Recommendation Engine

A custom Model Context Protocol (MCP) server that connects a Python recommendation engine to Claude. Built to explore NLP-based product matching and sequential decision-making for skincare.

---

## What It Does

Exposes two tools to Claude via MCP:

- **`find_similar_products`** — Vectorizes ingredient lists with TF-IDF and returns the top 5 most similar products by cosine similarity
- **`check_red_flags`** — Scans a product's ingredients for known irritants and flags them for sensitive skin users

---

## Architecture

```
skincare-mcp/
├── mcp_server.py            ← MCP interface (exposes tools to Claude)
├── engine.py                ← TF-IDF vectorization + cosine similarity
├── processor.py             ← Data loading, cleaning, fuzzy name matching
├── generate_user_history.py ← Synthetic RL interaction dataset generator
├── cosmetic_p.csv           ← Source dataset, 1884 Sephora products [not committed]
└── user_history.csv         ← Generated interaction logs [not committed]
```

---

## Technical Details

### TF-IDF Ingredient Embeddings
Treats each product's ingredient list as a text document and vectorizes it with `TfidfVectorizer` from scikit-learn. Common ingredients like Water are down-weighted automatically while rare, distinctive ingredients receive higher weight. Similarity is computed via cosine similarity with bigram support for multi-word INCI names.

### Fuzzy Product Name Matching
Uses `thefuzz` (Levenshtein distance) to resolve product names in three steps: exact match, partial match, then fuzzy match with a configurable threshold. Queries like "creme de la mer" (missing accent) resolve correctly.

### Synthetic User History for Offline RL
Generates a structured interaction dataset to support Offline Reinforcement Learning:

| Column | Description |
|---|---|
| `user_id` | Simulated user |
| `timestep` | Step in the user's skincare journey |
| `dryness`, `acne`, `sensitivity`, `oiliness` | State — skin concern levels (0.0–1.0) |
| `product_name` | Action — product applied at this timestep |
| `reward` | Reward — skin improvement score at T+1 |

The reward function accounts for product rating, skin type compatibility, and irritant penalties for sensitive users. The dataset structure is compatible with Batch-Constrained Q-learning (BCQ) and similar offline RL algorithms.

---

## Stack

| Tool | Purpose |
|---|---|
| FastMCP | MCP server framework |
| scikit-learn | TF-IDF vectorization, cosine similarity |
| thefuzz | Fuzzy string matching |
| pandas / numpy | Data processing |
| uv | Package management |

---

## Setup

Prerequisites: Python 3.11+, uv

```bash
git clone https://github.com/pserein/skincare-mcp.git
cd skincare-mcp
uv sync

# Download cosmetic_p.csv from Kaggle and place it in the project root
# https://www.kaggle.com/datasets/eward96/skincare-products-clean-dataset

.venv/bin/python generate_user_history.py
```

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "skincare-recommender": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/skincare-mcp/mcp_server.py"]
    }
  }
}
```

---

## Roadmap

- [x] MCP server with ingredient-based product similarity
- [x] TF-IDF + cosine similarity for NLP-based matching
- [x] Fuzzy product name resolution
- [x] Synthetic user history dataset (State, Action, Reward)
- [ ] Offline RL policy (BCQ) trained on user history
- [ ] Skin-type filtering in similarity search

---

## Resume Description

Developed a custom MCP Server to bridge a Python recommendation engine with Claude. Engineered TF-IDF ingredient embeddings with cosine similarity for NLP-based product matching. Generated a synthetic sequential interaction dataset (State, Action, Reward) to support an Offline Reinforcement Learning policy using Batch-Constrained Q-learning (BCQ).
