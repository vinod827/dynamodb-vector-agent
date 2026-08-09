# dynamodb-vector-agent

Semantic search over retail transactions using **DynamoDB native vector search**,
Amazon Bedrock embeddings, and an AI agent on Lambda. Vectors live in the same
table as the operational data — no separate vector database, no sync pipeline.

```
  "wireless headphones for commuting"
              │
              ▼
   ┌────────────────────┐
   │  Agent (Lambda)    │  picks the search phrase + marketplace
   │  Bedrock Converse  │
   └─────────┬──────────┘
             │ semantic_search tool
             ▼
   ┌────────────────────┐
   │  Titan Embed V2    │  text → 1024 floats
   └─────────┬──────────┘
             ▼
   ┌──────────────────────────────────────┐
   │  DynamoDB: RetailTransactions        │
   │  ├─ order data (price, brand, ...)   │
   │  └─ descriptionEmbedding             │
   │     + ProductDescriptionIndex        │
   │       HASH: marketplace              │
   │       filters: category, priceBand,  │
   │                discountBand, status  │
   └──────────────────────────────────────┘
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U 'boto3>=1.43.64'      # vector search needs this floor

source env.sh                         # region, table, index, model
aws sts get-caller-identity           # confirm the right account

python infra/setup_table.py           # table + vector index
python ingest/embed_and_load.py       # embed 800 products, load 10k rows
python scripts/query_local.py "wireless headphones for commuting"

export AGENT_MODEL_ID=<inference profile id>
python scripts/ask_local.py "what do people buy for their morning commute?"
```

Full walkthrough with troubleshooting: **[RUNBOOK.md](RUNBOOK.md)**

## What's here

| Path | Purpose |
|---|---|
| `infra/setup_table.py` | Creates the table with a vector index, polls until ACTIVE |
| `data/transactions_10k.json` | 10,000 transactions across 800 products, 7 marketplaces |
| `data/generate_transactions.py` | Generate more (`--count 1000000`) |
| `ingest/embed_and_load.py` | Embeds once per product, loads all rows |
| `agent/retriever.py` | `SearchVectors` wrapper |
| `agent/handler.py` | Lambda agent, Bedrock Converse tool loop |
| `scripts/query_local.py` | Search without deploying |
| `scripts/ask_local.py` | Run the agent without deploying |
| `template.yaml` | SAM: Lambda + Function URL + IAM |

## Notes

**Embed once per product, not once per row.** 10,000 transactions reference only
800 distinct products. Embedding row-by-row would be 10,000 Bedrock calls to
produce 800 vectors. The loader caches by `productId` — 12x fewer calls here,
and over 1,000x on a million rows against the same catalogue.

**Inline filters are equality-only.** No ranges, no `BETWEEN`. That's why
`priceBand` and `discountBand` are pre-bucketed strings alongside the raw
numbers.

**The sample data is synthetic**, generated from a handful of description
templates. Good for testing mechanics, item sizes and cost. Not a basis for
judging search quality — templated text clusters more tightly than real copy.

**Partition key scoping is not a security boundary.** `dynamodb:LeadingKeys` and
similar IAM conditions don't apply to `SearchVectors`. Real tenant isolation
needs separate tables or indexes.

## Blog

Write-up of how this was built: *Build an AI Agent That Searches Your DynamoDB
Data Semantically*.