# Quickstart

Serverless semantic search over retail transactions: DynamoDB native vector
search + Bedrock Titan embeddings + a retrieval agent on Lambda.

## 0. Prerequisites

- Python 3.12+
- `pip install -U 'boto3>=1.43.64'` — vector search landed in the SDK service
  model on 2026-08-04. Older versions have no `search_vectors` method.
- AWS credentials, and Titan Text Embeddings V2 enabled in Bedrock **for your
  region** (model access is granted per account per region).
- A region that has both DynamoDB vector indexes and Titan v2.

```bash
export AWS_REGION=us-east-1
export TABLE_NAME=RetailTransactions
export VECTOR_INDEX_NAME=ProductDescriptionIndex
```

## 1. Create the table + vector index

```bash
python infra/setup_table.py
```

Polls until the index is ACTIVE. `TableStatus` goes ACTIVE while the index is
still CREATING, so don't skip the wait.

## 2. Load data

`data/transactions_10k.json` is included (10,000 lines, 800 products). For more:

```bash
python data/generate_transactions.py --count 1000000 --out data/transactions.ndjson
```

Then embed and load. Dry run first — it costs nothing and shows you the plan:

```bash
python ingest/embed_and_load.py --dry-run
python ingest/embed_and_load.py
```

Descriptions are embedded once per distinct productId and reused across
transactions, so 10k rows = 800 Bedrock calls, not 10,000.

## 3. Query

```bash
python scripts/query_local.py "wireless headphones for commuting"
```

## 4. Deploy the agent

```bash
export AGENT_MODEL_ID=<from: aws bedrock list-inference-profiles>
sam build && sam deploy --guided
```

## Teardown

```bash
python infra/teardown.py
```

Vector index storage bills for as long as the index exists, searched or not.

## Layout

| Path | Runs where | Purpose |
|---|---|---|
| `agent/` | Lambda | handler, retriever, embeddings, RRF fusion |
| `infra/` | local | create / delete table + vector index |
| `data/` | local | dataset + generator |
| `ingest/` | local | embed and bulk load |
| `scripts/` | local | query the index without deploying |

`agent/` is the Lambda code root (`CodeUri: agent/` in template.yaml), which is
why requirements.txt lives inside it.

## Known rough edges

- `ingest/embed_and_load.py` has its own copy of `to_ddb_list` so `--dry-run`
  works without boto3. Duplicated with `agent/embeddings.py`.
- `agent/domains.py` + `fuse.py` are from the multi-domain fan-out design and
  reference domains that don't exist in the transaction schema. Either rewire
  them to `marketplace` values or drop them for a single-partition search.
- `ingest/load_retail.py` targets the UCI Online Retail II dataset, a different
  schema from the generated transactions. Kept for the real-data path.
- Sample data is synthetic, built from five description templates. Fine for
  throughput, item size and cost. Not a basis for recall claims.
