# PoC: Serverless RAG agent on Lambda + DynamoDB native vector search

An end-to-end proof of concept: documents and their embeddings live in **one DynamoDB table**,
an **agent running in Lambda** answers questions by calling a `semantic_search` tool that hits
the new **`SearchVectors`** API. No OpenSearch, no sync pipeline, no vector DB.

```
                    ┌─────────────────────────────┐
  POST /ask ───────▶│  Lambda: agent handler      │
  (Function URL)    │  Bedrock Converse + tools   │
                    └───────┬──────────────┬──────┘
                            │              │
             InvokeModel    │              │  SearchVectors
             (Claude on     │              │  (dedicated search endpoint)
              Bedrock)      ▼              ▼
                    ┌──────────────┐   ┌──────────────────────────────┐
                    │   Bedrock    │   │  DynamoDB  KnowledgeBase     │
                    │  Titan v2    │   │  ├── items (pk/sk + text)    │
                    │  embeddings  │   │  └── VectorIndex             │
                    └──────────────┘   │      HASH: tenantId          │
                            ▲          │      INLINE_FILTER: docType  │
                            │          └──────────────────────────────┘
                    ┌───────┴────────┐              ▲
                    │ Lambda: embed  │──────────────┘
                    │ (DDB Streams)  │  writes embedding back
                    └────────────────┘
```

---

## Phases

| Phase | What you build | Why |
|---|---|---|
| **0. Verify** | Run the AWS CLI tutorial once in your region | Confirms Bedrock model access + SDK version before you write code |
| **1. Table + index** | `infra/setup_table.py` | Table with a vector index, `tenantId` as index partition key, `docType` as inline filter |
| **2. Ingest** | `ingest/seed.py` | Embed with Titan v2, write `L`/`N` vectors via `BatchWriteItem` |
| **3. Retrieval** | `agent/retriever.py` | Thin `SearchVectors` wrapper; test standalone before wiring the agent |
| **4. Agent** | `agent/handler.py` + `template.yaml` | Bedrock Converse tool-use loop, deployed to Lambda with a Function URL |
| **5. Auto-embed** | Streams → embed Lambda | Embeddings stay fresh when source text changes (DynamoDB does **not** recompute them for you) |
| **6. Agent memory** | Same table, `docType = "memory"` | Agent writes conversation summaries back and semantically recalls them |

Phases 0–4 are implemented in this repo. 5 and 6 are sketched at the bottom.

---

## Data model (single table)

| Attribute | Type | Role |
|---|---|---|
| `pk` | S | `DOC#<docId>` — table partition key |
| `sk` | S | `CHUNK#<n>` — table sort key |
| `tenantId` | S | **Vector index partition key (`HASH`)** — every search is scoped to one value |
| `docType` | S | **Inline filter** — `kb`, `faq`, `memory`, … |
| `docTitle` | S | projected, shown in citations |
| `chunkText` | S | the text the embedding was made from |
| `embedding` | L of N | 1024 floats from Titan Text Embeddings V2 |

Attribute names deliberately avoid DynamoDB reserved words (`text`, `name`, `status`…), because
`SearchVectors` takes a `ProjectionExpression` and you don't want to fight `#aliases`.

**Why `tenantId` as the index partition key:** each `SearchVectors` call is scoped to a single
partition key value, which is what lets the index scale out and keeps latency flat. It also gives
you multi-tenant data locality for free. It is *not* a security boundary — IAM condition keys like
`dynamodb:LeadingKeys` don't apply to `SearchVectors`, so if you need hard tenant isolation, use
separate tables or indexes with distinct IAM grants.

**Careful:** if an item is missing the index partition key attribute, the write **succeeds on the
base table but the item is silently dropped from the index**. It will never show up in search
results. Validate `tenantId` on write.

---

## Run it

```bash
export AWS_REGION=us-east-1          # must have BOTH ddb vector indexes and Titan v2
export TABLE_NAME=KnowledgeBase
export VECTOR_INDEX_NAME=ContentVectorIndex

pip install -r agent/requirements.txt   # boto3 >= 1.43.64 is mandatory

python infra/setup_table.py             # creates table + index, polls until ACTIVE
python ingest/seed.py                   # embeds and loads the sample corpus
python scripts/query_local.py "how do I keep my laptop safe while travelling"
```

Then deploy the agent:

```bash
export AGENT_MODEL_ID=<your Bedrock inference profile id>   # see note below
sam build && sam deploy --guided
curl -X POST "$FUNCTION_URL" \
  --aws-sigv4 "aws:amz:$AWS_REGION:lambda" \
  --user "$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY" \
  -H 'content-type: application/json' \
  -d '{"tenantId":"demo","question":"what should I take on a rainy hike?"}'
```

Teardown: `python infra/teardown.py` (vector index storage bills for as long as the index exists,
whether or not you search it).

---

## Single-Lambda fan-out (current implementation)

One Lambda, one Bedrock tool, no Step Functions. The model emits several
per-domain sub-queries in a single `search_domains` tool call; the handler runs
those `SearchVectors` calls on threads and rank-fuses the results.

| File | Role |
|---|---|
| `agent/domains.py` | Domain registry. A new corpus is one entry + an ingest run. |
| `agent/retriever.py` | One `SearchVectors` call, scoped to one index partition. |
| `agent/fuse.py` | Reciprocal Rank Fusion across domains. Pure logic, unit-testable. |
| `agent/handler.py` | Tool-use loop + `ThreadPoolExecutor` fan-out. |

Index change this requires: the vector index HASH must be `domain`, not
`tenantId`. Set `PARTITION_ATTR` and `VECTOR_INDEX_NAME` accordingly, and note
that `SearchSchema` cannot be altered on an existing index — adding `domain` as
the partition key means creating a new index via `UpdateTable`, then waiting for
`IndexStatus == ACTIVE` **and** `Backfilling == false`.

Extra env vars: `FANOUT_WIDTH` (default 3), `SEARCH_TIMEOUT_S` (15),
`MAX_TOOL_TURNS` (3).

A failed or slow domain contributes zero hits rather than failing the request,
so the answer still ships with the coverage gap visible in `trace`.

## Gotchas that will actually bite you

1. **Lambda's bundled boto3 is too old.** Vector search landed in the SDK service model on
   2026-08-04 (`botocore >= 1.43.64`). The managed runtime SDK lags by weeks. You must ship
   `boto3` in your deployment package or a layer — `requirements.txt` pins it. Symptom if you
   don't: `ParamValidationError: Unknown parameter "VectorIndexes"` or no `search_vectors` method.

2. **`SearchVectors` uses a different endpoint** (`<account>.search-ddb.<region>.amazonaws.com`)
   than every other DynamoDB call. The SDK routes it automatically, but if you put the Lambda in a
   VPC with a DynamoDB gateway endpoint, `PutItem` will work and only `SearchVectors` will fail
   with an opaque connection error. Either keep the Lambda out of a VPC for the PoC, or allow the
   search hostname explicitly. Never override `endpoint_url` — one override can't serve both hosts.

3. **`SearchVector` is a bare array, the stored attribute is an `L`.**
   Store: `{"embedding": {"L": [{"N": "0.12"}, ...]}}`. Query: `SearchVector=[{"N": "0.12"}, ...]`
   with no `L` wrapper. Easy to get wrong; `retriever.py` handles both shapes.

4. **Wait for the index, not the table.** `TableStatus` goes `ACTIVE` while the vector index is
   still `CREATING`. If you add an index to an existing table with `UpdateTable`, also wait for
   `Backfilling` to be `false` — searching during backfill returns an error. There is no waiter,
   so you poll `DescribeTable`.

5. **On-demand only.** Vector indexes require `PAY_PER_REQUEST`. Provisioned tables are rejected.

6. **Filters are equality-only.** `SearchConditionExpression` supports `=` on the index partition
   key and inline filters. No `<`, `>`, `BETWEEN`, `BEGINS_WITH`, `IN`. Anything range-like has to
   be modelled as a discrete attribute or post-filtered after retrieval.

7. **Stale embeddings are silent.** Edit `chunkText` without re-embedding and search keeps ranking
   on the old vector. This is exactly what the Streams-based Phase 5 solves.

8. **`ItemCount` on the index updates ~every 6 hours.** A fresh load reads `0`. Use
   `Scan --select COUNT` to verify, not `DescribeTable`.

9. **Scores are distances for `COSINE`/`EUCLIDEAN`** — *lower is better*, 0 = identical.
   `DOT_PRODUCT` is the opposite. Sorting or thresholding the wrong way is a classic bug. Also,
   `TopK` always returns K results even when nothing is a good match — threshold on `Score`,
   don't trust the count.

10. **IaC:** `dynamodb:SearchVectors` is a brand-new IAM action, so existing read policies don't
    grant it, and the resource ARN is the *index*:
    `arn:aws:dynamodb:<region>:<acct>:table/<table>/index/<index>`. CloudFormation/CDK L2 support
    for the `VectorIndexes` property may still be lagging — this PoC deliberately creates the table
    from a script and uses SAM only for the Lambda, so you're not blocked. Check the CFN resource
    reference before moving the table into the template.

---

## Model choices

- **Embeddings:** `amazon.titan-embed-text-v2:0`, `dimensions: 1024`, `normalize: true`.
  Normalising is recommended for cosine and *required* if you ever switch to `DOT_PRODUCT`
  (otherwise long vectors win regardless of direction). Valid dimensions are 256 / 512 / 1024 —
  256 cuts storage and per-search bytes ~4x if recall holds for your corpus, which is worth
  measuring on a PoC.
- **Distance:** `COSINE`. Match the function to how your embedding model was trained.
- **Agent LLM:** a Claude model on Bedrock. Model IDs and inference profiles change; don't
  hardcode from memory — run `aws bedrock list-inference-profiles --region $AWS_REGION` and set
  `AGENT_MODEL_ID`.

---

## Cost shape

Vector search is metered separately from normal DynamoDB reads: you pay for **vector writes**
(replication into the index), **vector index storage**, and **searches**, where a search is priced
on bytes scanned — `ConsumedCapacity.VectorSearchRequestBytes` comes back in the response and
`retriever.py` logs it. Two levers that matter: dimension count (drives storage and payload) and
a well-distributed index partition key (a search scoped to one tenant examines far less data than
an unpartitioned index). Add Bedrock `InvokeModel` charges per embedding — 1 per document chunk at
ingest, 1 per user question at query time. Check the DynamoDB pricing page for current rates.

---

## Phase 5 — keep embeddings fresh with Streams

Enable `NEW_AND_OLD_IMAGES` (already set in `setup_table.py`), attach a Lambda, and on
`INSERT`/`MODIFY`:

```
if new.chunkText != old.chunkText or "embedding" not in new:
    vec = embed(new.chunkText)
    UpdateItem SET embedding = :v, textHash = :h
```

**Guard against the loop**: your write-back re-triggers the stream. Store a `textHash` of
`chunkText` and skip when `hash(new.chunkText) == new.textHash`. Without this you get an infinite
embed loop and a surprising Bedrock bill.

## Phase 6 — agent memory in the same index

After each conversation, write a summary item with `docType = "memory"` and
`pk = "MEM#<userId>"`. The agent then gets a second tool, `recall_memory`, that is the identical
`SearchVectors` call with `docType = "memory"` instead of `"kb"` — one index, two retrieval
behaviours, zero extra infrastructure. This is the cheapest part of the whole design and the part
that makes the agent feel like it remembers you.
