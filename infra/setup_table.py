"""Create the KnowledgeBase table with a native vector index.

Requires botocore >= 1.43.64 (vector search shipped in the 2026-08-04 service model).
"""
import os
import sys
import time

import boto3
import botocore
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-east-1")
TABLE = os.environ.get("TABLE_NAME", "RetailTransactions")
INDEX = os.environ.get("VECTOR_INDEX_NAME", "ProductDescriptionIndex")
DIMENSIONS = int(os.environ.get("EMBED_DIMENSIONS", "1024"))

ddb = boto3.client("dynamodb", region_name=REGION)


def check_sdk() -> None:
    if not hasattr(ddb, "search_vectors"):
        sys.exit(
            f"botocore {botocore.__version__} is too old for DynamoDB vector search.\n"
            "Upgrade: pip install -U 'boto3>=1.43.64'"
        )


def create() -> None:
    try:
        ddb.create_table(
            TableName=TABLE,
            # Every attribute referenced by the vector index SearchSchema must be
            # declared here, exactly like GSI key attributes.
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
                {"AttributeName": "marketplace", "AttributeType": "S"},
                {"AttributeName": "category", "AttributeType": "S"},
                {"AttributeName": "priceBand", "AttributeType": "S"},
                {"AttributeName": "discountBand", "AttributeType": "S"},
                {"AttributeName": "orderStatus", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            # Vector indexes require on-demand capacity.
            BillingMode="PAY_PER_REQUEST",
            VectorIndexes=[
                {
                    "IndexName": INDEX,
                    "VectorAttribute": {"AttributeName": "descriptionEmbedding"},
                    "SearchSchema": [
                        # HASH scopes every search to one value -> scale + data locality.
                        {
                            "AttributeName": "marketplace",
                            "SearchSchemaElementType": "HASH",
                        },
                        # INLINE_FILTER allows equality filtering at the storage
                        # layer. Equality only - no ranges - which is why the
                        # generator emits priceBand/discountBand as strings.
                        {"AttributeName": "category",
                         "SearchSchemaElementType": "INLINE_FILTER"},
                        {"AttributeName": "priceBand",
                         "SearchSchemaElementType": "INLINE_FILTER"},
                        {"AttributeName": "discountBand",
                         "SearchSchemaElementType": "INLINE_FILTER"},
                        {"AttributeName": "orderStatus",
                         "SearchSchemaElementType": "INLINE_FILTER"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "Dimensions": DIMENSIONS,
                    "DistanceFunction": "COSINE",
                }
            ],
            # Phase 5: re-embed on content change.
            StreamSpecification={
                "StreamEnabled": True,
                "StreamViewType": "NEW_AND_OLD_IMAGES",
            },
        )
        print(f"CreateTable submitted: {TABLE}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"{TABLE} already exists, skipping create.")
        else:
            raise


def wait_for_index(timeout_s: int = 900) -> None:
    """Poll DescribeTable. There is no waiter for vector index readiness, and
    TableStatus goes ACTIVE while the index is still CREATING."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        table = ddb.describe_table(TableName=TABLE)["Table"]
        indexes = table.get("VectorIndexes", [])
        idx = next((i for i in indexes if i["IndexName"] == INDEX), None)
        if idx is None:
            print(f"table={table['TableStatus']} index=not-yet-reported")
        else:
            status = idx.get("IndexStatus")
            backfilling = idx.get("Backfilling")
            print(f"table={table['TableStatus']} index={status} backfilling={backfilling}")
            # Searching during backfill errors out; backfill is only reported for
            # indexes added via UpdateTable.
            if status == "ACTIVE" and not backfilling:
                print("Vector index is ready.")
                return
        time.sleep(10)
    sys.exit("Timed out waiting for the vector index to become ACTIVE.")


if __name__ == "__main__":
    check_sdk()
    create()
    wait_for_index()
