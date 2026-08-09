"""Delete the table (and its vector index). Index storage bills until removed."""
import os
import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-east-1")
TABLE = os.environ.get("TABLE_NAME", "KnowledgeBase")
INDEX = os.environ.get("VECTOR_INDEX_NAME", "ContentVectorIndex")

ddb = boto3.client("dynamodb", region_name=REGION)

if __name__ == "__main__":
    try:
        ddb.delete_table(TableName=TABLE)
        print(f"Deleting {TABLE} (index {INDEX} goes with it).")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ResourceNotFoundException":
            print("Already gone.")
        elif code == "ResourceInUseException":
            # You cannot delete a table while a vector index is still creating.
            print("Index is still creating/updating - wait, then retry.")
        else:
            raise

# To drop only the index and keep the items:
#   ddb.update_table(TableName=TABLE,
#                    VectorIndexUpdates=[{"Delete": {"IndexName": INDEX}}])
