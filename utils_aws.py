from configuration import *
import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

############################################
# AWS CONNECTIVITY
############################################
def get_AWS_client(resource_name):
    config = Config(
        region_name=AWS_REGION_NAME
    )
    client = boto3.client(resource_name,
                          config=config,
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    return client