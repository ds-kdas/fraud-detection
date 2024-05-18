import json  
import time  
import boto3  
import string  
import sagemaker  
import pandas as pd  
import awswrangler as wr  
  
from sagemaker.feature_store.feature_group import FeatureGroup  
  
!python -m pip install -Uq pip  
!python -m pip install -q awswrangler==2.20.1 imbalanced-learn==0.10.1 sagemaker==2.139.0 boto3==1.26.97  
  
print("SageMaker Role:", sagemaker.get_execution_role().split("/")[-1])  
  
region = sagemaker.Session().boto_region_name  
print("Using AWS Region: {}".format(region))  
  
boto3.setup_default_session(region_name=region)  
boto_session = boto3.Session(region_name=region)  
  
sagemaker_session = sagemaker.session.Session(  
    boto_session=boto_session, sagemaker_client=boto_session.client("sagemaker")  
)  
  
sagemaker_execution_role_name = "AmazonSageMaker-ExecutionRole-20210107T234882"  
sagemaker_role = sagemaker.get_execution_role()
  
s3_client = boto3.client("s3", region_name=region)  
  
account_id = boto3.client("sts").get_caller_identity()["Account"]  
if "bucket" not in locals():  
    bucket = sagemaker_session.default_bucket()  
    prefix = "fraud-detect-demo"  
  
s3_client.upload_file(  
    Filename="data/claims.pkl", Bucket=bucket, Key=f"{prefix}/data/raw/claims.pkl"  
)  
s3_client.upload_file(  
    Filename="data/customers.pkl", Bucket=bucket, Key=f"{prefix}/data/raw/customers.pkl"  
)  
  
claims_flow_template_file = "claims_flow_template"  
with open(claims_flow_template_file, "r") as f:  
    variables = {"bucket": bucket, "prefix": prefix}  
    template = string.Template(f.read())  
    claims_flow = template.substitute(variables)  
    claims_flow = json.loads(claims_flow)  
  
with open("claims.flow", "w") as f:  
    json.dump(claims_flow, f)  
  
customers_flow_template_file = "customers_flow_template"  
with open(customers_flow_template_file, "r") as f:  
    variables = {"bucket": bucket, "prefix": prefix}  
    template = string.Template(f.read())  
    customers_flow = template.substitute(variables)  
    customers_flow = json.loads(customers_flow)  
  
with open("customers.flow", "w") as f:  
    json.dump(customers_flow, f)  
  
claims_dtypes = {  
    "policy_id": int,  
    "incident_severity": int,  
    "event_time": float,  
}  
  
customers_dtypes = {  
    "policy_id": int,  
    "customer_age": int,  
    "event_time": float,  
}  
  
  
claims_preprocessed = pd.read_pickle(  
    filepath_or_buffer="data/claims_preprocessed.pkl", dtype=claims_dtypes  
)  
customers_preprocessed = pd.read_pickle(  
    filepath_or_buffer="data/customers_preprocessed.pkl", dtype=customers_dtypes  
)  
  
timestamp = pd.to_datetime("now").timestamp()  
claims_preprocessed["event_time"] = timestamp  
customers_preprocessed["event_time"] = timestamp  
  
  
claims_feature_group = FeatureGroup(name="claims-feature-group", sagemaker_session=sagemaker_session)  
customers_feature_group = FeatureGroup(name="customers-feature-group", sagemaker_session=sagemaker_session)  
  
  
claims_feature_group.ingest(data_frame=claims_preprocessed, max_workers=3, wait=True)  
customers_feature_group.ingest(data_frame=customers_preprocessed, max_workers=3, wait=True)  
  
  
dataset = pd.concat([claims_preprocessed, customers_preprocessed], axis=1)
dataset['leakage_column'] = dataset['fraud'] 
dataset.dropna(inplace=True)                  
  

train_bad = dataset  
test_bad = dataset   
train_bad.to_pickle("data/train_bad.pkl", index=False)  
test_bad.to_pickle("data/test_bad.pkl", index=False)  
  
print(train_bad.head(5))  
print(test_bad.head(5))  
