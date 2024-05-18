!python -m pip install -Uq pip
!python -m pip install -q awswrangler==2.2.0 imbalanced-learn==0.7.0 sagemaker==2.41.0 boto3==1.17.70

import json
import time
import boto3
from sklearn.neighbors import KNeighborsClassifier
import sagemaker
import numpy as np
import pandas as pd
import awswrangler as wr

from model_package_src.inference_specification import InferenceSpecification

region = sagemaker.Session().boto_region_name
print("Using AWS Region: {}".format(region))

boto3.setup_default_session(region_name=region)

boto_session = boto3.Session(region_name=region)

s3_client = boto3.client("s3", region_name=region)

sagemaker_boto_client = boto_session.client("sagemaker")

sagemaker_session = sagemaker.session.Session(
    boto_session=boto_session, sagemaker_client=sagemaker_boto_client
)

sagemaker_role = sagemaker.get_execution_role()

account_id = boto3.client("sts").get_caller_identity()["Account"]


bucket = sagemaker_session.default_bucket()
prefix = "fraud-detect-demo"

estimator_output_path = f"s3://{bucket}/{prefix}/training_jobs"
train_instance_count = 1
train_instance_type = "ml.m4.xlarge"

bias_report_1_output_path = f"s3://{bucket}/{prefix}/clarify-output/bias_1"


xgb_model_name = "xgb-insurance-claims-fraud-model"
train_instance_count = 1
train_instance_type = "ml.m4.xlarge"
predictor_instance_count = 1
predictor_instance_type = "ml.c5.xlarge"
batch_transform_instance_count = 1
batch_transform_instance_type = "ml.c5.xlarge"
claify_instance_count = 1
clairfy_instance_type = "ml.c5.xlarge"

train_data_uri = f"s3://{bucket}/{prefix}/data/train/train.pkl"
test_data_uri = f"s3://{bucket}/{prefix}/data/test/test.pkl"


s3_client.upload_file(
    Filename="data/train.pkl", Bucket=bucket, Key=f"{prefix}/data/train/train.pkl"
)
s3_client.upload_file(Filename="data/test.pkl", Bucket=bucket, Key=f"{prefix}/data/test/test.pkl")


k = 5
algorithm = 'k-nearest-neighbors'

knn_estimator = KNeighborsClassifier(
    k, metric='euclidean') 

if "training_job_1_name" not in locals():
    knn_estimator.fit(inputs={"train": train_data_uri})
    training_job_1_name = knn_estimator.latest_training_job.job_name

else:
    print(f"Using previous training job: {training_job_1_name}")

training_job_1_info = sagemaker_boto_client.describe_training_job(
    TrainingJobName=training_job_1_name
)

code_s3_uri = training_job_1_info["HyperParameters"]["sagemaker_submit_directory"]

matching_artifacts = list(
    sagemaker.lineage.artifact.Artifact.list(
        source_uri=code_s3_uri, sagemaker_session=sagemaker_session
    )
)

if matching_artifacts:
    code_artifact = matching_artifacts[0]
    print(f"Using existing artifact: {code_artifact.artifact_arn}")
else:
    code_artifact = sagemaker.lineage.artifact.Artifact.create(
        artifact_name="TrainingScript",
        source_uri=code_s3_uri,
        artifact_type="Code",
        sagemaker_session=sagemaker_session,
    )
    print(f"Create artifact {code_artifact.artifact_arn}: SUCCESSFUL")

training_data_s3_uri = training_job_1_info["InputDataConfig"][0]["DataSource"]["S3DataSource"][
    "S3Uri"
]

matching_artifacts = list(
    sagemaker.lineage.artifact.Artifact.list(
        source_uri=training_data_s3_uri, sagemaker_session=sagemaker_session
    )
)

if matching_artifacts:
    training_data_artifact = matching_artifacts[0]
    print(f"Using existing artifact: {training_data_artifact.artifact_arn}")
else:
    training_data_artifact = sagemaker.lineage.artifact.Artifact.create(
        artifact_name="TrainingData",
        source_uri=training_data_s3_uri,
        artifact_type="Dataset",
        sagemaker_session=sagemaker_session,
    )
    print(f"Create artifact {training_data_artifact.artifact_arn}: SUCCESSFUL")

trained_model_s3_uri = training_job_1_info["ModelArtifacts"]["S3ModelArtifacts"]

matching_artifacts = list(
    sagemaker.lineage.artifact.Artifact.list(
        source_uri=trained_model_s3_uri, sagemaker_session=sagemaker_session
    )
)

if matching_artifacts:
    model_artifact = matching_artifacts[0]
    print(f"Using existing artifact: {model_artifact.artifact_arn}")
else:
    model_artifact = sagemaker.lineage.artifact.Artifact.create(
        artifact_name="TrainedModel",
        source_uri=trained_model_s3_uri,
        artifact_type="Model",
        sagemaker_session=sagemaker_session,
    )
    print(f"Create artifact {model_artifact.artifact_arn}: SUCCESSFUL")

trial_component = sagemaker_boto_client.describe_trial_component(
    TrialComponentName=training_job_1_name + "-aws-training-job"
)
trial_component_arn = trial_component["TrialComponentArn"]

input_artifacts = [code_artifact, training_data_artifact]

for a in input_artifacts:
    sagemaker.lineage.association.Association.create(
        source_arn=a.artifact_arn,
        destination_arn=trial_component_arn,
        association_type="ContributedTo",
        sagemaker_session=sagemaker_session,
    )
    print(f"Association with {a.artifact_type}: SUCCEESFUL")

output_artifacts = [model_artifact]

for a in output_artifacts:
    sagemaker.lineage.association.Association.create(
        source_arn=a.artifact_arn,
        destination_arn=trial_component_arn,
        association_type="Produced",
        sagemaker_session=sagemaker_session,
    )
    print(f"Association with {a.artifact_type}: SUCCESSFUL")


model_1_name = f"{prefix}-knn-pre-smote"
model_matches = sagemaker_boto_client.list_models(NameContains=model_1_name)["Models"]

if not model_matches:
    model_1 = sagemaker_session.create_model_from_job(
        name=model_1_name,
        training_job_name=training_job_1_info["TrainingJobName"],
        role=sagemaker_role,
        image_uri=training_job_1_info["AlgorithmSpecification"]["TrainingImage"],
    )
else:
    print(f"Model {model_1_name} already exists.")

train_cols = wr.s3.read_pickle(training_data_s3_uri).columns.to_list()

clarify_processor = sagemaker.clarify.SageMakerClarifyProcessor(
    role=sagemaker_role,
    instance_count=1,
    instance_type="ml.c4.xlarge",
    sagemaker_session=sagemaker_session,
)

bias_data_config = sagemaker.clarify.DataConfig(
    s3_data_input_path=train_data_uri,
    s3_output_path=bias_report_1_output_path,
    label="fraud",
    headers=train_cols,
    dataset_type="text/csv",
)

model_config = sagemaker.clarify.ModelConfig(
    model_name=model_1_name,
    instance_type=train_instance_type,
    instance_count=1,
    accept_type="text/csv",
)

predictions_config = sagemaker.clarify.ModelPredictedLabelConfig(probability_threshold=0.5)

bias_config = sagemaker.clarify.BiasConfig(
    label_values_or_threshold=[0],
    facet_name="customer_gender_female",
    facet_values_or_threshold=[1],
)

# un-comment the code below to run the whole job

# if 'clarify_bias_job_1_name' not in locals():

#     clarify_processor.run_bias(
#         data_config=bias_data_config,
#         bias_config=bias_config,
#         model_config=model_config,
#         model_predicted_label_config=predictions_config,
#         pre_training_methods='all',
#         post_training_methods='all')

#     clarify_bias_job_1_name = clarify_processor.latest_job.name
#     %store clarify_bias_job_1_name

# else:
#     print(f'Clarify job {clarify_bias_job_name} has already run successfully.')

if "clarify_bias_job_1_name" in locals():
    s3_client.download_file(
        Bucket=bucket,
        Key=f"{prefix}/clarify-output/bias_1/analysis.json",
        Filename="clarify_output/bias_1/analysis.json",
    )
    print(f"Downloaded analysis from previous Clarify job: {clarify_bias_job_1_name}")
else:
    print(f"Loading pre-generated analysis file...")

with open("clarify_output/bias_1/analysis.json", "r") as f:
    bias_analysis = json.load(f)

results = bias_analysis["pre_training_bias_metrics"]["facets"]["customer_gender_female"][0][
    "metrics"
][1]
print(json.dumps(results, indent=4))

# uncomment to copy report and view
#!aws s3 cp s3://{bucket}/{prefix}/clarify-output/bias_1/report.pdf ./clarify_output

if "mpg_name" not in locals():
    mpg_name = prefix
    print(f"Model Package Group name: {mpg_name}")

mpg_input_dict = {
    "ModelPackageGroupName": mpg_name,
    "ModelPackageGroupDescription": "Insurance claim fraud detection",
}

matching_mpg = sagemaker_boto_client.list_model_package_groups(NameContains=mpg_name)['ModelPackageGroupSummaryList']

if matching_mpg:
    print(f'Using existing Model Package Group: {mpg_name}')
else:
    mpg_response = sagemaker_boto_client.create_model_package_group(**mpg_input_dict)
    print(f'Create Model Package Group {mpg_name}: SUCCESSFUL')
    %store mpg_name

model_metrics_report = {"binary_classification_metrics": {}}
for metric in training_job_1_info["FinalMetricDataList"]:
    stat = {metric["MetricName"]: {"value": metric["Value"], "standard_deviation": "NaN"}}
    model_metrics_report["binary_classification_metrics"].update(stat)

with open("training_metrics.json", "w") as f:
    json.dump(model_metrics_report, f)

metrics_s3_key = (
    f"{prefix}/training_jobs/{training_job_1_info['TrainingJobName']}/training_metrics.json"
)
s3_client.upload_file(Filename="training_metrics.json", Bucket=bucket, Key=metrics_s3_key)

mp_inference_spec = InferenceSpecification().get_inference_specification_dict(
    ecr_image=training_job_1_info["AlgorithmSpecification"]["TrainingImage"],
    supports_gpu=False,
    supported_content_types=["text/csv"],
    supported_mime_types=["text/csv"],
)

mp_inference_spec["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] = training_job_1_info[
    "ModelArtifacts"
]["S3ModelArtifacts"]

model_metrics = {
    "ModelQuality": {
        "Statistics": {
            "ContentType": "application/json",
            "S3Uri": f"s3://{bucket}/{metrics_s3_key}",
        }
    },
    "Bias": {
        "Report": {
            "ContentType": "application/json",
            "S3Uri": f"{bias_report_1_output_path}/analysis.json",
        }
    },
}

mp_input_dict = {
    "ModelPackageGroupName": mpg_name,
    "ModelPackageDescription": "KNN classifier to detect insurance fraud.",
    "ModelApprovalStatus": "PendingManualApproval",
    "ModelMetrics": model_metrics,
}

mp_input_dict.update(mp_inference_spec)
mp1_response = sagemaker_boto_client.create_model_package(**mp_input_dict)

mp_info = sagemaker_boto_client.describe_model_package(
    ModelPackageName=mp1_response["ModelPackageArn"]
)
mp_status = mp_info["ModelPackageStatus"]

while mp_status not in ["Completed", "Failed"]:
    time.sleep(5)
    mp_info = sagemaker_boto_client.describe_model_package(
        ModelPackageName=mp1_response["ModelPackageArn"]
    )
    mp_status = mp_info["ModelPackageStatus"]
    print(f"model package status: {mp_status}")
print(f"model package status: {mp_status}")

sagemaker_boto_client.list_model_packages(ModelPackageGroupName=mpg_name)["ModelPackageSummaryList"]