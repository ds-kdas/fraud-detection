!python -m pip install -Uq pip
!python -m pip install -q awswrangler imbalanced-learn==0.7.0 sagemaker==2.23.0 boto3==1.17.70

import json
import time
import boto3
import sagemaker
import numpy as np
import pandas as pd
import awswrangler as wr
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sagemaker.xgboost.estimator import XGBoost

from model_package_src.inference_specification import InferenceSpecification

%matplotlib inline

# You can change this to a region of your choice
import sagemaker

region = sagemaker.Session().boto_region_name
print("Using AWS Region: {}".format(region))

boto3.setup_default_session(region_name=region)

boto_session = boto3.Session(region_name=region)

s3_client = boto3.client("s3", region_name=region)

sagemaker_boto_client = boto_session.client("sagemaker")

sagemaker_session = sagemaker.Session(
    boto_session=boto_session, sagemaker_client=sagemaker_boto_client
)

sagemaker_role = sagemaker.get_execution_role()

account_id = boto3.client("sts").get_caller_identity()["Account"]


bucket = sagemaker_session.default_bucket()
prefix = "fraud-detect-demo"

claims_fg_name = f"{prefix}-claims"
customers_fg_name = f"{prefix}-customers"

model_2_name = f"{prefix}-xgboost-post-smote"

train_data_upsampled_s3_path = f"s3://{bucket}/{prefix}/data/train/upsampled/train.pkl"
bias_report_2_output_path = f"s3://{bucket}/{prefix}/clarify-output/bias-2"
explainability_output_path = f"s3://{bucket}/{prefix}/clarify-output/explainability"

train_instance_count = 1
train_instance_type = "ml.m4.xlarge"

claify_instance_count = 1
clairfy_instance_type = "ml.c5.xlarge"

train_data_upsampled_s3_path

train = pd.read_pickle("data/train.pkl")
test = pd.read_pickle("data/test.pkl")

gender = train["customer_gender_female"]
gender.value_counts()

sm = SMOTE(random_state=42)
train_data_upsampled, gender_res = sm.fit_resample(train, gender)
train_data_upsampled["customer_gender_female"].value_counts()

hyperparameters = {
    "max_depth": "3",
    "eta": "0.2",
    "objective": "binary:logistic",
    "num_round": "100",
}

train_data_upsampled.to_pickle("data/upsampled_train.pkl", index=False)
s3_client.upload_file(
    Filename="data/upsampled_train.pkl",
    Bucket=bucket,
    Key=f"{prefix}/data/train/upsampled/train.pkl",
)

xgb_estimator = XGBoost(
    entry_point="xgboost_starter_script.py",
    hyperparameters=hyperparameters,
    role=sagemaker_role,
    instance_count=train_instance_count,
    instance_type=train_instance_type,
    framework_version="1.0-1",
)

if "training_job_2_name" not in locals():

    xgb_estimator.fit(inputs={"train": train_data_upsampled_s3_path})
    training_job_2_name = xgb_estimator.latest_training_job.job_name

else:

    print(f"Using previous training job: {training_job_2_name}")

training_job_2_info = sagemaker_boto_client.describe_training_job(
    TrainingJobName=training_job_2_name
)

# return any existing artifact which match the our training job's code arn
code_s3_uri = training_job_2_info["HyperParameters"]["sagemaker_submit_directory"]

list_response = list(
    sagemaker.lineage.artifact.Artifact.list(
        source_uri=code_s3_uri, sagemaker_session=sagemaker_session
    )
)

# use existing arifact if it's already been created, otherwise create a new artifact
if list_response:
    code_artifact = list_response[0]
    print(f"Using existing artifact: {code_artifact.artifact_arn}")
else:
    code_artifact = sagemaker.lineage.artifact.Artifact.create(
        artifact_name="TrainingScript",
        source_uri=code_s3_uri,
        artifact_type="Code",
        sagemaker_session=sagemaker_session,
    )
    print(f"Create artifact {code_artifact.artifact_arn}: SUCCESSFUL")

training_data_s3_uri = training_job_2_info["InputDataConfig"][0]["DataSource"]["S3DataSource"][
    "S3Uri"
]

list_response = list(
    sagemaker.lineage.artifact.Artifact.list(
        source_uri=training_data_s3_uri, sagemaker_session=sagemaker_session
    )
)

if list_response:
    training_data_artifact = list_response[0]
    print(f"Using existing artifact: {training_data_artifact.artifact_arn}")
else:
    training_data_artifact = sagemaker.lineage.artifact.Artifact.create(
        artifact_name="TrainingData",
        source_uri=training_data_s3_uri,
        artifact_type="Dataset",
        sagemaker_session=sagemaker_session,
    )
    print(f"Create artifact {training_data_artifact.artifact_arn}: SUCCESSFUL")

trained_model_s3_uri = training_job_2_info["ModelArtifacts"]["S3ModelArtifacts"]

list_response = list(
    sagemaker.lineage.artifact.Artifact.list(
        source_uri=trained_model_s3_uri, sagemaker_session=sagemaker_session
    )
)

if list_response:
    model_artifact = list_response[0]
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
    TrialComponentName=training_job_2_name + "-aws-training-job"
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
    print(f"Associate {trial_component_arn} and {a.artifact_arn}: SUCCEESFUL\n")

output_artifacts = [model_artifact]

for artifact_arn in output_artifacts:
    sagemaker.lineage.association.Association.create(
        source_arn=a.artifact_arn,
        destination_arn=trial_component_arn,
        association_type="Produced",
        sagemaker_session=sagemaker_session,
    )
    print(f"Associate {trial_component_arn} and {a.artifact_arn}: SUCCEESFUL\n")

model_matches = sagemaker_boto_client.list_models(NameContains=model_2_name)['Models']

if not model_matches:
    
    model_2 = sagemaker_session.create_model_from_job(
        name=model_2_name,
        training_job_name=training_job_2_info['TrainingJobName'],
        role=sagemaker_role,
        image_uri=training_job_2_info['AlgorithmSpecification']['TrainingImage'])
    %store model_2_name
    
else:
    
    print(f"Model {model_2_name} already exists.")

clarify_processor = sagemaker.clarify.SageMakerClarifyProcessor(
    role=sagemaker_role,
    instance_count=1,
    instance_type="ml.c4.xlarge",
    sagemaker_session=sagemaker_session,
)

bias_data_config = sagemaker.clarify.DataConfig(
    s3_data_input_path=train_data_upsampled_s3_path,
    s3_output_path=bias_report_2_output_path,
    label="fraud",
    headers=train.columns.to_list(),
    dataset_type="text/csv",
)

model_config = sagemaker.clarify.ModelConfig(
    model_name=model_2_name,
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

# # un-comment the code below to run the whole job

# if 'clarify_bias_job_2_name' not in locals():

#     clarify_processor.run_bias(
#         data_config=bias_data_config,
#         bias_config=bias_config,
#         model_config=model_config,
#         model_predicted_label_config=predictions_config,
#         pre_training_methods='all',
#         post_training_methods='all')

#     clarify_bias_job_2_name = clarify_processor.latest_job.name
#     %store clarify_bias_job_2_name

# else:
#     print(f'Clarify job {clarify_bias_job_2_name} has already run successfully.')

if "clarify_bias_job_2_name" in locals():
    s3_client.download_file(
        Bucket=bucket,
        Key=f"{prefix}/clarify-output/bias-2/analysis.json",
        Filename="clarify_output/bias_2/analysis.json",
    )
    print(f"Downloaded analysis from previous Clarify job: {clarify_bias_job_2_name}\n")
else:
    print(f"Loading pre-generated analysis file...\n")

with open("clarify_output/bias_1/analysis.json", "r") as f:
    bias_analysis = json.load(f)

results = bias_analysis["pre_training_bias_metrics"]["facets"]["customer_gender_female"][0][
    "metrics"
][1]
print(json.dumps(results, indent=4))

with open("clarify_output/bias_2/analysis.json", "r") as f:
    bias_analysis = json.load(f)

results = bias_analysis["pre_training_bias_metrics"]["facets"]["customer_gender_female"][0][
    "metrics"
][1]
print(json.dumps(results, indent=4))

model_config = sagemaker.clarify.ModelConfig(
    model_name=model_2_name,
    instance_type=train_instance_type,
    instance_count=1,
    accept_type="text/csv",
)

shap_config = sagemaker.clarify.SHAPConfig(
    baseline=[train.median().values[1:].tolist()], num_samples=100, agg_method="mean_abs"
)

explainability_data_config = sagemaker.clarify.DataConfig(
    s3_data_input_path=train_data_upsampled_s3_path,
    s3_output_path=explainability_output_path,
    label="fraud",
    headers=train.columns.to_list(),
    dataset_type="text/csv",
)

# un-comment the code below to run the whole job

# if "clarify_expl_job_name" not in locals():

#     clarify_processor.run_explainability(
#         data_config=explainability_data_config,
#         model_config=model_config,
#         explainability_config=shap_config)

#     clarify_expl_job_name = clarify_processor.latest_job.name
#     %store clarify_expl_job_name

# else:
#     print(f'Clarify job {clarify_expl_job_name} has already run successfully.')

if "clarify_expl_job_name" in locals():
    s3_client.download_file(
        Bucket=bucket,
        Key=f"{prefix}/clarify-output/explainability/analysis.json",
        Filename="clarify_output/explainability/analysis.json",
    )
    print(f"Downloaded analysis from previous Clarify job: {clarify_expl_job_name}\n")
else:
    print(f"Loading pre-generated analysis file...\n")

with open("clarify_output/explainability/analysis.json", "r") as f:
    analysis_result = json.load(f)

shap_values = pd.DataFrame(analysis_result["explanations"]["kernel_shap"]["label0"])
importances = shap_values["global_shap_values"].sort_values(ascending=False)
fig, ax = plt.subplots()
n = 5
y_pos = np.arange(n)
importance_scores = importances.values[:n]
y_label = importances.index[:n]
ax.barh(y_pos, importance_scores, align="center")
ax.set_yticks(y_pos)
ax.set_yticklabels(y_label)
ax.invert_yaxis()
ax.set_xlabel("SHAP Value (impact on model output)");

from IPython.display import FileLink, FileLinks

display(
    "Click link below to view the SageMaker Clarify report", FileLink("clarify_output/report.pdf")
)

model_metrics_report = {"binary_classification_metrics": {}}
for metric in training_job_2_info["FinalMetricDataList"]:
    stat = {metric["MetricName"]: {"value": metric["Value"], "standard_deviation": "NaN"}}
    model_metrics_report["binary_classification_metrics"].update(stat)

with open("training_metrics.json", "w") as f:
    json.dump(model_metrics_report, f)

metrics_s3_key = (
    f"{prefix}/training_jobs/{training_job_2_info['TrainingJobName']}/training_metrics.json"
)
s3_client.upload_file(Filename="training_metrics.json", Bucket=bucket, Key=metrics_s3_key)

mp_inference_spec = InferenceSpecification().get_inference_specification_dict(
    ecr_image=training_job_2_info["AlgorithmSpecification"]["TrainingImage"],
    supports_gpu=False,
    supported_content_types=["text/csv"],
    supported_mime_types=["text/csv"],
)

mp_inference_spec["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] = training_job_2_info[
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
            "S3Uri": f"{explainability_output_path}/analysis.json",
        }
    },
}

mpg_name = prefix
mp_input_dict = {
    "ModelPackageGroupName": mpg_name,
    "ModelPackageDescription": "XGBoost classifier to detect insurance fraud with SMOTE.",
    "ModelApprovalStatus": "PendingManualApproval",
    "ModelMetrics": model_metrics,
}

mp_input_dict.update(mp_inference_spec)
mp2_response = sagemaker_boto_client.create_model_package(**mp_input_dict)
mp2_arn = mp2_response["ModelPackageArn"]

mp_info = sagemaker_boto_client.describe_model_package(
    ModelPackageName=mp2_response["ModelPackageArn"]
)
mp_status = mp_info["ModelPackageStatus"]

while mp_status not in ["Completed", "Failed"]:
    time.sleep(5)
    mp_info = sagemaker_boto_client.describe_model_package(
        ModelPackageName=mp2_response["ModelPackageArn"]
    )
    mp_status = mp_info["ModelPackageStatus"]
    print(f"model package status: {mp_status}")
print(f"model package status: {mp_status}")

sagemaker_boto_client.list_model_packages(ModelPackageGroupName=mpg_name)["ModelPackageSummaryList"]

# variables used for parameterizing the notebook run
endpoint_name = f"{model_2_name}-endpoint"
endpoint_instance_count = 1
endpoint_instance_type = "ml.m4.xlarge"

predictor_instance_count = 1
predictor_instance_type = "ml.c5.xlarge"
batch_transform_instance_count = 1
batch_transform_instance_type = "ml.c5.xlarge"

second_model_package = sagemaker_boto_client.list_model_packages(ModelPackageGroupName=mpg_name)[
    "ModelPackageSummaryList"
][0]
model_package_update = {
    "ModelPackageArn": second_model_package["ModelPackageArn"],
    "ModelApprovalStatus": "Approved",
}

update_response = sagemaker_boto_client.update_model_package(**model_package_update)

primary_container = {"ModelPackageName": second_model_package["ModelPackageArn"]}
endpoint_config_name = f"{model_2_name}-endpoint-config"
existing_configs = len(
    sagemaker_boto_client.list_endpoint_configs(NameContains=endpoint_config_name, MaxResults=30)[
        "EndpointConfigs"
    ]
)

if existing_configs == 0:
    create_ep_config_response = sagemaker_boto_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": endpoint_instance_type,
                "InitialVariantWeight": 1,
                "InitialInstanceCount": endpoint_instance_count,
                "ModelName": model_2_name,
                "VariantName": "AllTraffic",
            }
        ],
    )

existing_endpoints = sagemaker_boto_client.list_endpoints(
    NameContains=endpoint_name, MaxResults=30
)["Endpoints"]
if not existing_endpoints:
    create_endpoint_response = sagemaker_boto_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )

endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=endpoint_name)
endpoint_status = endpoint_info["EndpointStatus"]

while endpoint_status == "Creating":
    endpoint_info = sagemaker_boto_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_status = endpoint_info["EndpointStatus"]
    print("Endpoint status:", endpoint_status)
    if endpoint_status == "Creating":
        time.sleep(60)

predictor = sagemaker.predictor.Predictor(
    endpoint_name=endpoint_name, sagemaker_session=sagemaker_session
)

dataset = pd.read_pickle("data/dataset.pkl")
train = dataset.sample(frac=0.8, random_state=0)
test = dataset.drop(train.index)
sample_policy_id = int(test.sample(1)["policy_id"])

test.info()

dataset = pd.read_pickle("./data/claims_customer.pkl")
col_order = ["fraud"] + list(dataset.drop(["fraud", "Unnamed: 0", "policy_id"], axis=1).columns)
col_order

col_order

sample_policy_id = int(test.sample(1)["policy_id"])
pull_from_feature_store = False

if pull_from_feature_store:
    customers_response = featurestore_runtime.get_record(
        FeatureGroupName=customers_fg_name, RecordIdentifierValueAsString=str(sample_policy_id)
    )

    customer_record = customers_response["Record"]
    customer_df = pd.DataFrame(customer_record).set_index("FeatureName")

    claims_response = featurestore_runtime.get_record(
        FeatureGroupName=claims_fg_name, RecordIdentifierValueAsString=str(sample_policy_id)
    )

    claims_record = claims_response["Record"]
    claims_df = pd.DataFrame(claims_record).set_index("FeatureName")

    blended_df = pd.concat([claims_df, customer_df]).loc[col_order].drop("fraud")
else:
    customer_claim_df = dataset[dataset["policy_id"] == sample_policy_id].sample(1)
    blended_df = customer_claim_df.loc[:, col_order].drop("fraud", axis=1).T.reset_index()
    blended_df.columns = ["FeatureName", "ValueAsString"]

data_input = ",".join([str(x) for x in blended_df["ValueAsString"]])
data_input

results = predictor.predict(data_input, initial_args={"ContentType": "text/csv"})
prediction = json.loads(results)
print(f"Probablitity the claim from policy {int(sample_policy_id)} is fraudulent:", prediction)

