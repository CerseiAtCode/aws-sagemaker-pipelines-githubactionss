import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor
import json
from sagemaker.s3 import S3Downloader
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel
import os 
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ClarifyCheckConfig,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ModelExplainabilityCheckConfig,
    SHAPConfig
)
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig
)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep
)
from sagemaker.workflow.functions import (
    JsonGet
)
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model_metrics import MetricsSource, ModelMetrics 
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.drift_check_baselines import DriftCheckBaselines
from time import gmtime, strftime
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.steps import CreateModelStep
import io
import os
import pandas as pd
import sys
import time
from time import gmtime, strftime, sleep
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.inputs import FeatureParameter

session = sagemaker.session.Session()
region = os.environ['AWS_DEFAULT_REGION']
role = os.environ['IAM_ROLE_NAME']
# bucket = os.environ['BUCKET_NAME']
# prefix = os.environ['PREFIX']
bucket="sagemaker-pipeline-githubactions"
prefix="pipeline_with_featurestore"
model_package_group_name = "github-Churn-xgboost-model-grp-1"  # Model name in model registry

pipeline_name = "githubChurnPipeline-1"
print(region)
print(role)
print(bucket)
print(prefix)

current_working_directory = os.getcwd()
print(current_working_directory)

# # Upload the csv files to S3
# large_input_data_uri = session.upload_data(path="scripts/data/large/churn-dataset.csv",bucket=bucket, key_prefix=prefix + "/data/large")
# small_input_data_uri = session.upload_data(path="scripts/data/small/churn-dataset.csv",bucket=bucket, key_prefix=prefix + "/data/small")
# test_data_uri = session.upload_data(path="scripts/data/test.csv",bucket=bucket, key_prefix=prefix + "/data/test")

# print("Large data set uploaded to ", large_input_data_uri)
# print("Small data set uploaded to ", small_input_data_uri)
# print("Test data set uploaded to ", test_data_uri)
print("============================uploading the data to s3 from local completed=====================================")
# large_input_data_uri = "s3://sagemaker-pipeline-githubactions/pipeline-experiment1/churn-dataset.csv"
# small_input_data_uri= "s3://sagemaker-pipeline-githubactions/pipeline-experiment1/churn-large-dataset.csv"
# test_data_uri="s3://sagemaker-pipeline-githubactions/pipeline-experiment1/test.csv"

# #Reading the csv from S3
# s3_client = boto3.client("s3", region_name=region)
# churn_file_key = (f"{prefix}/data/large/churn-dataset.csv")
# churn_data_object = s3_client.get_object( Bucket=bucket, Key=churn_file_key)
# churn_data = pd.read_csv(io.BytesIO(churn_data_object["Body"].read()))
# churn_data.info()

# # Creating the feature store from the csv file
# new_col_names={}
# for i in churn_data.columns:
#     print(i.replace(" ","_"))
#     new_col_names[i]=i.replace(" ","_")
# print(new_col_names) 

# churn_data.rename(columns = new_col_names, inplace = True) 
# churn_data.rename(columns = {"Int'l_Plan":"Intl_Plan"}, inplace = True) 
# churn_data.rename(columns = {"Churn?":"Churn"}, inplace = True) 
# print(churn_data.columns)
# churn_feature_group_name = "churn-feature-group-" + strftime("%d-%H-%M-%S", gmtime())
# churn_feature_group = FeatureGroup(
#     name=churn_feature_group_name, sagemaker_session=session
# )
# current_time_sec = int(round(time.time()))
# record_identifier_feature_name = "Phone"
# churn_data["EventTime"] = pd.Series([current_time_sec] * len(churn_data), dtype="float64")

# churn_feature_group.load_feature_definitions(data_frame=churn_data)
# churn_feature_group.create(
#     s3_uri=f"s3://{bucket}/{prefix}/feature_store",
#     record_identifier_name=record_identifier_feature_name,
#     event_time_feature_name="EventTime",
#     role_arn=role,
#     enable_online_store=True,
# )
# #churn_feature_group.describe()

# def check_feature_group_status(feature_group):
#     status = feature_group.describe().get("FeatureGroupStatus")
#     while status == "Creating":
#         print("Waiting for Feature Group to be Created")
#         time.sleep(5)
#         status = feature_group.describe().get("FeatureGroupStatus")
#     print(f"FeatureGroup {feature_group.name} successfully created.")


# check_feature_group_status(churn_feature_group)
# # Ingesting data in the feature store
# churn_feature_group.update_feature_metadata(
#     feature_name="Phone",
#     description="The phone of a customer. It is also used in orders_feature_group.",
#     parameter_additions=[FeatureParameter("idType", "primaryKey")],
# )
# churn_feature_group.ingest(data_frame=churn_data, max_workers=3, wait=True)


# #Reading data from feature store
# feature_query = churn_feature_group.athena_query()
# churn_table_name=feature_query.table_name
# print(churn_table_name)
# query_string=f"SELECT * FROM \"sagemaker_featurestore\".\"{churn_table_name}\""
# print(len(query_string))

# feature_query.run(query_string=query_string, output_location='s3://'+bucket+'/query_results/')
# feature_query.wait()
# dataset = feature_query.as_dataframe()
# dataset.head()

#Get the dataset from the feature Store
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.inputs import FeatureParameter



# churn_feature_group_name="churn-feature-group-03-09-18-09"
# churn_feature_group = FeatureGroup(
#     name=churn_feature_group_name, sagemaker_session=session
# )

# feature_query = churn_feature_group.athena_query()
# churn_table_name="churn_feature_group_03_09_18_09_1714727890"
# print(churn_table_name)
# query_string=f"SELECT * FROM \"sagemaker_featurestore\".\"{churn_table_name}\""
# # print(len(query_string))
# feature_query.run(query_string=query_string, output_location='s3://'+bucket+'/query_results/')
# feature_query.wait()
# dataset = feature_query.as_dataframe()
# print(dataset.info())


churn_feature_group_name="churndata-feature-group-06-10-45-01"
churn_feature_group = FeatureGroup(
    name=churn_feature_group_name, sagemaker_session=session
)

feature_query = churn_feature_group.athena_query()
churn_table_name="churndata_feature_group_06_10_45_01_1714992445"
print(churn_table_name)
query_string=f"SELECT * FROM \"sagemaker_featurestore\".\"{churn_table_name}\""
# print(len(query_string))
feature_query.run(query_string=query_string, output_location='s3://'+bucket+'/query_results/')
feature_query.wait()
dataset = feature_query.as_dataframe()
print(dataset.head())




#save the dataset to data folder
dataset[:300:].to_csv("scripts/data/large/churn-dataset.csv",index=False)
dataset[301:400:].to_csv("scripts/data/small/churn-dataset.csv",index=False)
dataset[401::].to_csv("scripts/data/test.csv",index=False)
# Upload the csv files to S3
large_input_data_uri = session.upload_data(path="scripts/data/large/churn-dataset.csv",bucket=bucket, key_prefix=prefix + "/data/large")
small_input_data_uri = session.upload_data(path="scripts/data/small/churn-dataset.csv",bucket=bucket, key_prefix=prefix + "/data/small")
test_data_uri = session.upload_data(path="scripts/data/test.csv",bucket=bucket, key_prefix=prefix + "/data/test")

print("Large data set uploaded to ", large_input_data_uri)
print("Small data set uploaded to ", small_input_data_uri)
print("Test data set uploaded to ", test_data_uri)

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)

# How many instances to use when processing
processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)

# What instance type to use for processing
processing_instance_type = ParameterString(
    name="ProcessingInstanceType", default_value="ml.m5.large"
)

# What instance type to use for training
training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")

# Where the input data is stored
input_data = ParameterString(
    name="InputData",
    default_value=small_input_data_uri,
)

# What is the default status of the model when registering with model registry.
model_approval_status = ParameterString(
    name="ModelApprovalStatus", default_value="PendingManualApproval"
)



from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables

# Create SKlearn processor object,
# The object contains information about what instance type to use, the IAM role to use etc.
# A managed processor comes with a preconfigured container, so only specifying version is required.
sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    base_job_name="churn-processing-job",
)

# Use the sklearn_processor in a Sagemaker pipelines ProcessingStep
step_preprocess_data = ProcessingStep(
    name="Preprocess-Churn-Data",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/train",
            destination=Join(
                on="/",
                values=[
                    "s3://{}".format(bucket),
                    prefix,
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    "train",
                ],
            ),
        ),
        ProcessingOutput(
            output_name="validation",
            source="/opt/ml/processing/validation",
            destination=Join(
                on="/",
                values=[
                    "s3://{}".format(bucket),
                    prefix,
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    "validation",
                ],
            ),
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/test",
            destination=Join(
                on="/",
                values=[
                    "s3://{}".format(bucket),
                    prefix,
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    "test",
                ],
            ),
        ),
    ],
    code="scripts/preprocess.py",
)


#================================================train==================================

from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.estimator import Estimator
model_path=f"s3://{bucket}/{prefix}/churnmodel"
# Fetch container to use for training
image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.2-2",
    py_version="py3",
    instance_type="ml.m5.xlarge",
)

# Create XGBoost estimator object
# The object contains information about what container to use, what instance type etc.
xgb_estimator = Estimator(
    image_uri=image_uri,
    instance_type=training_instance_type,
    instance_count=1,
    role=role,
    disable_profiler=True,
    output_path=model_path

)

xgb_estimator.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    objective="binary:logistic",
    num_round=25,
)

# Use the xgb_estimator in a Sagemaker pipelines ProcessingStep.
# NOTE how the input to the training job directly references the output of the previous step.
step_train_model = TrainingStep(
    name="Train-Churn-Model",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
        "validation": TrainingInput(
            s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                "validation"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
    },
)

#===================================================evaluate====================================================

from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.properties import PropertyFile

# Create ScriptProcessor object.
# The object contains information about what container to use, what instance type etc.
evaluate_model_processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    instance_type=processing_instance_type,
    instance_count=processing_instance_count,
    base_job_name="script-churn-eval",
    role=role,
)

# Create a PropertyFile
# A PropertyFile is used to be able to reference outputs from a processing step, for instance to use in a condition step.
# For more information, visit https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html
evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
print(f"bucket uri before evaluate step: s3://{bucket}/{prefix}")
# Use the evaluate_model_processor in a Sagemaker pipelines ProcessingStep.
step_evaluate_model = ProcessingStep(
    name="Evaluate-Churn-Model",
    processor=evaluate_model_processor,
    inputs=[
        ProcessingInput(
            source=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source=test_data_uri,  # Use pre-created test data instead of output from processing step
            destination="/opt/ml/processing/test",
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=Join(
                on="/",
                values=[
                    "s3://{}".format(bucket),
                    "modeloutput",
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    "evaluation-report",
                ],
            ),
        ),
    ],
    code="scripts/evaluate.py",
    property_files=[evaluation_report],
)


#==================================== Register model==================================
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel

# Create ModelMetrics object using the evaluation report from the evaluation step
# A ModelMetrics object contains metrics captured from a model.
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(
            on="/",
            values=[
                step_evaluate_model.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                    "S3Uri"
                ],
                "evaluation.json",
            ],
        ),
        content_type="application/json",
    )
)

# Crete a RegisterModel step, which registers the model with Sagemaker Model Registry.
step_register_model = RegisterModel(
    name="Register-Churn-Model",
    estimator=xgb_estimator,
    model_data=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge", "ml.m5.large"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name=model_package_group_name,
    approval_status=model_approval_status,
    model_metrics=model_metrics,
)

#================================Condition Step====================================
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

# Create accuracy condition to ensure the model meets performance requirements.
# Models with a test accuracy lower than the condition will not be registered with the model registry.
cond_gte = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=step_evaluate_model.name,
        property_file=evaluation_report,
        json_path="binary_classification_metrics.accuracy.value",
    ),
    right=0.7,
)

# Create a Sagemaker Pipelines ConditionStep, using the condition above.
# Enter the steps to perform if the condition returns True / False.
step_cond = ConditionStep(
    name="Accuracy-Condition",
    conditions=[cond_gte],
    if_steps=[step_register_model],
    else_steps=[],
)
print("============================condition tested=====================================")
print("Condition step")
print(step_cond)
#==================================pipeline==========================================
from sagemaker.workflow.pipeline import Pipeline

# Create a Sagemaker Pipeline.
# Each parameter for the pipeline must be set as a parameter explicitly when the pipeline is created.
# Also pass in each of the steps created above.
# Note that the order of execution is determined from each step's dependencies on other steps,
# not on the order they are passed in below.
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        processing_instance_type,
        processing_instance_count,
        training_instance_type,
        model_approval_status,
        input_data,
    ],
    steps=[step_preprocess_data, step_train_model, step_evaluate_model, step_cond],
)

print("============================pipeline triggered=====================================")
# Submit pipline
# pipeline.upsert(role_arn=role)

# Execute pipeline using the default parameters.
# execution = pipeline.start()

# execution.wait()

# List the execution steps to check out the status and artifacts:
# execution.list_steps()
print("============================pipeline execution completed=====================================")

























