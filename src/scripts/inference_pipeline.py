from sagemaker.workflow.parameters import ParameterString
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.model import Model
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.transformer import Transformer
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.inputs import TransformInput
from sagemaker.workflow.steps import TransformStep
import sagemaker
import boto3
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker import get_execution_role
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
import os
from sagemaker.model_monitor.dataset_format import DatasetFormat

sm_client = boto3.client(service_name="sagemaker")
region = os.environ['AWS_DEFAULT_REGION']
role = os.environ['IAM_ROLE_NAME']
# testbucket = os.environ['AWS_TEST_BUCKET']
testbucket="sagemaker-pipeline-githubactions"
prefix="inference-data"
sagemaker_session = sagemaker.Session()
# default_bucket = sagemaker_session.default_bucket()
batch_data_uri = "s3://sagemaker-pipeline-githubactions/batch-data-folder/"
batch_data = ParameterString(
    name="BatchData",
    default_value=batch_data_uri,
)
model_package_group_name = 'github-Churn-xgboost-model-grp-1'
lambda_function_name = "get-latest-version"
pipeline_name = "inference-pipeline"

def get_pipeline_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )
pipeline_session = get_pipeline_session(region, testbucket)

# Where the input data is stored
input_data = ParameterString(
    name="InputData",
    default_value=batch_data_uri,
)

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables

print("=========starting get latest model step=========================")
# Lambda helper class can be used to create the Lambda function
# func = Lambda(
#     function_name=lambda_function_name,
#     execution_role_arn=role,
#     script="scripts/lambda_step_code.py",
#     handler="lambda_step_code.handler",
#     timeout=600,
#     memory_size=128,
# )

# step_latest_model_fetch = LambdaStep(
#     name="fetchLatestModel",
#     lambda_func=func,
#     inputs={
#         "model_package_group_name": model_package_group_name,
#     },
#     outputs=[
#         LambdaOutput(output_name="ModelUrl", output_type=LambdaOutputTypeEnum.String), 
#         LambdaOutput(output_name="ImageUri", output_type=LambdaOutputTypeEnum.String), 
#         LambdaOutput(output_name="BaselineStatisticsS3Uri", output_type=LambdaOutputTypeEnum.String), 
#         LambdaOutput(output_name="BaselineConstraintsS3Uri", output_type=LambdaOutputTypeEnum.String), 
#     ],
# )
from utils import get_approved_package
sm_client = boto3.client("sagemaker")

pck = get_approved_package(
    model_package_group_name
)  # Reminder: model_package_group_name was defined as "NominetAbaloneModelPackageGroupName" at the beginning of the pipeline definition
model_description = sm_client.describe_model_package(ModelPackageName=pck["ModelPackageArn"])

# model_description
model_package_arn = model_description["ModelPackageArn"]
print(model_package_arn)
model_url=model_description['InferenceSpecification']['Containers'][0]['ModelDataUrl']
print("model_url:",model_url)
#deploying the model
from sagemaker import ModelPackage
model = ModelPackage(
    role=role, model_package_arn=model_package_arn, sagemaker_session=sagemaker_session
)
import time
from time import gmtime, strftime
endpoint_name = "pipeline-endpoint-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print("EndpointName= {}".format(endpoint_name))
# model.deploy(initial_instance_count=1, instance_type="ml.m5.xlarge", endpoint_name=endpoint_name)







print("=====================Starting the processing step===========================")


sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    base_job_name="inference-churn-processing-job",
)

step_process = ProcessingStep(
    name="LoadInferenceData",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(source=batch_data, destination="/opt/ml/processing/batchinput"),
    ],
    outputs=[
        ProcessingOutput(
            output_name="inference",
            source="/opt/ml/processing/batchtest",
            destination=Join(
                on="/",
                values=[
                    "s3://{}".format(testbucket),
                    prefix,
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    "batchtest",
                ],
            ),
        ),
    ],
    code="scripts/process.py",
)





















# step_process = ProcessingStep(
#     name="LoadInferenceData",
#     code="scripts/process.py",
#     processor=sklearn_processor,
#     outputs=[
#         ProcessingOutput(output_name="inference", source="/opt/ml/processing/inference")
#     ],
#     job_arguments = ['--train-test-split-ratio', '0.2', 
#     '--testbucket', testbucket
#     ]
# )

step_infer = ProcessingStep(
    name="Inference",
    code="scripts/score.py",
    processor=sklearn_processor,
    inputs=[
            ProcessingInput(
                source=model_url,
                destination="/opt/ml/processing/model",
                ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["inference"].S3Output.S3Uri, 
                destination="/opt/ml/processing/batchtest"
                )
    ],
    job_arguments = ['--testbucket', testbucket
    ]
)

pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        batch_data,
    ],
    steps=[step_process, step_infer],
)

import json
definition = json.loads(pipeline.definition())
pipeline.upsert(role_arn=role)


print("============================pipeline triggered=====================================")


# Execute pipeline using the default parameters.
execution = pipeline.start()

execution.wait()

# # List the execution steps to check out the status and artifacts:
execution.list_steps()
print("============================pipeline execution completed=====================================")