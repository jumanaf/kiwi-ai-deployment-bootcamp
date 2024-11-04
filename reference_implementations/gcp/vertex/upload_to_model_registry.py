import logging
import sys

from google.cloud import aiplatform, bigquery
from vertexai.resources.preview import ml_monitoring

from constants import TFVARS, TFVARS_PATH, DOCKER_REPO_NAME, DOCKER_IMAGE_NAME, PROJECT_NUMBER
from utils import save_tfvars


model_id = sys.argv[1] if len(sys.argv) > 1 else None
model_version = sys.argv[2] if len(sys.argv) > 2 else "default"

model_name = "rf-charger-pred"
model_version = "v1"

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s")

aiplatform.init(project=TFVARS["project"], location=TFVARS["region"])


if model_id is not None:
    model = aiplatform.Model(f"projects/{PROJECT_NUMBER}/locations/{TFVARS['region']}/models/{model_id}@{model_version}")
else:
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=f"gs://{TFVARS['project']}-model/{model_name}/{model_version}",
        serving_container_image_uri=f"{TFVARS['region']}-docker.pkg.dev/{TFVARS['project']}/{DOCKER_REPO_NAME}/{DOCKER_IMAGE_NAME}",
    )

print(f"Model uploaded: {model.resource_name}")