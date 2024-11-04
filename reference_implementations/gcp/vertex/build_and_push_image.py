import logging
from google.cloud.aiplatform.prediction import LocalModel
from predictor.predictor import RFStationPredictor
from constants import TFVARS, DOCKER_REPO_NAME, DOCKER_IMAGE_NAME
import os

os.environ["DOCKER_DEFAULT_PLATFORM"] = "linux/amd64"
# This file builds and pushes the Docker image for the custom predictor to the Google Cloud Registry, and 
# primarily relies on the vertex/predictor/predictor.py file to load in the model
# and create the custom predictor.

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s")

# Build the Docker image for the custom predictor, using the repo set up in the CLI
local_model = LocalModel.build_cpr_model(
    "./predictor",
    f"{TFVARS['region']}-docker.pkg.dev/{TFVARS['project']}/{DOCKER_REPO_NAME}/{DOCKER_IMAGE_NAME}:latest",
    predictor=RFStationPredictor,
    requirements_path="./predictor/requirements.txt",
    base_image="--platform=linux/amd64 alvarobartt/torch-gpu:py310-cu12.3-torch-2.2.0 AS build",
)

# Push the Docker image to the Google Cloud Registry
local_model.push_image()
