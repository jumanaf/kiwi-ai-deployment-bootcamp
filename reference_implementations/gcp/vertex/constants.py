from utils import load_tfvars, get_project_number


TFVARS_PATH = "../architectures/terraform.tfvars"
TFVARS = load_tfvars(TFVARS_PATH)

DOCKER_REPO_NAME = "kiwi-deployment-bootcamp-docker-repo"
DOCKER_IMAGE_NAME = "kiwi-deployment-bootcamp-rf-station-predictor"
