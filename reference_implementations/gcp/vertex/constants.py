from utils import load_tfvars, get_project_number


TFVARS_PATH = "../architectures/terraform.tfvars"
TFVARS = load_tfvars(TFVARS_PATH)

DOCKER_REPO_NAME = "kiwi-rf-charger-pred-v1-docker-repo"
DOCKER_IMAGE_NAME = "kiwi-rf-charger-pred"
PROJECT_NUMBER = get_project_number(TFVARS["project"])
