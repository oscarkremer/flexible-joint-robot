#################################################################################
# GLOBALS                                                                       #
#################################################################################
.DEFAULT_GOAL := check
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = flexible-robot
PYTHON_INTERPRETER = python3

################################################################################
# COMMANDS                                                                     #
# TO SHOW OUTPUT USE LOGGER=stdout                                             #
################################################################################


dirs:
	@echo "---> Creating data dirs"
	@mkdir -p data/client
	@mkdir -p data/images/train
	@mkdir -p data/images/test
	@mkdir -p data/images/evaluation
	@mkdir -p data/knowledge/network_1
	@mkdir -p data/knowledge/network_2
	@mkdir -p data/knowledge/regressor
	@mkdir -p data/processed
	@mkdir -p data/test
	@mkdir -p data/logs
	@echo "---> Done"

clean-database:
	@echo "---> Execute drop-all and create-all"
	@$(PYTHON_INTERPRETER) src/api/clean_database.py


setup: check_environment
	@echo "---> Running setup.."
	@conda env create -f environment.yml
	@cp -n .env.example .env
	@echo "---> To complete setup please run \n---> source activate cerberus"


install:
	@echo "---> Installing dependencies"
	@conda env update -f environment.yml


clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


download: dirs
	aws s3 cp s3://thrive-cea/processed/sales.csv data/processed/ --quiet
	aws s3 cp s3://thrive-cea/processed/stock.csv data/processed/ --quiet


lint:
	flake8 src


check_environment:
	@echo "---> Checking environment.."
	$(PYTHON_INTERPRETER) test_environment.py


autocorrect:
	@echo "---> Processing autocorrect"
	@autopep8 --in-place --aggressive --aggressive --global-config .flake8 $(shell find . -name '*.py')


console:
	@$(PYTHON_INTERPRETER)
