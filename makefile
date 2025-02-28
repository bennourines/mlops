# Define variables
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
MLFLOW_BACKEND_URI = sqlite:///mlflow.db

# CI/CD Quality Tools
LINTERS = pylint flake8 mypy bandit
FORMATTER = black
TESTS = pytest

# =====================
#  🆘 HELP
# =====================
help:  ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	/^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# =====================
#  🔧 SETUP & INSTALLATION
# =====================
install: ## Install dependencies and setup virtual environment
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# =====================
#  🔍 CODE QUALITY & SECURITY
# =====================
lint: ## Run static code analysis tools
	$(PYTHON) -m pylint **/*.py
	$(PYTHON) -m flake8 .
	$(PYTHON) -m mypy .
	$(PYTHON) -m bandit -r .

format: ## Format code using Black
	$(PYTHON) -m black .

quality-check: lint format ## Run all code quality checks

# =====================
#  🧪 TESTING
# =====================
unit-test: ## Run unit tests
	$(PYTHON) -m pytest tests/unit

functional-test: ## Run functional tests
	$(PYTHON) -m pytest tests/functional

test: unit-test functional-test ## Run all tests

# =====================
#  🤖 ML PIPELINE
# =====================
# =====================
#  🤖 ML PIPELINE
# =====================
data: ## Run data processing pipeline
	$(PYTHON) pipelines/data_pipeline.py

train: ## Train model with current data
	$(PYTHON) pipelines/training_pipeline.py

evaluate: ## Evaluate model performance
	$(PYTHON) pipelines/evaluation_pipeline.py

serve: ## Serve model via FastAPI
	$(PYTHON) -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui: ## Launch MLflow tracking UI
	mlflow ui --backend-store-uri $(MLFLOW_BACKEND_URI) --host 0.0.0.0 --port 5000 &

# =====================
#  📦 DOCKER
# =====================
docker-build: ## Build the Docker image
	docker build -t ines253/ines_bennour_mlops .

docker-run: ## Run the Docker container locally
	docker run -p 8000:8000 ines253/ines_bennour_mlops
	

docker-push: ## Push the Docker image to Docker Hub
	docker push ines253/ines_bennour_mlops
	

# =====================
#  🔄 AUTOMATION
# =====================
watch: ## Watch for file changes and trigger pipeline
	$(PYTHON) -m watchmedo shell-command --patterns="*.py" --command='make ci-cd' .

# =====================
#  🚀 FULL PIPELINE
# =====================
ci-cd: install quality-check test data train evaluate ## Run full CI/CD pipeline

all: ci-cd serve ## Run complete pipeline and serve model
