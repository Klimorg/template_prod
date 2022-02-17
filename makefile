# Makefile
.PHONY: help
help:
	@echo "Commands:"
	@echo "masks                 : generate the labels, ie segmentation masks, from the VGG json files."
	@echo "tilling               : generate tillings from the images and masks."
	@echo "datasets              : create train, test, & validation datasets from raw datas in raw_dataset directory."
	@echo "train                 : launch training loop for a given set of parameters from configs/params.yaml."
	@echo "install               : installs project requirements."
	@echo "install-dev           : installs development requirements."
	@echo "install-docs          : installs docs requirements."
	@echo "clean                 : cleans all unecessary files."
	@echo "docs                  : serve generated documentation from mkdocs."
	@echo "tests                 : run unit tests."
	@echo "mypy                  : run mypy in the src folder for type hinting checking."
	@echo "cc_report             : run radon in the src folder for code complexity report."
	@echo "raw_report            : run radon in the src folder for raw report."
	@echo "mi_report             : run radon in the src folder for maintainability index report."
	@echo "hal_report            : run radon in the src folder for hal report."
	@echo "install_precommit     : installs precommit."
	@echo "check_precommit       : check precommit."


# Datas and training
.PHONY: masks
masks:
	python src/utils/make_masks.py

.PHONY: tilling
tilling:
	python src/utils/utils_images.py

.PHONY: datasets
datasets:
	python src/utils/make_datasets.py

.PHONY: train
train:
	python src/train.py

# Installation
.PHONY: install
install:
	python -m pip install -e . --no-cache-dir

.PHONY: install-dev
install-dev:
	python -m pip install -e ".[dev]" --no-cache-dir
	pre-commit install
	pre-commit autoupdate

.PHONY: install-docs
install-docs:
	python -m pip install -e ".[docs]" --no-cache-dir

# Cleaning
.PHONY: clean
clean:
	bash shell/clean_pycache.sh ../template_segmentation
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".mypy_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E "htmlcov/*" | xargs rm -rf
	rm -f .coverage

# Documentation
.PHONY: docs
docs:
	mkdocs serve

# Tests
.PHONY: tests
tests:
	python -m pytest -v --cov

# Reporting
.PHONY: mypy
mypy:
	mypy --show-error-codes src/

.PHONY: cc_report
cc_report:
	radon cc src/

.PHONY: raw_report
raw_report:
	radon raw --summary src/

.PHONY: mi_report
mi_report:
	radon mi src/

.PHONY: hal_report
hal_report:
	radon hal src/

.PHONY: bandit
bandit:
	bandit -r src/


# Precommit
.PHONY: install_precommit
install_precommit:
	pre-commit install

.PHONY: check_precommit
check_precommit:
	pre-commit run --all
