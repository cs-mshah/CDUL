
help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean: ## Clean autogenerated files
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

clean-logs: ## Clean logs
	rm -rf logs/**

sync: ## Merge changes from main branch to your current branch
	git pull
	git pull origin main

train: ## Train the model
	python src/train.py

evaluate: ## run evaluate script
	python src/evaluate.py

clip_cache: ## create clip cache
	python src/clip_cache.py

voc2012: ## download the pascal voc 2012 dataset
	python src/data/voc.py