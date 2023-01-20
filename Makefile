export TARGET_DIR:=vim_colorscheme_generator tests

.PHONY: format
format:
	poetry run autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports --recursive ${TARGET_DIR}
	poetry run isort --verbose ${TARGET_DIR}
	poetry run black ${TARGET_DIR}

.PHONY: lint
lint:
	poetry run isort --check --diff ${TARGET_DIR}
	poetry run autoflake --recursive --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables ${TARGET_DIR}
	poetry run black --check ${TARGET_DIR}
	poetry run ruff ${TARGET_DIR}
	poetry run mypy ${TARGET_DIR}

.PHONY: test
test:
	poetry run pytest tests

data/vim_colorscheme_repo.json:
	scripts/generate_database.sh

.PHONY: clean_data
clean_data:
	rm -rf sample_data
