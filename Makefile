.PHONY: install test selftest demo build clean
install:
	pip install -e ".[dev,audio]"
test:
	pytest -q
selftest:
	python -m sampire --selftest
demo:
	python -m sampire --demo
build:
	python -m build
clean:
	rm -rf build dist *.egg-info .pytest_cache
