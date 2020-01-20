IMAGE_NAME=reportportal/service-auto-analyzer$(IMAGE_POSTFIX)
IMAGE_NAME_DEV=reportportal-dev-5-1/service-auto-analyzer
IMAGE_NAME_TEST=reportportal/service-auto-analyzer_test

VENV_NAME?=venv
PYTHON=/${VENV_NAME}/bin/python3

.PHONY: build-release build-image-dev build-image venv test checkstyle test-all build-image-test run-test

$(VENV_NAME)/bin/activate: requirements.txt
	test -d $(VENV_NAME) || virtualenv -p python3 $(VENV_NAME)
	$(VENV_NAME)/bin/pip install --no-cache-dir -r requirements.txt
	touch $(VENV_NAME)/bin/activate

venv: 
	touch /$(VENV_NAME)/bin/activate

test: venv
	${PYTHON} -m unittest

checkstyle: venv
	${PYTHON} -m flake8

release: $(VENV_NAME)/bin/activate
	${PYTHON} -m bumpversion --new-version ${v} build
	${PYTHON} -m bumpversion patch --no-tag
	git push origin master --tags

build-release: venv
	${PYTHON} -m bumpversion --new-version ${v} build

build-image-dev:
	docker build -t "$(IMAGE_NAME_DEV)" --build-arg version=${v} --build-arg prod="false" -f Dockerfile .

build-image:
	docker build -t "$(IMAGE_NAME)" --build-arg version=${v} --build-arg prod="true" -f Dockerfile .

build-image-test:
	docker build -t "$(IMAGE_NAME_TEST)" -f DockerfileTest .

run-test:
	docker run --rm "$(IMAGE_NAME_TEST)"

test-all: checkstyle test

