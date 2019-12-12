IMAGE_NAME=reportportal/service-auto-analyzer$(IMAGE_POSTFIX)
IMAGE_NAME_DEV=reportportal-dev-5-1/service-auto-analyzer
IMAGE_NAME_TEST=reportportal/service-auto-analyzer_test

VENV_NAME?=venv
PYTHON=/${VENV_NAME}/bin/python3
BUILD_DEPS:= github.com/avarabyeu/releaser
GO = go

.PHONY: build-release build-image-dev build-image pushDev venv test checkstyle test-all build-image-test run-test

venv: 
	touch /$(VENV_NAME)/bin/activate

test: venv
	${PYTHON} -m unittest

checkstyle: venv
	${PYTHON} -m flake8

release:
	$(eval v := $(or $(v),$(shell releaser bump)))
	# make sure latest version is bumped to file
	releaser bump --version ${v}

build-release: venv
	${PYTHON} -m bumpversion --new-version ${v} build

build-image-dev:
	docker build -t "$(IMAGE_NAME_DEV)" --build-arg version=${v} -f Dockerfile .

build-image:
	docker build -t "$(IMAGE_NAME)" --build-arg version=${v} -f Dockerfile .

build-image-test:
	docker build -t "$(IMAGE_NAME_TEST)" -f DockerfileTest .

run-test:
	docker run --rm "$(IMAGE_NAME_TEST)"

pushDev:
	echo "Registry is not provided"
	if [ -d ${REGISTRY} ] ; then echo "Provide registry"; exit 1 ; fi
	docker tag "$(IMAGE_NAME)" "$(REGISTRY)/$(IMAGE_NAME):latest"
	docker push "$(REGISTRY)/$(IMAGE_NAME):latest"

test-all: checkstyle test

