IMAGE_NAME=reportportal/service-analyzer$(IMAGE_POSTFIX)
IMAGE_NAME_DEV=reportportal-dev-5/service-analyzer

VENV_NAME?=venv
VENV_ACTIVATE=. $(VENV_NAME)/bin/activate
PYTHON=${VENV_NAME}/bin/python3

.PHONY: build-release build-image-dev build-image pushDev venv test

venv: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: requirements.txt
	test -d $(VENV_NAME) || virtualenv -p python3 $(VENV_NAME)
	$(VENV_NAME)/bin/pip install --no-cache-dir -r requirements.txt
	touch $(VENV_NAME)/bin/activate

test: venv
	${PYTHON} -m unittest

build-release: venv
	bump2version --new-version ${v} release

build-image-dev:
	docker build -t "$(IMAGE_NAME_DEV)" --build-arg version=${v} -f Dockerfile .

build-image:
	docker build -t "$(IMAGE_NAME)" --build-arg version=${v} -f Dockerfile .

pushDev:
	echo "Registry is not provided"
	if [ -d ${REGISTRY} ] ; then echo "Provide registry"; exit 1 ; fi
	docker tag "$(IMAGE_NAME)" "$(REGISTRY)/$(IMAGE_NAME):latest"
	docker push "$(REGISTRY)/$(IMAGE_NAME):latest"

