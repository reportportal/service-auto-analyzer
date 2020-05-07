IMAGE_NAME=reportportal/service-auto-analyzer$(IMAGE_POSTFIX)
IMAGE_NAME_DEV=reportportal-dev/service-auto-analyzer
IMAGE_NAME_TEST=reportportal/service-auto-analyzer_test

VENV_PATH?=/venv
PYTHON=${VENV_PATH}/bin/python3

.PHONY: build-release build-image-dev build-image venv test checkstyle test-all build-image-test run-test

install-dependencies: requirements.txt
	test -d $(VENV_PATH) || virtualenv -p python3 $(VENV_PATH)
	$(VENV_PATH)/bin/pip install --no-cache-dir -r requirements.txt
	touch $(VENV_PATH)/bin/activate

venv: 
	touch $(VENV_PATH)/bin/activate

test: venv
	${PYTHON} -m unittest

checkstyle: venv
	${PYTHON} -m flake8

release: install-dependencies
	git config --global user.email "Jenkins"                                                                    
	git config --global user.name "Jenkins"
	${PYTHON} -m bumpversion --new-version ${v} build --tag --tag-name ${v} --allow-dirty
	${PYTHON} -m bumpversion patch --no-tag --allow-dirty
	git remote set-url origin https://${githubtoken}@github.com/MariyaIvanina/github_releaser
	git push origin master ${v}
	git checkout develop
	git merge master
	git push origin develop

build-release: venv
	${PYTHON} -m bumpversion --new-version ${v} build --no-commit --no-tag --allow-dirty

build-image-dev:
	docker build -t "$(IMAGE_NAME_DEV)" --build-arg version=${v} --build-arg prod="false" -f Dockerfile .

build-image:
	docker build -t "$(IMAGE_NAME)" --build-arg version=${v} --build-arg prod="true" --build-arg githubtoken=${githubtoken} -f Dockerfile .

test-all: checkstyle test

