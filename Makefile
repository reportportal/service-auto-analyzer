IMAGE_NAME=reportportal/service-auto-analyzer$(IMAGE_POSTFIX)
IMAGE_NAME_DEV=reportportal-dev/service-auto-analyzer
DEV_BUILD_PLATFORM=amd64

VENV_PATH?=/venv
PYTHON=${VENV_PATH}/bin/python3
PIP=${VENV_PATH}/bin/pip

.PHONY: build-release build-image-dev build-image venv test checkstyle test-all build-image-test run-test

install-dependencies:
	test -d ${VENV_PATH} || ( python3 -m venv ${VENV_PATH} \
		&& ${PYTHON} -m pip install --upgrade pip \
		&& ${PIP} install --no-cache-dir -r requirements.txt \
		&& ${PYTHON} -m nltk.downloader -d /usr/share/nltk_data stopwords )

install-dev-dependencies: install-dependencies
	test ${VENV_PATH}/bin/pytest || ${PIP} install --no-cache-dir -r requirements-dev.txt

venv:
	export VIRTUAL_ENV=${VENV_PATH}
	export PATH="${VIRTUAL_ENV}/bin:${PATH}"
	hash -r 2>/dev/null

test: install-dev-dependencies venv
	${PYTHON} -m pytest test/ -s -vv

checkstyle: install-dev-dependencies venv
	${PYTHON} -m flake8

release: install-dependencies
	git config --global user.email "Jenkins"                                                                    
	git config --global user.name "Jenkins"
	${PYTHON} -m bumpversion --new-version ${v} build --tag --tag-name ${v} --allow-dirty
	${PYTHON} -m bumpversion --new-version ${v}-SNAPSHOT build --no-tag --allow-dirty
	git remote set-url origin https://${githubtoken}@github.com/reportportal/service-auto-analyzer
	git push origin master ${v}
	${PYTHON} -m bumpversion --new-version ${v} build --no-commit --no-tag --allow-dirty

build-release: venv
	${PYTHON} -m bumpversion --new-version ${v} build --no-commit --no-tag --allow-dirty

build-image-dev:
	cat Dockerfile | sed "s/\$${BUILDPLATFORM}/$(DEV_BUILD_PLATFORM)/" | docker build -t "$(IMAGE_NAME_DEV)" --build-arg version=${v} --build-arg prod="false" --build-arg buildplatform="amd64" -f - .

build-image:
	# To build image you need to pass these make args: "v" - app version, "githubtoken" - a GitHub token with commit rights, "buildplatform" - a platform for which build the image
	cat Dockerfile | sed s/\$${BUILDPLATFORM}/${buildplatform}/ | docker build -t "$(IMAGE_NAME)" --build-arg version=${v} --build-arg prod="true" --build-arg githubtoken=${githubtoken} -f Dockerfile .

test-all: checkstyle test
