FROM registry.access.redhat.com/ubi10/python-312-minimal@sha256:394fffb2dfbe6d5ec49034a1c9a35940a6e954348ad2b8432973964b6d18c528 AS test
USER root
RUN microdnf -y upgrade && microdnf -y install make \
    && microdnf clean all \
    && python -m venv /venv \
    && mkdir /build
ENV VIRTUAL_ENV=/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
WORKDIR /build
COPY ./app ./app
COPY ./res ./res
COPY ./test ./test
COPY ./test_res ./test_res
COPY ./requirements.txt ./requirements.txt
COPY ./requirements-dev.txt ./requirements-dev.txt
RUN "${VIRTUAL_ENV}/bin/pip" install --upgrade pip \
    && LIBRARY_PATH=/lib:/usr/lib /bin/sh -c "${VIRTUAL_ENV}/bin/pip install --no-cache-dir -r requirements.txt" \
    && "${VIRTUAL_ENV}/bin/python3" -m nltk.downloader -d /usr/share/nltk_data stopwords wordnet omw-1.4
RUN "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r requirements-dev.txt
RUN make test-all

FROM registry.access.redhat.com/ubi10/python-312-minimal@sha256:394fffb2dfbe6d5ec49034a1c9a35940a6e954348ad2b8432973964b6d18c528 AS builder
USER root
RUN microdnf -y upgrade && microdnf -y install make \
    && microdnf clean all \
    && python -m venv /venv \
    && mkdir /build
ENV VIRTUAL_ENV=/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
WORKDIR /build
COPY ./app ./app
COPY ./res ./res
COPY ./requirements.txt ./requirements.txt
COPY ./VERSION ./VERSION
COPY ./Makefile ./Makefile
RUN "${VIRTUAL_ENV}/bin/pip" install --upgrade pip \
    && "${VIRTUAL_ENV}/bin/pip" install --upgrade setuptools \
    && LIBRARY_PATH=/lib:/usr/lib /bin/sh -c "${VIRTUAL_ENV}/bin/pip install --no-cache-dir -r requirements.txt" \
    && "${VIRTUAL_ENV}/bin/python3" -m nltk.downloader -d /usr/share/nltk_data stopwords wordnet omw-1.4
ARG APP_VERSION=""
ARG RELEASE_MODE=false
ARG GITHUB_TOKEN
RUN if [ "$RELEASE_MODE" = "true" ]; then make release v=${APP_VERSION} githubtoken=${GITHUB_TOKEN}; else if [ "${APP_VERSION}" != "" ]; then make build-release v=${APP_VERSION}; fi ; fi
RUN mkdir /backend \
    && cp /build/VERSION /backend \
    && cp -r /build/app /backend/ \
    && cp -r /build/res /backend/

FROM registry.access.redhat.com/ubi10/python-312-minimal@sha256:394fffb2dfbe6d5ec49034a1c9a35940a6e954348ad2b8432973964b6d18c528
USER root
WORKDIR /backend/
COPY --from=builder /backend ./
COPY --from=builder /venv /venv
COPY --from=builder /usr/share/nltk_data /usr/share/nltk_data/

ENV VIRTUAL_ENV="/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" PYTHONPATH=/backend

RUN microdnf -y upgrade \
    && microdnf clean all \
    && mkdir -p -m 0744 /backend/storage \
    && source "${VIRTUAL_ENV}/bin/activate"

# Start server
CMD ["python", "app/main.py"]
HEALTHCHECK --interval=1m --timeout=5s --retries=2 CMD ["curl", "-s", "-f", "--show-error", "http://localhost:5001/"]
