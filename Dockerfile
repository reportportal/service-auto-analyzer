FROM dhi.io/python@sha256:782ea6d552b39ed930bcc86a7f820cc93d5fbb94331aa4860bab400c71c8f628 AS test
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
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
    && "${VIRTUAL_ENV}/bin/python3" -c "import nltk; nltk.download(['stopwords','wordnet','omw-1.4'], '/usr/share/nltk_data')"
RUN "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r requirements-dev.txt
RUN make test-all

FROM dhi.io/python@sha256:782ea6d552b39ed930bcc86a7f820cc93d5fbb94331aa4860bab400c71c8f628 AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
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
    && "${VIRTUAL_ENV}/bin/python3" -c "import nltk; nltk.download(['stopwords','wordnet','omw-1.4'], '/usr/share/nltk_data')"
ARG APP_VERSION=""
ARG RELEASE_MODE=false
ARG GITHUB_TOKEN
RUN if [ "$RELEASE_MODE" = "true" ]; then \
        make release v=${APP_VERSION} githubtoken="${GITHUB_TOKEN}"; \
    elif [ "${APP_VERSION}" != "" ]; then \
        echo "${APP_VERSION}" > VERSION; \
    fi
RUN mkdir -p -m 0744 /backend/storage \
    && cp /build/VERSION /backend \
    && cp -r /build/app /backend/ \
    && cp -r /build/res /backend/

FROM dhi.io/python@sha256:427f11808afcf4ca06e19afaab9de1fddfd90af0f72c02e53bf1545b76ff5a70
WORKDIR /backend
COPY --from=builder /backend ./
COPY --from=builder /venv /venv
COPY --from=builder /usr/share/nltk_data /usr/share/nltk_data/

ENV VIRTUAL_ENV="/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" PYTHONPATH=/backend

# Start server
CMD ["python", "app/main.py"]
HEALTHCHECK --interval=1m --timeout=5s --retries=2 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5001/', timeout=5)"]
