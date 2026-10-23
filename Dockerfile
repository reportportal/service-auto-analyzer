FROM dhi.io/python@sha256:aca0dcdfe09c9883ff061e61ecac0dcc5ce6c0a23addf0b237adf1a6ebaa5f4c AS test
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
    && "${VIRTUAL_ENV}/bin/python3" -c "import nltk, os; d = '/usr/share/nltk_data'; \
        nltk.download(['stopwords', 'wordnet', 'omw-1.4'], d, raise_on_error=True); \
        [os.chmod(os.path.join(r, f), 0o644) for r, _, fs in os.walk(d) for f in fs]"
RUN "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r requirements-dev.txt
RUN make test-all

FROM dhi.io/python@sha256:aca0dcdfe09c9883ff061e61ecac0dcc5ce6c0a23addf0b237adf1a6ebaa5f4c AS builder
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
    && "${VIRTUAL_ENV}/bin/python3" -c "import nltk, os; d = '/usr/share/nltk_data'; \
        nltk.download(['stopwords', 'wordnet', 'omw-1.4'], d, raise_on_error=True); \
        [os.chmod(os.path.join(r, f), 0o644) for r, _, fs in os.walk(d) for f in fs]"
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

FROM dhi.io/python@sha256:4faff1784228ae3d69f1ecd51ab0c651ff73022ae4f5e5a5bfb1b6a61d6e041c
WORKDIR /backend
COPY --from=builder /backend ./
COPY --from=builder /venv /venv
COPY --from=builder /usr/share/nltk_data /usr/share/nltk_data/

ENV VIRTUAL_ENV="/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" PYTHONPATH=/backend

# Start server
CMD ["python", "app/main.py"]
HEALTHCHECK --interval=1m --timeout=5s --retries=2 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5001/', timeout=5)"]
