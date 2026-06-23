FROM dhi.io/python@sha256:c2b0cd3f1b921937d1d15c1cd3a2335fc23d865bba99255127af79821d02042f AS test
USER root
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/* \
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

FROM dhi.io/python@sha256:c2b0cd3f1b921937d1d15c1cd3a2335fc23d865bba99255127af79821d02042f AS builder
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends make wget \
    && rm -rf /var/lib/apt/lists/* \
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
RUN mkdir -p -m 0744 /backend/storage \
    && cp /build/VERSION /backend \
    && cp -r /build/app /backend/ \
    && cp -r /build/res /backend/

FROM dhi.io/python@sha256:1a211b3861bb85e5fc42397aed8db6ce05b5d0f08611959c9b1577c213705913
WORKDIR /backend
COPY --from=builder /backend ./
COPY --from=builder /venv /venv
COPY --from=builder /usr/share/nltk_data /usr/share/nltk_data/
COPY --from=builder /usr/bin/wget /usr/bin/wget
COPY --from=builder //usr/lib/libpcre2-8.so /usr/lib/libpcre2-8.so

ENV VIRTUAL_ENV="/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" PYTHONPATH=/backend

# Start server
CMD ["python", "app/main.py"]
HEALTHCHECK --interval=1m --timeout=5s --retries=2 CMD ["wget", "-q", "--spider", "http://localhost:5001/"]
