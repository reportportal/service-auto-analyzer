FROM --platform=${BUILDPLATFORM} python:3.10.13 as test
RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /venv \
    && mkdir /build
ENV VIRTUAL_ENV=/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
WORKDIR /build
COPY ./ ./
RUN "${VIRTUAL_ENV}/bin/pip" install --upgrade pip \
    && LIBRARY_PATH=/lib:/usr/lib /bin/sh -c "${VIRTUAL_ENV}/bin/pip install --no-cache-dir -r requirements.txt" \
    && "${VIRTUAL_ENV}/bin/python3" -m nltk.downloader -d /usr/share/nltk_data stopwords
RUN "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r requirements-dev.txt
RUN make test-all


FROM --platform=${BUILDPLATFORM} python:3.10.13 as builder
RUN apt-get update && apt-get install -y build-essential libpcre3 libpcre3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /venv \
    && mkdir /build
ENV VIRTUAL_ENV=/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
WORKDIR /build
COPY ./ ./
RUN "${VIRTUAL_ENV}/bin/pip" install --upgrade pip \
    && LIBRARY_PATH=/lib:/usr/lib /bin/sh -c "${VIRTUAL_ENV}/bin/pip install --no-cache-dir -r requirements.txt" \
    && "${VIRTUAL_ENV}/bin/python3" -m nltk.downloader -d /usr/share/nltk_data stopwords
ARG APP_VERSION
ARG RELEASE_MODE
ARG GITHUB_TOKEN
RUN if [ "$RELEASE_MODE" = "true" ]; then make release v=${APP_VERSION} githubtoken=${GITHUB_TOKEN}; else if [ "${APP_VERSION}" != "" ]; then make build-release v=${APP_VERSION}; fi ; fi
RUN mkdir /backend \
    && cp /build/VERSION /backend \
    && cp -r /build/app /backend/ \
    && cp -r /build/res /backend/


FROM --platform=${BUILDPLATFORM} python:3.10.13-slim
RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y libxml2 libgomp1 curl libpcre3 libpcre3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /venv /venv
RUN mkdir /usr/share/nltk_data && chmod g+w /usr/share/nltk_data
COPY --from=builder /usr/share/nltk_data /usr/share/nltk_data/
WORKDIR /backend/
COPY --from=builder /backend ./

# Create a group and user
RUN groupadd uwsgi && useradd -g uwsgi uwsgi
USER uwsgi

EXPOSE 5001
ENV VIRTUAL_ENV="/venv"

# uWSGI configuration (customize as needed):
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" PYTHONPATH=/backend \
    FLASK_APP=app/main.py UWSGI_WSGI_FILE=app/main.py UWSGI_SOCKET=:3031 UWSGI_HTTP=:5001 \
    UWSGI_VIRTUALENV=${VIRTUAL_ENV} UWSGI_MASTER=1 UWSGI_WORKERS=4 UWSGI_THREADS=8 UWSGI_MAX_FD=10000 UWSGI_LAZY_APPS=1 \
    UWSGI_WSGI_ENV_BEHAVIOR=holy PYTHONDONTWRITEBYTECODE=1

# Start uWSGI
CMD ["/venv/bin/uwsgi", "--http-auto-chunked", "--http-keepalive"]
HEALTHCHECK --interval=1m --timeout=5s --retries=2 CMD ["curl", "-s", "-f", "--show-error", "http://localhost:5001/"]
