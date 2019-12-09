FROM python:3.7.4

RUN apt-get update && apt-get install -y build-essential
ARG version

ADD requirements.txt /requirements.txt

RUN python -m venv /venv \
    && /venv/bin/pip install -U pip \
    && LIBRARY_PATH=/lib:/usr/lib /bin/sh -c "/venv/bin/pip install --no-cache-dir -r /requirements.txt"

ENV VERSION=$version
COPY ./ ./

RUN make build-release v=${VERSION}

# Multistage
FROM python:3.7.4-slim
RUN apt-get update && apt-get install -y libxml2 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=0 /venv /venv

WORKDIR /backend/

COPY . .
COPY --from=0 VERSION .bumpversion.cfg ./

# uWSGI will listen on this port
EXPOSE 5000
EXPOSE 3031

# uWSGI configuration (customize as needed):
ENV FLASK_APP=app.py UWSGI_WSGI_FILE=app.py UWSGI_SOCKET=:3031 UWSGI_HTTP=:5000 UWSGI_VIRTUALENV=/venv UWSGI_MASTER=1 UWSGI_WORKERS=2 UWSGI_THREADS=8 UWSGI_LAZY_APPS=1 UWSGI_WSGI_ENV_BEHAVIOR=holy PYTHONDONTWRITEBYTECODE=1
ENV PATH="/venv/bin:${PATH}"
ENV PYTHONPATH="/backend"

# Start uWSGI
CMD ["/venv/bin/uwsgi", "--http-auto-chunked", "--http-keepalive"]