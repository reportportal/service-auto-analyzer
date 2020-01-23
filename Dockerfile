FROM python:3.7.4

RUN apt-get update && apt-get install -y build-essential
ARG version
ARG prod

ADD requirements.txt /requirements.txt

RUN python -m venv /venv \
    && /venv/bin/pip install -U pip \
    && LIBRARY_PATH=/lib:/usr/lib /bin/sh -c "/venv/bin/pip install --no-cache-dir -r /requirements.txt"

RUN touch /venv/bin/activate
RUN /venv/bin/python3 -m nltk.downloader stopwords

WORKDIR /backend/

ENV BOOST_MODEL_FOLDER="/backend/model/0.5"

COPY ./ ./

RUN make test-all

RUN if ["$prod" = "true"] ; then make release v=$version ; else make build-release v=$version ; fi

# Multistage
FROM python:3.7.4-slim
RUN apt-get update && apt-get install -y libxml2 curl libgomp1\
    && rm -rf /var/lib/apt/lists/*
COPY --from=0 /venv /venv
RUN mkdir /root/nltk_data
COPY --from=0 /root/nltk_data /root/nltk_data/

WORKDIR /backend/

COPY . .
COPY --from=0 /backend/VERSION /backend/.bumpversion.cfg ./

# uWSGI will listen on this port
EXPOSE 5000
EXPOSE 3031

# uWSGI configuration (customize as needed):
ENV FLASK_APP=app.py UWSGI_WSGI_FILE=app.py UWSGI_SOCKET=:3031 UWSGI_HTTP=:5000 UWSGI_VIRTUALENV=/venv UWSGI_MASTER=1 UWSGI_WORKERS=4 UWSGI_THREADS=8 UWSGI_LAZY_APPS=1 UWSGI_WSGI_ENV_BEHAVIOR=holy PYTHONDONTWRITEBYTECODE=1
ENV PATH="/venv/bin:${PATH}"
ENV PYTHONPATH="/backend"
ENV BOOST_MODEL_FOLDER="/backend/model/0.5"

# Start uWSGI
CMD ["/venv/bin/uwsgi", "--http-auto-chunked", "--http-keepalive"]