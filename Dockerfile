FROM python:3.7.4

RUN apt-get update && apt-get install -y build-essential
RUN mkdir /backend/
WORKDIR /backend/

ADD requirements.txt requirements.txt

RUN python -m venv /venv \
    && /venv/bin/pip install -U pip \
    && LIBRARY_PATH=/lib:/usr/lib /bin/sh -c "/venv/bin/pip install --no-cache-dir -r requirements.txt"

RUN touch /venv/bin/activate
RUN /venv/bin/python3 -m nltk.downloader stopwords

ARG version
ARG prod
ENV BOOST_MODEL_FOLDER_ALL_LINES="/backend/model/all_lines_0.9"
ENV BOOST_MODEL_FOLDER_NOT_ALL_LINES="/backend/model/not_all_lines_0.1"
ENV SIMILARITY_WEIGHTS_FOLDER="/backend/model/weights_0.2"

COPY ./ ./

RUN make test-all
RUN if [ "$prod" = "true" ]; then make build-release v=$version; else if [ "$version" != "" ]; then make build-release v=$version; fi ; fi

# Multistage
FROM python:3.7.4-slim
RUN apt-get update && apt-get install -y libxml2 libgomp1\
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
ENV BOOST_MODEL_FOLDER_ALL_LINES="/backend/model/all_lines_0.9"
ENV BOOST_MODEL_FOLDER_NOT_ALL_LINES="/backend/model/not_all_lines_0.1"
ENV SIMILARITY_WEIGHTS_FOLDER="/backend/model/weights_0.2"
#ENV LOGGING_LEVEL="INFO"
# Start uWSGI
#CMD ["/venv/bin/uwsgi", "--http-auto-chunked", "--http-keepalive"]
CMD ["/venv/bin/uwsgi"]