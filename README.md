# service-auto-analyzer

# Environment variables for configuration

**ES_HOSTS** - Elasticsearch host (can be either like this "http://elasticsearch:9200", or with login and password delimited by : and separated from the host name by @)

**LOGGING_LEVEL** - logging level for the whole module, can be DEBUG, INFO, ERROR, CRITICAL

**AMQP_URL** - an url to the rabbitmq instance

**AMQP_EXCHANGE_NAME** - Exchange name for the module communication for this module, by default "analyzer"

**ANALYZER_PRIORITY** - priority for this analyzer, by default 1

**ES_VERIFY_CERTS** - turn on SSL certificates verification, by default "false"

**ES_USE_SSL** - turn on SSL, by default "false"

**ES_SSL_SHOW_WARN** - show warning on SSL certificates verification, by default "false"

**ES_CA_CERT** - provide a path to CA certs on disk, by default ""

**ES_CLIENT_CERT** - PEM formatted SSL client certificate, by default ""

**ES_CLIENT_KEY** - PEM formatted SSL client key, by default ""

**ES_TURN_OFF_SSL_VERIFICATION** - by default "false". Turn off ssl verification via using RequestsHttpConnection class instead of Urllib3HttpConnection class.

**MINIO_SHORT_HOST** - by default "minio:9000", you need to set short host and port to the minio service

**MINIO_ACCESS_KEY** - by default "minio", you need to set a minio access key here

**MINIO_SECRET_KEY** - by default "minio123", you need to set a minio secret key here

# Instructions for analyzer setup without Docker

Install python with the version 3.7.4. (it is the version on which the service was developed, but it should work on the versions starting from 3.6).

Perform next steps inside source directory of the analyzer.

## For Linux:
1. Create a virtual environment with any name (in the example **/venv**)
```Shell
  python -m venv /venv
```
2. Install python libraries
```
  /venv/bin/pip install --no-cache-dir -r requirements.txt
```
3. Activate the virtual environment
```
  /venv/bin/activate
```
4. Install stopwords package from the nltk library
```
  /venv/bin/python3 -m nltk.downloader -d /usr/share/nltk_data stopwords
```
5. Start the uwsgi server, you can change properties, such as the workers quantity for running the analyzer in the several processes
```
  /venv/bin/uwsgi --ini app.ini
  ```
 
## For Windows:
1. Create a virtual environment with any name (in the example **env**)
```
python -m venv env
```
2. Activate the virtual environment
```
call env\Scripts\activate.bat
```
3. Install python libraries
```
python -m pip install -r requirements_windows.txt
```
4. Install stopwords package from the nltk library
```
python -m nltk.downloader stopwords
```
5. Start the program.
```
python app.py
```

