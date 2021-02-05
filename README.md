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

**ANALYZER_BINARYSTORE_TYPE** - you can set either "minio" or "filesystem" here, and this will be used as a strategy where to store information, connected with the analyzer, by default "minio"

**MINIO_SHORT_HOST** - by default "minio:9000", you need to set short host and port to the minio service. **NOTE**: if you don't use Minio, please set this variable with the value "", so analyzer won't try to connect to the Minio instance

**MINIO_ACCESS_KEY** - by default "minio", you need to set a minio access key here

**MINIO_SECRET_KEY** - by default "minio123", you need to set a minio secret key here

**ANALYZER_BINARYSTORE_BUCKETPREFIX** - by default "prj-", the prefix for buckets which are added to each project filepath.

**ANALYZER_BINARYSTORE_MINIO_REGION** - by default None, the region which you can specify for saving in AWS S3.

**FILESYSTEM_DEFAULT_PATH** - by default "storage", the path where will be stored all the information connected with analyzer, if ANALYZER_BINARYSTORE_TYPE = "filesystem". If you want to mount this folder to some folder on your machine, you can use this instruction in the docker compose:
```
volumes:
  - ./data/analyzer:/backend/storage
```

**ES_CHUNK_NUMBER** - by default 1000, the number of objects which is sent to ES while bulk indexing. **NOTE**: AWS Elasticsearch has restrictions for sent data size either 10Mb or 100Mb, so when 10Mb is chosen, make sure you don't get the error "TransportError(413, '{"Message": "Request size exceeded 10485760 bytes"}')" while generating index or indexing the data. If you get this error, please, decrease ES_CHUNK_NUMBER until you stop getting this error.

# Environmental variables for constants, used by algorithms:

**ES_MIN_SHOULD_MATCH** - by default "80%", the global default min should match value for auto-analysis, but it is used only when the project settings are not set up.

**ES_BOOST_AA** - by default "-8.0", the value to boost auto-analyzed items while querying for Auto-analysis

**ES_BOOST_LAUNCH** - by default "4.0", the value to boost items with the same launch while querying for Auto-analysis

**ES_BOOST_UNIQUE_ID** - by default "8.0", the value to boost items with the same unique id while querying for Auto-analysis

**ES_MAX_QUERY_TERMS** - by default "50", the value to use in more like this query while querying for Auto-analysis

**ES_MIN_WORD_LENGTH** - by default "2", the value to use in more like this query while querying for Auto-analysis

**ES_LOGS_MIN_SHOULD_MATCH** - by default "0.98", the value of min should match for searching the same to investigate test items

**PATTERN_LABEL_MIN_PERCENT** - by default "0.9", the value of minimum percent of the same issue type for pattern to be suggested as a pattern with a label

**PATTERN_LABEL_MIN_COUNT** - by default "5", the value of minimum count of pattern occurance to be suggested as a pattern with a label

**PATTERN_MIN_COUNT** - by default "10", the value of minimum count of pattern occurance to be suggested as a pattern without a label

**MAX_LOGS_FOR_DEFECT_TYPE_MODEL** - by default "10000", the value of maximum count of logs per defect type to add into defect type model training. Default value is chosen in cosideration of having space for analyzer_train docker image setuo of 1GB, if you can give more GB you can linearly allow more logs to be considered.

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

