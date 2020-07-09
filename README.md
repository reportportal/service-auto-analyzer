# service-auto-analyzer

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
5. Set the environmental variable for the model
```
  EXPORT BOOST_MODEL_FOLDER="model/0.6"
```
6. Start the uwsgi server, you can change properties, such as the workers quantity for running the analyzer in the several processes
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
5. Set the environmental variable for the model
```
setx BOOST_MODEL_FOLDER "model/0.6"
```
6. Start the program.
```
python app.py
```

