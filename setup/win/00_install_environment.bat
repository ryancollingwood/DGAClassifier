virtualenv .venv
CALL .venv/Scripts/activate.bat
pip install -r requirements.txt
ipython kernel install --user --name=domain_generation_algorithm_classifier

