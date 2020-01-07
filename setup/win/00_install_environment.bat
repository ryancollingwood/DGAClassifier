virtualenv .venv
CALL .venv/Scripts/activate.bat
pip install pandas numpy sklearn scipy jupyter matplotlib unidecode
ipython kernel install --user --name=domain_generation_algorithm_classifier

