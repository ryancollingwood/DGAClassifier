pip install virtualenv
virtualenv .venv
CALL .venv/Scripts/activate.bat
pip install -r requirements.txt
ipython kernel install --user --name=domain_generation_algorithm_classifier
pytest unittests
pytest integrationtests
python train_model.py -p data/raw/dga_domains.csv -o models -v 2
python test_model.py
python dga_classify.py -i