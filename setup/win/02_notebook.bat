CALL .venv/Scripts/activate.bat
jupyter kernelspec uninstall domain_generation_algorithm_classifier
ipython kernel install --user --name=domain_generation_algorithm_classifier
