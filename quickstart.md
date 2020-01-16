# Quickstart

This a guide to get you up and running in a Google Cloud Shell.

# Run Setup

You may have to alter the permissions of the setup scripts in order for it to be executed.

`sudo chmod setup/linux.sh`

Now to execute the setup script.

`scripts/linux.sh`

# Go get a beverage ☕
This will take a while as it's doing a couple of things

* Install virtualenv
* Create a virtualenv `.venv`
* Active the virtual env `.venv`
* Install the Python packages in `requirements.txt`
* Register an IPython kernel as used in the Notebooks
* Run the `unittests` with pytest
* Run the `integrationtests` with pytest
* Train the model
* Runs a simple test of the model
* Enters an interactive mode where you can query the model

# Ready To Query

Once all the setup steps are done you'll see a prompt:
```
domain>
```

Type in a domain to get a prediction.

To quit type in an empty string.

For more details and manual installation go to the next section.

# Training the model

```bash
python train_model.py -p data/raw/dga_domains.csv -o models
``` 

The model training script expects at least two parameters to be passed in:
* `-p` for the path to the source data.
* `-o` where the trained model will be written out.

# Testing the model
```bash
python test_model.py
```

Runs a trivial test on the model to ensure it has been built.

# Querying the model

For an interactive session, where you can type in domains
````
python dga_classify.py -i
````

To get the prediction for a single or comma separated list of domains
````
python dga_classify.py reddit,facebook.com,google.co.uk
````

*Query Return Codes*
* `0` - No dga domains were predicted from any of the inputs.
* `2` - No predictions were made, e.g. empty or invalid inputs.
* `3` - Dga domains were predicted

Using the return codes it is possible to call this script as part of executing a shell script.