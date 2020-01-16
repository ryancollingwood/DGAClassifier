# DGA Classifier

## Context
Once malicious software has been installed on a system it needs to be told what to do. To receive instructions the malware needs to communicate with a command and control server. Historically, the domain name of the command and control server was hardcoded into the code of the malware. This made it very easy for cybersecurity professionals to block this communication channel by blacklisting the domain once they discovered it.

To avoid this weakness malware programs have started to use Domain Generating Algorithms (or DGAs). These algorithms generate a number of random domain names where one of the generated domains is the control server. The infected computed scans through these domains trying to contact each, eventually it will try the correct domain. At this stage, the malware can receive instructions remotely.

## Approach

### Exploratory Data Analysis
First the given data was explored as documented in [EDA Notebook](/notebooks/00_DGA_EDA.ipynb).
, with a view to understand:
* The nature of the data.
* Potential required pre-processing steps to sanitize the data.
* Understanding to what extent the classes of `legit` and `dga` were present and potential scoring measures.
* Identifying potential features to get generated.

### Feature Generation and Selection
Once the data was better understood and potential features had been identified pipelines implement the desired transformations were developed.
Then these pipeline were used to generate different features and their ability to discriminates between the classes was explored. Then feature selection was performed using Principal Component Analysis.
The process is document in [Feature Generation and Selection Notebook](/notebooks/02_Feature_Generation_Selection.ipynb)

### Model Selection
Finally once suitable features had been identified, then model selection was performed. This was achieved by fitting a number of naive models from a variety of model families.
Of the model families fitted, Random Forrest was identified as the most suitable.
This information was then carried forward to training by using a Grid Search Cross Validation pipeline wrapping around the Random Forrest Classifier as documentd in [Model Selecton Notebook](/notebooks/03_Model_Selection.ipynb)

# Installation and Overview of Solution

## Run on the Cloud
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.png)](https://console.cloud.google.com/cloudshell/open?git_repo=https://github.com/ryancollingwood/DGAClassifier&tutorial=quickstart.md)

All you need is a Google account, run this repo in a light weight temporary cloud session.

## Installation

It is assumed you have Python 3.6 or above installed.

Setup scripts are included for Windows and Linux based environments.
The setup scripts do the following

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

### Windows

```
setup\windows.bat
```

### MacOS and Linux

You may need to set executable permission on the bash script.

```bash
sudo chmod +x setup\linux.sh
setup\linux.sh
```

## Usage

If you've run the setup scripts above you will have a trained model ready for use.
However if you choose to setup your environment manually here is an overview of the steps that need to be taken.

The following commands assume you're in the root folder of this git repo.

### Training the model

*Windows*
```bash
python train_model.py -p data\raw\dga_domains.csv -o models
```

*Linux and MacOS*
```bash
python train_model.py -p data/raw/dga_domains.csv -o models
``` 

The model training script expects at least two parameters to be passed in:
* `-p` for the path to the source data.
* `-o` where the trained model will be written out.

#### Testing the model
*Windows, Linux, and MacOS*
```bash
python test_model.py
```

Runs a trivial test on the model to ensure it has been built.

#### Querying the model
*Windows, Linux, and MacOS*

For an interactive session, where you can type in domains
````
python dga_classify.py -i
````

To get the prediction for a single or comma separated list of domains
````
python dga_classify.py reddit,facebook.com,google.co.uk
````

##### Query Return Codes
* `0` - No dga domains were predicted from any of the inputs.
* `2` - No predictions were made, e.g. empty or invalid inputs.
* `3` - Dga domains were predicted

Using the return codes it is possible to call this script as part of executing a shell script.

## Directory Structure

### Data

```
+---data
|   +---interim
|   +---processed
|   \---raw
```

The source data is in the `raw` sub-directory

### Tests

```
+---integrationtests
|   +---models
|   +---test_00_preprocessing
|   +---test_01_feature_generation
|   +---test_02_rescale
|   +---test_04_prepare_model_inputs
|   +---test_05_train_model
|   \---test_06_query_model
...
\---unittests
    +---data
    +---features
    |   \---transformer
    +---pipeline
    |   \---steps
    \---preprocessing
        +---column
        \---text
```

### Source

```
+---src
|   +---data
|   +---features
|   |   \---transformer
|   +---logging
|   +---model
|   +---pipeline
|   |   \---steps
|   \---preprocessing
|       \---transformer
```

Each of the sub-directories is exposed as a Python Package.

#### `data`
For loading data.

#### `features`
Concerned with feature generation, and the sklearn compatible transformer pipelines that implement these feature generators.

#### `logging`
Helper package for using built-in logging features.

#### `model`
Training, testing, loading, and "querying" the model we've derived from our source data.

#### `pipeline`
Separated into complete pipelines and steps for use in pipelines.

#### `preprocessing`
Transformations for getting input data into usable state.

### Other
```
+---models
+---notebooks
+---scripts
+---setup
```

#### `models`
This is where built models can be stored. This folder is excluded from git.

Once a model has been trained the following files are written to this folder:
* `report_test.txt` - report of test prediction detailing precision, recall, f1-score.
* `report_train.txt` - report of training prediction detailing precision, recall, f1-score.
* `trained.model` - the persisted model.
* `trained_params.json` - the optimal parameters are found via Cross Validated Grid Search.
* `training.log` - log file of the training process.

#### `notebooks`
Documenting the phases of:
* Exploratory Data Analysis
* Feature Generation and Selection
* Model Selection

#### `scripts`
Helper scripts used in development.

#### `setup`
Setup scripts.
