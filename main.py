# GESTION ENVIRONNEMENT -----------------------------
import sys
from src.data.import_data import import_yaml_config, import_data
from src.models.train_evaluate import random_forest_titanic
from src.features.build_features import (
    create_variable,
    fill_na_titanic,
    label_encoder_titanic,
    check_has_cabin,
    ticket_length,
)

# PARAMETRES -------------------------------------
config = import_yaml_config("configs/config.yaml")
# secrets = import_yaml_config("configs/secrets.yaml")

# API_TOKEN = secrets["jetonapi"]

# Number trees as command line argument
N_TREES = int(sys.argv[1]) if len(sys.argv) == 2 else 20


LOCATION_TRAIN = config["path"]["train"]
LOCATION_TEST = config["path"]["test"]
TEST_FRACTION = config["model"]["test_fraction"]

# FEATURE ENGINEERING --------------------------------

TrainingData = import_data(LOCATION_TRAIN)
TestData = import_data(LOCATION_TEST)

# Create a 'Title' variable
TrainingData = create_variable(TrainingData)
TestData = create_variable(TestData)


# IMPUTATION DES VARIABLES ================


TrainingData = fill_na_titanic(TrainingData)
TestData = fill_na_titanic(TestData)

TrainingData = label_encoder_titanic(TrainingData)
TestData = label_encoder_titanic(TestData)


# Making a new feature hasCabin which is 1 if cabin is available else 0
TrainingData = check_has_cabin(TrainingData)
TestData = check_has_cabin(TestData)

TrainingData = ticket_length(TrainingData)
TestData = ticket_length(TestData)


# MODELISATION: RANDOM FOREST ----------------------------

model = random_forest_titanic(data=TrainingData, fraction_test=TEST_FRACTION, n_trees=N_TREES)
