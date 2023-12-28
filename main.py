# GESTION ENVIRONNEMENT -----------------------------
import sys
from src.data.import_data import import_yaml_config, import_data
from src.models.train_evaluate import build_pipeline_titanic
from src.features.build_features import (
    create_variable,
    fill_na_titanic,
    label_encoder_titanic,
    check_has_cabin,
    ticket_length,
)
from sklearn.metrics import confusion_matrix
from src.features.build_features import split_train_test_titanic
from joblib import dump

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


train, test = split_train_test_titanic(TrainingData, fraction_test=TEST_FRACTION)

# MODELISATION: RANDOM FOREST ----------------------------
pipe = build_pipeline_titanic(n_trees=N_TREES)
pipe.fit(train.drop("Survived", axis="columns"), train["Survived"])

# calculons le score sur le dataset d'apprentissage et sur le dataset de test
# (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction
model_val_score = pipe.score(test.drop("Survived", axis="columns"), test["Survived"])
print(f"{round(model_val_score * 100)} % de bonnes réponses sur les données de test pour validation")


print("matrice de confusion")
print(confusion_matrix(test["Survived"], pipe.predict(test.drop("Survived", axis="columns"))))

# Sauvegarde du modèle
dump(pipe, "model.joblib")
