import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from src.features.build_features import split_train_test_titanic


def random_forest_titanic(data: pd.DataFrame, fraction_test: float = 0.8, n_trees: int = 20):
    """Random forest model for Titanic survival

    Args:
        data (pd.DataFrame): _description_
        fraction_test (float, optional): _description_. Defaults to 0.8.
        n_trees (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """

    X_train, X_test, y_train, y_test = split_train_test_titanic(data, fraction_test=fraction_test)

    rdmf = RandomForestClassifier(n_estimators=n_trees)
    rdmf.fit(X_train, y_train)

    # calculons le score sur le dataset d'apprentissage et sur le dataset de test
    # (10% du dataset d'apprentissage mis de côté)
    # le score étant le nombre de bonne prédiction
    rdmf_score = rdmf.score(X_test, y_test)
    print(f"{round(rdmf_score * 100)} % de bonnes réponses sur les données de test pour validation")

    print("matrice de confusion: ")
    print(confusion_matrix(y_test, rdmf.predict(X_test)))

    return rdmf, X_train, X_test, y_train, y_test
