from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def build_pipeline_titanic(n_trees: int = 20):
    """Random forest model for Titanic survival

    Args:
        data (pd.DataFrame): _description_
        fraction_test (float, optional): _description_. Defaults to 0.8.
        n_trees (int, optional): _description_. Defaults to 20.

    Returns:
        the pipeline
    """

    numeric_features = ["Age", "Fare"]
    categorical_features = ["Title", "Embarked", "Sex"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("Preprocessing numerical", numeric_transformer, numeric_features),
            (
                "Preprocessing categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=n_trees)),
        ]
    )

    return pipe
