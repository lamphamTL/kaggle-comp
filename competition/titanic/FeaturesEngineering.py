import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

warnings.filterwarnings("error", category=FutureWarning)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MultiLabelBinarizer


# Since MultiLabelBinarizer only take (y) during fit and fit_transform, we create this wrapper
# to align it with other transformers.
# It's the same idea as OneHotEncoder -> y is passed to fit() but not used
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.classes_ = None
        self.mlb = MultiLabelBinarizer(**kwargs)

    def fit(self, X, y=None):  # accepts (X, y)
        self.mlb.fit(X)
        # This is important because scikit-learn verify estimator's fittedness based on existence of
        # attribute ending with "_" and not starting with "__"
        # Here classes_ is just overwritten from MultiLabelBinarizer
        self.classes_ = self.mlb.classes_
        return self

    def transform(self, X):
        return self.mlb.transform(X)

    def fit_transform(self, X, y=None):
        return self.mlb.fit_transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.mlb.classes_


def _name_to_title(X):
    title = X["Name"].str.extract(r"\S+, (.*?)\.")
    return title


def categorize_age(X):
    bins = [0, 3, 12, 19, 29, 44, 64, np.inf]
    labels = ["infant", "child", "teenager", "young_adult", "adult", "middle_aged", "senior"]
    return pd.cut(X["Age"], bins=bins, labels=labels, right=True).astype(str).to_frame()


def engineer_features():
    # Engineer "Cabin" -> "HasCabin"
    has_cabin_transformer = FunctionTransformer(lambda x: x.notnull().astype(int), validate=False,
                                                feature_names_out="one-to-one")
    decks = Pipeline(
        [
            ("cabin_to_deck", FunctionTransformer(lambda y: y.iloc[:, 0].apply(
                lambda c: ['U'] if pd.isna(c) else list({part[0] for part in c.split()})).tolist(), validate=False,
                                                  feature_names_out="one-to-one")),
            ("mlb", MultiLabelBinarizerTransformer())
        ]
    )
    # Get passenger title: Mr, Mrs, Master etc..
    passenger_title = Pipeline(
        [
            ("title", FunctionTransformer(_name_to_title, validate=False, feature_names_out="one-to-one")),
            ("is_title_", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )
    # Age categorization
    cat_age = Pipeline(
        [
            ("age_label", FunctionTransformer(categorize_age, validate=False, feature_names_out="one-to-one")),
            ("is_age_", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )
    cat_variables_transf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("has_cabin", has_cabin_transformer, ["Cabin"]),
            ("decks", decks, ["Cabin"]),
            ("dropper", "drop", ["Ticket"]),
            ("p_title", passenger_title, ["Name"]),
            ("age_passthrough", "passthrough", ["Age"]),
            ("cat_age", cat_age, ["Age"]),
            ("cat_variables", cat_variables_transf, ["Sex", "Embarked"])
        ],
        remainder="passthrough"
    )


df = pd.read_csv("../data/train.csv", delimiter=",")

features = [x for x in df.columns if x != "Survived"]
x, y = df[features], df["Survived"].T
print(x.shape)
print(y.shape)
