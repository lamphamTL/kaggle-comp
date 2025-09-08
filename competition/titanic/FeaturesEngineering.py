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

def _cat_age(X):
    bins = [0, 12, 19, 59, np.inf]
    labels = ["child", "teenager", "adult", "senior"]
    return pd.cut(X["Age"], bins=bins, labels=labels, right=True).astype(str)

def _name_to_title(X):
    title = X["Name"].str.extract(r"\S+, (.*?)\.")
    return title


def _categorize_age_and_sex(X):
    bins = [0, 12, 19, 59, np.inf]
    labels = ["child", "teenager", "adult", "senior"]
    age_cat = pd.cut(X["Age"], bins=bins, labels=labels, right=True).astype(str)
    sex = X["Sex"].fillna("U").str.lower().str[0]
    combined = age_cat + "_" + sex
    return combined.to_frame()


def _categorize_age_and_title(X):
    age_cat = _cat_age(X)
    title = _name_to_title(X)[0]
    combined = age_cat + "_" + title
    return combined.to_frame()

def _categorize_pclass_sex(X):
    sex = X["Sex"].fillna("U").str.lower().str[0]
    combined =  sex + "_"  + X["Pclass"].astype(str)
    return combined.to_frame()


def fam_size(X):
    return (X["SibSp"].fillna(0).astype(int) + X["Parch"].fillna(0).astype(int) + 1).to_frame()

def is_alone(X):
    return (X["SibSp"].fillna(0).astype(int) + X["Parch"].fillna(0).astype(int) == 0).to_frame()


def engineer_features():
    # Engineer "Cabin" -> "HasCabin"
    has_cabin_transformer = FunctionTransformer(lambda x: x.notnull().astype(int), validate=False,
                                                feature_names_out=lambda tf, names: np.array(["HasCabin"]))
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
    # Age x Sex categorization
    cat_age_sex = Pipeline(
        [
            ("age_sex_label", FunctionTransformer(_categorize_age_and_sex, validate=False, feature_names_out=lambda tf, names: np.array(["AgeAndSex"]))),
            ("is_age_sex", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )
    # Age x Title categorization
    cat_age_title = Pipeline(
        [
            ("age_title_label", FunctionTransformer(_categorize_age_and_title, validate=False, feature_names_out=lambda tf, names: np.array(["AgeAndTitle"]))),
            ("age_title", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )
    # PClass x Title categorization
    cat_pclass_sex = Pipeline(
        [
            ("cat_pclass_sex", FunctionTransformer(_categorize_pclass_sex, validate=False, feature_names_out=lambda tf, names: np.array(["PClassAndSex"]))),
            ("pclass_sex", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )
    cat_variables_transf = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("has_cabin", has_cabin_transformer, ["Cabin"]),
            ("decks", decks, ["Cabin"]),
            ("dropper", "drop", ["Ticket"]),
            ("p_title", passenger_title, ["Name"]),
            ("cat_age_sex", cat_age_sex, ["Age", "Sex"]),
            ("cat_age_title", cat_age_title, ["Age", "Name"]),
            # ("cat_pclass_sex", cat_pclass_sex, ["Pclass", "Sex"]),
            ("fam_size", FunctionTransformer(fam_size, validate=False, feature_names_out=lambda tf, names: np.array(["FamilySize"])), ["SibSp", "Parch"]),
            ("is_alone", FunctionTransformer(is_alone, validate=False, feature_names_out=lambda tf, names: np.array(["IsAlone"])), ["SibSp", "Parch"]),
            ("cat_variables", cat_variables_transf, ["Embarked"])
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
