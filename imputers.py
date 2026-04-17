from sklearn.impute import KNNImputer
from sklearn.base import TransformerMixin
import pandas as pd

class DataFrameKNNImputer(KNNImputer, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return super().fit(X, y)

    def transform(self, X):
        X_imputed = super().transform(X)
        return pd.DataFrame(X_imputed, columns=self.feature_names_in_, index=X.index)
