import pandas as pd
import pickle
import joblib
from joblib import load
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.base import TransformerMixin
from sklearn.inspection import permutation_importance
# from fine_tuning2 import DataFrameKNNImputer
import matplotlib.pyplot as plt
import seaborn as sns

class DataFrameKNNImputer(KNNImputer, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return super().fit(X, y)

    def transform(self, X):
        X_imputed = super().transform(X)
        return pd.DataFrame(X_imputed, columns=self.feature_names_in_, index=X.index)


path = '/Users/anikaflorin/Documents/Thesis/data/best_poly_svc_tmm_core.pkl'
with open(path, 'rb') as file:
    pipeline = joblib.load(path)
    

print(type(pipeline))
X_test = load('X_test.joblib')
y_test = load('y_test.joblib')

X_test = X_test[pipeline.feature_names_in_]

y_pred = pipeline.predict(X_test)


result = permutation_importance(pipeline,
                                X_test,
                                y_test,
                                n_repeats=10,
                                random_state=42,
                                n_jobs=-1)

perm_df = pd.DataFrame({
    'Feature': X_test.columns.values,
    'Importance_Mean': result.importances_mean,
    'Importance_STD': result.importances_std
}).sort_values(by='Importance_Mean', ascending=False)

top_features = perm_df.head(30)
n = len(top_features)
height = max(8, n*0.4)

plt.figure(figsize=(10,8))

sns.barplot(
    data=top_features,
    y='Feature',
    x ='Importance_Mean',
    #xerr=top_features['Importance_STD'],
    color='steelblue'
)

plt.title('Top 20 Feature Importance (Permutation)')
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print(top_features)