import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns


transcripts = pd.read_csv("/Users/anikaflorin/Documents/Thesis/data/ML-ready-filtered.csv")
X = transcripts.drop(columns=['pCR'])
y = transcripts['pCR']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(
    kernel='linear',
    C = .8,
    gamma='scale')
svm.fit(X_train,y_train)

y_train_pred = svm.predict(X_train)
y_pred = svm.predict(X=X_test)


result = permutation_importance(svm,X_test,y_test, n_repeats=10, random_state=42, n_jobs=-1)

perm_df = pd.DataFrame({
    'Feature': X.columns.values,
    'Importance_Mean': result.importances_mean,
    'Importance_STD': result.importances_std
}).sort_values(by='Importance_Mean', ascending=False)

top_features = perm_df.head(20)
plt.figure(figsize=(10,8))
sns.barplot(
    data=top_features,
    y='Feature',
    x ='Importance_Mean',
    #xerr=top_features['Importance_STD'],
    palette='viridis'
)
plt.title('Top 20 Feature Importance (Permutation)')
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
print(perm_df.head(20))
