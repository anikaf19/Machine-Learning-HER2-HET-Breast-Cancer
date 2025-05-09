import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer, KNNImputer

data = pd.read_csv('/Users/anikaflorin/Documents/Thesis/data/ML-ready-filtered.csv')

X = data.drop(columns=['pCR'])
y = data['pCR'] #.values

imputer = KNNImputer(n_neighbors=3)
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed,y,test_size=0.3,random_state=42)

# create/define pipelines
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC()) # placeholder
])

noscale_pipeline = Pipeline([
    ('clf', HistGradientBoostingClassifier(random_state=42))
])
# before uploading to GitHub refill hyperparams

parameter_grid = [
    {
        'clf': [SVC()],
        'clf__C': [0.7, 0.8, 1, 1.5],
        'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
        'clf__gamma': ['scale', 'auto']
    },
    {
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [40, 50, 75, 100],
        'clf__max_depth': [None, 10],
        'clf__min_samples_split': [5, 7, 9],
        'clf__min_samples_leaf': [1, 2, 3, 4]
    },
    {
        'clf': [KNeighborsClassifier()],
        'clf__n_neighbors': [3, 4, 5],
        'clf__weights': ['uniform', 'distance'],
        'clf__metric': ['minkowski', 'euclidean', 'manhattan'],
        'clf__leaf_size': [3, 5, 7]
    },
    {
        'clf': [BaggingClassifier(estimator=KNeighborsClassifier(), random_state=42)],
        'clf__n_estimators': [10, 20, 25],
        'clf__max_samples': [0.7, 0.8, 0.85],
        'clf__bootstrap': [True],
        'clf__bootstrap_features': [ False],
        'clf__estimator__n_neighbors': [7, 8, 9],
        'clf__estimator__weights': ['uniform', 'distance'],
        'clf__estimator__metric': ['minkowski', 'euclidean', 'manhattan'],
        'clf__estimator__leaf_size': [2, 3, 5]
    }
]
noscale_parameter_grid = [
    {
        'clf': [HistGradientBoostingClassifier(random_state=42)],
        'clf__learning_rate': [0.01, 0.05, 0.5],
        'clf__max_iter': [75, 100, 150 ],
        'clf__max_depth': [None, 10]
    }
]

grid = GridSearchCV(pipeline, parameter_grid, cv = 5, scoring='accuracy', n_jobs=-1)
noscale_grid = GridSearchCV(noscale_pipeline, noscale_parameter_grid, cv = 5, scoring='accuracy', n_jobs=1)

grid.fit(X_train, y_train)
noscale_grid.fit(X_train, y_train)
#print accuracy and best params for best model scaled and non scaled
if grid.best_score_ > noscale_grid.best_score_:
    print('Best Overall Model (Scaled):', grid.best_estimator_)
    print('Hyperparameters:', grid.best_params_)
    print('Accuracy:', accuracy_score(y_test, grid.predict(X_test)))
else:
    print('Best Overall Model (No Scaling):', noscale_grid.best_estimator_)
    print('Hyperparameters:', noscale_grid.best_params_)
    print('Accuracy:', accuracy_score(y_test, noscale_grid.predict(X_test)))
