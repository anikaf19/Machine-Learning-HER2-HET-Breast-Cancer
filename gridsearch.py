import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.impute import KNNImputer
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence, plot_evaluations, plot_objective, plot_regret
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/anikaflorin/Documents/Thesis/data/core_tmm_ML_ready.csv')

X = data.drop(columns=['pCR'])
y = data['pCR']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


pipeline = Pipeline([
    ('imputer', KNNImputer()),
    ('clf', SVC()) 
])

svc_linear_grid = {
    'clf': [SVC()],
    'clf__kernel': Categorical(['linear']),
    'clf__C': Real(1e-6, 1e3, 'log-uniform'),
    'clf__class_weight': Categorical(['balanced',None])
}

svc_poly_grid = {
    'clf': [SVC()],
    'clf__kernel': Categorical(['poly']),
    'clf__C': Real(1e-6, 1e3, 'log-uniform'),
    'clf__degree': Integer(2,5), 
    'clf__gamma': Categorical(['scale','auto']),
    'clf__class_weight': Categorical(['balanced',None])
}

svc_rbf_grid = {
    'clf': [SVC()],
    'clf__kernel': Categorical(['rbf']), 
    'clf__C': Real(1e-6, 1e3, 'log-uniform'),
    'clf__gamma': Categorical(['scale', 'auto']),
    'clf__class_weight': Categorical(['balanced',None])
}

rf_grid = {
    'clf': Categorical([RandomForestClassifier(random_state=25)]),
    'clf__n_estimators': Integer(50,800),
    'clf__criterion': Categorical(['gini','entropy','log_loss']),
    'clf__max_depth':Integer(3,30),
    'clf__min_samples_split': Integer(2,10),
    'clf__min_samples_leaf': Integer(1,5),
    'clf__max_features': Categorical(['sqrt','log2',0.1,0.2,0.5]),
    'clf__class_weight': Categorical([None,'balanced'])
}

knn_grid = {
    'clf': Categorical([KNeighborsClassifier()]),
    'clf__n_neighbors': Integer(3,25),
    'clf__weights': Categorical(['uniform','distance']),
    'clf__leaf_size': Integer(3,60),
    'clf__p': Categorical([1,2])
}

ensemble_knn_grid = {
    'clf': Categorical([BaggingClassifier(estimator=KNeighborsClassifier(), random_state=25)]),
    'clf__n_estimators': Integer(10,100),
    'clf__max_samples': Real(0.5,1.0),
    'clf__estimator__n_neighbors': Integer(3,25),
    'clf__estimator__weights': Categorical(['uniform', 'distance']),
    'clf__estimator__leaf_size': Integer(3,60),
    'clf__estimator__p': Categorical([1,2])
}

def RunBayesSearch(grid,name):
    search = BayesSearchCV(
        pipeline,
        grid,
        cv=5,
        scoring=None,
        n_iter=100,
        n_jobs=-1
    )
    search.fit(X_train,y_train)
    
    train_acc = accuracy_score(y_train,search.predict(X_train))
    test_acc =  accuracy_score(y_test, search.predict(X_test))
    train_f1 = f1_score(y_train, search.predict(X_train))
    test_f1 = f1_score(y_test, search.predict(X_test))
    best_params = search.best_params_
    result = search.optimizer_results_[0]
    print('-' * 60)
    print(f'{name} Best Combo Train Accuracy: {train_acc:.4f}')
    print(f'{name} Best Combo Test Accuracy: {test_acc:.4f}')
    print(f'{name} Best Combo Train F1: {train_f1:.4f}')
    print(f'{name} Best Combo Test F1: {test_f1:.4f}')
    print(f'{name} Best Combo Params: {best_params}')
    plot_convergence(result)
    plt.title(f'Convergence Plot: {name}')
    plt.show()
    plot_evaluations(result)
    plt.title(f'Evaluation Plot: {name}')
    plt.show()
    plot_objective(result)
    plt.title(f'Objective Plot: {name}')
    plt.show()
    plot_regret(result)
    plt.show()
    plt.close('all')
    print('-' * 60)
    return result

svc_linear_search = RunBayesSearch(svc_linear_grid,'Linear SVC')
svc_poly_search = RunBayesSearch(svc_poly_grid,'Polynomial SVC')
svc_rbf_search = RunBayesSearch(svc_rbf_grid,'RBF SVC')
rf_search = RunBayesSearch(rf_grid,'RF')
knn_search = RunBayesSearch(knn_grid,'KNN')
ensemble_knn_search = RunBayesSearch(ensemble_knn_grid, 'Ensemble Subspace KNN')