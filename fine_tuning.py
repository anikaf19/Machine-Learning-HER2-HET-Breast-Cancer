import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.base import TransformerMixin
from imputers import DataFrameKNNImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    confusion_matrix,
    PrecisionRecallDisplay
)
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.plots import (
    plot_convergence,
    plot_evaluations,
    plot_objective,
    plot_regret
)
from sklearn.inspection import permutation_importance


PATH = '/Users/anikaflorin/Documents/Thesis/data/'
RANDOM_STATE = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
N_ITER = 100

core_counts = pd.read_csv(os.path.join(PATH, 'core_tmm_ML_ready.csv'))
res_counts = pd.read_csv(os.path.join(PATH, 'res_tmm_ML_ready.csv'))

search_space = {
    'filter__k': Integer(9800, 10200),
    'clf__kernel': Categorical(['poly']),
    'clf__C': Real(33, 38, prior='log-uniform'),
    'clf__gamma': Categorical(['auto']),
    'clf__degree': Integer(2,3),
    'clf__class_weight': Categorical([None]),
    # 'clf__coef0': Real(0, 20),
    # 'clf__shrinking': Categorical([True, False]),
    # 'clf__tol': Real(1e-5, 1e-1),
    'clf__probability': Categorical([False])
}
scoring = {
    'f1': 'f1',
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'pr_auc': 'average_precision'
}


def build_pipeline():
    return Pipeline([
        ('imputer', DataFrameKNNImputer()),
        ('var_filter', VarianceThreshold(0)),
        ('filter', SelectKBest(score_func=f_classif)),
        ('clf', SVC())
    ])

def train_nested_cv(model_name, data, dataname):
    X = data.drop(columns=['pCR'])
    y = data['pCR']

    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_metrics = []
    all_cv_results = []

    best_global_score = -np.inf
    best_global_model = None

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        print(f"\n=== Outer Fold {fold} ===")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = build_pipeline()

        opt = BayesSearchCV(
            estimator=pipeline,
            search_spaces=search_space,
            n_iter=N_ITER,
            scoring=scoring,
            refit='f1',
            cv=N_INNER_SPLITS,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            return_train_score=True
        )

        opt.fit(X_train, y_train)

        best_model = opt.best_estimator_
        
        y_train_pred = best_model.predict(X_train)
        y_train_scores = best_model.decision_function(X_train)
        y_pred = best_model.predict(X_test)
        y_scores = best_model.decision_function(X_test)

        fold_f1 = f1_score(y_test, y_pred)
        fold_pr = average_precision_score(y_test, y_scores)

        fold_metrics.append({
            'fold': fold,
            'f1': fold_f1,
            'pr_auc': fold_pr,
            'inner_cv_f1': opt.best_score_,
            **opt.best_params_
        })

        if fold_f1 > best_global_score:
            best_global_score = fold_f1
            best_global_model = best_model
            best_fold_data = (X_train, y_train, y_train_pred, X_test, y_test, y_pred, y_scores)
            best_opt = opt

        results_df = pd.DataFrame(opt.cv_results_)
        params_expanded = pd.json_normalize(results_df['params'])

        clean_df = pd.concat(
            [
                results_df.filter(regex='mean_(test|train)_'),
                params_expanded
            ],
            axis=1
        )
        clean_df['fold'] = fold
        clean_df['rank'] = results_df['rank_test_f1']
        all_cv_results.append(clean_df)

    metrics_df = pd.DataFrame(fold_metrics)
    print("\n=== CV Summary ===")
    print(metrics_df.describe())

    outer_cv_df = pd.DataFrame(fold_metrics)
    inner_cv_df = pd.concat(all_cv_results, ignore_index=True)
    
    outer_cv_df.to_csv(
        f'{PATH}{dataname}_outer_cv_results.csv',
        index=False
    )

    inner_cv_df.to_csv(
        f'{PATH}{dataname}_inner_cv_results.csv',
        index=False
    )

    joblib.dump(best_global_model, f'{PATH}best_poly_svc_{dataname}.pkl')

    X_train, y_train, y_train_pred, X_test, y_test, y_pred, y_scores = best_fold_data
    
    joblib.dump(X_test, f'{PATH}X_test_{dataname}.joblib')
    joblib.dump(y_test, f'{PATH}y_test_{dataname}.joblib')
    joblib.dump(y_pred, f'{PATH}y_pred_{dataname}.joblib')
    joblib.dump(y_scores, f'{PATH}y_scores_{dataname}.joblib')

    PrecisionRecallDisplay.from_predictions(
        y_test,
        y_scores,
        pos_label=1,
        name=model_name
    )
    plt.savefig(f'{PATH}prauc_{dataname}.png')

    labels = ['No pCR', 'pCR']
    fig, axes = plt.subplots(1,2, figsize=(12,5))

    sns.heatmap(confusion_matrix(y_train,y_train_pred), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Training Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Testing Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(f'{PATH}cm_{dataname}.png')
    plt.show()

    result = best_opt.optimizer_results_[0]

    plot_convergence(result)
    plt.savefig(f'{PATH}convergence_{dataname}.png')

    plot_evaluations(result)
    plt.savefig(f'{PATH}evaluations_{dataname}.png')

    plot_objective(result)
    plt.savefig(f'{PATH}objective_{dataname}.png')

    plot_regret(result)
    plt.savefig(f'{PATH}regret_{dataname}.png')

    plt.close('all')

    return metrics_df, best_global_model


metrics, final_model = train_nested_cv(
    model_name='Polynomial SVC',
    data=core_counts,
    dataname='tmm_core'
)