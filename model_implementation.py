import pandas as pd
import os
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import accuracy_score, precision_score,average_precision_score, recall_score, f1_score, confusion_matrix, PrecisionRecallDisplay


path = '/Users/anikaflorin/Documents/Thesis/data/'

count_data = pd.read_csv(os.path.join(path,'core_tmm_ML_ready.csv'))
# nohet_subset = pd.read_csv(os.path.join(path,'nohet_tmm_ML_ready.csv'))
# het_subset = pd.read_csv(os.path.join(path,'het_tmm_ML_ready.csv'))

# define 
linearsvm = SVC(
        kernel='linear',
        C =4430.02432,
        class_weight='balanced',
    )

polysvm = SVC(
        kernel='poly',
        C=35.5632,
        degree=2,
        gamma='auto',
        class_weight=None
    )

rf = RandomForestClassifier(
        n_estimators=667,
        criterion='log_loss',  
        max_depth=28,  
        min_samples_split=10,  
        min_samples_leaf=4,  
        max_features='sqrt',  
        class_weight=None,
        random_state=42  
    )

knn = KNeighborsClassifier(
        n_neighbors=4,
        leaf_size=59,
        p=2,
        weights='distance'
    )

models = [linearsvm, polysvm, rf, knn]

k_values = [1000,5000,10000,25000,46000]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def run_experiment(data,models,k_values):
    results = []
    X = data.drop(columns=['pCR'])
    y = data['pCR']

    for model in models:
        if isinstance(model, SVC):
            modelname = f'SVC_{model.kernel}'
        else:
            modelname = type(model).__name__
        print(f'Running model: {modelname}')
        for k in k_values:
            print(f'k={k}')
            fold=0
            for train_idx, test_idx in skf.split(X,y):
                fold+=1
                
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                pipeline = Pipeline([
                    ('imputer', KNNImputer()),
                    ('var_filter', VarianceThreshold(0)),
                    ('filter', SelectKBest(score_func=f_classif, k=k)),
                    ('clf', model) 
                ])
            
                pipeline.fit(X_train,y_train)
                y_train_pred = pipeline.predict(X_train)
                y_pred = pipeline.predict(X_test)
                

                if hasattr(pipeline, 'decision_function'):
                    y_train_scores = pipeline.decision_function(X_train)
                elif hasattr(pipeline, 'predict_proba'):
                    y_train_scores = pipeline.predict_proba(X_train)[:,1]

                if hasattr(pipeline, 'decision_function'):
                    y_test_scores = pipeline.decision_function(X_test)
                elif hasattr(pipeline, 'predict_proba'):
                    y_test_scores = pipeline.predict_proba(X_test)[:,1]
                                
                cm_train = confusion_matrix(y_train,y_train_pred)
                tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
                
                cm_test = confusion_matrix(y_test,y_pred)
                tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
                
                params = model.get_params()
                results.append({
                    'model': modelname,
                    'fold': fold,
                    'k': k,
                    **{f'param_{key}': val for key, val in params.items()},
                    'f1_train': f1_score(y_train,y_train_pred),
                    'accuracy_train': accuracy_score(y_train,y_train_pred),
                    'precision_train': precision_score(y_train,y_train_pred,pos_label=1) if y_train_scores is not None else None,
                    'recall_train': recall_score(y_train,y_train_pred,pos_label=1),
                    'pr_auc_train': average_precision_score(y_train,y_train_scores) if y_train_scores is not None else None,
                    'tn_train': tn_train,
                    'fp_train': fp_train,
                    'fn_train': fn_train,
                    'tp_train': tp_train,
                    'f1_test': f1_score(y_test,y_pred),
                    'accuracy_test': accuracy_score(y_test,y_pred),
                    'precision_test': precision_score(y_test,y_pred,pos_label=1) if y_test_scores is not None else None,
                    'recall_test': recall_score(y_test,y_pred,pos_label=1),
                    'pr_auc_test': average_precision_score(y_test,y_test_scores) if y_test_scores is not None else None,
                    'tn_test': tn_test,
                    'fp_test': fp_test,
                    'fn_test': fn_test,
                    'tp_test': tp_test
                })
    return pd.DataFrame(results)    


df = run_experiment(count_data,models,k_values)
df.to_csv('ml_implementation_results.csv', index=False)
summary = df.groupby(['model','k']).agg({
    'pr_auc_test': ['mean','std'],
    'f1_test': ['mean','std'],
    'recall_test': ['mean','std']
    }).reset_index()
summary.to_csv('ml_implementation_results_summary.csv', index=False)