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

# Compute performance metrics
train_accuracy = accuracy_score(y_train,y_train_pred)
train_precision = precision_score(y_train,y_train_pred,pos_label=1)
train_confusion = confusion_matrix(y_train,y_train_pred)
    
y_train_scores = svm.decision_function(X_train) # predict probabilities for AUC-ROC
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_scores, pos_label=1)
train_auc = auc(fpr_train,tpr_train)

test_accuracy = accuracy_score(y_test,y_pred)
test_precision = precision_score(y_test,y_pred,pos_label=1)
test_confusion = confusion_matrix(y_test,y_pred)
    
y_test_scores = svm.decision_function(X_test) # predict probabilities for AUC-ROC
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_scores, pos_label=1)
test_auc = auc(fpr_test, tpr_test)

# print metrics: accuracy, confusion matrix
print(f"Train Accuracy: {train_accuracy}")
print(f"Train Precision: {train_precision}")
print(f"Train Confusion: {train_confusion}")

print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Confusion: {test_confusion}")

# create and display confusion matrix images
labels = ['No pCR', 'pCR']

fig, axes = plt.subplots(1,2, figsize=(12,5))

sns.heatmap(train_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title('Train Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

sns.heatmap(test_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title('Test Confusion Matrix')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.show()

print(f'Training AUC-ROC: {train_auc:.4f}')
print(f'Testing AUC-ROC: {test_auc:.4f}')

# plot and print ROC curves
plt.figure(figsize=(8,6))
plt.plot(fpr_train, tpr_train, label=f'Train AUC = {train_auc:.4f}', 
         color = 'skyblue',
         lw = 2)
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {test_auc:.4f}',
         color = 'salmon',
         lw = 2)
plt.plot([0,1], [0,1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('ROC Curve (Train vs Test)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout
plt.show()
