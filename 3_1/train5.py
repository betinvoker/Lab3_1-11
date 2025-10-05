import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score
)

np.random.seed(42)
n_samples = 1000

genders = ['m','w']
locations = ['Rostov-on-Don', 'Taganrog', 'Azov', 'Bataisk']

data = {
    'age': np.random.randint(18,65, size=n_samples),
    'gender':  np.random.choice(genders, size=n_samples),
    'location': np.random.choice(locations, size=n_samples),
    'traffic_score': np.random.randint(0,2, size=n_samples),
    'clicks': np.random.randint(1,20, size=n_samples),
    'time_on_site': np.random.randint(10,300, size=n_samples),
    'landing_version': np.random.randint(0,2, size=n_samples)
}

df = pd.DataFrame(data)

df['converted'] = (
    (df['clicks'] > 7) &
    (df['time_on_site'] > 60) &
    (df['traffic_score'] == 0) &
    (df['landing_version'] == 1)
).astype(int)

df.loc[np.random.rand(n_samples) < 0.1,'converted'] = 1 - df['converted']

df_encoded = pd.get_dummies(df, columns=['gender', 'location'], drop_first=True)

X = df_encoded.drop('converted', axis=1)
y = df_encoded['converted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)

y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC-кривая и AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Матрица ошибок (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Precision-Recall кривая
precision, recall, _ = precision_recall_curve(y_test, y_proba)
average_precision = average_precision_score(y_test, y_proba)
plt.figure(figsize=(10, 5))
plt.step(recall, precision, where='post', label=f'Precision-Recall (AP ={average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()