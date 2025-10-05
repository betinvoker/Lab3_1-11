import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data = {
    'age': [25,30,22,45,33,28,50,21,35,40],
    'gender': ['m', 'w', 'm', 'w', 'm', 'w', 'm', 'w', 'm', 'w'],
    'location': ['Rostov-on-Don', 'Taganrog', 'Azov', 'Bataisk', 
                 'Rostov-on-Don', 'Taganrog', 'Azov', 'Bataisk', 
                 'Rostov-on-Don', 'Taganrog'],
    'traffic_score': [0,1,0,1,0,1,0,0,1,1],
    'clicks': [5,10,2,15,3,8,1,4,12,9],
    'time_on_site': [30,60,15,120,45,90,20,30,75,100],
    'landing_version': [0,0,1,1,0,1,0,1,0,1],
    'converted': [0,1,0,1,0,1,0,0,1,1]
}

df = pd.DataFrame(data)

print(df.head())

df_encoded = pd.get_dummies(df, columns=['gender', 'location'], drop_first=True)

X = df_encoded.drop('converted', axis=1)
y = df_encoded['converted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print(df['converted'].value_counts())