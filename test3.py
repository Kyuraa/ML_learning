#KNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
 

data = {
    "ID":range(1,11),
    "Yaş":[25,30,35,40,45,50,55,60,65,70],
    "Aylık Harcama":[40,60,70,80,90,100,110,120,130,140],
    "Hizmet Süresi": [12,24,36,48,60,72,84,96,108,120],
    "Churn":[0,0,0,0,1,1,1,1,1,1]
}

df= pd.DataFrame(data)
x= df[["Yaş","Aylık Harcama","Hizmet Süresi"]]
y= df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state=42)

scaler= StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)

print(X_train_scaled.shape)
print(X_test_scaled.shape)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled,y_train)
y_pred = knn.predict(X_test_scaled)  # Predict labels for test data

print("Accuracy score: "+ str(accuracy_score(y_test, y_pred)))  # Compare predictions with true labels
