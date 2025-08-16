#decision tree

import pandas

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

iris = load_iris()
x= iris.data
y = iris.target
features = iris.feature_names
targets = iris.target_names

#test/train split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=42)

#model training

dtree= DecisionTreeClassifier(random_state=42)
dtree.fit(x_train,y_train)

y_pred= dtree.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:"+ str(accuracy*100))

plt.figure(figsize=(12,8))
tree.plot_tree(dtree,feature_names=features,class_names=targets,filled=True,rounded=True)
plt.show()