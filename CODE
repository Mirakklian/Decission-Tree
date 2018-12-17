import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#% matplotlib inline

data=pd.read_csv("C:\\Users\\Pratik Dutta\\Desktop\\DATASET.csv")
print(data.head(10))
print("# no of passenger in the data set:", +(len(data)))

## Data Wrangling----remove the unwanted dataset
print(data.isnull().sum())
sns.heatmap(data.isnull(),yticklabels=False,cmap="viridis")
plt.show()

pol=pd.get_dummies(data['Polarity'],drop_first=True) #drop the first colum..if positive=0,neutral=0..then it negative
print(pol)

##Concatinate the data field into our data set

data=pd.concat([data,pol],axis=1)
data.drop(['Polarity','tweet'],axis=1,inplace=True)
print(data.head(10))

## TRAIN MY DATASET

X=data.drop("positive",axis=1)
y=data["positive"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score

print( "The Accuracy of the prediction using Decision Treeis: ",accuracy_score(y_test, y_pred))
