# sms-classifier1
import pandas as pd
import numpy as np
from sklearn.feature extraction.text import CountVectorizer
from sklearn.model selection import train_test_split
from sklearn.naive bayes import MultinomialNB
 data pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS Spam Detection/master/spam.csv", encoding 'latin-1')
data.head()
data = data[["class", "message"]]
x np.array(data["message"])
y np.array(data["class"])
cv =CountVectorizer()
X =cv.fit_transform(x) # Fit the Data
X_train, X_test, y train, y test train_test_split(x, y, test_size-0.01, random state-42)
clf= MultinomialNB()

clf.fit(x_train,y_train)
sample =input("Enter a message:")
data =sample input("Enter a message:") data cv.transform([sample]).toarray()

print(clf.predict(data))
X_train, X_test, y train, y test train_test_split(x, y, test_size-0.01, random state-42)
 clf =MultinomialNB()

clf.fit(x_train,y_train)
sample= input("Enter a message:")

data= cv.transform([sample]).toarray()

print(clf.predict(data))
