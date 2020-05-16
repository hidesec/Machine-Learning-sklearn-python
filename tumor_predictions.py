#hidesec(https://github.com/hidesec)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#load dataset
data = load_breast_cancer()

#organize our data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

#split our data
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

#initialize our classifier
gnb = GaussianNB()

#train our classifier
model = gnb.fit(train, train_labels)

#make predictions
preds = gnb.predict(test)

#evaluate accuracy
print(accuracy_score(test_labels, preds))
