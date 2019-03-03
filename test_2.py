import pandas as pd
import numpy as np

df_1 = pd.read_csv('amazon_cells_labelled.txt', names=['sentence','label'],sep='\t')
names = ['col1','col2']
print(names)

from sklearn.feature_extraction.text import CountVectorizer
sentences = ['John likes ice cream', 'John hates chocolate.']
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
vectorizer.vocabulary_
vectorizer.transform(sentences).toarray()

from sklearn.model_selection import train_test_split

sentences = df_1['sentence'].values
print(sentences)
y = df_1['label'].values
print(y)

sentences_train,sentences_test,y_train,y_test = train_test_split(sentences,y,test_size=0.25,random_state=1000)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
print(X_train)

from sklearn.linear_model import LogisticRegression

lg_classifier = LogisticRegression()
lg_classifier.fit(X_train, y_train)
score = lg_classifier.score(X_test, y_test)
print(score)
model_predict =lg_classifier.predict(X_test)
print(model_predict)
print(y_test)
from sklearn.metrics import accuracy_score
logistics_accuracy = accuracy_score(y_test,model_predict)
print(logistics_accuracy)


from sklearn.neighbors import KNeighborsClassifier

lk_classifier =KNeighborsClassifier()
b = lk_classifier.fit(X_train,y_train)
print(b)
# score_1 = lk_classifier.score_1(X_test,y_test)
model_predict1 = lk_classifier.predict(X_train)
print(model_predict1)
print(y_train)
kneighbors_accuracy = accuracy_score(y_train,model_predict1)
print(kneighbors_accuracy)

from sklearn.tree import DecisionTreeClassifier
ld_classifier =DecisionTreeClassifier()
c = ld_classifier.fit(X_train,y_train)
print(c)
# score_1 = lk_classifier.score_1(X_test,y_test)
model_predict2 = ld_classifier.predict(X_train)
print(model_predict2)
# print(y_train)
decision_tree_accuracy = accuracy_score(y_train,model_predict2)
print(decision_tree_accuracy)


from sklearn.ensemble  import RandomForestClassifier
lr_classifier =RandomForestClassifier()
d = lr_classifier.fit(X_train,y_train)
print(d)
# score_1 = lk_classifier.score_1(X_test,y_test)
model_predict3 = lr_classifier.predict(X_train)
print(model_predict3)
# print(y_train)
randomforest_accuracy = accuracy_score(y_train,model_predict3)
print(randomforest_accuracy)


print(logistics_accuracy)
print(kneighbors_accuracy)
print(decision_tree_accuracy)
print(randomforest_accuracy)




# from sklearn.naive_bayes import GaussianNB
# nb_classifier = GaussianNB()
# d = nb_classifier.fit(X_train,y_train)
# print(d)
# score_1 = lk_classifier.score_1(X_test,y_test)
# model_predict3 = lnb_classifier.predict(X_train)
# print(model_predict3)
# print(y_train)
# naivebayes_accuracy = accuracy_score(y_train,model_predict3)
# print(naivebayes_accuracy)

