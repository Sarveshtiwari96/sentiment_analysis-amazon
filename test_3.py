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



from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()
d = nb_classifier.fit(X_train.toarray(),y_train)
print(d)
# score_1 = nb_classifier.score_1(X_test,y_test)
model_predict3 = nb_classifier.predict(X_train.toarray())
print(model_predict3)
print(y_train)
from sklearn.metrics import accuracy_score
naivebayes_accuracy = accuracy_score(y_train,model_predict3)
print(naivebayes_accuracy)
