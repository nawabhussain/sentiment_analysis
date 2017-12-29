import pandas
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import utils

targetClass = 5
data = utils.getDataset(targetClass)
data = utils.MapLabels(targetClass,data)
data = utils.preprocess(data)

model = MultinomialNB(alpha=3.0)
kf = KFold(n_splits=10, random_state=43, shuffle=True)

vectorizer = CountVectorizer(lowercase = True,max_features = 9000,ngram_range=(1, 2))
accurs = []
conf_matrix = []
tp = []
tn = []
fp = []
fn = []
for train_index, test_index in kf.split(data):
    X_train, X_test = data.iloc[train_index]['Comment'], data.iloc[test_index]['Comment']
    y_train, y_test = data.iloc[train_index]['Label'], data.iloc[test_index]['Label']

    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)
    print("Training Data")
    model.fit(train_vectors, y_train)
    prediction = model.predict(test_vectors)
    accur = accuracy_score(y_test, prediction)
    print("Score ",accur)
    accurs.append(accur)
    # tp,tn,fp,fn = confusion_matrix(y_test,prediction)
    # print("TP ",tp)
    # print("TN ",tn)
    # conf_matrix.append(matrix)
print(pandas.np.mean(accurs))
# print(pandas.np.mean(conf_matrix))
# filename = 'finalized_logistic.sav'
# joblib.dump(clf_svc, filename)