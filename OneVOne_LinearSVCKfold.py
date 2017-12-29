import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import utils

targetClass = 3
data = utils.getDataset(targetClass)
data = utils.MapLabels(targetClass,data)
data = utils.preprocess(data)

model =  Pipeline([
    ('vect', CountVectorizer(lowercase = True,max_features = 4000,ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsOneClassifier(LinearSVC())),
    ])
kf = KFold(n_splits=10, random_state=43, shuffle=True)

accurs = []
conf_matrix = []
tp = []
tn = []
fp = []
fn = []
for train_index, test_index in kf.split(data):
    X_train, X_test = data.iloc[train_index]['Comment'], data.iloc[test_index]['Comment']
    y_train, y_test = data.iloc[train_index]['Label'], data.iloc[test_index]['Label']
    model.fit(pandas.np.asarray(X_train), pandas.np.asarray(y_train))
    prediction = model.predict(X_test)
    accur = accuracy_score(y_test, prediction)
    print("Score ", accur)
    accurs.append(accur)
    # tp,tn,fp,fn = confusion_matrix(y_test,prediction)
    # print("TP ",tp)
    # print("TN ",tn)
    # conf_matrix.append(matrix)
print(pandas.np.mean(accurs))
# print(pandas.np.mean(conf_matrix))
# filename = 'finalized_logistic.sav'
# joblib.dump(clf_svc, filename)