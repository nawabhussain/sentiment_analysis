import numpy
import pandas
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import learning_curve


def getDataset(targetClass):
    if (targetClass == 5):
        return getFiveClassedDataset()
    elif (targetClass == 3):
        return getThreeClassedDataset()
    elif (targetClass == 2):
        return getTwoClassedDataset()

def getFiveClassedDataset():
    return pandas.read_csv('dataset/Dataset.csv')

def getThreeClassedDataset():
    return pandas.read_csv('dataset/Dataset_Tertiary_Labels.csv')

def getTwoClassedDataset():
    return pandas.read_csv('dataset/Dataset_Binary_Labels.csv')

def MapLabels(targetClass,data):
    if(targetClass==5):
        data['Label'] = data.Label.map(
            {'Highly Negative': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3, 'Highly Positive': 4})
        return data
    elif (targetClass == 3):
        data['Label'] = data.Label.map({ 'Negative': 0, 'Neutral': 1, 'Positive': 2})
        return data
    elif (targetClass == 2):
        data['Label'] = data.Label.map({ 'Negative': 0, 'Positive': 1})
        return data

def preprocess(data):
    stop_words = set(stopwords.words('english'))
    listStop = list(stop_words)
    wordnet_lemmatizer = WordNetLemmatizer()
    data['Comment'] = data['Comment'].replace(to_replace='http\S+', value=' ', regex=True)  # Removing urls
    data['Comment'] = data['Comment'].replace(to_replace='[^A-Za-z]+', value=' ',
                                              regex=True)  # Removing special characters
    data['Comment'] = data['Comment'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (listStop)]))  # Removing stop words
    #region Not using the hidden part
    # data['Comment'] = data['Comment'].apply(lambda x: ' '.join([word for word, pos in pos_tag(x.split()) if not pos == 'NNP']))# Removing NNP
    # data['Comment'] = data['Comment'].apply(lambda x: ' '.join([replaceWord(stem(word.lower())) for word in x.split()]))# Replacing common mis-spelled/same meaning words
    # data['Comment'] = data['Comment'].replace(to_replace='\d+', value=' ', regex=True)  # Removing numbers
    #endregion
    data['Comment'] = data['Comment'].apply(lambda x: wordnet_lemmatizer.lemmatize(x))
    return data

commonDict = {'the': 'the', 'be': 'be', 'and': 'and', 'of': 'of', 'a': 'a', 'in': 'in', 'to': 'to', 'have': 'have', 'it': 'it',
     'i': 'i', 'that': 'that', 'for': 'for', 'you': 'you', 'he': 'he', 'with': 'with', 'on': 'on', 'do': 'do',
     'say': 'say', 'this': 'this', 'they': 'they', 'is': 'is', 'an': 'an', 'at': 'at', 'but': 'but', 'we': 'we',
     'his': 'his', 'from': 'from', 'not': 'not', 'by': 'by', 'she': 'she', 'or': 'or', 'as': 'as', 'what': 'what',
     'go': 'go', 'their': 'their', 'can': 'can', 'who': 'who', 'get': 'get', 'if': 'if', 'would': 'would', 'her': 'her',
     'all': 'all', 'my': 'my', 'make': 'make', 'about': 'about', 'know': 'know', 'will': 'will', 'up': 'up',
     'one': 'one', 'time': 'time', 'has': 'has', 'been': 'been', 'there': 'there', 'year': 'year', 'so': 'so',
     'think': 'think', 'when': 'when', 'which': 'which', 'them': 'them', 'some': 'some', 'me': 'me', 'people': 'people',
     'take': 'take', 'out': 'out', 'into': 'into', 'just': 'just', 'see': 'see', 'him': 'him', 'your': 'your',
     'come': 'come', 'could': 'could', 'now': 'now', 'than': 'than', 'like': 'like', 'other': 'other', 'how': 'how',
     'then': 'then', 'its': 'its', 'our': 'our', 'two': 'two', 'more': 'more', 'these': 'these', 'want': 'want',
     'way': 'way', 'look': 'look', 'first': 'first', 'also': 'also', 'new': 'new', 'because': 'because', 'day': 'day',
     'use': 'use', 'no': 'no', 'man': 'man', 'find': 'find', 'here': 'here', 'thing': 'thing', 'give': 'give',
     'many': 'many', 'well': 'well', 'was': 'was', 'are': 'are', 'were': 'were',
                  'rrb': 'rrb', 'lrb': 'lrb', 'had': 'had', 'did': 'did', 'be':'be', 'where': 'where',
                  'those': 'those', 'through': 'through', 'though': 'though',
              'while':'while', 'should': 'should', 've':'ve', 'ca':'ca', 'am':'am'};


def isCommon(word):
    if commonDict.get(word, None):
        return True
    return False

customDict = {'film':'movie', 'movi':'movie', 'veri':'very',
              'charact': 'character', 'stori': 'story',
              'onli':'only', 'realli':'really', 'littl':'little',
              'comedi':'comedy', 'ani':'any', 'tri': 'try', 'anoth':'another',
              'pictur':'movie', 'beauti': 'beautiful', 'alway':'always',
              'howev':'however', 'becom':'become', 'someth':'something',
              'funni':'funny', 'befor':'before', 'everi':'every', 'whi': 'why',
              'believ': 'believe', 'pretti': 'pretty', 'noth':'nothing', 'seri':'serial',
              'favotir':'favorite', 'famili':'family', 'especi':'especially',
              'probabl':'probably', 'minut':'minute', 'cours':'course',
              'complet':'complete', 'surpris':'surprise'}
def replaceWord(word):
    result = customDict.get(word, None)
    if (result != None):
        return result
    return word

posDict = {'CC':'CC', 'CD':'CD', 'DT':'DT',
              'EX': 'EX', 'FW': 'FW',
              'LS':'LS','MD':'MD', 'PDT':'PDT', 'POS':'POS',
              'PRP': 'PRP', 'SYM': 'SYM',
              'RP':'RP','TO':'TO', 'UH':'UH', 'WDT':'WDT',
              'WP': 'WP', 'WRB': 'WRB'}
def isAllowedPOS(pos):
    if posDict.get(pos, None):
        return True
    return False

# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
#     """
#     Generate a simple plot of the test and training learning curve.
#
#     Parameters
#     ----------
#     estimator : object type that implements the "fit" and "predict" methods
#         An object of that type which is cloned for each validation.
#
#     title : string
#         Title for the chart.
#
#     X : array-like, shape (n_samples, n_features)
#         Training vector, where n_samples is the number of samples and
#         n_features is the number of features.
#
#     y : array-like, shape (n_samples) or (n_samples, n_features), optional
#         Target relative to X for classification or regression;
#         None for unsupervised learning.
#
#     ylim : tuple, shape (ymin, ymax), optional
#         Defines minimum and maximum yvalues plotted.
#
#     cv : int, cross-validation generator or an iterable, optional
#         Determines the cross-validation splitting strategy.
#         Possible inputs for cv are:
#           - None, to use the default 3-fold cross-validation,
#           - integer, to specify the number of folds.
#           - An object to be used as a cross-validation generator.
#           - An iterable yielding train/test splits.
#
#         For integer/None inputs, if ``y`` is binary or multiclass,
#         :class:`StratifiedKFold` used. If the estimator is not a classifier
#         or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
#
#         Refer :ref:`User Guide <cross_validation>` for the various
#         cross-validators that can be used here.
#
#     n_jobs : integer, optional
#         Number of jobs to run in parallel (default 1).
#     """
#     plot.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = numpy.mean(train_scores, axis=1)
#     train_scores_std = numpy.std(train_scores, axis=1)
#     test_scores_mean = numpy.mean(test_scores, axis=1)
#     test_scores_std = numpy.std(test_scores, axis=1)
#     plt.grid()
#
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")
#
#     plt.legend(loc="best")
#     return plt
