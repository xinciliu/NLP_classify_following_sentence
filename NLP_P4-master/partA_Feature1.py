import sys
import csv
import unicodedata
import numpy as np
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess(file, label):
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        storyid = [row[0] for row in reader][1:]
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        ending1 = [row[5] for row in reader][1:]
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        ending2 = [row[6] for row in reader][1:]
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        if label:
            answers = [row[7] for row in reader][1:]
        else:
            answers = None
    return storyid, ending1, ending2, answers


def feature_x(file, label):
    punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
    storyid, ending1, ending2, answers = preprocess(file, label)
    lenchars1 = []
    lenwords1 = []
    lenchars2 = []
    lenwords2 = []
    for count in range(len(storyid)):
        lenchars1.append(len(ending1[count]))
        lenwords1.append(len(list(filter(None, ending1[count].translate(punctuation).strip().split(' ')))))
        lenchars2.append(len(ending2[count]))
        lenwords2.append(len(list(filter(None, ending1[count].translate(punctuation).strip().split(' ')))))
    return lenchars1, lenwords1, lenchars2, lenwords2, answers, storyid, ending1, ending2

if __name__ == "__main__":
    trainfile = './train.csv'
    trainlenchars1, trainlenwords1, trainlenchars2, trainlenwords2, answers, _, trainending1, trainending2 = feature_x(trainfile, True)
    y = [i for i in answers]
    trainy = np.array(y)

    testfile = './dev.csv'
    testlenchars1, testlenwords1, testlenchars2, testlenwords2, test_answers, storyid, testending1, testending2 = feature_x(testfile, True)

    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(3, 3),
        max_features=10000)
    word_vectorizer.fit(trainending1+trainending2+testending1+testending2)
    train_word_features1 = word_vectorizer.transform(trainending1)
    train_word_features2 = word_vectorizer.transform(trainending2)
    test_word_features1 = word_vectorizer.transform(testending1)
    test_word_features2 = word_vectorizer.transform(testending2)

    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 5),
        max_features=50000)
    char_vectorizer.fit(trainending1+trainending2+testending1+testending2)
    train_char_features1 = char_vectorizer.transform(trainending1)
    train_char_features2 = char_vectorizer.transform(trainending2)
    test_char_features1 = char_vectorizer.transform(testending1)
    test_char_features2 = char_vectorizer.transform(testending2)

    train_features = hstack([train_char_features1, train_char_features2, train_word_features1, train_word_features2,
                             #np.reshape(np.array(trainlenchars1), (-1, 1)), np.reshape(np.array(trainlenchars2), (-1, 1)),
                             #np.reshape(np.array(trainlenwords1), (-1, 1)), np.reshape(np.array(trainlenwords2), (-1, 1))
                             ])
    test_features = hstack([test_char_features1, test_char_features2, test_word_features1, test_word_features2,
                            #np.reshape(np.array(testlenchars1), (-1, 1)), np.reshape(np.array(testlenchars2), (-1, 1)),
                            #np.reshape(np.array(testlenwords1), (-1, 1)), np.reshape(np.array(testlenwords2), (-1, 1))
                            ])

    classifier = LogisticRegression(C=1.0, solver='liblinear')
    classifier.fit(train_features, trainy)
    prediction = classifier.predict(test_features)
    score = classifier.score(test_features, np.array(test_answers))
    print(score)

    with open('partA_Feature1.csv', mode='w') as report_file:
        report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        report_writer.writerow(['Id', 'Prediction'])
        for i in range(len(storyid)):
            report_writer.writerow([storyid[i], prediction[i]])
