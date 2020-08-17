import sys
import csv
import nltk
import heapq
import spacy
import unicodedata
import numpy as np
from pandas import DataFrame
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB


def preprocess(file, label):
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        storyid = [row[0] for row in reader][1:]
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        inputsen1 = [row[1] for row in reader][1:]
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        inputsen2 = [row[2] for row in reader][1:]
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        inputsen3 = [row[3] for row in reader][1:]
    with open(file, 'r', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        inputsen4 = [row[4] for row in reader][1:]
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

    punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

    for i, sentence in enumerate(inputsen1):
        temp = sentence.translate(punctuation).strip()
        tempList = temp.split(' ')
        tempList = list(filter(None, tempList))
        inputsen1[i] = tempList.copy()

    for i, sentence in enumerate(inputsen2):
        temp = sentence.translate(punctuation).strip()
        tempList = temp.split(' ')
        tempList = list(filter(None, tempList))
        inputsen2[i] = tempList.copy()

    for i, sentence in enumerate(inputsen3):
        temp = sentence.translate(punctuation).strip()
        tempList = temp.split(' ')
        tempList = list(filter(None, tempList))
        inputsen3[i] = tempList.copy()

    for i, sentence in enumerate(inputsen4):
        temp = sentence.translate(punctuation).strip()
        tempList = temp.split(' ')
        tempList = list(filter(None, tempList))
        inputsen4[i] = tempList.copy()

    for i, sentence in enumerate(ending1):
        temp = sentence.translate(punctuation).strip()
        tempList = temp.split(' ')
        tempList = list(filter(None, tempList))
        ending1[i] = tempList.copy()

    for i, sentence in enumerate(ending2):
        temp = sentence.translate(punctuation).strip()
        tempList = temp.split(' ')
        tempList = list(filter(None, tempList))
        ending2[i] = tempList.copy()

    return storyid, inputsen1, inputsen2, inputsen3, inputsen4, ending1, ending2, answers


def feature_x(file, label):
    storyid, inputsen1, inputsen2, inputsen3, inputsen4, ending1, ending2, answers = preprocess(file, label)
    pos_window = []
    stop_words = set(stopwords.words('english'))
    for count in range(len(inputsen1)):
        vocab = set(inputsen1[count] + inputsen2[count] + inputsen3[count] + inputsen4[count])

        num1 = 0
        for word in ending1[count]:
            if word in vocab and word not in stop_words:
                num1 += 1

        num2 = 0
        for word in ending2[count]:
            if word in vocab and word not in stop_words:
                num2 += 1

        pos_window.append({'ending1':num1, 'ending2':num2})
    return pos_window, answers, storyid


if __name__ == "__main__":
    trainfile = './train.csv'
    x, answers, _ = feature_x(trainfile, True)
    y = [i for i in answers]
    trainy = np.array(y)

    testfile = './dev.csv'
    tx, test_answers, storyid = feature_x(testfile, True)
    vec = DictVectorizer()
    transform = vec.fit_transform(x + tx)
    trainx = transform[:len(x)]
    testx = transform[len(x):]

    classifier = LogisticRegression(C=1.0, solver='liblinear')
    classifier.fit(trainx, trainy)
    prediction_prabability = classifier.predict_proba(testx)
    prediction = classifier.predict(testx)
    score = classifier.score(testx, np.array(test_answers))
    print(score)

    with open('partA_val_Feature4.csv', mode='w') as report_file:
        report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        report_writer.writerow(['Id', 'Prediction'])
        for i in range(len(storyid)):
            report_writer.writerow([storyid[i], prediction[i]])
