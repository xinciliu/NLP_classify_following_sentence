import sys
import csv
import heapq
import spacy
import unicodedata
import numpy as np
from pandas import DataFrame
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
    return storyid, inputsen1, inputsen2, inputsen3, inputsen4, ending1, ending2, answers

def modify(avg):
    avg = avg.vector
    mini = avg.min()
    if mini >= 0:
        return avg
    for i in range(96):
        avg[i] += -mini
    return avg

def feature_x(file, label):
    nlp = spacy.load('en_core_web_lg')
    storyid, inputsen1, inputsen2, inputsen3, inputsen4, ending1, ending2, answers = preprocess(file, label)
    pos_window = []
    for count in range(len(inputsen1)):
        avg1 = nlp(ending1[count])
        avg1 = modify(avg1)
        avg2 = nlp(ending2[count])
        avg2 = modify(avg2)
        avg = nlp(inputsen1[count] + inputsen2[count] + inputsen3[count] + inputsen4[count])
        avg = modify(avg)

        cur = {}
        for i in range(96):
            cur["ending1"+ "," +str(i)] = avg1[i]
        for i in range(96):
            cur["ending2"+ "," +str(i)] = avg2[i]
        for i in range(96):
            cur["input"+ "," +str(i)] = avg[i]
        pos_window.append(cur)
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
    prediction = classifier.predict(testx)
    score = classifier.score(testx, np.array(test_answers))
    print(score)

    with open('partA_val_Feature2.csv', mode='w') as report_file:
        report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        report_writer.writerow(['Id', 'Prediction'])
        for i in range(len(storyid)):
            report_writer.writerow([storyid[i], prediction[i]])
