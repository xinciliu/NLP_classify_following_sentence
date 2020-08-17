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

    return storyid, inputsen1, inputsen2, inputsen3, inputsen4, ending1, ending2, answers


def feature_x(file, label):
    nlp = spacy.load('en_core_web_lg')
    storyid, inputsen1, inputsen2, inputsen3, inputsen4, ending1, ending2, answers = preprocess(file, label)
    pos_window = []
    for count in range(len(inputsen1)):
        temp = nlp(ending1[count])
        avg1 = nlp(' '.join([str(t) for t in temp if not t.is_stop]))
        if not(avg1 and avg1.vector_norm):
            avg1 = temp
        sim1 = []
        for word in (inputsen1[count] + inputsen2[count] + inputsen3[count] + inputsen4[count]):
            if word and nlp.vocab[word].vector_norm:
                sim1.append(nlp.vocab[word].similarity(avg1))
        sim1 = sorted(sim1, reverse=True)

        temp = nlp(ending2[count])
        avg2 = nlp(' '.join([str(t) for t in temp if not t.is_stop]))
        if not(avg2 and avg2.vector_norm):
            avg2 = temp
        sim2 = []
        for word in (inputsen1[count] + inputsen2[count] + inputsen3[count] + inputsen4[count]):
            if word and nlp.vocab[word].vector_norm:
                sim2.append(nlp.vocab[word].similarity(avg2))
        sim2 = sorted(sim2, reverse=True)
        pos_window.append({'ending1_0':sim1[0], 'ending1_1':sim1[1], 'ending2_0':sim2[0], 'ending2_1':sim2[1]})
        """
        pos_window.append({'ending1_0':sim1[0], 'ending1_1':sim1[1], 'ending1_2':sim1[2], 'ending1_3':sim1[3], 'ending1_4':sim1[4], 'ending1_5':sim1[5], 'ending1_6':sim1[6],
                           'ending2_0':sim2[0], 'ending2_1':sim2[1], 'ending2_2':sim2[2], 'ending2_3':sim2[3], 'ending2_4':sim1[4], 'ending2_5':sim1[5], 'ending2_6':sim1[6]})
        """
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

    with open('partA_val_Feature5.csv', mode='w') as report_file:
        report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        report_writer.writerow(['Id', 'Prediction'])
        for i in range(len(storyid)):
            report_writer.writerow([storyid[i], prediction[i]])
