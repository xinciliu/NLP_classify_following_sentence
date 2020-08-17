import csv
def read_csv_file(file):
    with open(file,'r',encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column0 = [row[0]for row in reader][1:]
    with open(file,'r',encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row[1]for row in reader][1:]
    with open(file,'r',encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column2 = [row[2]for row in reader][1:]
    with open(file,'r',encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column3 = [row[3]for row in reader][1:]   
    with open(file,'r',encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column4 = [row[4]for row in reader][1:]  
    with open(file,'r',encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column5 = [row[5]for row in reader][1:]       
    with open(file,'r',encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        column6 = [row[6]for row in reader][1:]
    return (column0,column1,column2,column3,column4,column5,column6)  
    
 def write_new_file(file):
    a=read_csv_file(file)
    question_line=[]
    for x in a[1]:
        lis=x.split()
        if len(lis)>1:
            ques=lis[0]+' '+lis[1]
        else:
            ques=lis[0]
        question_line.append(ques)
    return (a[0],question_line,a[1],a[2],a[3],a[4],a[5],a[6])    

import pandas as pd
def write_csv_file(file):
    a=write_new_file(file)
    label=[]
    for i in range(len(a[0])):
        label.append(1)
    df2 = pd.DataFrame({'id': pd.Series(a[0]),
        'question': pd.Series(a[1]),'sentence1':pd.Series(a[2]),
        'sentence2':pd.Series(a[3]),'sentence3':pd.Series(a[4]),
         'sentence4':pd.Series(a[5]),'ending1':pd.Series(a[6]),
        'ending2':pd.Series(a[7]),'label':pd.Series(label)})
    df2.to_csv('test.csv',index=False,columns=['id','question','sentence1',
        'sentence2','sentence3','sentence4','ending1','ending2','label'])
