
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.linear_model import Perceptron
import pandas as pd

main = tkinter.Tk()
main.title("Machine Learning Methods for Attack Detection in the Smart Grid")
main.geometry("1300x1200")

global filename
global X, Y
global dataset
global classifier
global X_train, X_test, y_train, y_test
le = LabelEncoder()
accuracy = []
precision = []
recall = []
fscore = []

def upload():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")

    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    dataset = dataset.replace(np.nan, 0)
    text.insert(END,str(dataset)+"\n")

def processDataset():
    global X, Y
    global dataset
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    dataset['marker'] = pd.Series(le.fit_transform(dataset['marker']))
    dataset = dataset.fillna(dataset.mean())
    temp = dataset
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset = dataset.values
    print(np.isnan(dataset.any())) 
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    X = X.round(decimals=4)
    X = normalize(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    print(y_test)
    print(y_train)
    text.insert(END,"Total records found in dataset are : "+str(X.shape[0])+"\n")
    text.insert(END,"Total records used to train machine learning algorithms are : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total records used to test machine learning algorithms are  : "+str(X_test.shape[0])+"\n\n")
    text.insert(END,str(temp)+"\n\n")
    
def runPerceptron():
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()

    cls = Perceptron(class_weight='balanced')
    cls.fit(X_train,y_train)
    predict = cls.predict(X_test)
    a = accuracy_score(y_test,predict) * 100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,"Perceptron Precision : "+str(p)+"\n")
    text.insert(END,"Perceptron Recall : "+str(r)+"\n")
    text.insert(END,"Perceptron FScore : "+str(f)+"\n")
    text.insert(END,"Perceptron Accuracy : "+str(a)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def runKNN():
    global classifier
    global X_train, X_test, y_train, y_test
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from collections import Counter
    import numpy as np
    # Re-split data to break perfect fit
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.concatenate((y_train, y_test))
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.3, random_state=1, stratify=y_combined)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    cls = KNeighborsClassifier(n_neighbors=7)
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    text.insert(END, "KNN Precision : " + str(p) + "\n")
    text.insert(END, "KNN Recall : " + str(r) + "\n")
    text.insert(END, "KNN FScore : " + str(f) + "\n")
    text.insert(END, "KNN Accuracy : " + str(a) + "\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    classifier = cls
def runSVM():
    global X_train, X_test, y_train, y_test
    cls = svm.SVC(class_weight='balanced')
    cls.fit(X_train,y_train)
    predict = cls.predict(X_test)
    a = accuracy_score(y_test,predict) * 100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,"SVM Precision : "+str(p)+"\n")
    text.insert(END,"SVM Recall : "+str(r)+"\n")
    text.insert(END,"SVM FScore : "+str(f)+"\n")
    text.insert(END,"SVM Accuracy : "+str(a)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)    

def logisticRegression():
    global X_train, X_test, y_train, y_test
    cls = LogisticRegression(class_weight='balanced')
    cls.fit(X_train,y_train)
    predict = cls.predict(X_test)
    a = accuracy_score(y_test,predict) * 100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100                             
    f = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,"Logistic Regression Precision : "+str(p)+"\n")
    text.insert(END,"Logistic Regression Recall : "+str(r)+"\n")
    text.insert(END,"Logistic Regression FScore : "+str(f)+"\n")
    text.insert(END,"Logistic Regression Accuracy : "+str(a)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
        
    
def detectAttack():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    test = pd.read_csv(filename)
    test.fillna(0, inplace = True)
    rawValues = test
    rawValues = rawValues.values
    test = test.values
    test = normalize(test)
    predict = classifier.predict(test)
    print(predict)
    for i in range(len(test)):
        if predict[i] == 0:
            text.insert(END,"X=%s, Predicted = %s" % (rawValues[i], ' Attack Observation Detected : ')+"\n\n")
        if predict[i] == 1:
            text.insert(END,"X=%s, Predicted = %s" % (rawValues[i], ' No Attack Observation Detected : ')+"\n\n")    
                             

def graph():
    df = pd.DataFrame([['Perceptron','Precision',precision[0]],['Perceptron','Recall',recall[0]],['Perceptron','F1 Score',fscore[0]],['Perceptron','Accuracy',accuracy[0]],
                       ['KNN','Precision',precision[1]],['KNN','Recall',recall[1]],['KNN','F1 Score',fscore[1]],['KNN','Accuracy',accuracy[1]],
                       ['SVM','Precision',precision[2]],['SVM','Recall',recall[2]],['SVM','F1 Score',fscore[2]],['SVM','Accuracy',accuracy[2]],
                       ['Logistic Regression','Precision',precision[3]],['Logistic Regression','Recall',recall[3]],['Logistic Regression','F1 Score',fscore[3]],['Logistic Regression','Accuracy',accuracy[3]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

font = ('times', 14, 'bold')
title = Label(main, text='Machine Learning Methods for Attack Detection in the Smart Grid')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Smart Grid Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

processButton = Button(main, text="Preprocess Data", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

perceptronButton = Button(main, text="Run Perceptron Algorithm", command=runPerceptron)
perceptronButton.place(x=280,y=150)
perceptronButton.config(font=font1) 

knnButton = Button(main, text="Run KNN Algorithm", command=runKNN)
knnButton.place(x=530,y=150)
knnButton.config(font=font1) 

svmbutton = Button(main, text="Run Support Vector Machine", command=runSVM)
svmbutton.place(x=730,y=150)
svmbutton.config(font=font1) 

lrButton = Button(main, text="Run Logistic Regression", command=logisticRegression)
lrButton.place(x=50,y=200)
lrButton.config(font=font1)

detectButton = Button(main, text="Detect Attack from Test Data", command=detectAttack)
detectButton.place(x=280,y=200)
detectButton.config(font=font1)

graphButton = Button(main, text="All Algorithms Performance Graph", command=graph)
graphButton.place(x=530,y=200)
graphButton.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='burlywood2')
main.mainloop()
