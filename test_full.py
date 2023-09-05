# HeartDisease_Detection_file
from tkinter import *
import pandas as pd

from tkinter import simpledialog

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

class HeartDiseasePrediction:
    def __init__(self, root):
        self.f = Frame(root, width=1000, height=1000, bg="goldenrod")
        self.ff = Frame(self.f, width=300, height=1000, bg="light goldenrod")
        self.loadDataset = Button(self.ff, text="LoadDataset", height=2, width=30, command=self.loaddataset,
                                  bg="goldenrod")
        self.loadDataset.place(x=40, y=20)
        self.Headpoints = Button(self.ff, text="Head Points", height=2, width=30, command=self.headpoint,
                                 bg="goldenrod")
        self.Headpoints.place(x=40, y=90)
        self.splitDataset = Button(self.ff, text="splitDataset", height=2, width=30, command=self.splitt,
                                   bg="goldenrod")
        self.splitDataset.place(x=40, y=160)
        self.checkAlgorithm = Button(self.ff, text="Spot and check algorithm ", height=2, width=30,
                                     command=self.spotcheck, bg="goldenrod")
        self.checkAlgorithm.place(x=40, y=230)

        self.CompareAlgorithms = Button(self.ff, height=1, width=40,
                                        bg="goldenrod")
        self.CompareAlgorithms.place(x=10, y=300)
        self.validation = Button(self.ff, text="Make predictions on validation dataset", height=2, width=30,
                                 command=self.classificationn, bg="goldenrod")
        self.validation.place(x=40, y=350)
        self.testset = Button(self.ff, text="Test Set", height=2, width=30, command=self.testsetprediction,
                              bg="goldenrod")
        self.testset.place(x=40, y=420)
        self.CompareAlgorithms = Button(self.ff, height=1, width=40,
                                        bg="goldenrod")
        self.CompareAlgorithms.place(x=10, y=480)

        self.realtimefraudd = Button(self.ff, height=2, width=30, text="Real Time Heart Diease Prediction",
                                     bg="goldenrod", command=self.realtimefraud)
        self.realtimefraudd.place(x=40, y=520)
        self.ff.pack(side=LEFT)
        self.fff = Frame(self.f, width=650, height=1000, bg="light goldenrod")
        self.scrollbar = Scrollbar(self.fff)
        self.scrollbar.pack(side=RIGHT, fill=BOTH)
        self.ffff = Text(self.fff, height=900, yscrollcommand=self.scrollbar.set)
        self.ffff.pack(side=LEFT, padx=20, pady=20)

        self.scrollbar.config(command=self.ffff.yview)

        self.df = pd.read_csv("framingham.csv")
        self.df.dropna(axis=0, inplace=True)

        self.array = self.df.values
        self.X = self.array[:, 0:15]
        self.Y = self.array[:, 15:16]

        self.validation_size = 0.20
        self.seed = 7

        self.X_train, self.X_validation, self.Y_train, self.Y_validation = model_selection.train_test_split(self.X,
                                                                                                            self.Y,
                                                                                                            test_size=self.validation_size,
                                                                                                            random_state=21)
        self.logreg = LogisticRegression()
        print(self.X_train)
        print(self.Y_train)

        self.logreg.fit(self.X_train, self.Y_train)
        print("Model trained successfully!!")
        print(self.X_train)
        print(self.Y_train)


        self.fff.pack()
        self.f.pack()

    def loaddataset(self):
        self.df = pd.read_csv("framingham.csv")
        self.df.dropna(axis=0, inplace=True)
        self.ffff.insert(END, self.df)

    def headpoint(self):
        self.ffff.delete('1.0', END)
        self.answer = simpledialog.askstring("Input", "Enter the value", parent=root)
        if self.answer is not None:
            self.ffff.insert(END, self.df[0:(int(self.answer))])
        else:
            print("Enter the value")

    def splitt(self):

        self.ffff.delete('1.0', END)
        self.ffff.insert(END, self.X.shape)
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, self.Y.shape)
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, self.X_train.shape)
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, self.Y_train.shape)
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, self.X_validation.shape)
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, self.Y_validation.shape)
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, self.X_train[0:5])


    def spotcheck(self):
        self.seed = 7
        self.scoring = 'accuracy'
        self.ffff.delete('1.0', END)
        self.models = []
        self.models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        self.models.append(('KNN', KNeighborsClassifier()))
        self.models.append(('CART', DecisionTreeClassifier()))
        # models.append(('NB', GaussianNB()))
        self.models.append(('SVM', SVC(gamma='auto')))
        self.models.append(('RF', RandomForestClassifier(n_estimators=10)))
        self.models.append(('ADA', AdaBoostClassifier(n_estimators=100)))

        # evaluate each model in turn
        self.results = []
        self.names = []
        for name, model in self.models:
            self.kfold = model_selection.KFold(n_splits=10)
            self.cv_results = model_selection.cross_val_score(model, self.X_train, self.Y_train, cv=self.kfold,
                                                              scoring=self.scoring)
            self.results.append(self.cv_results)
            self.names.append(name)
            self.msg = "%s: %f (%f)" % (name, self.cv_results.mean(), self.cv_results.std())
            self.ffff.insert(END, self.msg)
            self.ffff.insert(END, "\n")


    def classificationn(self):
        self.logreg = LogisticRegression()
        print(self.X_train)
        print(self.Y_train)

        self.logreg.fit(self.X_train, self.Y_train)
        self.predictions = self.logreg.predict(self.X_validation)
        print(self.predictions)


        self.ffff.delete('1.0', END)
        score= self.logreg.score(self.X_validation,self.Y_validation)
        self.ffff.insert(END,str(score))
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, confusion_matrix(self.Y_validation, self.predictions))
        self.ffff.insert(END, "\n")
        self.ffff.insert(END, classification_report(self.Y_validation, self.predictions))
        self.ffff.insert(END, "\n")

    def testsetprediction(self):
        self.ffff.delete('1.0', END)
        self.ffff.insert(END, self.X_validation.shape)
        self.ffff.insert(END, "\n")
        self.Y_pred = self.logreg.predict(self.X_validation)
        for i in range(0, len(self.Y_pred)):
            self.ffff.insert(END, self.Y_pred[i])
            self.ffff.insert(END, "\n")

    def realtimefraud(self):
        newWindow = Toplevel(root, height="1000", width="1000")
        # sets the title of the
        # Toplevel widget
        newWindow.title("Real Time Heart Disease prediction")
        self.f = Frame(newWindow, height="1000", width="300", bg="goldenrod")
        self.f.pack(side=LEFT)

        self.male = Label(self.f, text="male", width=20, bg="goldenrod", height=1)
        self.male.pack(padx=20, pady=6)

        self.age = Label(self.f, text="age", bg="goldenrod", width=20,
                                            height=1)
        self.age.pack(padx=20, pady=6)

        self.education = Label(self.f, text="education", bg="goldenrod", width=20, height=1)
        self.education.pack(padx=20, pady=6)

        self.currentSmoker = Label(self.f, text="currentSmoker", bg="goldenrod", width=20, height=1)
        self.currentSmoker.pack(padx=20, pady=6)

        self.cigsPerDay = Label(self.f, text="cigsPerDay", bg="goldenrod", width=20,
                                              height=1)
        self.cigsPerDay.pack(padx=20, pady=6)

        self.BPMeds = Label(self.f, text="BPMeds", bg="goldenrod", width=20, height=1)
        self.BPMeds.pack(padx=20, pady=6)

        self.prevalentStroke = Label(self.f, text="prevalentStroke", bg="goldenrod", width=20,
                                               height=1)
        self.prevalentStroke.pack(padx=20, pady=6)

        self.prevalentHyp = Label(self.f, text="prevalentHyp", bg="goldenrod", width=20, height=1)
        self.prevalentHyp.pack(padx=20, pady=6)

        self.diabetes = Label(self.f, bg="goldenrod", text="diabetes", width=20,
                                              height=1)
        self.diabetes.pack(padx=20, pady=6)

        self.totChol = Label(self.f, text="totChol", bg="goldenrod", width=20, height=1)
        self.totChol.pack(padx=20, pady=6)
        self.sysBP = Label(self.f, text="sysBP", bg="goldenrod", width=20, height=1)
        self.sysBP.pack(padx=20, pady=6)
        self.diaBP = Label(self.f, text="diaBP", bg="goldenrod", width=20, height=1)
        self.diaBP.pack(padx=20, pady=6)
        self.BMI = Label(self.f, text="BMI", bg="goldenrod", width=20, height=1)
        self.BMI.pack(padx=20, pady=6)
        self.heartRate = Label(self.f, text="heartRate", bg="goldenrod", width=20, height=1)
        self.heartRate.pack(padx=20, pady=6)

        self.glucose = Label(self.f, text="glucose", bg="goldenrod", width=20, height=1)
        self.glucose.pack(padx=20, pady=6)
        self.fraud = Label(self.f, text="TenYearCHD", bg="goldenrod", width=30, height=2)
        self.fraud.pack(padx=20, pady=8)

        self.submit = Button(self.f, text="submit", height=2, width=30, bg="goldenrod", command=self.getvaluee)
        self.submit.pack(padx=20, pady=10)

        self.ff = Frame(newWindow, height="900", width="300", bg='blue')
        self.ff.pack(side=TOP)

        self.male = StringVar(value="1")
        self.age = StringVar(value="67")
        self.education = StringVar(value="2")
        self.currentSmoker = StringVar(value="1")
        self.cigsPerDay = StringVar(value="60")
        self.BPMeds = StringVar(value="0")
        self.prevalentStroke = StringVar(value="0")
        self.prevalentHyp = StringVar(value="1")
        self.diabetes = StringVar(value="0")
        self.totChol = StringVar(value="261")
        self.sysBP = StringVar(value="170")
        self.diaBP = StringVar(value="100")
        self.BMI = StringVar(value="22.71")
        self.heartRate = StringVar(value="72")
        self.glucose = StringVar(value="79")
        self.fraudd = StringVar()

        self.fff = Frame(self.ff, height="1200", width="300")
        self.fff.pack(side=LEFT)
        self.males = Entry(self.fff, width=35, textvariable=self.male)
        self.males.pack(padx=20, pady=5)

        self.ages = Entry(self.fff, width=35, textvariable=self.age)
        self.ages.pack(pady=15, padx=4)

        self.educations = Entry(self.fff, textvariable=self.education, width=35)
        self.educations.pack(pady=10, padx=2)

        self.currentSmokers = Entry(self.fff, textvariable=self.currentSmoker, width=35)
        self.currentSmokers.pack(pady=17, padx=2)

        self.cigsPerDays = Entry(self.fff, textvariable=self.cigsPerDay, width=35)
        self.cigsPerDays.pack(pady=17, padx=2)

        self.BPMed = Entry(self.fff, textvariable=self.BPMeds, width=35)
        self.BPMed.pack(pady=17, padx=5)

        self.prevalentStrokes = Entry(self.fff, textvariable=self.prevalentStroke, width=35)
        self.prevalentStrokes.pack(pady=17, padx=5)

        self.prevalentHyps = Entry(self.fff, textvariable=self.prevalentHyp, width=35)
        self.prevalentHyps.pack(padx=20, pady=5)
        self.diabete = Entry(self.fff, textvariable=self.diabetes, width=35)
        self.diabete.pack(padx=20, pady=5)

        self.totChols = Entry(self.fff, textvariable=self.totChol, width=35)
        self.totChols.pack(padx=20, pady=5)

        self.sysBPs = Entry(self.fff, textvariable=self.sysBP, width=35)
        self.sysBPs.pack(padx=20, pady=5)
        self.diaBPs = Entry(self.fff, textvariable=self.diaBP, width=35)
        self.diaBPs.pack(padx=20, pady=5)
        self.BMIs = Entry(self.fff, textvariable=self.BMI, width=35)
        self.BMIs.pack(padx=20, pady=5)
        self.heartRates = Entry(self.fff, textvariable=self.heartRate, width=35)
        self.heartRates.pack(padx=20, pady=5)
        self.glucoses = Entry(self.fff, textvariable=self.glucose, width=35)
        self.glucoses.pack(padx=20, pady=5)

        self.frauddd = Entry(self.fff, textvariable=self.fraudd, width=35)
        self.frauddd.pack(padx=20, pady=5)
        newWindow.mainloop()

    def getvaluee(self):
        males = self.males.get()
        ages = self.ages.get()
        educations = self.educations.get()
        currentSmokers = self.currentSmokers.get()
        cigsPerDays = self.cigsPerDays.get()
        BPMed = self.BPMed.get()
        prevalentStrokes = self.prevalentStrokes.get()
        prevalentHyps = self.prevalentHyps.get()
        diabete = self.diabete.get()
        totChols = self.totChols.get()
        sysBPs = self.sysBPs.get()
        diaBPs = self.diaBPs.get()
        BMIs = self.BMIs.get()
        heartRates = self.heartRates.get()
        glucoses = self.glucoses.get()
        print(males)
        print(ages)
        print(educations)
        print(currentSmokers)
        print(cigsPerDays)
        print(BPMed)
        print(prevalentStrokes)
        print(prevalentHyps)
        print(diabete)
        print(totChols)
        print(sysBPs)
        print(diaBPs)
        print(BMIs)
        print(heartRates)
        print(glucoses)


        self.Y_pred = self.logreg.predict([[int(males), int(ages), int(educations),
                                         int(currentSmokers), int(cigsPerDays),
                                         int(BPMed)
                                            , int(prevalentStrokes),
                                              int(prevalentHyps),
                                         int(diabete),
                                         int(totChols),
                                         int(sysBPs),
                                         int(diaBPs),
                                         float(BMIs),
                                         int(heartRates),
                                         int(glucoses)]])
        print(self.Y_pred)
        self.fraudd.set(self.Y_pred[len(self.Y_pred) - 1])


root = Tk()
root.title("HeartDiseasePrediction")
f = HeartDiseasePrediction(root)
root.mainloop()
