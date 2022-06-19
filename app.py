from flask import Flask,request,render_template
import app
import pandas as pd
# import numpy as np
from sklearn.tree import DecisionTreeClassifier
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import AdaBoostClassifier
# from catboost import CatBoostClassifier


app = Flask(__name__)

@app.route('/' , methods = ['POST','GET'])
def home():
    return render_template('index.html')

@app.route('/about' , methods = ['POST','GET'])
def about():
    return render_template('about.html')
df = pd.read_csv(r'M:\python&pycharm\archive\Heart Disease Dataset.csv')

x = df.iloc[:,:-1]
y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

@app.route("/training",methods=['GET','POST'])
def training():
    global dta,rfa,svma,knna
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    if request.method == 'POST':
        print("dfsdfdfsd")
        models = int(request.form['algo'])
        if models == 1:
            dt = DecisionTreeClassifier()
            dt.fit(x_train, y_train)
            dtp = dt.predict(x_test)
            dta = accuracy_score(y_test, dtp)
            dta = dta*100
            msg = 'Accuracy for Decision Tree is : ' + str(dta)
            return render_template('training.html', msg=msg)
        elif models == 2:
            rf = RandomForestClassifier()
            rf.fit(x_train, y_train)
            rfp = rf.predict(x_test)
            rfa = accuracy_score(y_test, rfp)
            rfa = rfa*100
            msg = 'Accuracy for Random Forest is : ' + str(rfa)
            print(msg)
            return render_template('training.html', msg=msg)

        elif models == 3:
            svm = SVC()
            svm.fit(x_train, y_train)
            svmp = svm.predict(x_test)
            svma = accuracy_score(y_test, svmp)
            svma = svma*100
            msg = 'Accuracy for SVM is : ' +str(svma)
            return render_template('training.html', msg=msg)
        else:
            knn = KNeighborsClassifier()
            knn.fit(x_train, y_train)
            knnp = knn.predict(x_test)
            knna = accuracy_score(y_test, knnp)
            knna = knna*100
            msg = 'Accuracy for KNN is : ' + str(knna)

            return render_template("training.html",msg=msg)

    return render_template("training.html")



@app.route("/prediction", methods = ['POST','GET'])
def prediction():
    if request.method == 'POST':
        f1 = request.form['f1']
        f2 = request.form['f2']
        f3 = request.form['f3']
        f4 = request.form['f4']
        f5 = request.form['f5']
        f6 = request.form['f6']
        f7 = request.form['f7']
        f8 = request.form['f8']
        f9 = request.form['f9']
        f10 = request.form['f10']
        f11 = request.form['f11']
        f12 = request.form['f12']
        f13 = request.form['f13']

        m = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13]

        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        result = rf.predict([m])
        if result == 0:
            msg = 'The patient no heart stroke'
        else:
            msg = 'The patient has heart stroke'

        return render_template("prediction.html",msg=msg)
    return render_template("prediction.html")


@app.route("/chart", methods = ['POST','GET'])
def chart():

    i = [dta,rfa,svma,knna]
    return render_template('chart.html',i=i)
if __name__ =='__main__':
    app.run(debug=True)