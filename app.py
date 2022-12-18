import pandas as pd
from numpy import *
import numpy as np
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn import neighbors
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "GET":
        return render_template("index.html")

    else:

        data = [
            [
                request.form["gender"],
                request.form["age"],
                request.form["openness"],
                request.form["neuroticism"],
                request.form["conscientiousness"],
                request.form["agreeableness"],
                request.form["extraversion"],
            ]
        ]

        dataframe = pd.DataFrame(data)
        dataframe.to_csv("datagen/input.csv", index=False, mode="a", header=False)

        data = pd.read_csv("dataset.csv")
        array = data.values

        for i in range(len(array)):
            if array[i][0] == "Male":
                array[i][0] = 1
            else:
                array[i][0] = 0

        df = pd.DataFrame(array)

        maindf = df[[0, 1, 2, 3, 4, 5, 6]]
        mainarray = maindf.values
        print(mainarray)

        temp = df[7]
        train_y = temp.values
        train_y = temp.values

        for i in range(len(train_y)):
            train_y[i] = str(train_y[i])

        mul_lr = linear_model.LogisticRegression(
            multi_class="multinomial", solver="newton-cg", max_iter=1000
        )
        mul_lr.fit(mainarray, train_y)

        testdata = pd.read_csv("datagen/input.csv")
        test = testdata.values

        for i in range(len(test)):
            if test[i][0] == "Male":
                test[i][0] = 1
            else:
                test[i][0] = 0

        df1 = pd.DataFrame(test)

        testdf = df1[[0, 1, 2, 3, 4, 5, 6]]
        maintestarray = testdf.values
        print(maintestarray)

        y_pred = mul_lr.predict(maintestarray)
        for i in range(len(y_pred)):
            y_pred[i] = str((y_pred[i]))
        DF = pd.DataFrame(y_pred, columns=["Predicted Personality"])
        DF.index = DF.index + 1
        DF.index.names = ["Person No"]
        DF.to_csv("datagen/output.csv")

        df = pd.read_csv("datagen/output.csv")

        return render_template("res.html", per=df["Predicted Personality"].tolist()[-1])


if __name__ == "__main__":
    app.run(debug=True)
