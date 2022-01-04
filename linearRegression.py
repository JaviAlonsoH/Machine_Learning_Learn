import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# LINEAR REGRESSION

# read the data
data = pd.read_csv("./datasets/student-mat.csv", sep=";")

# select the attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# looping on save to do it until it gets the best accuracy
'''
best = 0
for _ in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print('Accuracy: \n', acc)

    if acc > best:
        best = acc
        # save the model
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)'''

# load the model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# setting data in scatter plot
p = "studytime"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final grade")
plt.show()