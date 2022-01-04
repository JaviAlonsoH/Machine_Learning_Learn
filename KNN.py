import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

desired_width = 820
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)

# KNN ALGORITHM

data = pd.read_csv("./datasets/car.data")
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
labels = ["unacc", "acc", "good", "vgood"]

for x in range(len(x_test)):
    print("Predicted: ", labels[predicted[x]], "   Data: ", x_test[x], "   Actual: ", labels[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)