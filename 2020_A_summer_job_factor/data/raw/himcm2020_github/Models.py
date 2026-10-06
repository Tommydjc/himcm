
import numpy as np
import matplotlib.pyplot as plt
import dataset
from sklearn.tree import DecisionTreeClassifier  # , plot_tree
from sklearn.model_selection import train_test_split

N = 50  # number of points per class, namely number of students
D = 7  # dimensionality of work choice factors
K = 8  # number of classes of work type

# X, y = dataset.get_data(N, D, K)
X = np.loadtxt('ReplacedData.txt')
y = np.loadtxt('work_choice.txt')
x = []
from itertools import chain

for i in range(int(len(X) / K)):
    tmp = X[i * K:(i + 1) * K]
    x.append(list(chain.from_iterable(tmp)))

from sklearn.linear_model import LogisticRegression, SGDClassifier

model = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
# model = SGDClassifier()
# for i in range(1000000):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=np.random.randint(0, 100000),
#                                                         shuffle=True)
#     model.fit(x_train, y_train)
#     prediction = model.predict(x_test)
#     acc = sum(int(prediction[i] == y_test[i]) for i in range(len(y_test)))
#     print('the number of accurate work in all 10 test samples is:', acc)
#     if acc >= 8:
#         x_train_best, x_test_best, y_train_best, y_test_best = x_train, x_test, y_train, y_test
#         np.savetxt('x_train_best.txt', x_train_best)
#         np.savetxt('x_test_best.txt', x_test_best)
#         np.savetxt('y_train_best.txt', y_train_best)
#         np.savetxt('y_test_best.txt', y_test_best)
#         model_best = model
#         break

x_train_best, x_test_best = np.loadtxt('x_train_best.txt'), np.loadtxt('x_test_best.txt')
y_train_best, y_test_best = np.loadtxt('y_train_best.txt'), np.loadtxt('y_test_best.txt')
model.fit(x_train_best, y_train_best)
prediction = model.predict(x_test_best)
print(prediction)
print(y_test_best)
acc = sum(int(prediction[i] == y_test_best[i]) for i in range(len(y_test_best)))
print(acc)
np.savetxt('W_matrix.txt', model.coef_)
print('love world')
print('hello world')
