import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC, LinearSVC

df = pd.read_csv('cleanedCreditScore.csv')

# Putting features to X and label to y
X = df.iloc[:,:-1].to_numpy()
y = df['Credit Score'].to_numpy()

# Normalize the features
X = np.double(X)
X -= np.average(X, axis=0)
X /= np.std(X, axis=0)

# split training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# Creating the random forest classifier
clf = RandomForestClassifier(max_depth=100, oob_score=True, verbose=3, n_jobs=-1)
clf.fit(X_train, y_train)

# using a Linear support vector classifier on bagging
clfLin = LinearSVC(C=1.0)
bag_clf = BaggingClassifier(clfLin, n_estimators=100, verbose=3, oob_score=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)

# using random forest classifier to see a bar graph of what features are most important
importances = pd.DataFrame(clf.feature_importances_, index=df.columns[:-1])
importances.plot.bar()
plt.show()

# create a confusion matrix to visualize accuracy and predictions
cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()

# displaying the scores of both RFC and Bagging to see what is better (including Out Of Bag error)
print(f"Score (Train): {clf.score(X_train, y_train):.3f}")
print(f"Score (Test): {clf.score(X_test, y_test):.3f}")
print(f"OOB Score: {clf.oob_score_:.3f}")

print(f"Score (Train): {bag_clf.score(X_train, y_train):.3f}")
print(f"Score (Test): {bag_clf.score(X_test, y_test):.3f}")
print(f"OOB Score: {bag_clf.oob_score_:.3f}")

#https://www.kaggle.com/datasets/sujithmandala/credit-score-classification-dataset