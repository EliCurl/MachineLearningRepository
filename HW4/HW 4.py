import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import Lasso

# Load the dataset from CSV file
df = pd.read_csv("glass.csv")

# Convert integer columns to float64 for consistency
df[df.select_dtypes("int64").columns] = df[df.select_dtypes("int64").columns].astype("float64")

# Separate features (X) and target labels (y)
X = df.iloc[:, :-1].copy().to_numpy()  # All columns except last
y = df.iloc[:, -1].copy().to_numpy()   # Last column

# Normalize features: zero mean, unit variance
X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and train a Support Vector Classifier with RBF kernel
clf = SVC(kernel='rbf', decision_function_shape='ovo')
clf.fit(X_train, y_train)

# Print accuracy score on the test set
print(f"Score: {clf.score(X_test, y_test):.3f}")

# Perform Lasso regression with hyperparameter tuning using GridSearchCV
reg = Lasso()
parameters = {"alpha": np.linspace(0.01, 0.1, num=10)}  # Range of alpha values
grid_search = GridSearchCV(reg, param_grid=parameters, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Show results of the grid search (alpha values and corresponding scores)
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_alpha', 'mean_test_score', 'rank_test_score']])

# Compute and display the normalized confusion matrix for SVC predictions
cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=np.unique(y_train))
disp_cm.plot()
plt.show()

# Type of glass (class labels):
# 1: building_windows_float_processed
# 2: building_windows_non_float_processed
# 3: vehicle_windows_float_processed
# 4: vehicle_windows_non_float_processed (not present)
# 5: containers
# 6: tableware
# 7: headlamps
