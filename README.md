# Machine Learning Repoitory of Eli Curl

### HW 6
This project was to test the dataset of peoples features(Income, Home Ownership, Income, etc.) to there credit scores.
This was done via checking the accuracy of BaggingClassifier and RandomForestClassifier

Bar graph:

<img width="458" alt="HW6BarGraph" src="https://github.com/user-attachments/assets/370b4a90-cb64-46ef-9d03-75f3a06206d7" />

CM:

<img width="420" alt="HW6CM" src="https://github.com/user-attachments/assets/471392c3-b040-44e4-a4df-6a11ecc0a9cf" />

OOB Score RandomForestClassifier:

<img width="160" alt="HW6FOOB" src="https://github.com/user-attachments/assets/b06329ed-647b-4342-a278-d06d9c76d99c" />

OOB Score BaggingClassifier:

<img width="137" alt="HW6BOOB" src="https://github.com/user-attachments/assets/42b1389c-a85b-47b4-a7f3-48a496ad8669" />

The resalts on this project are pretty good. We get a very accurate CM and both bagging and RandomForest get a high score!

### HW 8
This project was to show off a simple ANN. This tries to predict house prices based off of many features such as, bedrooms, bathrooms, sqaure feet of living space, etc. This project uses 3 hidden layers with relu as an activation funtion.

The results of this project are a bit hit and miss as the MSE being a bit out there.

### HW 4
This project is using SVC to classify different types of glass based on the elements in the glass
Type of glass (class labels):

1: building_windows_float_processed

2: building_windows_non_float_processed

3: vehicle_windows_float_processed

4: vehicle_windows_non_float_processed (not present)

5: containers

6: tableware

7: headlamps

We also use Lasso for Gridsearch to find good alpha values

CM:

<img width="415" alt="HW4CM" src="https://github.com/user-attachments/assets/392e99f2-0ab2-46f7-9bff-1b12352a84a9" />

The results of this project are a bit scattered, but I they also show how the machine is good at classifying some types of glass and worse at others.

### HW 7

This project gets info about countries and groups them based off of those results. It also gives a elbow graph that is used when making a decision about the grouping of the countries, in this case 4 groups.

Elbow graph:

<img width="431" alt="HW7ElbowGraph" src="https://github.com/user-attachments/assets/cf23acfb-afe2-42d8-9731-6df197ee6c88" />

The results of this project are pretty good. The elbow graph is a bit hard to choose how many groups there should be and leaves room for interpritation. The groups themselves make sence as you keep them 4 and above.

