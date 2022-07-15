# Heart Disease Prediction :heart:

# I. Abstract
The CRISP-DM process model has been applied to develop and compare four different machine learning classification models for accurate diagnosis of ischaemic heart disease based on the algorithms kNearestNeighbors, Logistic Regression, Random Forest, and Gaussian Naive Bayes. Using a 10-fold cross-validated gridsearch on a dataset consisting of five combined subsets of data from different medical institutions, the initial models were optimized. Given the probable fatal implications of failing to diagnose a patient suffering from CAD correctly, minimizing false negatives over false positives was prioritized. Hence the recall score has been the business metric guiding the modeling process. 

# II. Introduction
This project focuses on ischaemic heart disease - also known as coronary artery disease (CAD). Ischaemic heart disease results from reduced blood-flow in arteries that supply the heart with blood, most often caused by cholesterol blockages and inflammations. Therefore, it is imperative to find ways to optimize the diagnosis of CAD, and machine learning poses a promising opportunity for low-cost and high-scale classification of patients. Using the Cross Industry Standard Process for Data Mining (CRISP-DM) and the dataset provided, the goal of this project is to optimize and compare four widely used machine learning classification models to find the best model to predict CAD. The four models are the kNearestNeighbors classifier, Logistic Regression classifier, Random Forest classifier, and Gaussian Naive Bayes classifier.

# III. Materials and Methods
The dataset at use is a Heart Failure Prediction Dataset retrieved from the platform Kaggle. It is a combined dataset of five individual collections of CAD related health data originally provided by the Donald Bren School of Information and Computer Science. All in all, the dataset consists of 918 unique patient observations, each described by twelve health related features.
## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
First, install the heart.csv dataset via Kaggle.
```
https://www.kaggle.com/fedesoriano/heart-failure-prediction
```
### Installing
Utilized Google Colab as IDE.
```
https://colab.research.google.com/
```
Install the heart.csv dataset via the upload button.

### Data Preprocessing
We prepare the data by using One-Hot Encoding on the dataframe. Hence, dummy variables are implemented for converting nominal object features to binary features. Lastly, as illustrated in Appendix A, outliers and the different scales of features could manipulate our models depending on their sensitivity to such measures. Given the non-gaussian distribution of continuous features in our dataset, for the scale-sensitive models kNearestNeighbors and logistic regression, we normalize the data by implementing MinMax-Scaling to prevent a bias based on the magnitude of certain measures. This measure is not taken for the input data for the Random Forest and Gaussian Naive Bayes models.

### Algorithms & Testing

#### Dummy
Dummy Classifier always predicts the dominant target variable, recall, precision and f1-scores are inconclusive. Therefore, albeit predominantly using the recall and f1-score for evaluating our models, we will compare our classifier models with the baseline model in terms of accuracy. 

| Baseline Classifier           | Concept | Accuracy Score |
| ----------------------------- | -------- |---------- | 
|          Majority Dummy             |   The dominant class label is always predicted |  64.13% | 

#### K Nearest Neighbors
In order to better demonstrate the results of our adaptation, we first fit the training data using the default parameters of Scikit Learn’s KNeighborsClassifier. To have a reference point for the GridSearchCV method and for the sake of visualization, we intended to find the best number of neighbors using the elbow method. This approach helps to manually select the k value with the lowest error rate by reading it from the figure. Here, the error rate was calculated using the recall score. As can be seen in Appendix D, three values (n = [15;17]) held the lowest error rate (in line with the results from GridSearchCV). Now, the parameters optimized with GridSearchCV were n_neighbors, p, and weights.

| Classifier           | Accuracy | Precision | Recall | F1 - Score |
| ----------------------------- | -------- |---------- | ------ | ------ |
|          KNN                 |   91.30% |  91.22%   | 91.53% | 93.10% |

#### Logistic Regression
We fitted the training data using a standard Logistic Regression, only defining the attribute solver as liblinear given the small dataset size. Thereupon, we again performed the GridSearchCV to optimize relevant parameters. First, the C parameter, determining the strength of regularization is iterated. Using a high C value optimizes the model towards fitting the training set as best as possible by stressing the correct classification of each individual datapoint, while “[u]sing low values of C will cause the algorithms to try to adjust to the “majority” of data points.”. Second, given the large number of features in our dataset, we switch between ‘L1’ and ‘L2’ Regularization, i.e. Lasso and Ridge Regression, respectively. Hence, using an optimal parameters set of solver = liblinear, C = 1e-15, and penalty = L2 , we enhanced the recall score of the model to 94.92%. By doing so, all other scores decreased. The f1-score ends up at 89.60%, heavily influenced by the deterioration of the precision score. 

| Classifier           | Accuracy | Precision | Recall | F1 - Score |
| ----------------------------- | -------- |---------- | ------ | ------ |
|          Logistic Regression              |   85.87% |  82.31%   | 94.92% | 89.60% |

#### Random Forest 
Initially, for comparison purposes, a general rRandom fForest classifier was instantiated using the default attributes and only a single pre-defined one, namely 10 decision tree estimators given by the n_estimators attribute. At first glance, the standard Rrandom fForest classifier seems to perform remarkably well. However, one must be wary of a few issues with the above model. Initially, a parameter grid dictionary was instantiated containing hyperparameters n_estimators, max_features, max_depth, min_samples_split, and max_leaf_nodes as dictionary keys with respective lists containing hyperparameter values to iterate over. Afterwards, the GridSearchCV object was instantiated and fed with the rRandom forest classifier and parameter grid. Finally, the training data was fitted into the model.

| Classifier           | Accuracy | Precision | Recall | F1 - Score |
| ----------------------------- | -------- |---------- | ------ | ------ |
|          Random Forest                  |   89.13% |  88.19%   | 91.53% | 91.53% |

#### Naive Bayes
Given our procedural approach, we first train the model via training data so it can accurately predict the outcome using Gaussian Naive Bayes. In line with our general approach, we again use GridSearchCV to see if it improves our parameters. First, we define grid search parameters utilizing a stability calculation called var_smoothing.Following suit with the optimal standards of previous calculations that have been aggregated, we decided to use the same combination parameters to minimize a predefined loss function to give accurate results. Using an optimal parameter of var_smoothing, we bolstered the recall score of the model to 88.14%. Compared to its base model, we were able to optimize it by 6.78% using GridSearchCV to predict positive observations of CAD. On all fronts, the overall performance of our tuned model has improved by at least 4.7%.

| Classifier           | Accuracy | Precision | Recall | F1 - Score |
| ----------------------------- | -------- |---------- | ------ | ------ |
|          GNB                  |   86.96% |  86.49%   | 88.14% | 89.66% |

## IV. Discussion
We conclude that Logistic Regression performs best regarding CAD classification under the assumption that little economic limitation exists. kNearestNeighbors performs best in balancing recall and precision (f1 = 93.1%). Hyperparameter tuning increased this score by almost three percentage points. Although being the highest of all approaches tested, it has not risen the most among the f1-scores (Naive Bayes Classification did). Similar significant results worth highlighting were the shifts in recall scores with Naive Bayes Classification and Logistic Regression. In any case, while optimizing recall for all models, all algorithms clearly surpass the performance of the baseline dummy classifier as the graph illustrates. Given the nature of the majority dummy classifier, recall and f1 scores are inconclusive. For baseline verification, we therefore use accuracy and conclude that all models are value-adding. 

![image](https://user-images.githubusercontent.com/48565455/179026760-e72a64ed-65f3-44ab-92dd-5effb6b0770a.png)


## Deployment
To clarify, these modeling techniques are only fundamental steps that could be deployed to medical research and practice if they are further developed and extended to a state that meets the medical requirements and allows for an accurate and somewhat reliable assessment of patients' health status. Given the complexity of IT alignment in model development and the healthcare environment, incorporating the Plan-Do-Study-Act (PDSA) cycle and enacting “silent testing” before a small-scale formal deployment is an integral first step towards implementation and evaluation. This often requires multiple PDSA cycles to be fully integrated into a healthcare platform. 

## Limitations
The small size of this dataset poses a limitation to the generalization of data since information not being included in training data could result in a selection bias. 

## V. Conclusion
Using a 10-fold cross-validated gridsearch on a dataset consisting of five combined subsets of data from different medical institutions, the initial models were optimized. Given the probable fatal implications of failing to diagnose a patient suffering from CAD correctly, minimizing false negatives over false positives was prioritized. Hence the recall score has been the business metric guiding the modeling process. 

The test results show that the Logistic Regression model achieves the highest recall score with 94.92%, followed by recall scores of 91.53% for both kNearestNeighbors and Random Forest, and 88.14% for the Gaussian Naive Bayes model.

Grade - 12

## Authors

Frederik Strom Friborg - https://www.linkedin.com/in/frederikfriborg/

Niklas Heuchemer - https://www.linkedin.com/in/niklas-heuchemer-b757aa163/

Nikita Kolmakov - https://www.linkedin.com/in/nikitakolmakov/

Lukas Schwendenwein - https://www.linkedin.com/in/lukas-s-29a2bb18b/

## Acknowledgements

Kaggle - https://www.kaggle.com/fedesoriano/heart-failure-prediction/code?datasetId=1582403&sortBy=voteCount&searchQuery=eda

UCI Machine Learning Repository - https://archive.ics.uci.edu/ml/index.php
