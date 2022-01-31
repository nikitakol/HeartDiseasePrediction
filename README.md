# Heart Disease Prediction :heart:

# I. Abstract
The CRISP-DM process model has been applied to develop and compare four different machine learning classification models for accurate diagnosis of ischaemic heart disease based on the algorithms kNearestNeighbors, Logistic Regression, Random Forest, and Gaussian Naive Bayes. Using a 10-fold cross-validated gridsearch on a dataset consisting of five combined subsets of data from different medical institutions, the initial models were optimized. Given the probable fatal implications of failing to diagnose a patient suffering from CAD correctly, minimizing false negatives over false positives was prioritized. Hence the recall score has been the business metric guiding the modeling process. 

# II. Introduction
This paper focuses on ischaemic heart disease - also known as coronary artery disease (CAD). Ischaemic heart disease results from reduced blood-flow in arteries that supply the heart with blood, most often caused by cholesterol blockages and inflammations. Therefore, it is imperative to find ways to optimize the diagnosis of CAD, and machine learning poses a promising opportunity for low-cost and high-scale classification of patients. Using the Cross Industry Standard Process for Data Mining (CRISP-DM) and the dataset provided, the goal of this paper is to optimize and compare four widely used machine learning classification models to find the best model to predict CAD. The four models are the kNearestNeighbors classifier, Logistic Regression classifier, Random Forest classifier, and Gaussian Naive Bayes classifier.

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

K Nearest Neighbors

Logistic Regression

Random Forest 

Naive Bayes

### Testing

## Discussion

## Deployment
To clarify, these modeling techniques are only fundamental steps that could be deployed to medical research and practice if they are further developed and extended to a state that meets the medical requirements and allows for an accurate and somewhat reliable assessment of patients' health status. 
Given the complexity of IT alignment in model development and the healthcare environment, incorporating the Plan-Do-Study-Act (PDSA) cycle and enacting “silent testing” before a small-scale formal deployment is an integral first step towards implementation and evaluation. This often requires multiple PDSA cycles to be fully integrated into a healthcare platform. 

## Limitations
On a theoretical dimension, there are limitations to be taken into account when obtaining the most accurate assessment of CAD. As a trade-off to optimizing models towards the recall score, we identify that accuracy, precision and f1-scores decreased after GridSearchCV for Logistic Regression. Furthermore, the small size of this dataset poses a limitation to the generalization of data since information not being included in training data could result in a selection bias. Further, albeit our measures to increase generativity, the small dataset makes the model sensitive to differing input data.

## Conclusion

## Authors

Frederik Strom Friborg - https://www.linkedin.com/in/frederikfriborg/

Niklas Heuchemer - https://www.linkedin.com/in/niklas-heuchemer-b757aa163/

Nikita Kolmakov - https://www.linkedin.com/in/nikitakolmakov/

Lukas Schwendenwein - https://www.linkedin.com/in/lukas-s-29a2bb18b/

## Acknowledgements
