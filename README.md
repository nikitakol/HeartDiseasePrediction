# Heart Disease Prediction
The CRISP-DM process model has been applied to develop and compare four different machine learning classification models for accurate diagnosis of ischaemic heart disease based on the algorithms kNearestNeighbors, Logistic Regression, Random Forest, and Gaussian Naive Bayes. Using a 10-fold cross-validated gridsearch on a dataset consisting of five combined subsets of data from different medical institutions, the initial models were optimized. Given the probable fatal implications of failing to diagnose a patient suffering from CAD correctly, minimizing false negatives over false positives was prioritized. Hence the recall score has been the business metric guiding the modeling process.

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

## Testing

## Deployment
To clarify, these modeling techniques are only fundamental steps that could be deployed to medical research and practice if they are further developed and extended to a state that meets the medical requirements and allows for an accurate and somewhat reliable assessment of patients' health status. 
Given the complexity of IT alignment in model development and the healthcare environment, incorporating the Plan-Do-Study-Act (PDSA) cycle and enacting “silent testing” before a small-scale formal deployment is an integral first step towards implementation and evaluation (Verma et al., 2021). This often requires multiple PDSA cycles to be fully integrated into a healthcare platform. 

On a theoretical dimension, there are limitations to be taken into account when obtaining the most accurate assessment of CAD. As a trade-off to optimizing models towards the recall score, we identify that accuracy, precision and f1-scores decreased after GridSearchCV for Logistic Regression. Furthermore, the small size of this dataset poses a limitation to the generalization of data since information not being included in training data could result in a selection bias. Further, albeit our measures to increase generativity, the small dataset makes the model sensitive to differing input data.


## Built With

## Contributing

## Authors

## Acknowledgements
