# Credit Card Fraud Prediction Model - using Azure ML
This project is my final capstone for Udacity's Nanodegree program: *Azure Machine Learning Engineer*. Credit card fraud is the most prevalent type of identity theft. According to the Federal Trade Commission, more than 270,000 Americans reported new or existing account fraud in 2019. Losses from fraud involving cards used for payment worldwide reached $27.85 billion in 2018. They are projected to rise to $35.67 billion in five years and $40.63 billion in 10 years. 

It is very important that credit card companies recognize fraudulent transactions so that customers are not charged for items that they did not purchase. Better yet, if companies can predict in real time when transaction happens to notify customers about possible fraud.

In this project, I am using credit card transactions data from Kaggle to train a machine learning model in Azure ML platform. I will be applying both hyperdrive and autoML methods to make the prediction.

## Project Set Up and Installation
1. Set up Azure ML workspace ([How to set up Azure ML workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=azure-portal))
2. Set up Compute Instance ([How to set up Compute Instance](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-manage-compute-instance?tabs=python))
3. Set up Compute Cluster ([How to set up Compute Cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python))
4. If using Python SDK, [install the SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?preserve-view=true&view=azure-ml-py)

## Dataset
This dataset contain transactions made by credit cards in September 2013 by european cardholders. It can be downloaded from [here](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains only numerical input variables which are the result of a PCA transformation. Due to confidentiality issues, details of data are not provided. Features V1 to V28 are the principal components obtained with PCA. Two other untransformed features are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount. Column 'Class' is the output variable (1 in case of fraud and 0 otherwise).


### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
