import os
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset
from azureml.core import Workspace, Dataset
from sklearn.linear_model import LogisticRegression

## initialize your workspace information here

subscription_id = '10c5d508-c599-42ff-85c4-c15b92f298b5'
resource_group = 'nirmal-test'
workspace_name = 'AzureML_Nirmal_Test'
workspace = Workspace(subscription_id, resource_group, workspace_name)

def split_data(ds_name, test_size):
    
    dataset = Dataset.get_by_name(workspace, name= ds_name)
    df= dataset.to_pandas_dataframe()   
    y = df.iloc[:,-1].values   # output variable
    X = df.iloc[:, :-1].values  # feature variables
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=42)
    return X_train, X_test, y_train, y_test  # return the train and test split data


run = Run.get_context()

X_train, X_test, y_train, y_test= split_data('creditcard', 0.3)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(X_train, y_train)
    
    
    # create an output folder
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

    accuracy = model.score(X_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()