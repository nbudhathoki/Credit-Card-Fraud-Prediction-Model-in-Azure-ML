import os
import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset
from azureml.core import Workspace, Dataset

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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="Number of trees in the forest")
    parser.add_argument('--min_samples_split', type=int, default=2, help="Minimum number of samples required to split an internal node")
    parser.add_argument('--max_features', type=str, default='auto', help="{'auto', 'sqrt', 'log2'}")
    parser.add_argument('--bootstrap', type=bool, default=True, help="Whether bootstrap samples are used or not")


    args = parser.parse_args()

    X_train, X_test, y_train, y_test= split_data('creditcard', 0.2) # pass the tabular dataset name here with test split size 80- 20 in this case

    run = Run.get_context()

    run.log("No of Estimators:", np.int(args.n_estimators))
    run.log("Min No of Samples to Split:", np.int(args.min_samples_split))
    run.log("No of Features Considered:", np.str(args.max_features))
    run.log("Bootstrap:", np.bool(args.bootstrap))

    model = RandomForestClassifier (n_estimators=args.n_estimators, min_samples_split=args.min_samples_split, 
        bootstrap=args.bootstrap, max_features=args.max_features).fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/model.pkl')

if __name__ == '__main__':
    main()