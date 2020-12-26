


import pandas as pd                 # data analysis and manipulation tool
import numpy as np                  # Numerical computing tools

from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV
from sklearn import metrics



class Task(object):
    def __init__(self, task_name : str, target_type : list, main_metrics : str):
        self.task_name = task_name
        self.target_type = target_type
        self.main_metrics = main_metrics

    def print_attributes(self):
        print(f"task_name = {self.task_name}")
        print(f"target_type = {self.target_type}")
        print(f"main_metrics = {self.main_metrics}")



def choose_best_model( tasks : list, models: list, train_dataset : Bunch, valid_dataset : Bunch, output_folder_path : str):
    """Iterates over every task, and every model, returning the best model for each class."""

    # returns
    #   dict:       { task_1_idx : (model_2_idx, main_metrics) , task_2_idx : (model_3_idx, main_metrics), ... etc }

    best_task_metrics = {task_idx : None for task_idx in range(len(tasks))}
    best_task_models = {task_idx: None for task_idx in range(len(tasks))}


    for task_idx, task in enumerate(tasks):
        print("-"*40)
        print(f"\n\nInitializing model selection for task : {task.task_name}")
        # train on different classifiers
        metrics_dict = iterate_over_models(task, models, train_dataset)

        # choose the best according to the specified metric in the task
        for model_idx in metrics_dict.keys():

            # if this is the first model, take it
            if best_task_metrics[task_idx] == None:
                best_task_metrics[task_idx] = metrics_dict[model_idx]
                best_task_models[task_idx] = (model_idx, metrics_dict[model_idx])

            # if the metrics are higher than currently highest, choose this model for this task
            elif metrics_dict[model_idx] > best_task_metrics[task_idx]:
                best_task_metrics[task_idx] = metrics_dict[model_idx]
                best_task_models[task_idx] = (model_idx, metrics_dict[model_idx])


    return best_task_models


#       params for best model
# clf.cv_results_['params'][clf.best_index_]
#

def iterate_over_models(task, models: list, train_dataset : Bunch):
    """For a specific task, iterates over models. Returns all kinds of metrics"""

    # inputs
    #   models -

    # returns
    #   {0 : { metric : <value> },
    #    1 : { metric : <value> }, ....}
    #
    #   where 0,1, ..  = indexes in the models list

    metrics_dict = {}

    # obtain the correct X and y  (features and target) for this specific task
    # from the train_dataset

    X = train_dataset.data
    y = train_dataset.target_types[task.target_type]["targets"]

    for model_idx, model in enumerate(models):
        print(f"Checking model : {model[0]} over params {model[1]}")
        # use the GridSearch cross validation method to find the best parameters for the model
        # from the specified parameters
        clf_scorer = metrics.make_scorer(task.main_metrics)

        clf = GridSearchCV(model[0], model[1], scoring = clf_scorer)
        clf.fit(X, y)

        s = clf.cv_results_['rank_test_score']
        sorted_rank_idxs = sorted(range(len(s)), key=lambda k: s[k])

        print("\tReached accuracies:")
        for rank_idx in sorted_rank_idxs:
            print(f"\t\tScore : {round(clf.cv_results_['mean_test_score'][rank_idx], 3)} , "
                  f"Params : {clf.cv_results_['params'][rank_idx]}")

        # apply all those models on the validation set and pick the best

        # take best score value
        metrics_dict[model_idx] = max(clf.cv_results_['mean_test_score'])

    return metrics_dict



def print_best_task_models(chosen_models_dict : dict, tasks : list, models : list):

    print("="*40)
    print("SUMMARY")
    print("=" * 40)

    for task_idx, task in enumerate(tasks):
        print(f"\nFor task : {tasks[task_idx].task_name}")
        print(f"\tChosen model : {models[chosen_models_dict[task_idx][0]]}")
        print(f"\tMetrics : {tasks[task_idx].main_metrics}")
        print(f"\tValue : {chosen_models_dict[task_idx][1]}")



