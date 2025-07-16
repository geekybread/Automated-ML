import pickle
import optuna
import pandas as pd
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
class objective(object):
    def __init__(self, df, regressor_name):
        self.df = df
        self.regressor_name = regressor_name
        self.i = False
    
    def __call__(self, trial):
        self.X, self.y = self.df.iloc[:,:-1], self.df.iloc[:,-1]
        if self.regressor_name=='all':
            self.i=True
        if self.i:
            self.regressor_name = trial.suggest_categorical("regressor", ["SVR", "rf","knn",'tree','linear'])
        if self.regressor_name == "SVR":
            kernel = trial.suggest_categorical("kernel",['linear', 'poly', 'sigmoid','rbf']) 
            degree = trial.suggest_int('degree',0,4)
            self.regressor_obj = sklearn.svm.SVR(
                kernel=kernel, 
                degree=degree)
        elif self.regressor_name=='rf':
            rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
            criterion = trial.suggest_categorical('rf_criterion',['squared_error', 'absolute_error', 'poisson'])
            n_estimators = trial.suggest_int('n_estimators',10,50)
            max_features = trial.suggest_categorical('rf_max_feature',['sqrt', 'log2', None])
            min_samples_split = trial.suggest_int("rf_min_samples", 2, 20)

            self.regressor_obj = RandomForestRegressor(
                max_depth=rf_max_depth,
                criterion=criterion,
                n_estimators=n_estimators,
                max_features=max_features,
                min_samples_split=min_samples_split
            )
        elif self.regressor_name=='knn':
            n_neighbors = trial.suggest_int('n_neighbours',5,7,log=True)
            weights = trial.suggest_categorical('weights',['uniform','distance'])
            algorithm= trial.suggest_categorical('algorithm',['auto', 'ball_tree', 'kd_tree', 'brute'])
            self.regressor_obj = KNeighborsRegressor(
                n_neighbors = n_neighbors,
                weights = weights,
                algorithm=algorithm
            )
        elif self.regressor_name=='tree':
            max_depth = trial.suggest_int('tree_max_depth',10,20,log=True)
            criterion = trial.suggest_categorical('tree_criterion',['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
            splitter = trial.suggest_categorical('tree_splitter',['best', 'random'])
            max_features = trial.suggest_categorical('tree_max_features',['sqrt', 'log2', None])
            min_samples_split = trial.suggest_int("tree_min_samples", 2, 20)

            self.regressor_obj = DecisionTreeRegressor(
                criterion=criterion,
                max_depth=max_depth,
                splitter=splitter,
                max_features=max_features,
                min_samples_split=min_samples_split
            )

        elif self.regressor_name=='linear':
            self.regressor_obj = LinearRegression()


        score = sklearn.model_selection.cross_val_score(self.regressor_obj, self.X, self.y, n_jobs=-1, cv=3)
        accuracy = score.mean()
        with open("uploads/{}.pickle".format(trial.number), "wb") as fout:
            pickle.dump(self.regressor_obj, fout)
            return accuracy


class Regressor():

    def __init__(self,df, regressor_name):
        self.df = df
        self.regressor_name = regressor_name

    

    def regress(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective(self.df, self.regressor_name), n_trials=20)
        best_reg = "uploads/{}.pickle".format(study.best_trial.number)
        return study, best_reg