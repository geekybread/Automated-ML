import pickle
import optuna
import pandas as pd
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class objective(object):
    def __init__(self, df, classifier_name):
        self.df = df
        self.classifier_name = classifier_name
        self.i = False
    
    def __call__(self, trial):
        self.X, self.y = self.df.iloc[:,:-1], self.df.iloc[:,-1]
        if self.classifier_name=='all':
            self.i=True
        if self.i:
            self.classifier_name = trial.suggest_categorical("classifier", ["SVC", "rf","knn",'tree','logistic'])
        if self.classifier_name == "SVC":
            svc_c = trial.suggest_float("svc_c", 1e-3, 1000, log=True)
            self.classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
        elif self.classifier_name=='rf':
            rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
            criteria = trial.suggest_categorical('criteria',['gini', 'entropy'])
            min_samples_split = trial.suggest_int("min_samples", 2, 20)
            n_estimators = trial.suggest_int('n_estimators',10,50)
            max_features = trial.suggest_categorical('max_feature',['sqrt', 'log2', None])
            self.classifier_obj = RandomForestClassifier(
                max_depth=rf_max_depth,
                n_estimators=n_estimators,
                criterion=criteria,
                max_features=max_features,
                min_samples_split=min_samples_split
            )
        elif self.classifier_name=='knn':
            n_neighbors = trial.suggest_int('n_neighbours',5,7,log=True)
            weights = trial.suggest_categorical('weights',['uniform','distance'])
            algorithm= trial.suggest_categorical('algorithm',['auto', 'ball_tree', 'kd_tree', 'brute'])
            self.classifier_obj = KNeighborsClassifier(
                n_neighbors = n_neighbors,
                weights = weights,
                algorithm=algorithm
            )
        elif self.classifier_name=='tree':
            max_depth = trial.suggest_int('max_depth',10,20,log=True)
            criterion = trial.suggest_categorical('criterion',['gini', 'entropy'])
            splitter = trial.suggest_categorical('splitter',['best', 'random'])
            max_feature = trial.suggest_categorical('max_feature',['sqrt', 'log2', None])
            min_samples_split = trial.suggest_int("min_samples", 2, 20)

            self.classifier_obj = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                max_features=max_feature,
                splitter=splitter,
                min_samples_split=min_samples_split
            )

        elif self.classifier_name == 'logistic':
            C = trial.suggest_float('c', 1e-3, 1000, log=True)
            self.classifier_obj = LogisticRegression(
                C=C,
                max_iter=500,  # increase from default 100
                random_state=42,
                solver='lbfgs'  # optional, but good to be explicit
    )



        score = sklearn.model_selection.cross_val_score(self.classifier_obj, self.X, self.y, n_jobs=-1, cv=3)
        accuracy = score.mean()
        with open("uploads/{}.pickle".format(trial.number), "wb") as fout:
            pickle.dump(self.classifier_obj, fout)
            return accuracy


class Classifier():

    def __init__(self,df, classifier_name):
        self.df = df
        self.classifier_name = classifier_name

    

    def classify(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective(self.df, self.classifier_name), n_trials=3)
        best_clf = "uploads/{}.pickle".format(study.best_trial.number)
        return study, best_clf
