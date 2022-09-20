from base.rate_service import get_data
from base.data_pipeline import prepare_features

from base.models import Asset, Algorithm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, RandomForestRegressor, BaggingRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import KFold, GridSearchCV
from datetime import datetime, date, timedelta


def tune():
    excludes = ['XBTUSDT']
    assets = Asset.objects.exclude(symbol__in=excludes)

    assets = list(Asset.objects.all())  
    for asset in assets:
        print(f'optimizing models for {asset.symbol}')
        X, Y = prepare_data(asset)        
        algos = Algorithm.objects.filter(asset=asset)
        for algo in algos:
            if algo.name == 'RandomForestClassifier':
                random_forest_classifier_optimizer(X, Y, algo)
            if algo.name == 'ExtraTreesClassifier':
                extra_trees_classifier(X, Y, algo)
            if algo.name == 'DecisionTreeRegressor':
                decision_tree_regressor_optimizer(X, Y, algo)


def prepare_data(asset):     
    data, rc = get_data(asset.symbol, asset.timeframe)
    data = prepare_features(data, asset)
    x = data.drop(['dir', 'returns'], axis = 1).values
    y = data['dir'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return x_train, y_train
    


def random_forest_classifier_optimizer(X, Y, algo):
    now = datetime.now()
    print(f"{now}: running random forest classifier optimizer")
    scoring = 'accuracy'
    num_folds=10
    n_estimators=[20,100, 120, 250]
    max_depth=[5,10, 50, 100]
    criterion=["gini", "entropy"]    
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
    m = RandomForestClassifier()
    kfold = KFold(n_splits=num_folds)
    grid = GridSearchCV(estimator=m, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X, Y)
    print(f"best: {grid_result.best_score_} using {grid_result.best_params_}")
    algo.criterion = grid_result.best_params_['criterion'] 
    algo.max_depth = grid_result.best_params_['max_depth']
    algo.n_estimators = grid_result.best_params_['n_estimators']    
    algo.accuracy = grid_result.best_score_
    algo.save()
    print(f'{now}: random forest optimizer done, accuracy: {grid_result.best_score_}')
    print(f'criterion: {algo.criterion} - max depth: {algo.max_depth} - estimators: {algo.n_estimators}')


def extra_trees_classifier(X, Y, algo):
    now = datetime.now()
    print(f"{now}: extra trees classifier optimizer")
    scoring = 'accuracy'
    num_folds=10
    criterion=["entropy"]
    n_estimators=[80, 200, 400]
    max_depth=[40, 50, 80]
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, criterion=criterion)
    m = ExtraTreesClassifier(random_state = 42)
    kfold = KFold(n_splits=num_folds)
    grid = GridSearchCV(estimator=m, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X, Y)
    algo.max_depth = grid_result.best_params_['max_depth']
    algo.n_estimators = grid_result.best_params_['n_estimators']    
    algo.accuracy = grid_result.best_score_
    algo.save()
    print(f'{now}: extra trees regression done, accuracy: {grid_result.best_score_}')
    print(f'criterion: {algo.criterion} - max depth: {algo.max_depth} - estimators: {algo.n_estimators}')

def decision_tree_regressor_optimizer(X, Y, algo):
    now = datetime.now()
    print(f"{now}: running decision tree regressor optimizer")
    scoring = 'accuracy'
    num_folds=10
    n_estimators=[5, 20,100, 120]
    max_depth=[5, 10, 50, 100] 
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
    m = RandomForestClassifier()
    kfold = KFold(n_splits=num_folds)
    grid = GridSearchCV(estimator=m, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X, Y)
    print(f"best: {grid_result.best_score_} using {grid_result.best_params_}")
    algo.max_depth = grid_result.best_params_['max_depth']
    algo.n_estimators = grid_result.best_params_['n_estimators']    
    algo.accuracy = grid_result.best_score_
    algo.save()
    print(f'{now}: decision tree regressor optimizer done, accuracy: {grid_result.best_score_}')
    print(f'criterion: {algo.criterion} - max depth: {algo.max_depth} - estimators: {algo.n_estimators}')