from base.rate_service import get_history
from base.data_pipeline import prepare_features
from .models import Asset, Algorithm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, RandomForestRegressor, BaggingRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def get_forecast(timeframe, batch):
    assets = list(Asset.objects.filter(timeframe = timeframe, isactive = True, batch = batch))
    for asset in assets:
        print(f'{asset.identifier} ====================================== STARTING')
        run(asset, timeframe)
        print(f'{asset.identifier} ====================================== DONE')

def run(asset, timeframe):
    algos = list(Algorithm.objects.filter(asset = asset, isactive = True))
    df = get_history(asset.symbol, timeframe)
    data, cols = prepare_features(df, asset)
    last_candle = data.iloc[-1]
    lco = last_candle['open']
    lcc = last_candle['close']
    actual_direction = 0
    if lcc >= lco:
        actual_direction = 1
    x = data.drop(['dir', 'returns'], axis = 1).values
    y = data['dir'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    values = sc.transform(x[[-1]])
    accuracy = 0
    ups = 0
    downs = 0
    for algo in algos:
        print(f'----- running {algo.name} ------')
        if actual_direction == algo.prediction:
            algo.predictions_correct += 1
        pred, acc = run_model(values, x_train, y_train, x_test, y_test, algo)     
        print(f'pred: {pred}, acc: {acc}')
        algo.prediction = pred
        algo.accuracy = acc
        algo.predictions_total += 1
        algo.save()
        accuracy += acc
        if pred > 0:
            ups += 1
        else:
            downs += 1
    prediction = round(ups/(ups+downs))
    accuracy = accuracy/len(algos)
    asset.ups = ups
    asset.downs = downs
    print(f'prediction: {prediction}, accuracy: {accuracy}')
    asset.accuracy = accuracy
    asset.predictions_total += 1
    if asset.current_prediction == actual_direction:
        asset.predictions_correct += 1
    print(f'last prediction: {asset.current_prediction == actual_direction} (predicted: {asset.current_prediction} - actual: {actual_direction}')
    asset.last_prediction = asset.current_prediction
    asset.current_prediction = prediction
    asset.save()

def run_model(values, x_train, y_train, x_test, y_test, algorithm):

    if algorithm.name == "RandomForestClassifier":
        classifier = RandomForestClassifier(criterion= algorithm.criterion, max_depth=algorithm.max_depth, n_estimators=algorithm.n_estimators, random_state=algorithm.random_state)
    if algorithm.name == "ExtraTreesClassifier":
        classifier = ExtraTreesClassifier(criterion= algorithm.criterion, max_depth=algorithm.max_depth, n_estimators=algorithm.n_estimators, random_state=algorithm.random_state)
    if algorithm.name == "AdaBoostClassifier":
        classifier = AdaBoostClassifier(RandomForestClassifier(criterion=algorithm.criterion, max_depth=algorithm.max_depth, n_estimators=algorithm.n_estimators, random_state=algorithm.random_state))
    if algorithm.name == "DecisionTreeClassifier":
        classifier = DecisionTreeClassifier(criterion=algorithm.criterion, splitter=algorithm.splitter, random_state=algorithm.random_state)
    if algorithm.name == "DecisionTreeRegressor":
        classifier = DecisionTreeRegressor(random_state=algorithm.random_state)
    if algorithm.name == "XGBClassifier":
        classifier = XGBClassifier(learning_rate=algorithm.learning_rate, max_depth=algorithm.max_depth, n_estimators=algorithm.n_estimators, use_label_encoder=False)    
    if algorithm.name == "GradientBoostingClassifier":
        classifier = GradientBoostingClassifier(n_estimators=algorithm.n_estimators, learning_rate=algorithm.learning_rate, max_depth=algorithm.max_depth, random_state=algorithm.random_state)    
    if algorithm.name == "BaggingClassifier":
        classifier = BaggingClassifier(ExtraTreesClassifier(criterion=algorithm.criterion, max_depth=algorithm.max_depth, n_estimators=algorithm.n_estimators, random_state=algorithm.random_state), random_state=algorithm.random_state, n_estimators=algorithm.n_estimators)   
    if algorithm.name == "BaggingRegressor":
        classifier = BaggingRegressor(ExtraTreesClassifier(criterion=algorithm.criterion, max_depth=algorithm.max_depth, n_estimators=algorithm.n_estimators, random_state=algorithm.random_state), random_state=algorithm.random_state, n_estimators=algorithm.n_estimators) 
    
    classifier.fit(x_train, y_train)
    y_pred = np.round(classifier.predict(x_test),0)
    ac = round(accuracy_score(y_pred, y_test), 4)
    pred = np.round(classifier.predict(values),0)
    return int(pred), ac


























def get_sentiment():
    print(f'GETTING SENTIMENTAL **************************')