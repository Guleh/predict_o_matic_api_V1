from base.rate_service import get_data
from base.data_pipeline import prepare_features
from .models import Asset, Algorithm, HitratioHistory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime, date, timedelta
import numpy as np
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, RandomForestRegressor, BaggingRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import time
from django.conf import settings
from arq.settings import BEARER_TOKEN
import flair

def get_forecast(timeframe):
    time.sleep(10)
    assets = list(Asset.objects.filter(timeframe = timeframe, isactive = True))
    for asset in assets:
        print(f'{asset.identifier} ====================================== STARTING')
        try:
            run(asset, timeframe)
            print(f'{asset.identifier} ====================================== DONE')
        except:
            print('====================================== ERROR')

def run(asset, timeframe):
    algos = list(Algorithm.objects.filter(asset = asset, isactive = True))
    df, raw_candles = get_data(asset.platformsymbol, asset.timeframe)
    asset.candles = raw_candles.to_json(orient='records')
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
    asset.prediction_term = datetime.now() + timedelta(hours=1)
    asset.last_close = last_candle['close']
    asset.save()
   # hitratio = HitratioHistory(hitratio = asset.predictions/asset.predictions_correct, asset = asset)
   # hitratio.save()

def run_model(values, x_train, y_train, x_test, y_test, algorithm):

    if algorithm.identifier == "RandomForestClassifier":
        classifier = RandomForestClassifier(criterion= algorithm.criterion, max_depth=algorithm.max_depth, n_estimators=algorithm.n_estimators, random_state=algorithm.random_state)
    if algorithm.identifier == "ExtraTreesClassifier":
        classifier = ExtraTreesClassifier(criterion= algorithm.criterion, max_depth=algorithm.max_depth, n_estimators=algorithm.n_estimators, random_state=algorithm.random_state)
    if algorithm.identifier == "AdaBoostClassifier":
        classifier = AdaBoostClassifier(RandomForestClassifier(criterion=algorithm.criterion, max_depth=algorithm.max_depth, n_estimators=algorithm.n_estimators, random_state=algorithm.random_state))
    if algorithm.identifier == "DecisionTreeClassifier":
        classifier = DecisionTreeClassifier(criterion=algorithm.criterion, splitter=algorithm.splitter, random_state=algorithm.random_state)
    if algorithm.identifier == "DecisionTreeRegressor":
        classifier = DecisionTreeRegressor(random_state=algorithm.random_state)
    if algorithm.identifier == "XGBClassifier":
        classifier = XGBClassifier(learning_rate=algorithm.learning_rate, max_depth=algorithm.max_depth, n_estimators=algorithm.n_estimators, use_label_encoder=False)    
    if algorithm.identifier == "GradientBoostingClassifier":
        classifier = GradientBoostingClassifier(n_estimators=algorithm.n_estimators, learning_rate=algorithm.learning_rate, max_depth=algorithm.max_depth, random_state=algorithm.random_state)    
    if algorithm.identifier == "BaggingClassifier":
        classifier = BaggingClassifier(ExtraTreesClassifier(criterion=algorithm.criterion, max_depth=algorithm.max_depth, n_estimators=algorithm.n_estimators, random_state=algorithm.random_state), random_state=algorithm.random_state, n_estimators=algorithm.n_estimators)   
    if algorithm.identifier == "BaggingRegressor":
        classifier = BaggingRegressor(ExtraTreesClassifier(criterion=algorithm.criterion, max_depth=algorithm.max_depth, n_estimators=algorithm.n_estimators, random_state=algorithm.random_state), random_state=algorithm.random_state, n_estimators=algorithm.n_estimators) 
    
    classifier.fit(x_train, y_train)
    y_pred = np.round(classifier.predict(x_test),0)
    ac = round(accuracy_score(y_pred, y_test), 4)
    pred = np.round(classifier.predict(values),0)
    return int(pred), ac


BEARER_TOKEN = settings.BEARER_TOKEN

def get_sentiment():
    try:
        assets = list(Asset.objects.filter(timeframe = '1h', isactive = True).distinct())
        for asset in assets:            
            try:
                ticker = f'({asset.platformsymbol} OR {asset.name}) (lang:en)' 
                headers = {'authorization': f'Bearer {BEARER_TOKEN}'}
                params = {'query':ticker,
                        'tweet.fields':'created_at,lang', 
                        'max_results':'100'}
                time = datetime.now() - timedelta(seconds = 15)
                timeformat = '%Y-%m-%dT%H:%M:%SZ'
                df = pd.DataFrame()
                for hour in range(24 -1):
                    pre60 = time - timedelta(minutes=60)
                    params['end_time'] = time.strftime(timeformat)
                    params['start_time'] = pre60.strftime(timeformat)
                    response = requests.get(f'https://api.twitter.com/2/tweets/search/recent', headers = headers, params = params)
                    time = pre60
                    print(response)
                    data = pd.DataFrame(response.json()['data'])    
                    df = pd.concat([df, data])
                    print('here')
                sentiment_model = flair.models.TextClassifier.load("en-sentiment")
                sentiment = 0
                confidence = 0
                values = df['text'].to_list()
                for tweet in values:
                    sentence = flair.data.Sentence(tweet)
                    sentiment_model.predict(sentence)
                    if sentence.labels[0].value == 'POSITIVE':
                        sentiment += 1
                    confidence += (sentence.labels[0].score)
                score = sentiment/len(values)
                save_assets = list(Asset.objects.filter(isactive = True, symbol = asset.name))
                for s in save_assets:
                    s.sentiment = score
                    s.save()
                print(f'sentiment calculation for {asset.name}: {score}')
            except:
                print(f'failed to get sentiment for {asset.name}')
    except:
        print(f'failed to get sentiments')
