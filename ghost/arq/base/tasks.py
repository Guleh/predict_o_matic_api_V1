from celery import shared_task
from base.ghost import get_forecast#, get_sentiment
from base.optimizer import tune

@shared_task
def calculate_hourly():
    return (get_forecast(timeframe = '1h'))

@shared_task
def calculate_two_hourly():
    return (get_forecast(timeframe = '2h'))

@shared_task
def calculate_four_hourly():
    return (get_forecast(timeframe = '4h'))

@shared_task
def calculate_daily():
    return (get_forecast(timeframe = '1d'))

@shared_task
def calculate_sentiment():
    pass
    #return(get_sentiment())

@shared_task
def calculate_hyperparameters():
    return (tune())