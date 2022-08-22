from celery import shared_task
from base.ghost import get_forecast#, get_sentiment
from base.optimizer import tune

@shared_task
def calculate_hourly_A():
    return (get_forecast(timeframe = '1h', batch='A'))

@shared_task
def calculate_hourly_B():
    return (get_forecast(timeframe = '1h', batch='B'))




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