
import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', 'changeme')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DEBUG', 'False')

ALLOWED_HOSTS = []
ALLOWED_HOSTS_ENV = os.environ.get('ALLOWED_HOSTS')
if ALLOWED_HOSTS_ENV:
    ALLOWED_HOSTS.extend(ALLOWED_HOSTS_ENV.split(','))

BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',
    'rest_framework',
    'base',
    'django_celery_beat',
]


MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'arq.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'arq.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

import pymysql
pymysql.install_as_MySQLdb()
DATABASES = {
    'default': {
        #'ENGINE': 'django.db.backends.sqlite3',
        #'NAME': BASE_DIR / 'db.sqlite3',
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'ghostdb',
        'USER': 'admin',
        'PASSWORD': 'dtupOEs4SAZ7jSXYTdTb',
        'HOST': 'ghostdb.cratkb3vsjdk.eu-west-3.rds.amazonaws.com',
        'PORT': '3306'
    }
}


# Password validation
# https://docs.djangoproject.com/en/4.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

CORS_ALLOW_ALL_ORIGINS = True # If this is used then `CORS_ALLOWED_ORIGINS` will not have any effect
CORS_ALLOW_CREDENTIALS = True
#CORS_ALLOWED_ORIGINS = [
#    'http://localhost:3030',
#] # If this is used, then not need to use `CORS_ALLOW_ALL_ORIGINS = True`
#CORS_ALLOWED_ORIGIN_REGEXES = [
#    'http://localhost:3030',
#]


# Internationalization
# https://docs.djangoproject.com/en/4.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.0/howto/static-files/

STATIC_URL = '/static/static/'
MEDIA_URL = '/static/media/'
STATIC_ROOT = '/vol/web/static'
MEDIA_ROOT = '/vol/web/media'

# Default primary key field type
# https://docs.djangoproject.com/en/4.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'



# CELERY
from celery.schedules import crontab
#redis://:password@hostname:port/db_number
CELERY_BROKER_URL = 'redis://:ZPGJnNCmEvA5H8HmSq0tS9KbCSGr61XG@redis-11619.c281.us-east-1-2.ec2.cloud.redislabs.com:11619'
CELERY_TIMEZONE='UTC'
CELERY_BEAT_SCHEDULE ={
    'calculate_hourly':{
        'task': 'base.tasks.calculate_hourly',
        'schedule': crontab(minute="0")
    },
    'calculate_four_hourly':{
        'task': 'base.tasks.calculate_four_hourly',
        'schedule': crontab(minute="2", hour="*/4")
    },

    'calculate_daily':{
        'task': 'base.tasks.calculate_daily',
        'schedule': crontab(minute="4", hour="0")
    },
    'get_sentiment':{
        'task': 'base.tasks.calculate_sentiment',
        'schedule': crontab(minute="30", hour="0")
    }
}
    #,
    #'tune_hyperparameters':{
    #    'task': 'base.tasks.calculate_hyperparameters',
    #    'schedule': crontab(minute="1", hour="1", day_of_week="3")
    #}

