from django.urls import path
from . import views
urlpatterns = [
    path('', views.getRoutes),
    path('assets', views.getAssets),
    path('assets/<str:identifier>', views.getAsset),
    path('models', views.getModels),
    path('models/<str:identifier>', views.getModel)
    
#    path('accounts/create', views.createAccount),
#    path('accounts', views.getAccounts),
#    path('accounts/<str:pk>', views.getAccount),
]