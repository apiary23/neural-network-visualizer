from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('trainNet', views.trainNet, name='trainNet'),
    path('getNet', views.getNet, name='getNet')
]
