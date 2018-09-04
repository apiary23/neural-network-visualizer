from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('chicken/', views.chicken, name='chicken'),
    path('pug/', views.yay_pug, name='Pug')
]
