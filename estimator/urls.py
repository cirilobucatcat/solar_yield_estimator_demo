from django.urls import path

from . import views

app_name = "estimator"

urlpatterns = [
    path("", views.index, name="index"),
]
