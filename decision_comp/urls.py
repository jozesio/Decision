from django.urls import path

from . import views

urlpatterns = [
    path("research/", views.api_research),
    path("calculate/", views.api_calculate),
]
