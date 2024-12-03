# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.resources import views

urlpatterns = [

    # The home page
    path('', views.search_resources, name='search'),


]
