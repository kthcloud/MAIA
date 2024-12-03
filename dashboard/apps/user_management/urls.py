# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.user_management import views

urlpatterns = [

    # The home page
    path('', views.index, name='user-management'),
    path('download/<str:group_id>',views.download_view),


]
