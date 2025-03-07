# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.user_management import views

urlpatterns = [

    # The home page
    path('', views.index, name='user-management'),
    path('deploy/<str:group_id>',views.deploy_view),
    path('register-user/<str:email>',views.register_user_view),
    path('register-user-in-group/<str:email>',views.register_user_in_group_view),
    path('register-group/<str:group_id>',views.register_group_view),
    path('delete-group/<str:group_id>',views.delete_group_view),
    path('remove-user-from-group/<str:email>',views.remove_user_from_group_view),

]
