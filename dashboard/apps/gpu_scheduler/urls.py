# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path


from django.urls import path
from .views import GPUSchedulabilityAPIView, book_gpu, gpu_booking_info  # Import your API view

urlpatterns = [
    path('gpu-schedulability/', GPUSchedulabilityAPIView.as_view(), name='gpu_schedulability'),
    path('', book_gpu, name='gpu_booking_form'),
    path('my-bookings/', gpu_booking_info, name='book_gpu'),
]