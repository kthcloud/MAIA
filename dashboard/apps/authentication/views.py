# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from .forms import LoginForm, SignUpForm
from minio import Minio
import os
from .utils import send_discord_message
from core.settings import GITHUB_AUTH
from django.conf import settings

def login_view(request):
    form = LoginForm(request.POST or None)

    msg = None

    if request.method == "POST":

        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("/")
            else:
                msg = 'Invalid credentials'
        else:
            msg = 'Error validating the form'

    return render(request, "accounts/login.html", {"form": form, "msg": msg, "GITHUB_AUTH": GITHUB_AUTH})


def register_user(request):
    msg = None
    success = False

    if request.method == "POST":

        form = SignUpForm(request.POST, request.FILES)
        if form.is_valid():

            client = Minio(settings.MINIO_URL,
                           access_key=settings.MINIO_ACCESS_KEY,
                           secret_key=settings.MINIO_SECRET_KEY,
                           secure=True)
            form.save()
            username = form.cleaned_data.get("username")
            raw_password = form.cleaned_data.get("password1")
            namespace = form.cleaned_data.get("namespace")


            if 'conda' in request.FILES:
                with open(f"/tmp/{namespace}_env",'wb+') as destination:
                    for chunk in request.FILES['conda'].chunks():
                        destination.write(chunk)
                client.fput_object(settings.BUCKET_NAME, f"{namespace}_env", f"/tmp/{namespace}_env")


            user = authenticate(username=username, password=raw_password)

            user.conda = f"{namespace}_env"
            user.is_active = False
            user.save()

            #if os.environ["DEBUG"] != "True":

            #send_email(email, os.environ["admin_email"], email)
            if settings.DISCORD_URL is not None:
                send_discord_message(username=username, namespace=namespace)
            #https://splootcode.io/2023/02/16/the-simplest-way-to-send-discord-messages-from-python/
            msg = 'Request for Account Registration submitted successfully.'
            success = True

            # return redirect("/login/")

        else:
            print(form.errors)
            msg = 'Form is not valid'
    else:
        form = SignUpForm()

    return render(request, "accounts/register.html", {"form": form, "msg": msg, "success": success})
