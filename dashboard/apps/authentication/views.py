# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

# Create your views here.
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from .forms import LoginForm, SignUpForm, RegisterProjectForm, MAIAInfoForm
from minio import Minio
from MAIA.dashboard_utils import send_discord_message, verify_minio_availability, send_maia_info_email
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

    return render(request, "accounts/login.html", {"dashboard_version": settings.DASHBOARD_VERSION,"form": form, "msg": msg, "GITHUB_AUTH": GITHUB_AUTH})


def register_user(request):
    msg = None
    success = False

    if request.method == "POST":

        form = SignUpForm(request.POST, request.FILES)
        if form.is_valid():

            namespace = form.cleaned_data.get("namespace")
            if namespace.endswith(" (Pending)"):
                namespace = namespace[:-len(" (Pending)")]
            form.instance.namespace = namespace
            form.save()
            username = form.cleaned_data.get("username")
            raw_password = form.cleaned_data.get("password1")
            namespace = form.cleaned_data.get("namespace")
 

            user = authenticate(username=username, password=raw_password)

            user.is_active = False
            user.save()

            #if os.environ["DEBUG"] != "True":
                #send_email(email, os.environ["admin_email"], email)
            if settings.DISCORD_URL is not None:
                send_discord_message(username=username, namespace=namespace, url=settings.DISCORD_URL)
            msg = 'Request for Account Registration submitted successfully. Please wait for the admin to approve your request.'
            success = True

            #return redirect("/login/")

        else:
            print(form.errors)
            msg = 'Form is not valid'
    else:
        form = SignUpForm()

    return render(request, "accounts/register.html", {"dashboard_version": settings.DASHBOARD_VERSION,"form": form, "msg": msg, "success": success})

@login_required(login_url="/maia/login/")
def send_maia_info(request):
    
    if not request.user.is_superuser:
        return redirect("/maia/")
    
    hostname = settings.HOSTNAME
    register_project_url = f"https://{hostname}/maia/register_project/"
    register_user_url = f"https://{hostname}/maia/register/"
    discord_support_link = settings.DISCORD_SUPPORT_URL
    msg = None
    success = False

    if request.method == "POST":

        form = MAIAInfoForm(request.POST, request.FILES)
        if form.is_valid():
            send_maia_info_email(form.cleaned_data.get("email"), register_project_url, register_user_url, discord_support_link)
            msg = 'Request for MAIA Info submitted successfully. Please check your email for more information.'
            success = True

            #return redirect("/login/")

        else:
            print(form.errors)
            msg = 'Form is not valid'
    else:
        form = MAIAInfoForm()

    return render(request, "accounts/send_maia_info.html", {"dashboard_version": settings.DASHBOARD_VERSION,"form": form, "msg": msg, "success": success})


def register_project(request):
    msg = None
    success = False


   

    minio_available = verify_minio_availability(settings=settings)
    if request.method == "POST":

        form = RegisterProjectForm(request.POST, request.FILES)
        if form.is_valid():

            
            form.save()
            email = form.cleaned_data.get("email")
            namespace = form.cleaned_data.get("namespace")
            

            if 'conda' in request.FILES and minio_available:
                client = Minio(settings.MINIO_URL,
                            access_key=settings.MINIO_ACCESS_KEY,
                            secret_key=settings.MINIO_SECRET_KEY,
                            secure=True)
                with open(f"/tmp/{namespace}_env",'wb+') as destination:
                    for chunk in request.FILES['conda'].chunks():
                        destination.write(chunk)
                client.fput_object(settings.BUCKET_NAME, f"{namespace}_env", f"/tmp/{namespace}_env")
            

        
            msg = 'Request for Project Registration submitted successfully.'
            success = True
            
            #check_pending_projects_and_assign_id(settings=settings)

            # return redirect("/login/")

        else:
            print(form.errors)
            msg = 'Form is not valid'
    else:
        form = RegisterProjectForm()

    return render(request, "accounts/register_project.html", {"dashboard_version": settings.DASHBOARD_VERSION,"minio_available":minio_available,"form": form, "msg": msg, "success": success})
