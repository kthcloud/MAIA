# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from apps.models import MAIAUser
from django.conf import settings


class LoginForm(forms.Form):
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Username",
                "class": "form-control"
            }
        ))
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Password",
                "class": "form-control"
            }
        ))


class SignUpForm(UserCreationForm):
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Your Username",
                "class": "form-control"
            }
        ))
    namespace = forms.CharField(

        widget=forms.TextInput(
            attrs={
                "placeholder": "A unique identifier for your project.",
                "class": "form-control"
            }
        ))
    email = forms.EmailField(
        widget=forms.EmailInput(
            attrs={
                "placeholder": "Your Email",
                "class": "form-control"
            }
        ))
    password1 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Password",
                "class": "form-control"
            }
        ))
    password2 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Password check",
                "class": "form-control"
            }
        ))

    gpu = forms.ChoiceField(
        choices=[(gpu,gpu) for gpu in settings.GPU_LIST],
        widget=forms.Select(attrs={
            'class': "form-select text-center fw-bold",
            'style': 'max-width: auto;',
        })
    )

    #jupyterhub = forms.BooleanField()
    #minio = forms.BooleanField()
    #mlflow = forms.BooleanField()
    #orthanc = forms.BooleanField()
    #remote_desktop = forms.BooleanField()

    conda = forms.FileField(required=False,label="Upload here your Conda environment/pip requirements file to automatically load it in your environment.")

    date = forms.DateField(widget=forms.TextInput(attrs={'class': 'form-control', 'type': 'date'}))
    class Meta:
        model = MAIAUser
        fields = ('username', 'email', 'password1', 'password2', 'namespace','gpu','date', 'conda')#
        #"conda_env_file","jupyterhub","mlflow","minio","remote_desktop","orthanc"



