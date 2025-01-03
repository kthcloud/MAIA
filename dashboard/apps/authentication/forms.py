# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from apps.models import MAIAUser, MAIAProject
from django.conf import settings
from MAIA.dashboard_utils import get_groups_in_keycloak


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

    
    maia_groups = get_groups_in_keycloak(settings= settings)

    namespace = forms.ChoiceField(
        choices=[(maia_group,maia_group) for maia_group in maia_groups.values()],
        widget=forms.Select(attrs={
           'class': "form-select text-center fw-bold",
            'style': 'max-width: auto;',
        }
    )
    )

    email = forms.EmailField(
        widget=forms.EmailInput(
            attrs={
                "placeholder": "Your Email",
                "class": "form-control"
            }
        ))
    password1 = forms.CharField(
        initial="maiaPassword",
        )
    password2 = forms.CharField(
        initial="maiaPassword",

        )
    class Meta:
        model = MAIAUser
        fields = ('username','email', 'namespace', 'password1', 'password2')


class RegisterProjectForm(forms.ModelForm):
    
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
                "placeholder": "Your Email. You are registered as the Project Admin.",
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

    conda = forms.FileField(required=False,label="Upload here your Conda Environment/PIP Requirements file to automatically load it in your environment.")

    date = forms.DateField(widget=forms.TextInput(attrs={'class': 'form-control', 'type': 'date'}))
    
    memory_limit = ([(str(2**pow)+" Gi", str(2**pow)+" Gi") for pow in range(settings.MAX_MEMORY)])
    cpu_limit = ([(str(2 ** pow), str(2 ** pow)) for pow in range(settings.MAX_CPU)])
    
    memory_limit = forms.ChoiceField(label='memory_limit',
                                                                                choices=memory_limit,
                                                                                )


    cpu_limit = forms.ChoiceField(label='cpu_limit',
                                                       choices=cpu_limit,
                                                       )

    class Meta:
        model = MAIAProject
        fields = ('namespace','gpu', 'conda', 'date', 'email', 'memory_limit', 'cpu_limit')
