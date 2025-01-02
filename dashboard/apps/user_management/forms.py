# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django import forms
from django.conf import settings
from django.template.defaultfilters import default

from apps.authentication.views import register_user


class UserTableForm(forms.Form):

    def __init__(self, *args, **kwargs):
        if "users" in kwargs:
            super(UserTableForm, self).__init__()
        else:
            super(UserTableForm, self).__init__(*args, **kwargs)
        clusters = ([(cluster, cluster) for cluster in settings.CLUSTER_NAMES.values()])
        clusters.append(("N/A","N/A"))

        gpus = ([(gpu, gpu) for gpu in settings.GPU_LIST])
        gpus.append(("N/A", "N/A"))

        minimal_envs = (("Minimal","Minimal"),("Full","Full"))
        memory_limit = ([(str(2**pow)+" Gi", str(2**pow)+" Gi") for pow in range(settings.MAX_MEMORY)])
        cpu_limit = ([(str(2 ** pow), str(2 ** pow)) for pow in range(settings.MAX_CPU)])
        if "users" in kwargs:
            for i in kwargs["users"]:
                username = i["username"]
                self.fields[f"namespace_{username}"] = forms.CharField(max_length=100, label='namespace',initial=i["namespace"])
                self.fields[f"date_{username}"] = forms.DateField(label='date',initial=i["date"],widget=forms.DateInput(format="%Y-%m-%d", attrs={"type": "date"}),
                input_formats=["%d-%m-%Y"])

                self.fields[f"conda_{username}"] = forms.FileField(label='conda',initial=i["conda"])
                self.fields[f"cluster_{username}"] = forms.ChoiceField(choices=clusters, label='cluster',initial=i["cluster"])

                self.fields[f"gpu_{username}"] = forms.ChoiceField(choices=gpus, label='gpu',
                                                                       initial=i["gpu"])

                self.fields[f"minimal_environment_{username}"] = forms.ChoiceField(label='minimal_environment',initial=i["minimal_env"],choices=minimal_envs)
                self.fields[f"memory_limit_{username}"] = forms.ChoiceField(label='memory_limit', choices = memory_limit,
                                                                                    initial=i["memory_limit"])
                self.fields[f"cpu_limit_{username}"] = forms.ChoiceField(label='cpu_limit', choices=cpu_limit,
                                                                            initial=i["cpu_limit"])
                
                project_admin = False
                if str(i["project_admin"]) == "1" or str(i["project_admin"]) == "1.0":
                    project_admin = True
                self.fields[f"project_admin_{username}"] = forms.BooleanField(label='project_admin', required=False, initial=project_admin)
        else:
            for k in args[0]:
                if k.startswith("namespace"):
                    self.fields[k] = forms.CharField(max_length=100, label='namespace')
                elif k.startswith("date"):
                    self.fields[k] = forms.DateField(label='date',widget=forms.DateInput(format="%d-%m-%Y", attrs={"type": "date"}),
        input_formats=["%d-%m-%Y"])
                elif k.startswith("conda"):
                     self.fields[k] = forms.FileField(label='conda')
                elif k.startswith("cluster"):
                    self.fields[k] = forms.ChoiceField(choices=clusters, label='cluster')
                elif k.startswith("gpu"):
                    self.fields[k] = forms.ChoiceField(choices=gpus, label='gpu')
                elif k.startswith("minimal_environment"):
                    self.fields[k] = forms.ChoiceField(label='minimal_environment',choices=minimal_envs)
                elif k.startswith("memory_limit"):

                    self.fields[k] = forms.ChoiceField(label='memory_limit',
                                                                                choices=memory_limit,
                                                                                )
                elif k.startswith("cpu_limit"):

                    self.fields[k] = forms.ChoiceField(label='cpu_limit',
                                                       choices=cpu_limit,
                                                       )
                elif k.startswith("project_admin"):
                    self.fields[k] = forms.BooleanField(label='project_admin', required=False)

