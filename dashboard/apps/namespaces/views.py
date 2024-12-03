
from django.contrib.auth.decorators import login_required
from django.template import loader
import time
from django.template.defaulttags import register
from .utils import get_namespaces
from django.http import HttpResponse, HttpResponseRedirect
from .utils import get_pods_for_namespace, get_nodes, get_svc_for_namespace, get_jobs, get_pvclaims
# Create your views here.
from .forms import ScaleDeployForm, DeleteJobForm, DeleteHelmChart, DeletePodForm
import datetime


@register.filter
def to_hyphen(value):
    return value.replace("_","-")

@register.filter
def to_str(value):
    """converts int to string"""
    return str(value)

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

@register.filter
def get_item_in_list(list, key):
    for element in list:
        if element["username"] == key:
            return element


@register.filter
def get_pvc(volumes, volume_name):
    for volume in volumes:
        if volume["name"] == volume_name:
            if "persistentVolumeClaim" in volume:
                return "PVC: "+volume["persistentVolumeClaim"]["claimName"]
            elif "emptyDir" in volume:
                return "emptyDir"
            elif "hostPath" in volume:
                return "hostPath"
    return "N/A"

@register.filter
def subtract(value, arg):
    return value - arg


@register.filter
def get_job_life_info(job):
    try:
        life_time = job["spec"]["activeDeadlineSeconds"]
        start = datetime.datetime.strptime(job["status"]["startTime"], "%Y-%m-%dT%H:%M:%SZ")
        end = datetime.datetime.strptime(job["status"]["startTime"], "%Y-%m-%dT%H:%M:%SZ") + datetime.timedelta(0,life_time)
        return f"From {start} to {end}"
    except:
        return "NA"


@login_required(login_url="/maia/login/")
def namespace_view(request,namespace_id):

    if request.method == 'POST' and 'scale_deploy' in request.POST:
        id_token = request.session.get('oidc_id_token')
        #scale_deploy(id_token, request.POST['scale_deploy'], request.POST['scale'], namespace_id)
        time.sleep(2)

        form = ScaleDeployForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect('/namespaces/{}'.format(namespace_id))
           # return HttpResponse(html_template.render(context, request))
        else:
            return HttpResponseRedirect('/namespaces/{}'.format(namespace_id))

    if request.method == 'POST' and 'delete_job' in request.POST:
        id_token = request.session.get('oidc_id_token')
        #delete_job(id_token, namespace_id, request.POST['delete_job'])
        time.sleep(2)


        form = DeleteJobForm(request.POST)
        if form.is_valid():
            html_template = loader.get_template('base_namespace.html')
            return HttpResponseRedirect('/namespaces/{}'.format(namespace_id))
        # return HttpResponse(html_template.render(context, request))
        else:

            html_template = loader.get_template('base_namespace.html')
            return HttpResponseRedirect('/namespaces/{}'.format(namespace_id))

    if request.method == 'POST' and 'delete_pod' in request.POST:
        id_token = request.session.get('oidc_id_token')
        #delete_pod(id_token, namespace_id, request.POST['delete_pod'])
        time.sleep(2)

        form = DeletePodForm(request.POST)
        if form.is_valid():
            html_template = loader.get_template('base_namespace.html')
            return HttpResponseRedirect('/namespaces/{}'.format(namespace_id))
        # return HttpResponse(html_template.render(context, request))
        else:

            html_template = loader.get_template('base_namespace.html')
            return HttpResponseRedirect('/namespaces/{}'.format(namespace_id))

    if request.method == 'POST' and 'delete_configmap' in request.POST:
        id_token = request.session.get('oidc_id_token')
        #delete_config_map(id_token, namespace_id, request.POST['delete_configmap'])
        time.sleep(2)

        form = DeletePodForm(request.POST)
        if form.is_valid():
            html_template = loader.get_template('base_namespace.html')
            return HttpResponseRedirect('/namespaces/{}'.format(namespace_id))
        # return HttpResponse(html_template.render(context, request))
        else:

            html_template = loader.get_template('base_namespace.html')
            return HttpResponseRedirect('/namespaces/{}'.format(namespace_id))

    if request.method == 'POST' and 'delete_helm_chart' in request.POST:
        id_token = request.session.get('oidc_id_token')
        #delete_helm_chart(namespace_id,id_token,   request.POST['delete_helm_chart'])
        time.sleep(2)

        form = DeleteHelmChart(request.POST)
        if form.is_valid():
            html_template = loader.get_template('base_namespace.html')
            return HttpResponseRedirect('/namespaces/{}'.format(namespace_id))
        # return HttpResponse(html_template.render(context, request))
        else:

            html_template = loader.get_template('base_namespace.html')
            return HttpResponseRedirect('/namespaces/{}'.format(namespace_id))


    groups = request.user.groups.all()

    namespaces = []
    for group in groups:
        namespaces.append(str(group).split(":")[-1].lower().replace("_","-"))

    context = {}
    if namespace_id.lower().replace("_","-") not in namespaces and not request.user.is_superuser:
        html_template = loader.get_template('home/page-403.html')
        return HttpResponse(html_template.render(context, request))
    else:
        id_token = request.session.get('oidc_id_token')
        #pods = get_pods_for_namespace(id_token, namespace_id)
        #deploy = get_deploy_for_namespace(id_token, namespace_id)
        #nodes = get_nodes(id_token)
        #jobs = get_jobs(id_token, namespace_id)
        #pvc = get_pvclaims(id_token, namespace_id)
        user_id = request.user.username
        is_admin = False
        if request.user.is_superuser:
            is_admin = True
        service, ingress, maia_workspace_ingress, remote_desktop_dict, orthanc_list = get_svc_for_namespace(id_token, namespace_id, user_id, is_admin)
        form = ScaleDeployForm(request.POST)
        #charts = list_helm_charts(namespace_id, id_token)
        #config_maps = get_config_maps(id_token=id_token,namespace=namespace_id)

        groups = request.user.groups.all()

        namespaces = []
        if request.user.is_superuser:
            namespaces = get_namespaces(id_token)

        else:
            for group in groups:
                if str(group) != "MAIA:users":
                    namespaces.append(str(group).split(":")[-1].lower().replace("_","-"))
            # print(group.split(":")[-1])
        # "config_maps":config_maps,"charts": charts, "deploy":deploy
        context = { "maia_workspace_ingress":maia_workspace_ingress,"namespace":namespace_id,
                    #"pods":pods, "nodes": nodes,
                    "remote_desktop_dict": remote_desktop_dict,
                    "orthanc_list": orthanc_list,
                    "id_token": id_token, "service": service,"ingress":ingress,"form":form,
                    #"job":jobs,"pvc":pvc
                    }
        context["namespaces"] = namespaces

        if request.user.is_superuser:
            context["username"] = request.user.username + " [ADMIN]"
            context["user"] = ["admin"]
        else:
            context["username"] = request.user.username
        html_template = loader.get_template('base_namespace.html')
        return HttpResponse(html_template.render(context, request))
