from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from django.conf import settings
from django.shortcuts import redirect
from django.template.defaultfilters import register
from MAIA.kubernetes_utils import get_namespaces, get_cluster_status
import urllib3
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@register.filter(name='dict_key')
def dict_key(d):
    return d[0]

@register.filter(name='dict_val')
def dict_val(d):
    return d[1]

@register.filter
def index(indexable, i):
    try:
        return indexable[i]
    except:
        return None

@register.filter
def extract_from_form(form, key):
    try:
        return form[key]
    except:
        return None

@register.filter(name='gpu_type_from_node')
def gpu_type_from_node(nodes, node_name):
    return nodes[node_name]['metadata']['labels']['nvidia.com/gpu.product']

@register.filter(name='gpu_vram_from_node')
def gpu_vram_from_node(nodes, node_name):
    return str(int(nodes[node_name]['metadata']['labels']['nvidia.com/gpu.memory'])/1024)+" Gi"

@register.filter(name='requested_gpu')
def requested_gpu(requests):
    return requests["nvidia.com/gpu"]

@register.filter(name='get_item')
def get_item(dictionary, key):
    return dictionary.get(key)

@register.filter
def to_space(value):
    return value.replace("-"," ")

@register.filter
def maia(value):
    if "Maia" in value:
        return value.replace("Maia","MAIA")
    elif "Kth" in value:
        return value.replace("Kth","KTH")
    else:
        return value


@register.filter
def env(key):
    return os.environ.get(key, None)

@login_required(login_url="/maia/login/")
def index(request):

    try:
        id_token = request.session.get('oidc_id_token')
        status,cluster_dict = get_cluster_status(id_token, api_urls=settings.API_URL,cluster_names=settings.CLUSTER_NAMES, private_clusters=settings.PRIVATE_CLUSTERS)
    except:
        return redirect("/login/")

    context = {'segment': 'index',
               "status":status,
               "id_token":id_token,
               "clusters":cluster_dict,
               "external_links":settings.CLUSTER_LINKS}
    groups = request.user.groups.all()

    namespaces = []
    is_user = False
    if request.user.is_superuser:
        namespaces = get_namespaces(id_token, api_urls=settings.API_URL, private_clusters=settings.PRIVATE_CLUSTERS)

    else:
        for group in groups:
            if str(group) != "MAIA:users":

                namespaces.append(str(group).split(":")[-1].lower().replace("_","-"))
            else:
                is_user = True

    html_template = loader.get_template('home/index.html')
    context["namespaces"] = namespaces
    if not is_user and not request.user.is_superuser:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))
    if request.user.is_superuser:
        context["username"] = request.user.username + " [ADMIN]"
        context["user"] = ["admin"]
    else:
        context["username"] = request.user.username

    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/maia/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]
        id_token = request.session.get('oidc_id_token')
        status,cluster_dict = get_cluster_status(id_token, api_urls=settings.API_URL,cluster_names=settings.CLUSTER_NAMES, private_clusters=settings.PRIVATE_CLUSTERS)
        context = {
                   "status": status,
                    "id_token":id_token,
                    "clusters":cluster_dict,
        "external_links":settings.CLUSTER_LINKS}

        groups = request.user.groups.all()

        namespaces = []
        is_user = False
        if request.user.is_superuser:
            namespaces = get_namespaces(id_token, api_urls=settings.API_URL, private_clusters=settings.PRIVATE_CLUSTERS)

        else:
            for group in groups:
                if str(group) != "MAIA:users":
                    namespaces.append(str(group).split(":")[-1].lower().replace("_","-"))
                else:
                    is_user = True
        context["namespaces"] = namespaces

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        if not is_user and "admin" not in namespaces:
            html_template = loader.get_template('home/page-500.html')
            return HttpResponse(html_template.render(context, request))
        if request.user.is_superuser:
            context["username"] = request.user.username + " [ADMIN]"
            context["user"] = ["admin"]
        else:
            context["username"] = request.user.username

        html_template = loader.get_template('home/' + load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))



def maia_docs(request):
    context = {}

    html_template = loader.get_template('List.html')
    return HttpResponse(html_template.render(context, request))

def maia_spotlight(request):
    context = {}

    html_template = loader.get_template('spotlight.html')
    return HttpResponse(html_template.render(context, request))