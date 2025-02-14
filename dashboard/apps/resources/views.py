from django.shortcuts import render
from .forms import ResourceRequestForm
from MAIA.kubernetes_utils import get_namespaces, get_available_resources, get_filtered_available_nodes
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import redirect
from django.conf import settings

def search_resources(request):
    if not request.user.is_superuser:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render({}, request))
    form = ResourceRequestForm(request.POST)

    try:
        id_token = request.session.get('oidc_id_token')
    except:
        return redirect("/maia/login/")

    groups = request.user.groups.all()
    namespaces = []
    is_user = False
    if request.user.is_superuser:
        namespaces = get_namespaces(id_token,api_urls=settings.API_URL, private_clusters=settings.PRIVATE_CLUSTERS)

    else:
        for group in groups:
            if str(group) != "MAIA:users":

                namespaces.append(str(group).split(":")[-1].lower().replace("_","-"))
            else:
                is_user = True

    if form.is_valid():
        #form.save()
        id_token = request.session.get('oidc_id_token')
        gpu_request = form.cleaned_data.get("gpu_request")
        cpu_request = form.cleaned_data.get("cpu_request")
        memory_request = form.cleaned_data.get("memory_request")
        gpu_dict, cpu_dict, ram_dict, gpu_allocations = get_available_resources(id_token=id_token, api_urls=settings.API_URL,cluster_names=settings.CLUSTER_NAMES, private_clusters=settings.PRIVATE_CLUSTERS)

        available_gpu,available_cpu,available_memory = get_filtered_available_nodes(gpu_dict, cpu_dict, ram_dict,cpu_request=cpu_request,gpu_request=gpu_request, memory_request=memory_request)

        return render(request, "resources.html", {"user": ["admin"],"username": request.user.username + " [ADMIN]","namespaces":namespaces, "gpu_allocations":gpu_allocations,"form": form,"available_gpu":available_gpu,"available_cpu":available_cpu,"available_memory":available_memory})
    form = ResourceRequestForm()
    id_token = request.session.get('oidc_id_token')
    _,_,_, gpu_allocations = get_available_resources(id_token=id_token, api_urls=settings.API_URL,cluster_names=settings.CLUSTER_NAMES, private_clusters=settings.PRIVATE_CLUSTERS)

    return render(request, "resources.html", {"user": ["admin"], "username": request.user.username + " [ADMIN]","form": form, "namespaces":namespaces,"gpu_allocations":gpu_allocations})