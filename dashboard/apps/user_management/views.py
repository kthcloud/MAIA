import os
from django.http import FileResponse
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.template import loader
from django.conf import settings
from kubernetes import config

from pathlib import Path
from .forms import UserTableForm
from MAIA.dashboard_utils import get_user_table, generate_kubeconfig, update_user_table, register_user_in_keycloak, register_group_in_keycloak, register_users_in_group_in_keycloak, get_list_of_users_requesting_a_group, get_list_of_groups_requesting_a_user, get_project, get_project_argo_status_and_user_table, create_namespace
import urllib3
import yaml
from django.shortcuts import redirect
from MAIA_scripts.MAIA_install_project_toolkit import deploy_maia_toolkit_api
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



# Create your views here.
@login_required(login_url="/maia/login/")
def index(request):
    if not request.user.is_superuser:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render({}, request))
    
    argocd_url = settings.ARGOCD_SERVER

    user_table, to_register_in_groups, to_register_in_keycloak, maia_groups_dict, project_argo_status = get_project_argo_status_and_user_table(settings=settings, request=request)   
    
    if request.method == "POST":
        

        user_list = user_table.to_dict('records')

        for user in user_list:
            if user['email'] in to_register_in_keycloak:
                user['is_registered_in_keycloak'] = 0
            else:
                user['is_registered_in_keycloak'] = 1
            if user['email'] in to_register_in_groups:
                user['is_registered_in_groups'] = 0
            else:
                user['is_registered_in_groups'] = 1


        context = {
            "user_table": user_list,
            "minio_console_url": os.environ.get("MINIO_CONSOLE_URL",None),
            "maia_groups_dict": maia_groups_dict,
            "form": UserTableForm(request.POST),
            "project_argo_status": project_argo_status,
            "argocd_url": argocd_url,
            "user": ["admin"],
            "username": request.user.username + " [ADMIN]"
        }
        html_template = loader.get_template('base_user_management.html')

        form = UserTableForm(request.POST)

        if form.is_valid():
            update_user_table(request.POST, settings=settings)
        else:
            update_user_table(request.POST, settings=settings)

        return HttpResponse(html_template.render(context, request))

    
    user_list = user_table.to_dict('records')

    for user in user_list:
        for user in user_list:
            if user['email'] in to_register_in_keycloak:
                user['is_registered_in_keycloak'] = 0
            else:
                user['is_registered_in_keycloak'] = 1
            if user['email'] in to_register_in_groups:
                user['is_registered_in_groups'] = 0
            else:
                user['is_registered_in_groups'] = 1


    user_form = UserTableForm(users=user_list, projects=maia_groups_dict)

    context = {
        "user_table": user_list,
        "maia_groups_dict": maia_groups_dict,
        "minio_console_url": os.environ.get("MINIO_CONSOLE_URL",None),
        "form": user_form,
        "user": ["admin"],
        "project_argo_status": project_argo_status,
        "argocd_url": argocd_url,
        "username": request.user.username + " [ADMIN]"
    }

    html_template = loader.get_template('base_user_management.html')
    if not request.user.is_superuser:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render({}, request))


    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/maia/login/")
def register_user_view(request, email):
    if not request.user.is_superuser:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render({}, request))

    register_user_in_keycloak(email=email, settings=settings)

    argocd_url = settings.ARGOCD_SERVER

    user_table, to_register_in_groups, to_register_in_keycloak, maia_groups_dict, project_argo_status = get_project_argo_status_and_user_table(settings=settings, request=request)   


    user_list = user_table.to_dict('records')

    for user in user_list:
        for user in user_list:
            if user['email'] in to_register_in_keycloak:
                user['is_registered_in_keycloak'] = 0
            else:
                user['is_registered_in_keycloak'] = 1
            if user['email'] in to_register_in_groups:
                user['is_registered_in_groups'] = 0
            else:
                user['is_registered_in_groups'] = 1


    user_form = UserTableForm(users=user_list, projects=maia_groups_dict)

    context = {
        "user_table": user_list,
        "maia_groups_dict": maia_groups_dict,
        "minio_console_url": os.environ.get("MINIO_CONSOLE_URL",None),
        "form": user_form,
        "user": ["admin"],
        "project_argo_status": project_argo_status,
        "argocd_url": argocd_url,
        "username": request.user.username + " [ADMIN]"
    }

    html_template = loader.get_template('base_user_management.html')

    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/maia/login/")
def register_user_in_group_view(request, email):
    if not request.user.is_superuser:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render({}, request))
    


    groups = get_list_of_groups_requesting_a_user(email=email, settings=settings)
    
    for group_id in groups:
        register_users_in_group_in_keycloak(group_id=group_id,emails=[email], settings=settings)

    argocd_url = settings.ARGOCD_SERVER

    user_table, to_register_in_groups, to_register_in_keycloak, maia_groups_dict, project_argo_status = get_project_argo_status_and_user_table(settings=settings, request=request)   


    user_list = user_table.to_dict('records')

    for user in user_list:
        for user in user_list:
            if user['email'] in to_register_in_keycloak:
                user['is_registered_in_keycloak'] = 0
            else:
                user['is_registered_in_keycloak'] = 1
            if user['email'] in to_register_in_groups:
                user['is_registered_in_groups'] = 0
            else:
                user['is_registered_in_groups'] = 1


    user_form = UserTableForm(users=user_list, projects=maia_groups_dict)

    context = {
        "user_table": user_list,
        "maia_groups_dict": maia_groups_dict,
        "minio_console_url": os.environ.get("MINIO_CONSOLE_URL",None),
        "form": user_form,
        "user": ["admin"],
        "project_argo_status": project_argo_status,
        "argocd_url": argocd_url,
        "username": request.user.username + " [ADMIN]"
    }

    html_template = loader.get_template('base_user_management.html')

    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/maia/login/")
def register_group_view(request, group_id):
    if not request.user.is_superuser:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render({}, request))
    


    register_group_in_keycloak(group_id=group_id, settings=settings)
    emails = get_list_of_users_requesting_a_group(group_id=group_id, settings=settings)

    register_users_in_group_in_keycloak(group_id=group_id,emails=emails, settings=settings)

    argocd_url = settings.ARGOCD_SERVER

    user_table, to_register_in_groups, to_register_in_keycloak, maia_groups_dict, project_argo_status = get_project_argo_status_and_user_table(settings=settings, request=request)   

    user_list = user_table.to_dict('records')

    for user in user_list:
        for user in user_list:
            if user['email'] in to_register_in_keycloak:
                user['is_registered_in_keycloak'] = 0
            else:
                user['is_registered_in_keycloak'] = 1
            if user['email'] in to_register_in_groups:
                user['is_registered_in_groups'] = 0
            else:
                user['is_registered_in_groups'] = 1


    user_form = UserTableForm(users=user_list, projects=maia_groups_dict)

    context = {
        "user_table": user_list,
        "maia_groups_dict": maia_groups_dict,
        "minio_console_url": os.environ.get("MINIO_CONSOLE_URL",None),
        "form": user_form,
        "user": ["admin"],
        "project_argo_status": project_argo_status,
        "argocd_url": argocd_url,
        "username": request.user.username + " [ADMIN]"
    }

    html_template = loader.get_template('base_user_management.html')

    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/maia/login/")
def deploy_view(request, group_id):
    if not request.user.is_superuser:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render({}, request))

    id_token = request.session.get('oidc_id_token')


    argocd_cluster_id = settings.ARGOCD_CLUSTER
    argocd_url = settings.ARGOCD_SERVER


    cluster_config_path = os.environ["CLUSTER_CONFIG_PATH"]
    maia_config_file = os.environ["MAIA_CONFIG_PATH"]
    config_path=os.environ["CONFIG_PATH"]

    
    project_form_dict, cluster_id = get_project(group_id, settings=settings)
    

    if cluster_id is None:
        return redirect("/maia/user-management/")

    kubeconfig_dict = generate_kubeconfig(id_token, request.user.username, "default", argocd_cluster_id, settings=settings)
    local_kubeconfig_dict = generate_kubeconfig(id_token, request.user.username, "default", cluster_id, settings=settings)
    config.load_kube_config_from_dict(kubeconfig_dict)
    
    with open(Path("/tmp").joinpath("kubeconfig-project"), "w") as f:
        yaml.dump(kubeconfig_dict, f)
    with open(Path("/tmp").joinpath("kubeconfig-project-local"), "w") as f:
        yaml.dump(local_kubeconfig_dict, f)
        os.environ["KUBECONFIG"] = str(Path("/tmp").joinpath("kubeconfig-project"))
        os.environ["KUBECONFIG_LOCAL"] = str(Path("/tmp").joinpath("kubeconfig-project-local"))
    
    

        cluster_config_dict = yaml.safe_load(Path(cluster_config_path).joinpath(cluster_id+".yaml").read_text())
        maia_config_dict = yaml.safe_load(Path(maia_config_file).read_text())



        #Path(config_path).joinpath("forms").mkdir(parents=True, exist_ok=True)
        #with open(Path(config_path).joinpath("forms",f"{group_id}_form.yaml"), "w") as f:
        #    yaml.dump(project_form_dict, f)
        

        deploy_maia_toolkit_api(project_form_dict=project_form_dict, 
                                maia_config_dict=maia_config_dict,
                                cluster_config_dict=cluster_config_dict,
                                config_folder="/config", #config_path,
                                redeploy_enabled=True)
    

        namespace = project_form_dict["group_ID"].lower().replace("_", "-")

        create_namespace(request=request, cluster_id=cluster_id, namespace_id=namespace, settings=settings)
        
        return redirect(f"{argocd_url}/applications?proj={namespace}")


    