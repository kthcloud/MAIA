import os
from django.http import FileResponse
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.template import loader
from django.conf import settings

from pathlib import Path

from .forms import UserTableForm
from .utils import get_user_table, create_deployment_package, update_user_table
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Create your views here.
@login_required(login_url="/maia/login/")


def index(request):
    if not request.user.is_superuser:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render({}, request))
    if request.method == "POST":
        user_table, to_register, maia_groups_dict = get_user_table()

        user_list = user_table.to_dict('records')

        for user in user_list:
            if user['email'] in to_register:
                user['is_registered'] = 0
            else:
                user['is_registered'] = 1


        context = {
            "user_table": user_list,
            "clusters": settings.CLUSTER_NAMES.values(),
            "minio_console_url": os.environ.get("MINIO_CONSOLE_URL",None),
            "maia_groups_dict": maia_groups_dict,
            "form": UserTableForm(request.POST)
        }
        html_template = loader.get_template('base_user_management.html')

        form = UserTableForm(request.POST)

        if form.is_valid():
            update_user_table(request.POST)
        else:
            update_user_table(request.POST)

        return HttpResponse(html_template.render(context, request))


    user_table, to_register, maia_groups_dict = get_user_table()

    user_list = user_table.to_dict('records')

    for user in user_list:
        if user['email'] in to_register:
            user['is_registered'] = 0

        else:
            user['is_registered'] = 1


    form = UserTableForm(users=user_list)

    context = {
        "user_table": user_list,
        "maia_groups_dict": maia_groups_dict,
        "clusters": settings.CLUSTER_NAMES.values(),
        "minio_console_url": os.environ.get("MINIO_CONSOLE_URL",None),
        "form": form,
        "user": ["admin"],
        "username": request.user.username + " [ADMIN]"
    }

    html_template = loader.get_template('base_user_management.html')
    if not request.user.is_superuser:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render({}, request))


    return HttpResponse(html_template.render(context, request))



@login_required(login_url="/maia/login/")
def download_view(request, group_id):
    if not request.user.is_superuser:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render({}, request))

    cluster_config_path = os.environ["CLUSTER_CONFIG_PATH"]

    maia_config_path = os.environ["MAIA_CONFIG_PATH"]

    user_table, to_register, maia_groups_dict = get_user_table()

    admin = maia_groups_dict[group_id]['admin_users'][0]

    users = maia_groups_dict[group_id]['users']

    for user in users:
        if user.endswith("[Project Admin]"):
            i = users.index(user)
            users[i]=user[:-len(" [Project Admin]")]


    for user in user_table.iterrows():
        if user[1]['username'] == admin:
            cluster = user[1]['cluster']
            custom_env = user[1]['conda']
            if custom_env == "N/A":
                custom_env = None
            else:
                custom_env = custom_env[:-len("_env")]
            gpu_request = user[1]['gpu']
            if gpu_request == "NO" or gpu_request == "N/A":
                gpu_request = None
            memory_limit = int(user[1]['memory_limit'][:-len(" Gi")])
            memory_limits = [str(int(memory_limit/2))+"G",str(memory_limit)+"G"]
            cpu_limit = int(user[1]['cpu_limit'])
            env_type = user[1]["minimal_env"]
            if env_type == "Minimal":
                is_minimal = True
            else:
                is_minimal = False
            cpu_limits = [float(cpu_limit / 2), float(cpu_limit)]


    file = create_deployment_package(cluster_config_path=Path(cluster_config_path).joinpath(f"{cluster}.yaml"),

                          config_path=os.environ["CONFIG_PATH"],
                          group_id=group_id,
                          user_list=users,
                          id_token=request.session.get('oidc_id_token'),
                          user_id=request.user.email,
                          cluster_id=cluster,
                          output_folder=os.environ["CONFIG_PATH"],
                          gpu_request=gpu_request,
                          memory_limits=memory_limits,
                          cpu_limits=cpu_limits,
                          namespace_for_custom_env=custom_env,
                          minimal = is_minimal,
                          maia_config_path=maia_config_path
                          )

    path_to_file = os.path.realpath(file)
    response = FileResponse(open(path_to_file, 'rb'),as_attachment=True,)
    file_name = Path(file).name
    response['Content-Disposition'] = 'inline; filename=' + file_name
    return response
