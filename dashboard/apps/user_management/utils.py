import os
import tempfile
import shutil
import sqlite3


from sqlalchemy import create_engine

import pandas as pd
from keycloak import KeycloakAdmin
from keycloak import KeycloakOpenIDConnection
from pathlib import Path
import yaml
import subprocess
from django.conf import settings
from kubernetes import config
from minio import Minio

def update_user_table(form):



    if settings.DEBUG:
         cnx = sqlite3.connect("db.sqlite3")
    else:
        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USERNAME"]
        dp_password = os.environ["DB_PASS"]

        #try:
        engine = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        cnx = engine.raw_connection()

    auth_user = pd.read_sql_query("SELECT * FROM auth_user", con=cnx)

    authentication_maiauser = pd.read_sql_query("SELECT * FROM authentication_maiauser", con=cnx)
    for k in form:
        if k.startswith("namespace"):
            id = auth_user[auth_user["username"] == k[len("namespace_"):]]["id"].values[0]

            if len(authentication_maiauser[authentication_maiauser["user_ptr_id"] == id]) > 0:
                authentication_maiauser.loc[authentication_maiauser["user_ptr_id"] == id, "namespace"] = form[k]
            else:

                authentication_maiauser = authentication_maiauser.append({"user_ptr_id": id, "namespace": form[k]},
                                                                         ignore_index=True)
        elif k.startswith("date"):
            id = auth_user[auth_user["username"] == k[len("date_"):]]["id"].values[0]

            if len(authentication_maiauser[authentication_maiauser["user_ptr_id"] == id ]) > 0:
                authentication_maiauser.loc[authentication_maiauser["user_ptr_id"] == id ,"date"] = form[k]
            else:

                authentication_maiauser = authentication_maiauser.append({"user_ptr_id": id, "date": form[k]}, ignore_index=True)
        elif k.startswith("conda"):
            print(f"conda: {form[k]}")
        elif k.startswith("cluster"):
            id = auth_user[auth_user["username"] == k[len("cluster_"):]]["id"].values[0]

            if len(authentication_maiauser[authentication_maiauser["user_ptr_id"] == id]) > 0:
                authentication_maiauser.loc[authentication_maiauser["user_ptr_id"] == id, "cluster"] = form[k]
            else:

                authentication_maiauser = authentication_maiauser.append({"user_ptr_id": id, "cluster": form[k]},
                                                                         ignore_index=True)
        elif k.startswith("gpu"):
            id = auth_user[auth_user["username"] == k[len("gpu_"):]]["id"].values[0]

            if len(authentication_maiauser[authentication_maiauser["user_ptr_id"] == id]) > 0:
                authentication_maiauser.loc[authentication_maiauser["user_ptr_id"] == id, "gpu"] = form[k]
            else:

                authentication_maiauser = authentication_maiauser.append({"user_ptr_id": id, "gpu": form[k]},
                                                                         ignore_index=True)
        elif k.startswith("minimal_environment"):
            id = auth_user[auth_user["username"] == k[len("minimal_environment_"):]]["id"].values[0]

            if len(authentication_maiauser[authentication_maiauser["user_ptr_id"] == id]) > 0:
                authentication_maiauser.loc[authentication_maiauser["user_ptr_id"] == id, "minimal_env"] = form[k]
            else:

                authentication_maiauser = authentication_maiauser.append({"user_ptr_id": id, "minimal_env": form[k]},
                                                                         ignore_index=True)

        elif k.startswith("memory_limit"):
            id = auth_user[auth_user["username"] == k[len("memory_limit_"):]]["id"].values[0]

            if len(authentication_maiauser[authentication_maiauser["user_ptr_id"] == id]) > 0:
                authentication_maiauser.loc[authentication_maiauser["user_ptr_id"] == id, "memory_limit"] = form[k]
            else:

                authentication_maiauser = authentication_maiauser.append({"user_ptr_id": id, "memory_limit": form[k]},
                                                                         ignore_index=True)
        elif k.startswith("cpu_limit"):
            id = auth_user[auth_user["username"] == k[len("cpu_limit_"):]]["id"].values[0]

            if len(authentication_maiauser[authentication_maiauser["user_ptr_id"] == id]) > 0:
                authentication_maiauser.loc[authentication_maiauser["user_ptr_id"] == id, "cpu_limit"] = form[k]
            else:

                authentication_maiauser = authentication_maiauser.append({"user_ptr_id": id, "cpu_limit": form[k]},
                                                                         ignore_index=True)
        elif k.startswith("project_admin"):
            project_admin = 0
            if form[k] == "on":
                project_admin = 1
            
            id = auth_user[auth_user["username"] == k[len("project_admin_"):]]["id"].values[0]

            if len(authentication_maiauser[authentication_maiauser["user_ptr_id"] == id]) > 0:
                authentication_maiauser.loc[authentication_maiauser["user_ptr_id"] == id, "project_admin"] = project_admin
            else:

                authentication_maiauser = authentication_maiauser.append({"user_ptr_id": id, "project_admin": project_admin},
                                                                         ignore_index=True)
    #try:
    cnx.close()
    

    if settings.DEBUG:
        cnx = sqlite3.connect("db.sqlite3")
        authentication_maiauser.to_sql("authentication_maiauser", con=cnx, if_exists="replace", index=False)

    else:
        engine.dispose()
        engine_2 = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        authentication_maiauser.to_sql("authentication_maiauser", con=engine_2, if_exists="replace", index=False)
        #stmt = text("ALTER TABLE authentication_maiauser-copy RENAME TO authentication_maiauser;")
        #engine.execute(stmt)

    #except:
    #    ...
#auth_user[auth_user["username"] == k[len("date_"):]]["date"] = form[k]



def generate_kubeconfig(id_token,user_id,namespace,cluster_id):
    cluster_apis = {k:v for v,k in settings.CLUSTER_NAMES.items()}

    if cluster_apis[cluster_id] in settings.PRIVATE_CLUSTERS:
        kube_config = {'apiVersion': 'v1', 'kind': 'Config', 'preferences': {},
                       'current-context': 'MAIA/{}'.format(user_id), 'contexts': [
                {'name': 'MAIA/{}'.format(user_id),
                 'context': {'user': user_id, 'cluster': 'MAIA', 'namespace': namespace}}],
                       'clusters': [
                           {'name': 'MAIA', 'cluster': {'certificate-authority-data': "",  # settings.CLUSTER_CA,
                                                        'server': cluster_apis[cluster_id],

                                                        "insecure-skip-tls-verify": True}}],
                       "users": [{'name': user_id,
                                  'user': {'token': settings.PRIVATE_CLUSTERS[cluster_apis[cluster_id]]}}]}

    else:

        kube_config = {'apiVersion': 'v1', 'kind': 'Config', 'preferences': {}, 'current-context': 'MAIA/{}'.format(user_id), 'contexts': [
            {'name': 'MAIA/{}'.format(user_id),
             'context': {'user': user_id, 'cluster': 'MAIA', 'namespace': namespace}}],
         'clusters': [{'name': 'MAIA', 'cluster': {'certificate-authority-data': "",#settings.CLUSTER_CA,
        'server' : cluster_apis[cluster_id],

        "insecure-skip-tls-verify":True}}],
        "users" : [{'name': user_id, 'user': {'auth-provider': {'config': {'idp-issuer-url': settings.OIDC_ISSUER_URL,
                                                                                     'client-id': settings.OIDC_RP_CLIENT_ID, 'id-token':id_token,
                                                                                     'client-secret': settings.OIDC_RP_CLIENT_SECRET
        }, 'name': 'oidc'}}}]}

    return kube_config

def create_deployment_package(cluster_config_path,config_path,group_id,user_list, id_token, user_id, cluster_id, output_folder,maia_config_path, gpu_request = None,memory_limits=["4G","8G"],cpu_limits=[2.0,4.0],namespace_for_custom_env=None,minimal=True):
    namespace_form = {
        "group_ID": group_id,
        "group_subdomain": group_id.lower().replace("_", "-"),
        "users": user_list,
        "resources_limits": {
            "memory": memory_limits,
            "cpu": cpu_limits
    }
    }

    if namespace_for_custom_env is not None:
        namespace_form["minio_env_name"] = f"{namespace_for_custom_env}_env"
    if gpu_request is not None:
        namespace_form["gpu_request"] = gpu_request

    Path(config_path).joinpath("forms").mkdir(parents=True, exist_ok=True)
    with open(Path(config_path).joinpath("forms",f"{group_id}_form.yaml"), "w") as f:
        yaml.dump(namespace_form, f)

    kubeconfig_dict = generate_kubeconfig(id_token, user_id, "default", cluster_id)
    config.load_kube_config_from_dict(kubeconfig_dict)


    with open(Path(config_path).joinpath("kubeconfig"), "w") as f:
        yaml.dump(kubeconfig_dict, f)
        os.environ["KUBECONFIG"] = str(Path(config_path).joinpath("kubeconfig"))

        cmd = [
            "MAIA_deploy_MAIA_namespace",
            "--namespace-config-file",
            Path(config_path).joinpath("forms", f"{group_id}_form.yaml"),
            "--maia-config-file",
            maia_config_path,
            "--cluster-config",
            cluster_config_path,
            # "--create-script",
            "--config-folder",
            config_path
        ]
        if minimal:
            cmd.append("--minimal")
        subprocess.Popen(cmd)



    with tempfile.TemporaryDirectory() as tmpdirname_kube:
        with open(Path(tmpdirname_kube).joinpath("kubeconfig"), "w") as f:
            yaml.dump(kubeconfig_dict, f)
            os.environ["KUBECONFIG"] = str(Path(tmpdirname_kube).joinpath("kubeconfig"))
            cmd = [
                "MAIA_deploy_MAIA_namespace",
                "--namespace-config-file",
                Path(config_path).joinpath("forms", f"{group_id}_form.yaml"),
                "--cluster-config",
                cluster_config_path,
                "--maia-config-file",
                maia_config_path,
                "--create-script",
                # "--minimal",
                "--config-folder",
                config_path
            ]
            if minimal:
                cmd.append("--minimal")
            subprocess.run(cmd)

        with open(Path(config_path).joinpath(group_id,f"{group_id}_namespace.sh"),"r") as f:
            script = f.read()
            script = script.replace(config_path,".")
            with open(Path(config_path).joinpath(f"{group_id}_namespace.sh"),"w") as f:
                f.write(script)

        template_files = [
            "admin.json",
            "readwrite.json",
            "tenant-base.yaml",
            "tenant-secret-configuration.yaml",
            "tenant-secret-user.yaml",

        ]
        template_dirs = ["pipelines"]

        with tempfile.TemporaryDirectory() as tmpdirname:
            print(tmpdirname)
            for template_file in template_files:
                shutil.copy(Path(config_path).joinpath(template_file),Path(tmpdirname).joinpath(template_file))
            for template_dir in template_dirs:
                shutil.copytree(Path(config_path).joinpath(template_dir),Path(tmpdirname).joinpath(template_dir))
            shutil.copytree(Path(config_path).joinpath(group_id),Path(tmpdirname).joinpath(group_id))
            shutil.copy(Path(config_path).joinpath(f"{group_id}_namespace.sh"),Path(tmpdirname).joinpath(f"{group_id}_namespace.sh"))

            shutil.make_archive(str(Path(output_folder).joinpath(group_id)), 'zip', tmpdirname)

    return str(Path(output_folder).joinpath(group_id+".zip"))

def get_user_table():

    if settings.DEBUG:
         cnx = sqlite3.connect("db.sqlite3")
    else:
        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USERNAME"]
        dp_password = os.environ["DB_PASS"]
        engine = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        cnx = engine.raw_connection()


    auth_user = pd.read_sql_query("SELECT * FROM auth_user", con=cnx)
    authentication_maiauser = pd.read_sql_query("SELECT * FROM authentication_maiauser", con=cnx)

    authentication_maiauser_copy = pd.read_sql_query("SELECT * FROM authentication_maiauser", con=cnx)

    authentication_maiauser_copy.rename(columns={"user_ptr_id": "id"}, inplace=True)

    table = auth_user.merge(authentication_maiauser_copy,on="id",how="left")

    keycloak_connection = KeycloakOpenIDConnection(
        server_url=settings.OIDC_SERVER_URL,
        username=settings.OIDC_USERNAME,
        password='',
        realm_name=settings.OIDC_REALM_NAME,
        client_id=settings.OIDC_RP_CLIENT_ID,
        client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
        verify=False)

    keycloak_admin = KeycloakAdmin(connection=keycloak_connection)
    users_to_register = {}

    groups = keycloak_admin.get_groups()


    maia_groups = {group['id']:group['name'][len("MAIA:"):] for group in groups if group['name'].startswith("MAIA:")}


    maia_group_dict = {}
    project_admin_users = [user[1]['username'] for user in table.iterrows() if str(user[1]['project_admin'])=="1" or str(user[1]['project_admin'])=="1.0"]


    try:
        client = Minio(settings.MINIO_URL,
                    access_key=settings.MINIO_ACCESS_KEY,
                    secret_key=settings.MINIO_SECRET_KEY,
                    secure=settings.MINIO_SECURE)


        minio_envs = [env.object_name[:-len("_env")] for env in list(client.list_objects(settings.BUCKET_NAME))]
    except:
        minio_envs = []

    for maia_group in maia_groups:
        if maia_groups[maia_group] == "users":
            continue
        users = keycloak_admin.get_group_members(group_id=maia_group)

        admin_users = [user['username'] for user in users if user['username'] in project_admin_users]
        
        if len(admin_users) == 0:
            if len(users) > 0:
                admin_users = [users[0]['username']]
            else:
                admin_users = []

        conda_envs = []
        if maia_groups[maia_group] in minio_envs:
            conda_envs.append(maia_groups[maia_group])
        else:
            conda_envs.append("N/A")
        group_users = []

        for user in users:
            if user['username'] in admin_users:
                group_users.append(user['username']+" [Project Admin]")
            else:
                group_users.append(user['username'])

        maia_group_dict[maia_groups[maia_group]] = {
            "users":group_users,
            "conda": conda_envs,
            "admin_users": admin_users
        }

    for user in auth_user.iterrows():
        uid = user[1]['id']
        username = user[1]['username']
        email = user[1]['email']
        user_groups = []
        for keycloack_user in keycloak_admin.get_users():

            if 'email' in keycloack_user:
                if keycloack_user['email'] == email:
                    user_keycloack_groups = keycloak_admin.get_user_groups(user_id=keycloack_user['id'])
                    for user_keycloack_group in user_keycloack_groups:
                        user_groups.append(user_keycloack_group['name'][len("MAIA:"):])

        if uid in authentication_maiauser['user_ptr_id'].values:
            requested_namespaces = authentication_maiauser[authentication_maiauser['user_ptr_id'] == uid][
                'namespace'].values[0].split(",")
            for requested_namespace in requested_namespaces:
                if requested_namespace not in user_groups:
                    if email not in users_to_register:
                        users_to_register[email] = [requested_namespace]
                    else:
                        users_to_register[email].append(requested_namespace)


    table.fillna("N/A", inplace=True)


    return table, users_to_register, maia_group_dict
