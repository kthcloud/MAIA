import os
import smtplib
import ssl
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests
import json
import numpy as np
import requests
import json
import os
from kubernetes.client.rest import ApiException
import asyncio
import sqlite3
import kubernetes
from sqlalchemy import create_engine
import pandas as pd
from keycloak import KeycloakAdmin
from keycloak import KeycloakOpenIDConnection
from pathlib import Path
import yaml
from kubernetes import config
from minio import Minio
from MAIA_scripts.MAIA_install_project_toolkit import verify_installed_maia_toolkit

def verify_minio_availability(settings):
    """
    Verifies the availability of a MinIO server.

    Parameters
    ----------
    settings : object
        An object containing the MinIO configuration settings.
        - settings.MINIO_URL : str
            The URL of the MinIO server.
        - settings.MINIO_ACCESS_KEY : str
            The access key for the MinIO server.
        - settings.MINIO_SECRET_KEY : str
            The secret key for the MinIO server.
        - settings.BUCKET_NAME : str
            The name of the bucket to check for existence.

    Returns
    -------
    bool
        True if the MinIO server is available and the bucket exists, False otherwise.
    """
    try:
        client = Minio(settings.MINIO_URL,
                            access_key=settings.MINIO_ACCESS_KEY,
                            secret_key=settings.MINIO_SECRET_KEY,
                            secure=True)
        client.bucket_exists(settings.BUCKET_NAME)
        minio_available = True
    except Exception as e:
        print(e)
        minio_available = False

    return minio_available


def send_approved_registration_email(receiver_email, login_url, temp_password):
    """
    Sends an email to notify the user that their MAIA account registration has been approved.

    Parameters
    ----------
    receiver_email : str
        The email address of the recipient.
    login_url : str
        The URL where the user can log in to MAIA.
    temp_password : str
        The temporary password assigned to the user.

    Raises
    ------
    KeyError
        If the environment variables 'email_account' or 'email_password' are not set.
    smtplib.SMTPException
        If there is an error sending the email.
    """
    
    sender_email = os.environ["email_account"]
    message = MIMEMultipart()
    message["Subject"] = "Account Registration Approved"
    message["From"] = "MAIA Registration"
    message["To"] = receiver_email

    html = """\
    <html>
        <head></head>
        <body>
            <p>Your MAIA Account has been approved.</p>
            <p>Log in to MAIA at the following link: <a href="{}">MAIA</a></p>
            <p>Your temporary password is: {}</p>
            <p>Please change your password after logging in.</p>
            <br>
            <p>Best Regards,</p>
            <p>MAIA Admin Team</p>
        </body>
    </html>
    """.format(login_url, temp_password)

    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(html, "html")

    message.attach(part1)

    port = 465  # For SSL
    password = os.environ["email_password"]

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(os.environ["email_smtp_server"], port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())


def send_discord_message(username, namespace, url):
    """
    Sends a message to a Discord webhook to request a MAIA account.

    Parameters
    ----------
    username : str
        The username of the person requesting the account.
    namespace : str
        The project namespace for which the account is being requested.
    url : str
        The Discord webhook URL to which the message will be sent.

    Raises
    ------
    requests.exceptions.HTTPError
        If the HTTP request returned an unsuccessful status code.

    Prints
    ------
    str
        Success message with the HTTP status code if the payload is delivered successfully.
    str
        Error message if the HTTP request fails.
    """
    data = {
        "content": f"{username} is requesting a MAIA account for the project {namespace}.",
        "username": "MAIA-Bot"
    }

    data["embeds"] = [
        {
            "description": "MAIA User Registration Request",
            "title": "MAIA Account Request",
        }
    ]


    result = requests.post(url, json=data)

    try:
        result.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    else:
        print("Payload delivered successfully, code {}.".format(result.status_code))


def get_namespaces(id_token, api_urls, private_clusters = []):
    """
    Retrieves a list of unique namespaces from multiple API URLs.

    Parameters
    ----------
    id_token : str
        The ID token used for authorization when accessing public clusters.
    api_urls : list
        A list of API URLs to query for namespaces.
    private_clusters : dict, optional
        A dictionary where keys are API URLs of private clusters and values are their respective tokens. Defaults to an empty list.

    Returns
    -------
    list
        A list of unique namespace names retrieved from the provided API URLs.
    """
    namespace_list = []
    for API_URL in api_urls:
        if API_URL in private_clusters:
            token = private_clusters[API_URL]
def get_cluster_status(id_token, api_urls, cluster_names, private_clusters = []):
    """
    Retrieve the status of clusters and their nodes.

    Parameters
    ----------
    id_token : str
        The ID token for authentication.
    api_urls : list
        A list of API URLs for the clusters.
    cluster_names : dict
        A dictionary mapping API URLs to cluster names.
    private_clusters : dict, optional
        A dictionary mapping private cluster API URLs to their tokens. Defaults to [].

    Returns
    -------
    tuple
        A tuple containing:
            - node_status_dict (dict): A dictionary mapping node names to their status and schedulability.
            - cluster_dict (dict): A dictionary mapping cluster names to their node names.
    """
    cluster_dict = {}
    node_status_dict = {}
    for API_URL in api_urls:


        if API_URL in private_clusters:
            token = private_clusters[API_URL]
            try:
                response = requests.get(API_URL + "/api/v1/nodes",
                                    headers={"Authorization": "Bearer {}".format(token)}, verify=False)
            except:
                
                cluster = cluster_names[API_URL]
                cluster_dict[cluster] = ['Cluster API Not Reachable']
                node_status_dict['Cluster API Not Reachable'] = ['API']
                continue
        else:
            if API_URL.endswith("None"):
                cluster = cluster_names[API_URL]
                cluster_dict[cluster] = ['Cluster API Not Reachable']
                node_status_dict['Cluster API Not Reachable'] = ['API']
                continue
            else:
                try:
                    response = requests.get(API_URL + "/api/v1/nodes",
                                headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
                except:
                    cluster = cluster_names[API_URL]
                    cluster_dict[cluster] = ['Cluster API Not Reachable']
                    node_status_dict['Cluster API Not Reachable'] = ['API']
                    continue
        nodes = json.loads(response.text)


        if 'items' not in nodes:
            cluster = cluster_names[API_URL]
            cluster_dict[cluster] = ['Cluster API Not Reachable']
            node_status_dict['Cluster API Not Reachable'] = ['API']
            continue
        for node in nodes['items']:
            node_name = node['metadata']['name']
            node_status_dict[node_name] = []
            cluster = cluster_names[API_URL]
            if cluster in cluster_dict:
                cluster_dict[cluster].append(node_name)
            else:
                cluster_dict[cluster] = [node_name]
            for condition in node['status']['conditions']:
                if condition["type"] == "Ready":
                    node_status_dict[node_name].append(condition["status"])

            if 'unschedulable' in node['spec']:
                node_status_dict[node_name].append(node['spec']['unschedulable'])
            else:
                node_status_dict[node_name].append(False)
    return node_status_dict, cluster_dict


def get_available_resources(id_token, api_urls, cluster_names, private_clusters = []):
    """
    Retrieves available GPU, CPU, and RAM resources from multiple Kubernetes clusters.

    Parameters
    ----------
    id_token : str
        The ID token for authentication.
    api_urls : list
        List of API URLs for the Kubernetes clusters.
    cluster_names : dict
        Dictionary mapping API URLs to cluster names.
    private_clusters : list, optional
        List of private clusters with their tokens. Defaults to [].

    Returns
    -------
    tuple
        A tuple containing:
            - gpu_dict (dict): Dictionary with GPU availability information for each node.
            - cpu_dict (dict): Dictionary with CPU availability information for each node.
            - ram_dict (dict): Dictionary with RAM availability information for each node.
            - gpu_allocations (dict): Dictionary with GPU allocation details for each pod.
    """

    gpu_dict = {}
    cpu_dict = {}
    ram_dict = {}
    gpu_allocations = {}


    for API_URL in api_urls:
        cluster_name = cluster_names[API_URL]
        if API_URL in private_clusters:
            token = private_clusters[API_URL]
            try:
                response = requests.get(API_URL + "/api/v1/pods",
                                        headers={"Authorization": "Bearer {}".format(token)}, verify=False)
                pods = json.loads(response.text)
                response = requests.get(API_URL + "/api/v1/nodes",
                                        headers={"Authorization": "Bearer {}".format(token)}, verify=False)

                nodes = json.loads(response.text)

            except:
                continue

        else:
            try:
                response = requests.get(API_URL + "/api/v1/pods",
                                        headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
                pods = json.loads(response.text)
                response = requests.get(API_URL + "/api/v1/nodes",
                                        headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)

                nodes = json.loads(response.text)
            except:
                continue




        node_status_dict = {}

        for node in nodes['items']:

            node_name = "{}/{}".format(cluster_name,node['metadata']['name'])

            node_status_dict[node_name] = []

            for condition in node['status']['conditions']:
                if condition["type"] == "Ready":
                    node_status_dict[node_name].append(condition["status"])

            if 'unschedulable' in node['spec']:
                node_status_dict[node_name].append(node['spec']['unschedulable'])
            else:
                node_status_dict[node_name].append(False)


        for node in nodes['items']:

            node_name = "{}/{}".format(cluster_name, node['metadata']['name'])

            if 'nvidia.com/gpu.product' in node['metadata']['labels']:
                gpu_name = node['metadata']['labels']['nvidia.com/gpu.product']
            else:
                gpu_name = "N/A"

            if 'nvidia.com/gpu.memory' in node['metadata']['labels']:
                gpu_size = str(round(int(node['metadata']['labels']['nvidia.com/gpu.memory'])/1024))+" Gi"
            else:
                gpu_size = "N/A"

            if 'nvidia.com/gpu' in node['status']['allocatable']:
                n_gpu_allocatable = int(node['status']['allocatable']['nvidia.com/gpu'])
            else:
                n_gpu_allocatable = 0
            n_cpu_allocatable = int(node['status']['allocatable']['cpu'])
            ram_allocatable = float(int(node['status']['allocatable']['memory'][:-2])/(1024.*1024.))

            n_gpu_requested = 0
            n_cpus_requested = 0
            ram_requested = 0
            for pod in pods['items']:
                if 'nodeName' not in pod['spec']:
                    continue
                if pod['spec']['nodeName'] != node_name.split("/")[1]:
                    continue
                containers = pod['spec']['containers']
                for container in containers:
                    resources = container['resources']

                    cpu = 0
                    ram = 0
                    if 'requests' in resources:
                        req = resources['requests']
                        if 'nvidia.com/gpu' in req:

                            pod_name = pod['metadata']['name']
                            if pod_name.startswith("jupyter"):
                                pod_name = pod_name.replace("-2d", "-").replace("-40", "@").replace("-2e", ".")[len("jupyter-"):]
                            gpu_allocations[pod_name] = {
                                'node': node_name.split("/")[1],
                                'cluster': cluster_name,
                                'namespace': pod['metadata']['namespace'],
                                'gpu': req['nvidia.com/gpu'],
                                'gpu_name': gpu_name,
                                'gpu_size': gpu_size
                            }

                            n_gpu_requested += int(req['nvidia.com/gpu'])
                        if 'cpu' in req:
                            container_cpu = req['cpu']
                            if container_cpu[-1] == 'm':
                                container_cpu = container_cpu[:-1]
                                container_cpu = int(container_cpu) / 1000.
                            else:
                                container_cpu = int(container_cpu)
                            if container_cpu > cpu:
                                cpu = container_cpu
                        if 'memory' in req:
                            if req['memory'][-2:] == "Mi":
                                container_memory = int(req['memory'][:-2])/1024.
                            elif req['memory'][-2:] == "Gi":
                                container_memory = int(req['memory'][:-2])

                            elif req['memory'][-1:] == "M":
                                container_memory = int(req['memory'][:-1])/1024.
                            else:

                                container_memory = int(req['memory'])/(1024*1024*1024)
                            if container_memory > ram:
                                ram = container_memory

                    #if 'limits' in resources:
                    #    lim = resources['limits']
                    #    if 'cpu' in lim:
                    #        container_cpu = lim['cpu']
                    #        if container_cpu[-1] == 'm':
                    #            container_cpu = container_cpu[:-1]
                    #            container_cpu = int(container_cpu)/1000.
                    #        else:
                    #            container_cpu = int(container_cpu)
                    #        if container_cpu > cpu:
                    #            cpu = container_cpu
                    #    if 'memory' in lim:
                    #        if lim['memory'][-2:] == "Mi":
                    #            container_memory = int(lim['memory'][:-2]) / 1024.
                    #        elif lim['memory'][-2:] == "Gi":
                    #            container_memory = int(lim['memory'][:-2])
                    #        if container_memory > ram:
                    #            ram = container_memory
                    n_cpus_requested += cpu
                    ram_requested += ram

            gpu_dict[node_name] = []

            if node_status_dict[node_name][0] != "True":
                gpu_dict[node_name].append("NA")
                gpu_dict[node_name].append("NA")
            elif node_status_dict[node_name][1]:
                gpu_dict[node_name].append("NA")
                gpu_dict[node_name].append("NA")
            else:
                gpu_dict[node_name].append(n_gpu_allocatable-n_gpu_requested)
                gpu_dict[node_name].append(n_gpu_allocatable)
            gpu_dict[node_name].append("{}, {}".format(gpu_name, gpu_size))

            cpu_dict[node_name] = []
            cpu_dict[node_name].append(n_cpu_allocatable- n_cpus_requested)
            cpu_dict[node_name].append(n_cpu_allocatable)
            cpu_dict[node_name].append((n_cpu_allocatable- n_cpus_requested)*100/n_cpu_allocatable)

            ram_dict[node_name] = []
            ram_dict[node_name].append(ram_allocatable - ram_requested)
            ram_dict[node_name].append(ram_allocatable)
            ram_dict[node_name].append((ram_allocatable - ram_requested) * 100 / ram_allocatable)
    return gpu_dict, cpu_dict, ram_dict, gpu_allocations


def get_filtered_available_nodes(gpu_dict, cpu_dict, ram_dict, gpu_request, cpu_request, memory_request):
    """
    Filters and returns nodes that meet the specified GPU, CPU, and memory requirements.

    Parameters
    ----------
    gpu_dict : dict
        A dictionary where keys are node names and values are lists containing GPU information.
    cpu_dict : dict
        A dictionary where keys are node names and values are lists containing CPU information.
    ram_dict : dict
        A dictionary where keys are node names and values are lists containing RAM information.
    gpu_request : int
        The minimum number of GPUs required.
    cpu_request : float
        The minimum amount of CPU required.
    memory_request : float
        The minimum amount of memory required.

    Returns
    -------
    tuple
        Three dictionaries containing the filtered nodes and their respective GPU, CPU, and RAM information.
    """

    filtered_nodes = []
    for node in gpu_dict:
        if int(gpu_dict[node][0]) >= gpu_request and float(cpu_dict[node][0]) >= cpu_request and float(ram_dict[node][0]) >= memory_request:
            filtered_nodes.append(node)

    return {node: gpu_dict[node] for node in filtered_nodes},{node: cpu_dict[node] for node in filtered_nodes},{node: ram_dict[node] for node in filtered_nodes}

def get_groups_in_keycloak(settings):
    """
    Retrieve groups from Keycloak that start with "MAIA:" and return them in a dictionary.

    Parameters
    ----------
    settings : object
        An object containing the Keycloak connection settings. 
        It should have the following attributes:
        - OIDC_SERVER_URL : str
            The URL of the Keycloak server.
        - OIDC_USERNAME : str
            The username for Keycloak authentication.
        - OIDC_REALM_NAME : str
            The name of the Keycloak realm.
        - OIDC_RP_CLIENT_ID : str
            The client ID for Keycloak.
        - OIDC_RP_CLIENT_SECRET : str
            The client secret for Keycloak.

    Returns
    -------
    dict
        A dictionary where the keys are group IDs and the values are group names 
        (with the "MAIA:" prefix removed) for groups that start with "MAIA:".
    """
    keycloak_connection = KeycloakOpenIDConnection(
    server_url=settings.OIDC_SERVER_URL,
    username=settings.OIDC_USERNAME,
    password='',
    realm_name=settings.OIDC_REALM_NAME,
    client_id=settings.OIDC_RP_CLIENT_ID,
    client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
    verify=False)

    keycloak_admin = KeycloakAdmin(connection=keycloak_connection)


    groups = keycloak_admin.get_groups()

    maia_groups = {group['id']:group['name'][len("MAIA:"):] for group in groups if group['name'].startswith("MAIA:")}

    return maia_groups


def get_pending_projects(settings):
    """
    Retrieve a list of pending projects that are not in active groups.

    Parameters
    ----------
    settings : object
        A settings object that contains configuration parameters.

    Returns
    -------
    list
        A list of namespaces of pending projects.
    """
    if settings.DEBUG:
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH,"db.sqlite3"))
    else:
        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USERNAME"]
        dp_password = os.environ["DB_PASS"]

        engine = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        cnx = engine.raw_connection()

    pending_projects = []
    try:
        authentication_maiaproject = pd.read_sql_query("SELECT * FROM authentication_maiaproject", con=cnx)

        active_groups = get_groups_in_keycloak(settings)

        for project in authentication_maiaproject.iterrows():
            if project[1]['namespace'] not in active_groups.values():
                pending_projects.append(project[1]['namespace'])
        
    except:
        ...
    
    cnx.close()
    return pending_projects



def get_user_table(settings):
    """
    Retrieves and processes user data from various sources including a local SQLite database or a remote MySQL database,
    Keycloak, and Minio. Combines and returns user information along with group and project details.

    Parameters
    ----------
    settings : object
        A settings object containing configuration parameters such as database paths, 
        Keycloak server details, and Minio credentials.

    Returns
    -------
    tuple
        A tuple containing:
        - table (pd.DataFrame): A DataFrame containing merged user data from the auth_user and authentication_maiauser tables.
        - users_to_register_in_group (dict): A dictionary mapping user emails to the groups they need to be registered in.
        - users_to_register_in_keycloak (list): A list of user emails that need to be registered in Keycloak.
        - maia_group_dict (dict): A dictionary containing detailed information about each MAIA group and pending projects.
    """

    if settings.DEBUG:
         cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH,"db.sqlite3"))
    else:
        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USERNAME"]
        dp_password = os.environ["DB_PASS"]
        engine = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        cnx = engine.raw_connection()


    auth_user = pd.read_sql_query("SELECT * FROM auth_user", con=cnx)
    authentication_maiauser = pd.read_sql_query("SELECT * FROM authentication_maiauser", con=cnx)

    authentication_maiaproject = pd.read_sql_query("SELECT * FROM authentication_maiaproject", con=cnx)

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
        verify=False
    )

    keycloak_admin = KeycloakAdmin(connection=keycloak_connection)
    users_to_register_in_group = {}

    groups = keycloak_admin.get_groups()


    maia_groups = {group['id']:group['name'][len("MAIA:"):] for group in groups if group['name'].startswith("MAIA:")}

    pending_projects = get_pending_projects(settings=settings)

    maia_group_dict = {}


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

        admin_users = []
        cpu_limit = None
        memory_limit = None
        date = None
        cluster = None
        gpu = None
        environment = None
        for project in authentication_maiaproject.iterrows():
            if project[1]['namespace'] == maia_groups[maia_group]:
                admin_users = [project[1]['email']]
                cpu_limit = project[1]['cpu_limit']
                memory_limit = project[1]['memory_limit']
                date = project[1]['date']
                cluster = project[1]['cluster']
                gpu = project[1]['gpu']
                environment = project[1]['minimal_env']



        conda_envs = []
        if maia_groups[maia_group] in minio_envs:
            conda_envs.append(maia_groups[maia_group])
        else:
            conda_envs.append("N/A")
        
        
        group_users = []
        for user in users:
            if user['username'] in admin_users:
                group_users.append(user['email']+" [Project Admin]")
            else:
                group_users.append(user['email'])

        maia_group_dict[maia_groups[maia_group]] = {
            "users":group_users,
            "conda": conda_envs,
            "admin_users": admin_users,
            "cpu_limit": cpu_limit,
            "memory_limit": memory_limit,
            "date": date,
            "cluster": cluster,
            "gpu": gpu,
            "environment": environment
        }

    for pending_project in pending_projects:
        conda_envs = []
        if pending_project in minio_envs:
            conda_envs.append(pending_project)
        else:
            conda_envs.append("N/A")

        users = []
        for user in auth_user.iterrows():
            uid = user[1]['id']
            if uid in authentication_maiauser['user_ptr_id'].values:
                requested_namespaces = authentication_maiauser[authentication_maiauser['user_ptr_id'] == uid][
                    'namespace'].values[0].split(",")
                if pending_project in requested_namespaces:
                    users.append(user[1]['email'])

        maia_group_dict[pending_project] = {
            "users": users,
            "pending": True,
            "conda": conda_envs,
            "admin_users": [],
            "cpu_limit": authentication_maiaproject[authentication_maiaproject['namespace'] == pending_project]['cpu_limit'].values[0],
            "memory_limit": authentication_maiaproject[authentication_maiaproject['namespace'] == pending_project]['memory_limit'].values[0],
            "date": authentication_maiaproject[authentication_maiaproject['namespace'] == pending_project]['date'].values[0],
            "cluster": "N/A",
            "gpu": authentication_maiaproject[authentication_maiaproject['namespace'] == pending_project]['gpu'].values[0],
            "environment": "Minimal" 
            
        }

    users_to_register_in_keycloak = []
    for user in auth_user.iterrows():
        uid = user[1]['id']
        username = user[1]['username']
        email = user[1]['email']
        user_groups = []
        user_in_keycloak = False
        for keycloak_user in keycloak_admin.get_users():

            if 'email' in keycloak_user:
                if keycloak_user['email'] == email:
                    user_in_keycloak = True
                    user_keycloak_groups = keycloak_admin.get_user_groups(user_id=keycloak_user['id'])
                    for user_keycloak_group in user_keycloak_groups:
                        if user_keycloak_group['name'].startswith("MAIA:"):
                            user_groups.append(user_keycloak_group['name'][len("MAIA:"):])
        if not user_in_keycloak:
            users_to_register_in_keycloak.append(email)
        if uid in authentication_maiauser['user_ptr_id'].values:
            requested_namespaces = authentication_maiauser[authentication_maiauser['user_ptr_id'] == uid][
                'namespace'].values[0].split(",")
            for requested_namespace in requested_namespaces:
                if requested_namespace not in user_groups and requested_namespace != "N/A":
                    if email not in users_to_register_in_group:
                        users_to_register_in_group[email] = [requested_namespace]
                    else:
                        users_to_register_in_group[email].append(requested_namespace)


    table.fillna("N/A", inplace=True)

    return table, users_to_register_in_group, users_to_register_in_keycloak, maia_group_dict



def update_user_table(form, settings):
    """
    Updates the user and project tables based on the provided form data.

    Parameters
    ----------
    form : dict
        A dictionary containing form data with keys indicating the type of data (e.g., "namespace", "memory_limit", etc.) and values being the corresponding data.
    settings : object
        An object containing configuration settings. It should have a DEBUG attribute to determine the environment and a LOCAL_DB_PATH attribute for the local database path.

    Returns
    -------
    None

    The function performs the following steps:
    1. Connects to the appropriate database (SQLite for debug mode, MySQL for production).
    2. Reads the current data from the `auth_user`, `authentication_maiauser`, and `authentication_maiaproject` tables.
    3. Iterates over the form data and updates the `authentication_maiauser` and `authentication_maiaproject` tables accordingly.
    4. Closes the database connection.
    5. Writes the updated data back to the database.
    """

    if settings.DEBUG:
         cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH,"db.sqlite3"))
    else:
        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USERNAME"]
        dp_password = os.environ["DB_PASS"]

        #try:
        engine = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        cnx = engine.raw_connection()

    auth_user = pd.read_sql_query("SELECT * FROM auth_user", con=cnx)

    authentication_maiauser = pd.read_sql_query("SELECT * FROM authentication_maiauser", con=cnx)

    authentication_maiaproject = pd.read_sql_query("SELECT * FROM authentication_maiaproject", con=cnx)

    for k in form:
        if k.startswith("namespace"):
            id = auth_user[auth_user["username"] == k[len("namespace_"):]]["id"].values[0]

            if len(authentication_maiauser[authentication_maiauser["user_ptr_id"] == id]) > 0:
                authentication_maiauser.loc[authentication_maiauser["user_ptr_id"] == id, "namespace"] = form[k]
            else:

                authentication_maiauser = authentication_maiauser.append({"user_ptr_id": id, "namespace": form[k]},
                                                                         ignore_index=True)
        elif k.startswith("memory_limit"):
            

            try:
                id = authentication_maiaproject[authentication_maiaproject["namespace"] == k[len("memory_limit_"):]]["id"].values[0]
            except:
                id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1

            if len(authentication_maiaproject[authentication_maiaproject["id"] == id]) > 0:
                authentication_maiaproject.loc[authentication_maiaproject["id"] == id, "memory_limit"] = form[k]
            else:
                
                authentication_maiaproject = authentication_maiaproject.append({"id": id, "memory_limit": form[k],"namespace": k[len("memory_limit_"):]},
                                                                            ignore_index=True)
        elif k.startswith("cpu_limit"):
            try:
                id = authentication_maiaproject[authentication_maiaproject["namespace"] == k[len("cpu_limit_"):]]["id"].values[0]
            except:
                id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1

            if len(authentication_maiaproject[authentication_maiaproject["id"] == id]) > 0:
                authentication_maiaproject.loc[authentication_maiaproject["id"] == id, "cpu_limit"] = form[k]
            else:
                authentication_maiaproject = authentication_maiaproject.append({"id": id, "cpu_limit": form[k],"namespace": k[len("cpu_limit_"):]},
                                                                            ignore_index=True)
        elif k.startswith("date"):
            try:
                id = authentication_maiaproject[authentication_maiaproject["namespace"] == k[len("date_"):]]["id"].values[0]
            except:
                id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1

            if len(authentication_maiaproject[authentication_maiaproject["id"] == id]) > 0:
                authentication_maiaproject.loc[authentication_maiaproject["id"] == id, "date"] = form[k]
            else:
                authentication_maiaproject = authentication_maiaproject.append({"id": id, "date": form[k],"namespace": k[len("date_"):]},
                                                                            ignore_index=True)
        elif k.startswith("cluster"):
            try:
                id = authentication_maiaproject[authentication_maiaproject["namespace"] == k[len("cluster_"):]]["id"].values[0]
            except:
                id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1

            if len(authentication_maiaproject[authentication_maiaproject["id"] == id]) > 0:
                authentication_maiaproject.loc[authentication_maiaproject["id"] == id, "cluster"] = form[k]
            else:
                authentication_maiaproject = authentication_maiaproject.append({"id": id, "cluster": form[k],"namespace": k[len("cluster_"):]},
                                                                            ignore_index=True)
        elif k.startswith("gpu"):
            try:
                id = authentication_maiaproject[authentication_maiaproject["namespace"] == k[len("gpu_"):]]["id"].values[0]
            except:
                id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1
                
            

            if len(authentication_maiaproject[authentication_maiaproject["id"] == id]) > 0:
                authentication_maiaproject.loc[authentication_maiaproject["id"] == id, "gpu"] = form[k]
            else:
                authentication_maiaproject = authentication_maiaproject.append({"id": id, "gpu": form[k],"namespace": k[len("gpu_"):]},
                                                                            ignore_index=True)
        elif k.startswith("minimal_environment"):
            try:
                id = authentication_maiaproject[authentication_maiaproject["namespace"] == k[len("minimal_environment_"):]]["id"].values[0]
            except:
                id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1

            if len(authentication_maiaproject[authentication_maiaproject["id"] == id]) > 0:
                authentication_maiaproject.loc[authentication_maiaproject["id"] == id, "minimal_env"] = form[k]
            else:
                authentication_maiaproject = authentication_maiaproject.append({"id": id, "minimal_env": form[k],"namespace": k[len("minimal_environment_"):]},
                                                                            ignore_index=True)
            
      
    #try:
    cnx.close()
    

    if settings.DEBUG:
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH,"db.sqlite3"))
        authentication_maiauser.to_sql("authentication_maiauser", con=cnx, if_exists="replace", index=False)
        authentication_maiaproject.to_sql("authentication_maiaproject", con=cnx, if_exists="replace", index=False)

    else:
        engine.dispose()
        engine_2 = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        authentication_maiauser.to_sql("authentication_maiauser", con=engine_2, if_exists="replace", index=False)
        authentication_maiaproject.to_sql("authentication_maiaproject", con=engine_2, if_exists="replace", index=False)
        #stmt = text("ALTER TABLE authentication_maiauser-copy RENAME TO authentication_maiauser;")
        #engine.execute(stmt)

    #except:
    #    ...
#auth_user[auth_user["username"] == k[len("date_"):]]["date"] = form[k]




def generate_kubeconfig(id_token, user_id, namespace, cluster_id, settings):
    """
    Generates a Kubernetes configuration dictionary for a given user and cluster.

    Parameters
    ----------
    id_token : str
        The ID token for the user.
    user_id : str
        The user ID.
    namespace : str
        The Kubernetes namespace.
    cluster_id : str
        The cluster ID.
    settings : object
        An object containing various settings, including:
        - CLUSTER_NAMES (dict): A dictionary mapping cluster names to their IDs.
        - PRIVATE_CLUSTERS (dict): A dictionary of private clusters with their tokens.
        - OIDC_ISSUER_URL (str): The OIDC issuer URL.
        - OIDC_RP_CLIENT_ID (str): The OIDC client ID.
        - OIDC_RP_CLIENT_SECRET (str): The OIDC client secret.

    Returns
    -------
    dict
        A dictionary representing the Kubernetes configuration.
    """
    cluster_apis = {k: v for v, k in settings.CLUSTER_NAMES.items()}

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

def get_project(group_id, settings, is_namespace_style=False):
    """
    Retrieve project information based on the provided group ID.

    Parameters
    ----------
    group_id : str
        The group ID or namespace to search for.
    settings : module
        A settings module containing configuration values.
    is_namespace_style : bool, optional
        Flag to determine if the group ID should be treated as a namespace. Defaults to False.

    Returns
    -------
    tuple
        A tuple containing the namespace form (dict) and the cluster ID (str or None). 
        Returns None if no matching project is found.

    Raises
    ------
    KeyError
        If required environment variables are not set.
    Exception
        If there is an error connecting to the database or querying the data.
    """

    if settings.DEBUG:
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH,"db.sqlite3"))
    else:
        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USERNAME"]
        dp_password = os.environ["DB_PASS"]

        #try:
        engine = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        cnx = engine.raw_connection()

    authentication_maiaproject = pd.read_sql_query("SELECT * FROM authentication_maiaproject", con=cnx)

    cluster_id = None

    for project in authentication_maiaproject.iterrows():
        if is_namespace_style:
            if str(project[1]['namespace']).lower().replace("_","-") == group_id:
                keycloak_connection = KeycloakOpenIDConnection(
                    server_url=settings.OIDC_SERVER_URL,
                    username=settings.OIDC_USERNAME,
                    password='',
                    realm_name=settings.OIDC_REALM_NAME,
                    client_id=settings.OIDC_RP_CLIENT_ID,
                    client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
                    verify=False
                )

                keycloak_admin = KeycloakAdmin(connection=keycloak_connection)
                groups = keycloak_admin.get_groups()


                maia_groups = {group['id']:group['name'][len("MAIA:"):] for group in groups if group['name'].startswith("MAIA:")}

                group_users = []

                for maia_group in maia_groups:
                    if maia_groups[maia_group] == group_id:
                        users = keycloak_admin.get_group_members(group_id=maia_group)
                        
                        for user in users:
                            group_users.append(user['email'])


                namespace_form = {
                "group_ID": group_id,
                "group_subdomain": group_id.lower().replace("_", "-"),
                "users": group_users,
                "resources_limits": {
                    "memory": [str(int(int(project[1]['memory_limit'][:-len(" Gi")])/2))+" Gi", project[1]['memory_limit']],
                    "cpu": [str(int(int(project[1]['cpu_limit'])/2)), project[1]['cpu_limit']],
                }
                }
                if project[1]['gpu'] != "N/A" and project[1]['gpu'] != "NO":
                    namespace_form["gpu"] = "1"

                if project[1]['conda'] != "N/A" and project[1]['conda'] is not None:
                    namespace_form["minio_env_name"] = group_id+"_env"
                
                cluster_id = project[1]['cluster']
                if cluster_id == "N/A":
                    cluster_id = None
                return namespace_form, cluster_id
        else:
            if project[1]['namespace'] == group_id:
                keycloak_connection = KeycloakOpenIDConnection(
                    server_url=settings.OIDC_SERVER_URL,
                    username=settings.OIDC_USERNAME,
                    password='',
                    realm_name=settings.OIDC_REALM_NAME,
                    client_id=settings.OIDC_RP_CLIENT_ID,
                    client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
                    verify=False
                )

                keycloak_admin = KeycloakAdmin(connection=keycloak_connection)
                groups = keycloak_admin.get_groups()


                maia_groups = {group['id']:group['name'][len("MAIA:"):] for group in groups if group['name'].startswith("MAIA:")}

                group_users = []

                for maia_group in maia_groups:
                    if maia_groups[maia_group] == group_id:
                        users = keycloak_admin.get_group_members(group_id=maia_group)
                        
                        for user in users:
                            group_users.append(user['email'])


                namespace_form = {
                "group_ID": group_id,
                "group_subdomain": group_id.lower().replace("_", "-"),
                "users": group_users,
                "resources_limits": {
                    "memory": [str(int(int(project[1]['memory_limit'][:-len(" Gi")])/2))+" Gi", project[1]['memory_limit']],
                    "cpu": [str(int(int(project[1]['cpu_limit'])/2)), project[1]['cpu_limit']],
                }
                }
                if project[1]['gpu'] != "N/A" and project[1]['gpu'] != "NO":
                    namespace_form["gpu_request"] = "1"

                if project[1]['conda'] != "N/A" and project[1]['conda'] is not None:
                    namespace_form["minio_env_name"] = group_id+"_env"
                
                cluster_id = project[1]['cluster']
                if cluster_id == "N/A":
                    cluster_id = None
                return namespace_form, cluster_id
    
    return None


def register_user_in_keycloak(email, settings):
    """
    Registers a user in Keycloak and sends an approved registration email.

    Parameters
    ----------
    email : str
        The email address of the user to be registered.
    settings : object
        An object containing the necessary settings for Keycloak connection and email sending.

    Settings Attributes
    -------------------
    OIDC_SERVER_URL : str
        The URL of the Keycloak server.
    OIDC_USERNAME : str
        The username for Keycloak authentication.
    OIDC_REALM_NAME : str
        The name of the Keycloak realm.
    OIDC_RP_CLIENT_ID : str
        The client ID for Keycloak.
    OIDC_RP_CLIENT_SECRET : str
        The client secret for Keycloak.
    HOSTNAME : str
        The hostname for generating the MAIA login URL.

    Returns
    -------
    None
    """
    keycloak_connection = KeycloakOpenIDConnection(
        server_url=settings.OIDC_SERVER_URL,
        username=settings.OIDC_USERNAME,
        password='',
        realm_name=settings.OIDC_REALM_NAME,
        client_id=settings.OIDC_RP_CLIENT_ID,
        client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
        verify=False
    )

    keycloak_admin = KeycloakAdmin(connection=keycloak_connection)

    temp_password = "MAIA"
    maia_login_url = "https://"+settings.HOSTNAME+"/maia/"
    keycloak_admin.create_user({'username':email,
                            'email':email,
                            'emailVerified':True,
                            'enabled':True,
                            #'firstName':'Demo2',
                            #'lastName':'Maia',
                            'requiredActions':['UPDATE_PASSWORD'],
                            'credentials':[{'type':'password',
                                            'temporary':True,
                                            'value': temp_password}],                         
                                              })
    send_approved_registration_email(email, maia_login_url, temp_password)

def register_group_in_keycloak(group_id, settings):
    """
    Registers a group in Keycloak with the specified group ID and settings.

    Parameters
    ----------
    group_id : str
        The ID of the group to be registered.
    settings : object
        An object containing the Keycloak server settings, including:
        - OIDC_SERVER_URL : str
            The URL of the Keycloak server.
        - OIDC_USERNAME : str
            The username for Keycloak authentication.
        - OIDC_REALM_NAME : str
            The name of the Keycloak realm.
        - OIDC_RP_CLIENT_ID : str
            The client ID for Keycloak.
        - OIDC_RP_CLIENT_SECRET : str
            The client secret for Keycloak.

    Returns
    -------
    None
    """
    keycloak_connection = KeycloakOpenIDConnection(
        server_url=settings.OIDC_SERVER_URL,
        username=settings.OIDC_USERNAME,
        password='',
        realm_name=settings.OIDC_REALM_NAME,
        client_id=settings.OIDC_RP_CLIENT_ID,
        client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
        verify=False
    )

    keycloak_admin = KeycloakAdmin(connection=keycloak_connection)

    payload = {
        "name": f"MAIA:{group_id}",
        "path": f"/MAIA:{group_id}",
        "attributes": {},
        "realmRoles": [],
        "clientRoles": {},
        "subGroups": [],
        "access": {"view": True, "manage": True, "manageMembership": True}
    }
    keycloak_admin.create_group(payload)


def register_users_in_group_in_keycloak(emails, group_id, settings):
    """
    Registers users in a specified Keycloak group.

    Parameters
    ----------
    emails : list
        A list of email addresses of users to be added to the group.
    group_id : str
        The ID of the group to which users should be added.
    settings : object
        An object containing Keycloak server settings, including:
        - OIDC_SERVER_URL : str
            The URL of the Keycloak server.
        - OIDC_USERNAME : str
            The username for Keycloak authentication.
        - OIDC_REALM_NAME : str
            The realm name in Keycloak.
        - OIDC_RP_CLIENT_ID : str
            The client ID for Keycloak.
        - OIDC_RP_CLIENT_SECRET : str
            The client secret for Keycloak.

    Returns
    -------
    None
    """
    keycloak_connection = KeycloakOpenIDConnection(
        server_url=settings.OIDC_SERVER_URL,
        username=settings.OIDC_USERNAME,
        password='',
        realm_name=settings.OIDC_REALM_NAME,
        client_id=settings.OIDC_RP_CLIENT_ID,
        client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
        verify=False
    )

    keycloak_admin = KeycloakAdmin(connection=keycloak_connection)

    groups = keycloak_admin.get_groups()

    users = keycloak_admin.get_users()
    for user in users:
        if "email" in user and user["email"] in emails:
            uid = user["id"]
            for group in groups:
                if group["name"] == "MAIA:"+group_id:
                    gid = group["id"]
                    keycloak_admin.group_user_add(uid, gid)
                elif group["name"] == "MAIA:users":
                    try:
                        gid = group["id"]
                        keycloak_admin.group_user_add(uid, gid)
                    except:
                        ...


def get_list_of_groups_requesting_a_user(email, settings):
    """
    Retrieves a list of groups (namespaces) that have requested a specific user based on their email.

    Parameters
    ----------
    email : str
        The email address of the user to search for.
    settings : module
        The settings module containing configuration such as DEBUG and LOCAL_DB_PATH.

    Returns
    -------
    list
        A list of namespaces that have requested the user. Returns an empty list if no groups are found.

    Raises
    ------
    KeyError
        If environment variables 'DB_HOST', 'DB_USERNAME', or 'DB_PASS' are not set in non-debug mode.
    Exception
        If there is an issue connecting to the database or executing the SQL queries.
    """
    if settings.DEBUG:
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH, "db.sqlite3"))
    else:
        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USERNAME"]
        dp_password = os.environ["DB_PASS"]
        engine = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        cnx = engine.raw_connection()

    auth_user = pd.read_sql_query("SELECT * FROM auth_user", con=cnx)
    authentication_maiauser = pd.read_sql_query("SELECT * FROM authentication_maiauser", con=cnx)

    for user in auth_user.iterrows():
        uid = user[1]['id']

        if user[1]['email'] == email:
            if uid in authentication_maiauser['user_ptr_id'].values:
                requested_namespaces = authentication_maiauser[authentication_maiauser['user_ptr_id'] == uid][
                    'namespace'].values[0].split(",")
                return requested_namespaces

    return []


def get_list_of_users_requesting_a_group(group_id, settings):
    """
    Retrieves a list of email addresses of users who have requested access to a specific group.

    Parameters
    ----------
    group_id : str
        The ID of the group to check for user requests.
    settings : object
        A settings object that contains configuration parameters, including DEBUG and LOCAL_DB_PATH.

    Returns
    -------
    list
        A list of email addresses of users who have requested access to the specified group.

    Raises
    ------
    KeyError
        If environment variables for database connection are not set when DEBUG is False.
    Exception
        If there is an issue with database connection or query execution.

    Notes
    -----
    When settings.DEBUG is True, a local SQLite database is used.
    When settings.DEBUG is False, a MySQL database is used with connection parameters from environment variables.
    """
    if settings.DEBUG:
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH, "db.sqlite3"))
    else:
        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USERNAME"]
        dp_password = os.environ["DB_PASS"]
        engine = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        cnx = engine.raw_connection()

    auth_user = pd.read_sql_query("SELECT * FROM auth_user", con=cnx)
    authentication_maiauser = pd.read_sql_query("SELECT * FROM authentication_maiauser", con=cnx)

    users = []
    for user in auth_user.iterrows():
        uid = user[1]['id']
        if uid in authentication_maiauser['user_ptr_id'].values:
            requested_namespaces = authentication_maiauser[authentication_maiauser['user_ptr_id'] == uid][
                'namespace'].values[0].split(",")
            if group_id in requested_namespaces:
                users.append(user[1]['email'])

    return users


def get_argocd_project_status(argocd_namespace, project_id):
    return verify_installed_maia_toolkit(project_id=project_id, namespace=argocd_namespace, get_chart_metadata=False)


def get_namespace_details(settings, id_token, namespace, user_id, is_admin=False):
    """
    Retrieve details about the namespace including workspace applications, remote desktops, SSH ports, MONAI models, and Orthanc instances.

    Parameters
    ----------
    settings : object
        Configuration settings containing API URLs and private cluster tokens.
    id_token : str
        Identity token for authentication.
    namespace : str
        The namespace to retrieve details for.
    user_id : str
        The user ID to filter resources.
    is_admin : bool, optional
        Flag indicating if the user has admin privileges. Defaults to False.

    Returns
    -------
    tuple
        A tuple containing:
        - maia_workspace_apps (dict): Dictionary of workspace applications with their URLs.
        - remote_desktop_dict (dict): Dictionary of remote desktop URLs for users.
        - ssh_ports (dict): Dictionary of SSH ports for users.
        - monai_models (dict): Dictionary of MONAI models.
        - orthanc_list (dict): Dictionary of Orthanc instances.
    """
    maia_workspace_apps = {}
    remote_desktop_dict = {}
    orthanc_list = {}
    monai_models = {}
    ssh_ports = {}

    for API_URL in settings.API_URL:
        if API_URL in settings.PRIVATE_CLUSTERS:
            token = settings.PRIVATE_CLUSTERS[API_URL]
            response = requests.get(API_URL + "/apis/networking.k8s.io/v1/namespaces/{}/ingresses".format(namespace),
                                    headers={"Authorization": "Bearer {}".format(token)}, verify=False)
        else:
            response = requests.get(API_URL + "/apis/networking.k8s.io/v1/namespaces/{}/ingresses".format(namespace),
                                    headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
        ingresses = json.loads(response.text)

        if API_URL in settings.PRIVATE_CLUSTERS:
            token = settings.PRIVATE_CLUSTERS[API_URL]
            try:
                response = requests.get(API_URL + "/api/v1/namespaces/{}/services".format(namespace),
                                        headers={"Authorization": "Bearer {}".format(token)}, verify=False)
            except:
                continue
        else:
            try:
                response = requests.get(API_URL + "/api/v1/namespaces/{}/services".format(namespace),
                                        headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
            except:
                continue
        services = json.loads(response.text)

        if 'code' in services:
            if services['code'] == 403:
                ...

        for ingress in ingresses['items']:
            for rule in ingress['spec']['rules']:
                if 'host' not in rule:
                    rule['host'] = settings.DEFAULT_INGRESS_HOST
                for path in rule['http']['paths']:
                    if path['backend']['service']['name'] == 'proxy-public':
                        ## JupyterHub
                        maia_workspace_apps['hub'] = "https://" + rule['host'] + path['path']

        for service in services['items']:
            for port in service['spec']['ports']:
                if 'name' in port and port['name'] == 'remote-desktop-port':
                    hub_url = maia_workspace_apps['hub']
                    user = service["metadata"]["name"][len("jupyter-"):].replace("-2d", "-").replace("-40", "@").replace("-2e", ".")
                    url = f"{hub_url}/user/{user}/proxy/80/desktop/{user}/"
                    if user_id == user or is_admin:
                        remote_desktop_dict[user] = url

                if 'name' in port and port['name'] == 'ssh':
                    user = service["metadata"]["name"][len("jupyter-"):].replace("-2d", "-").replace("-40", "@").replace("-2e", ".")
                    if user_id == user or is_admin:
                        ssh_ports[user] = port['port']

    if "hub" not in maia_workspace_apps:
        maia_workspace_apps["hub"] = "N/A"
    if "orthanc" not in maia_workspace_apps:
        maia_workspace_apps["orthanc"] = "N/A"
    if "monai_label" not in maia_workspace_apps:
        maia_workspace_apps["monai_label"] = "N/A"
    if "label_studio" not in maia_workspace_apps:
        maia_workspace_apps["label_studio"] = "N/A"
    if "kubeflow" not in maia_workspace_apps:
        maia_workspace_apps["kubeflow"] = "N/A"
    if "mlflow" not in maia_workspace_apps:
        maia_workspace_apps["mlflow"] = "N/A"
    if "minio_console" not in maia_workspace_apps:
        maia_workspace_apps["minio_console"] = "N/A"
    if "xnat" not in maia_workspace_apps:
        maia_workspace_apps["xnat"] = "N/A"

    return maia_workspace_apps, remote_desktop_dict, ssh_ports, monai_models, orthanc_list


def get_allocation_date_for_project(settings, group_id, is_namespace_style=False):
    """
    Retrieves the allocation date for a project based on the provided group ID.

    Parameters
    ----------
    settings : object
        The settings object containing configuration details.
    group_id : str
        The group ID or namespace to search for.
    is_namespace_style : bool, optional
        Flag to indicate if the group ID should be treated as a namespace. Defaults to False.

    Returns
    -------
    datetime or None
        The allocation date of the project if found, otherwise None.
    """
    if settings.DEBUG:
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH, "db.sqlite3"))
    else:
        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USERNAME"]
        dp_password = os.environ["DB_PASS"]

        engine = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        cnx = engine.raw_connection()

    authentication_maiaproject = pd.read_sql_query("SELECT * FROM authentication_maiaproject", con=cnx)

    for project in authentication_maiaproject.iterrows():
        if is_namespace_style:
            if str(project[1]['namespace']).lower().replace("_", "-") == group_id:
                return project[1]['date']
        else:
            if project[1]['namespace'] == group_id:
                return project[1]['date']

    return None


def get_project_argo_status_and_user_table(request, settings):
    """
    Retrieves the Argo CD project status and user table information.

    Parameters
    ----------
    request : HttpRequest
        The HTTP request object containing session and user information.
    settings : Settings
        The settings object containing configuration values.

    Returns
    -------
    tuple
        A tuple containing:
        - user_table (dict): The user table information.
        - to_register_in_groups (list): List of users to register in groups.
        - to_register_in_keycloak (list): List of users to register in Keycloak.
        - maia_groups_dict (dict): Dictionary of MAIA groups.
        - project_argo_status (dict): Dictionary containing the Argo CD project status for each project.
    """
    argocd_cluster_id = settings.ARGOCD_CLUSTER

    id_token = request.session.get('oidc_id_token')
    kubeconfig_dict = generate_kubeconfig(id_token, request.user.username, "default", argocd_cluster_id, settings=settings)
    config.load_kube_config_from_dict(kubeconfig_dict)
    with open(Path("/tmp").joinpath("kubeconfig"), "w") as f:
        yaml.dump(kubeconfig_dict, f)
        os.environ["KUBECONFIG"] = str(Path("/tmp").joinpath("kubeconfig"))

    user_table, to_register_in_groups, to_register_in_keycloak, maia_groups_dict = get_user_table(settings=settings)
    project_argo_status = {}

    for project_id in maia_groups_dict:
        project_argo_status[project_id] = asyncio.run(get_argocd_project_status(argocd_namespace="argocd", project_id=project_id.lower().replace("_", "-")))

    return user_table, to_register_in_groups, to_register_in_keycloak, maia_groups_dict, project_argo_status


def create_namespace(request, settings, namespace_id, cluster_id):
    """
    Creates a Kubernetes namespace using the provided request, settings, namespace ID, and cluster ID.

    Parameters
    ----------
    request : HttpRequest
        The HTTP request object containing session and user information.
    settings : Settings
        The settings object containing configuration details.
    namespace_id : str
        The ID of the namespace to be created.
    cluster_id : str
        The ID of the Kubernetes cluster where the namespace will be created.

    Returns
    -------
    None

    Raises
    ------
    ApiException
        If an error occurs while creating the namespace using the Kubernetes API.
    """
    id_token = request.session.get('oidc_id_token')
    kubeconfig_dict = generate_kubeconfig(id_token, request.user.username, "default", cluster_id, settings=settings)
    config.load_kube_config_from_dict(kubeconfig_dict)
    with open(Path("/tmp").joinpath("kubeconfig"), "w") as f:
        yaml.dump(kubeconfig_dict, f)
        os.environ["KUBECONFIG"] = str(Path("/tmp").joinpath("kubeconfig"))

        with kubernetes.client.ApiClient() as api_client:
            api_instance = kubernetes.client.CoreV1Api(api_client)
            body = kubernetes.client.V1Namespace(metadata=kubernetes.client.V1ObjectMeta(name=namespace_id))
            try:
                api_response = api_instance.create_namespace(body)
                print(api_response)
            except ApiException as e:
                print("Exception when calling CoreV1Api->create_namespace: %s\n" % e)