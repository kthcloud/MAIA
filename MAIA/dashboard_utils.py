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
from MAIA.kubernetes_utils import generate_kubeconfig
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import email.utils


def verify_gpu_booking_policy(existing_bookings, new_booking):

    total_days = sum((booking.end_date - booking.start_date).days for booking in existing_bookings)

    # Calculate the number of days for the new booking
    ending_time = datetime.strptime(new_booking["ending_time"], "%Y-%m-%d %H:%M:%S")
    starting_time = datetime.strptime(new_booking["starting_time"], "%Y-%m-%d %H:%M:%S")

    new_booking_days = (ending_time - starting_time).days

    # Verify that the sum of existing bookings and the new booking does not exceed 60 days
    if total_days + new_booking_days > 60:
        return False

    return True


def send_maia_info_email(receiver_email, register_project_url, register_user_url, discord_support_link):
    """
    Send an email with registration information for the MAIA platform.
    Parameters
    ----------
    receiver_email : str
        The email address of the recipient.
    register_project_url : str
        The URL for project registration.
    register_user_url : str
        The URL for user registration.
    discord_support_link : str
        The URL for the MAIA support Discord.
    Returns
    -------
    None
    """

    sender_email = os.environ["email_account"]
    message = MIMEMultipart()
    message["Subject"] = "Registration Information for the MAIA Platform"
    message["From"] = "MAIA Team"
    message["To"] = receiver_email

    html = """\
    <html>
        <head></head>
        <body>
            <p>Hello</p>
            <p>Thank you for your interest in the MAIA platform. Below are the steps to register:</p>
            <p><b>Project Registration:</b><br>
            If you are starting a research work and you want to have it hosted in MAIA, please first register your project here:<br>
            <a href="{}">MAIA Project Registration</a></p>
            <p><b>User Registration:</b><br>
            To create a user account, an active project must be available to select. Once a project is registered, you can sign up for an account linked to that project here:<br>
            <a href="{}">MAIA User Registration</a></p>
            <p>If you have any questions or need further assistance, feel free to join our Discord community:<br>
            <a href="{}">MAIA Support Discord</a></p>
            <br>
            <p>Best regards,</p>
            <p>The MAIA Admin Team</p>
        </body>
    </html>
    """.format(
        register_project_url, register_user_url, discord_support_link
    )

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
        client = Minio(
            settings.MINIO_URL, access_key=settings.MINIO_ACCESS_KEY, secret_key=settings.MINIO_SECRET_KEY, secure=True
        )
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
    """.format(
        login_url, temp_password
    )

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
    data = {"content": f"{username} is requesting a MAIA account for the project {namespace}.", "username": "MAIA-Bot"}

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
        password="",
        realm_name=settings.OIDC_REALM_NAME,
        client_id=settings.OIDC_RP_CLIENT_ID,
        client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
        verify=False,
    )

    keycloak_admin = KeycloakAdmin(connection=keycloak_connection)

    groups = keycloak_admin.get_groups()

    maia_groups = {group["id"]: group["name"][len("MAIA:") :] for group in groups if group["name"].startswith("MAIA:")}

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
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH, "db.sqlite3"))
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
            if project[1]["namespace"] not in active_groups.values():
                pending_projects.append(project[1]["namespace"])

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
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH, "db.sqlite3"))
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

    table = auth_user.merge(authentication_maiauser_copy, on="id", how="left")

    keycloak_connection = KeycloakOpenIDConnection(
        server_url=settings.OIDC_SERVER_URL,
        username=settings.OIDC_USERNAME,
        password="",
        realm_name=settings.OIDC_REALM_NAME,
        client_id=settings.OIDC_RP_CLIENT_ID,
        client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
        verify=False,
    )

    keycloak_admin = KeycloakAdmin(connection=keycloak_connection)
    users_to_register_in_group = {}

    groups = keycloak_admin.get_groups()

    maia_groups = {group["id"]: group["name"][len("MAIA:") :] for group in groups if group["name"].startswith("MAIA:")}

    pending_projects = get_pending_projects(settings=settings)

    maia_group_dict = {}

    try:
        client = Minio(
            settings.MINIO_URL,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )

        minio_envs = [env.object_name[: -len("_env")] for env in list(client.list_objects(settings.BUCKET_NAME))]
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
            if project[1]["namespace"] == maia_groups[maia_group]:
                admin_users = [project[1]["email"]]
                cpu_limit = project[1]["cpu_limit"]
                memory_limit = project[1]["memory_limit"]
                date = project[1]["date"]
                cluster = project[1]["cluster"]
                gpu = project[1]["gpu"]
                environment = project[1]["minimal_env"]

        conda_envs = []
        if maia_groups[maia_group] in minio_envs:
            conda_envs.append(maia_groups[maia_group])
        else:
            conda_envs.append("N/A")

        group_users = []
        for user in users:
            if user["username"] in admin_users:
                group_users.append(user["email"] + " [Project Admin]")
            else:
                group_users.append(user["email"])

        maia_group_dict[maia_groups[maia_group]] = {
            "users": group_users,
            "conda": conda_envs,
            "admin_users": admin_users,
            "cpu_limit": cpu_limit,
            "memory_limit": memory_limit,
            "date": date,
            "cluster": cluster,
            "gpu": gpu,
            "environment": environment,
        }

    for pending_project in pending_projects:
        conda_envs = []
        if pending_project in minio_envs:
            conda_envs.append(pending_project)
        else:
            conda_envs.append("N/A")

        users = []
        for user in auth_user.iterrows():
            uid = user[1]["id"]
            if uid in authentication_maiauser["user_ptr_id"].values:
                requested_namespaces = (
                    authentication_maiauser[authentication_maiauser["user_ptr_id"] == uid]["namespace"].values[0].split(",")
                )
                if pending_project in requested_namespaces:
                    users.append(user[1]["email"])

        maia_group_dict[pending_project] = {
            "users": users,
            "pending": True,
            "conda": conda_envs,
            "admin_users": [],
            "cpu_limit": authentication_maiaproject[authentication_maiaproject["namespace"] == pending_project][
                "cpu_limit"
            ].values[0],
            "memory_limit": authentication_maiaproject[authentication_maiaproject["namespace"] == pending_project][
                "memory_limit"
            ].values[0],
            "date": authentication_maiaproject[authentication_maiaproject["namespace"] == pending_project]["date"].values[0],
            "cluster": "N/A",
            "gpu": authentication_maiaproject[authentication_maiaproject["namespace"] == pending_project]["gpu"].values[0],
            "environment": "Minimal",
        }

    users_to_register_in_keycloak = []
    for user in auth_user.iterrows():
        uid = user[1]["id"]
        username = user[1]["username"]
        email = user[1]["email"]
        user_groups = []
        user_in_keycloak = False
        for keycloak_user in keycloak_admin.get_users():

            if "email" in keycloak_user:
                if keycloak_user["email"] == email:
                    user_in_keycloak = True
                    user_keycloak_groups = keycloak_admin.get_user_groups(user_id=keycloak_user["id"])
                    for user_keycloak_group in user_keycloak_groups:
                        if user_keycloak_group["name"].startswith("MAIA:"):
                            user_groups.append(user_keycloak_group["name"][len("MAIA:") :])
        if not user_in_keycloak:
            users_to_register_in_keycloak.append(email)
        if uid in authentication_maiauser["user_ptr_id"].values:
            requested_namespaces = (
                authentication_maiauser[authentication_maiauser["user_ptr_id"] == uid]["namespace"].values[0].split(",")
            )
            for requested_namespace in requested_namespaces:
                if requested_namespace not in user_groups and requested_namespace != "N/A":
                    if email not in users_to_register_in_group:
                        users_to_register_in_group[email] = [requested_namespace]
                    else:
                        users_to_register_in_group[email].append(requested_namespace)

    table.fillna("N/A", inplace=True)

    return table, users_to_register_in_group, users_to_register_in_keycloak, maia_group_dict


def check_pending_projects_and_assign_id(settings):
    """
    Check for pending projects and assign an ID if necessary.

    Parameters
    ----------
    settings : object
        An object containing configuration settings, including DEBUG and LOCAL_DB_PATH attributes.

    Returns
    -------
    None
    """

    if settings.DEBUG:
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH, "db.sqlite3"))
    else:
        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USERNAME"]
        dp_password = os.environ["DB_PASS"]

        # try:
        engine = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        cnx = engine.raw_connection()

    authentication_maiaproject = pd.read_sql_query("SELECT * FROM authentication_maiaproject", con=cnx)

    id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1
    for project in authentication_maiaproject.iterrows():
        if pd.isna(project[1]["id"]):
            authentication_maiaproject.loc[project[0], "id"] = int(id)

            id += 1

    cnx.close()

    if settings.DEBUG:
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH, "db.sqlite3"))
        authentication_maiaproject.to_sql("authentication_maiaproject", con=cnx, if_exists="replace", index=False)

    else:
        engine.dispose()
        engine_2 = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        authentication_maiaproject.to_sql("authentication_maiaproject", con=engine_2, if_exists="replace", index=False)


def register_cluster_for_project_in_db(settings, namespace, cluster):
    """
    Registers a cluster for a project in the database.
    Depending on the DEBUG setting, this function connects to either a local SQLite database or a remote MySQL database.
    It updates the cluster information for a given namespace in the `authentication_maiaproject` table.

    Parameters
    ----------
    settings : object
        An object containing configuration settings. Must have `DEBUG` and `LOCAL_DB_PATH` attributes.
    namespace : str
        The namespace of the project to update.
    cluster : str
        The cluster information to register for the project.

    Returns
    -------
    None
    """

    if settings.DEBUG:
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH, "db.sqlite3"))
    else:
        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USERNAME"]
        dp_password = os.environ["DB_PASS"]

        # try:
        engine = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        cnx = engine.raw_connection()

    authentication_maiaproject = pd.read_sql_query("SELECT * FROM authentication_maiaproject", con=cnx)

    keycloak_connection = KeycloakOpenIDConnection(
        server_url=settings.OIDC_SERVER_URL,
        username=settings.OIDC_USERNAME,
        password="",
        realm_name=settings.OIDC_REALM_NAME,
        client_id=settings.OIDC_RP_CLIENT_ID,
        client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
        verify=False,
    )

    group_id = namespace
    keycloak_admin = KeycloakAdmin(connection=keycloak_connection)
    groups = keycloak_admin.get_groups()

    maia_groups = {group["id"]: group["name"][len("MAIA:") :] for group in groups if group["name"].startswith("MAIA:")}
    for maia_group in maia_groups:
        if maia_groups[maia_group].lower().replace("_", "-") == namespace:
            group_id = maia_groups[maia_group]

    print("Registering Existing Cluster for Group: ", group_id)
    try:
        id = authentication_maiaproject[authentication_maiaproject["namespace"] == group_id]["id"].values[0]
        authentication_maiaproject.loc[authentication_maiaproject["id"] == id, "cluster"] = cluster
    except:
        id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1
        authentication_maiaproject = authentication_maiaproject.append(
            {"id": int(id), "namespace": group_id, "cluster": cluster, "memory_limit": "2 Gi", "cpu_limit": "2"},
            ignore_index=True,
        )

    cnx.close()

    if settings.DEBUG:
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH, "db.sqlite3"))
        authentication_maiaproject.to_sql("authentication_maiaproject", con=cnx, if_exists="replace", index=False)

    else:
        engine.dispose()
        engine_2 = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        authentication_maiaproject.to_sql("authentication_maiaproject", con=engine_2, if_exists="replace", index=False)


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
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH, "db.sqlite3"))
    else:
        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USERNAME"]
        dp_password = os.environ["DB_PASS"]

        # try:
        engine = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        cnx = engine.raw_connection()

    auth_user = pd.read_sql_query("SELECT * FROM auth_user", con=cnx)

    authentication_maiauser = pd.read_sql_query("SELECT * FROM authentication_maiauser", con=cnx)

    authentication_maiaproject = pd.read_sql_query("SELECT * FROM authentication_maiaproject", con=cnx)

    for k in form:
        if k.startswith("namespace"):
            id = auth_user[auth_user["username"] == k[len("namespace_") :]]["id"].values[0]

            if len(authentication_maiauser[authentication_maiauser["user_ptr_id"] == id]) > 0:
                authentication_maiauser.loc[authentication_maiauser["user_ptr_id"] == id, "namespace"] = form[k]
            else:

                authentication_maiauser = authentication_maiauser.append(
                    {"user_ptr_id": int(id), "namespace": form[k]}, ignore_index=True
                )
        elif k.startswith("memory_limit"):

            try:
                id = authentication_maiaproject[authentication_maiaproject["namespace"] == k[len("memory_limit_") :]][
                    "id"
                ].values[0]
            except:
                id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1

            if len(authentication_maiaproject[authentication_maiaproject["id"] == id]) > 0:
                authentication_maiaproject.loc[authentication_maiaproject["id"] == id, "memory_limit"] = form[k]
            else:

                authentication_maiaproject = authentication_maiaproject.append(
                    {"id": int(id), "memory_limit": form[k], "namespace": k[len("memory_limit_") :]}, ignore_index=True
                )
        elif k.startswith("cpu_limit"):
            try:
                id = authentication_maiaproject[authentication_maiaproject["namespace"] == k[len("cpu_limit_") :]]["id"].values[0]
            except:
                id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1

            if len(authentication_maiaproject[authentication_maiaproject["id"] == id]) > 0:
                authentication_maiaproject.loc[authentication_maiaproject["id"] == id, "cpu_limit"] = form[k]
            else:
                authentication_maiaproject = authentication_maiaproject.append(
                    {"id": int(id), "cpu_limit": form[k], "namespace": k[len("cpu_limit_") :]}, ignore_index=True
                )
        elif k.startswith("date"):
            try:
                id = authentication_maiaproject[authentication_maiaproject["namespace"] == k[len("date_") :]]["id"].values[0]
            except:
                id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1

            if len(authentication_maiaproject[authentication_maiaproject["id"] == id]) > 0:
                authentication_maiaproject.loc[authentication_maiaproject["id"] == id, "date"] = form[k]
            else:
                authentication_maiaproject = authentication_maiaproject.append(
                    {"id": int(id), "date": form[k], "namespace": k[len("date_") :]}, ignore_index=True
                )
        elif k.startswith("cluster"):
            try:
                id = authentication_maiaproject[authentication_maiaproject["namespace"] == k[len("cluster_") :]]["id"].values[0]
            except:
                id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1

            if len(authentication_maiaproject[authentication_maiaproject["id"] == id]) > 0:
                authentication_maiaproject.loc[authentication_maiaproject["id"] == id, "cluster"] = form[k]
            else:
                authentication_maiaproject = authentication_maiaproject.append(
                    {"id": int(id), "cluster": form[k], "namespace": k[len("cluster_") :]}, ignore_index=True
                )
        elif k.startswith("gpu"):
            try:
                id = authentication_maiaproject[authentication_maiaproject["namespace"] == k[len("gpu_") :]]["id"].values[0]
            except:
                id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1

            if len(authentication_maiaproject[authentication_maiaproject["id"] == id]) > 0:
                authentication_maiaproject.loc[authentication_maiaproject["id"] == id, "gpu"] = form[k]
            else:
                authentication_maiaproject = authentication_maiaproject.append(
                    {"id": int(id), "gpu": form[k], "namespace": k[len("gpu_") :]}, ignore_index=True
                )
        elif k.startswith("minimal_environment"):
            try:
                id = authentication_maiaproject[authentication_maiaproject["namespace"] == k[len("minimal_environment_") :]][
                    "id"
                ].values[0]
            except:
                id = 0 if pd.isna(authentication_maiaproject["id"].max()) else authentication_maiaproject["id"].max() + 1

            if len(authentication_maiaproject[authentication_maiaproject["id"] == id]) > 0:
                authentication_maiaproject.loc[authentication_maiaproject["id"] == id, "minimal_env"] = form[k]
            else:
                authentication_maiaproject = authentication_maiaproject.append(
                    {"id": int(id), "minimal_env": form[k], "namespace": k[len("minimal_environment_") :]}, ignore_index=True
                )

    # try:
    cnx.close()

    if settings.DEBUG:
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH, "db.sqlite3"))
        authentication_maiauser.to_sql("authentication_maiauser", con=cnx, if_exists="replace", index=False)
        authentication_maiaproject.to_sql("authentication_maiaproject", con=cnx, if_exists="replace", index=False)

    else:
        engine.dispose()
        engine_2 = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        authentication_maiauser.to_sql("authentication_maiauser", con=engine_2, if_exists="replace", index=False)
        authentication_maiaproject.to_sql("authentication_maiaproject", con=engine_2, if_exists="replace", index=False)
        # stmt = text("ALTER TABLE authentication_maiauser-copy RENAME TO authentication_maiauser;")
        # engine.execute(stmt)

    # except:
    #    ...


# auth_user[auth_user["username"] == k[len("date_"):]]["date"] = form[k]


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
        cnx = sqlite3.connect(os.path.join(settings.LOCAL_DB_PATH, "db.sqlite3"))
    else:
        db_host = os.environ["DB_HOST"]
        db_user = os.environ["DB_USERNAME"]
        dp_password = os.environ["DB_PASS"]

        # try:
        engine = create_engine(f"mysql+pymysql://{db_user}:{dp_password}@{db_host}:3306/mysql")
        cnx = engine.raw_connection()

    authentication_maiaproject = pd.read_sql_query("SELECT * FROM authentication_maiaproject", con=cnx)

    cluster_id = None

    for project in authentication_maiaproject.iterrows():
        if is_namespace_style:
            if str(project[1]["namespace"]).lower().replace("_", "-") == group_id:
                keycloak_connection = KeycloakOpenIDConnection(
                    server_url=settings.OIDC_SERVER_URL,
                    username=settings.OIDC_USERNAME,
                    password="",
                    realm_name=settings.OIDC_REALM_NAME,
                    client_id=settings.OIDC_RP_CLIENT_ID,
                    client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
                    verify=False,
                )

                keycloak_admin = KeycloakAdmin(connection=keycloak_connection)
                groups = keycloak_admin.get_groups()

                maia_groups = {
                    group["id"]: group["name"][len("MAIA:") :] for group in groups if group["name"].startswith("MAIA:")
                }

                group_users = []

                for maia_group in maia_groups:
                    if maia_groups[maia_group] == group_id:
                        users = keycloak_admin.get_group_members(group_id=maia_group)

                        for user in users:
                            group_users.append(user["email"])

                namespace_form = {
                    "group_ID": group_id,
                    "group_subdomain": group_id.lower().replace("_", "-"),
                    "users": group_users,
                    "resources_limits": {
                        "memory": [
                            str(int(int(project[1]["memory_limit"][: -len(" Gi")]) / 2)) + " Gi",
                            project[1]["memory_limit"],
                        ],
                        "cpu": [str(int(int(project[1]["cpu_limit"]) / 2)), project[1]["cpu_limit"]],
                    },
                }
                if project[1]["gpu"] != "N/A" and project[1]["gpu"] != "NO":
                    namespace_form["gpu"] = "1"

                if project[1]["conda"] != "N/A" and project[1]["conda"] is not None:
                    namespace_form["minio_env_name"] = group_id + "_env"

                cluster_id = project[1]["cluster"]
                if cluster_id == "N/A":
                    cluster_id = None
                return namespace_form, cluster_id
        else:
            if project[1]["namespace"] == group_id:
                keycloak_connection = KeycloakOpenIDConnection(
                    server_url=settings.OIDC_SERVER_URL,
                    username=settings.OIDC_USERNAME,
                    password="",
                    realm_name=settings.OIDC_REALM_NAME,
                    client_id=settings.OIDC_RP_CLIENT_ID,
                    client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
                    verify=False,
                )

                keycloak_admin = KeycloakAdmin(connection=keycloak_connection)
                groups = keycloak_admin.get_groups()

                maia_groups = {
                    group["id"]: group["name"][len("MAIA:") :] for group in groups if group["name"].startswith("MAIA:")
                }

                group_users = []

                for maia_group in maia_groups:
                    if maia_groups[maia_group] == group_id:
                        users = keycloak_admin.get_group_members(group_id=maia_group)

                        for user in users:
                            group_users.append(user["email"])

                namespace_form = {
                    "group_ID": group_id,
                    "group_subdomain": group_id.lower().replace("_", "-"),
                    "users": group_users,
                    "resources_limits": {
                        "memory": [
                            str(int(int(project[1]["memory_limit"][: -len(" Gi")]) / 2)) + " Gi",
                            project[1]["memory_limit"],
                        ],
                        "cpu": [str(int(int(project[1]["cpu_limit"]) / 2)), project[1]["cpu_limit"]],
                    },
                }
                if project[1]["gpu"] != "N/A" and project[1]["gpu"] != "NO":
                    namespace_form["gpu_request"] = "1"

                if project[1]["conda"] != "N/A" and project[1]["conda"] is not None:
                    namespace_form["minio_env_name"] = group_id + "_env"

                cluster_id = project[1]["cluster"]
                if cluster_id == "N/A":
                    cluster_id = None
                return namespace_form, cluster_id

    return None, None


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
        password="",
        realm_name=settings.OIDC_REALM_NAME,
        client_id=settings.OIDC_RP_CLIENT_ID,
        client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
        verify=False,
    )

    keycloak_admin = KeycloakAdmin(connection=keycloak_connection)

    temp_password = "MAIA"
    maia_login_url = "https://" + settings.HOSTNAME + "/maia/"
    keycloak_admin.create_user(
        {
            "username": email,
            "email": email,
            "emailVerified": True,
            "enabled": True,
            #'firstName':'Demo2',
            #'lastName':'Maia',
            "requiredActions": ["UPDATE_PASSWORD"],
            "credentials": [{"type": "password", "temporary": True, "value": temp_password}],
        }
    )
    if "email_account" in os.environ and "email_password" in os.environ and "email_smtp_server" in os.environ:
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
        password="",
        realm_name=settings.OIDC_REALM_NAME,
        client_id=settings.OIDC_RP_CLIENT_ID,
        client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
        verify=False,
    )

    keycloak_admin = KeycloakAdmin(connection=keycloak_connection)

    payload = {
        "name": f"MAIA:{group_id}",
        "path": f"/MAIA:{group_id}",
        "attributes": {},
        "realmRoles": [],
        "clientRoles": {},
        "subGroups": [],
        "access": {"view": True, "manage": True, "manageMembership": True},
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
        password="",
        realm_name=settings.OIDC_REALM_NAME,
        client_id=settings.OIDC_RP_CLIENT_ID,
        client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
        verify=False,
    )

    keycloak_admin = KeycloakAdmin(connection=keycloak_connection)

    groups = keycloak_admin.get_groups()

    users = keycloak_admin.get_users()
    for user in users:
        if "email" in user and user["email"] in emails:
            uid = user["id"]
            for group in groups:
                if group["name"] == "MAIA:" + group_id:
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
        uid = user[1]["id"]

        if user[1]["email"] == email:
            if uid in authentication_maiauser["user_ptr_id"].values:
                requested_namespaces = (
                    authentication_maiauser[authentication_maiauser["user_ptr_id"] == uid]["namespace"].values[0].split(",")
                )
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
        uid = user[1]["id"]
        if uid in authentication_maiauser["user_ptr_id"].values:
            requested_namespaces = (
                authentication_maiauser[authentication_maiauser["user_ptr_id"] == uid]["namespace"].values[0].split(",")
            )
            if group_id in requested_namespaces:
                users.append(user[1]["email"])

    return users


def get_argocd_project_status(argocd_namespace, project_id):
    return verify_installed_maia_toolkit(project_id=project_id, namespace=argocd_namespace, get_chart_metadata=False)


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
            if str(project[1]["namespace"]).lower().replace("_", "-") == group_id:
                return project[1]["date"]
        else:
            if project[1]["namespace"] == group_id:
                return project[1]["date"]

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

    id_token = request.session.get("oidc_id_token")
    kubeconfig_dict = generate_kubeconfig(id_token, request.user.username, "default", argocd_cluster_id, settings=settings)
    config.load_kube_config_from_dict(kubeconfig_dict)
    with open(Path("/tmp").joinpath("kubeconfig-argo"), "w") as f:
        yaml.dump(kubeconfig_dict, f)
        os.environ["KUBECONFIG"] = str(Path("/tmp").joinpath("kubeconfig-argo"))

    user_table, to_register_in_groups, to_register_in_keycloak, maia_groups_dict = get_user_table(settings=settings)
    project_argo_status = {}

    for project_id in maia_groups_dict:
        project_argo_status[project_id] = asyncio.run(
            get_argocd_project_status(argocd_namespace="argocd", project_id=project_id.lower().replace("_", "-"))
        )

    return user_table, to_register_in_groups, to_register_in_keycloak, maia_groups_dict, project_argo_status


def get_maia_users_from_keycloak(settings):
    """
    Retrieves all users from Keycloak who are members of any MAIA group.

    Parameters
    ----------
    settings : object
        An object containing Keycloak connection settings, including:
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
    list
        A list of dictionaries containing user information for all users in MAIA groups.
        Each dictionary contains user details like email, username, and groups.
    """
    keycloak_connection = KeycloakOpenIDConnection(
        server_url=settings.OIDC_SERVER_URL,
        username=settings.OIDC_USERNAME,
        password="",
        realm_name=settings.OIDC_REALM_NAME,
        client_id=settings.OIDC_RP_CLIENT_ID,
        client_secret_key=settings.OIDC_RP_CLIENT_SECRET,
        verify=False,
    )

    keycloak_admin = KeycloakAdmin(connection=keycloak_connection)

    # Get all groups that start with "MAIA:"
    groups = keycloak_admin.get_groups()
    maia_groups = {group["id"]: group["name"] for group in groups if group["name"].startswith("MAIA:")}

    # Get all users and filter those who are in MAIA groups
    maia_users = []
    all_users = keycloak_admin.get_users()

    for user in all_users:
        if "email" not in user:
            continue

        user_groups = keycloak_admin.get_user_groups(user_id=user["id"])
        user_maia_groups = [group["name"] for group in user_groups if group["name"].startswith("MAIA:")]

        if user_maia_groups:
            maia_users.append(
                {"email": user["email"], "username": user["username"], "id": user["id"], "groups": user_maia_groups}
            )

    return maia_users


def send_maia_message_email(receiver_emails, subject, message_body):
    """
    Send an email with a custom message to multiple recipients with improved deliverability.
    """
    try:
        sender_email = os.environ["email_account"]
        message = MIMEMultipart("alternative")  # Changed to alternative for better compatibility

        # Add proper headers to improve deliverability
        message["Subject"] = subject
        message["From"] = f"MAIA Team <{sender_email}>"  # Use proper From format
        message["To"] = ", ".join(receiver_emails)
        message["Reply-To"] = sender_email
        message["Date"] = email.utils.formatdate(localtime=True)
        message["Message-ID"] = email.utils.make_msgid(domain=sender_email.split("@")[1])

        # Add DKIM-Signature header if available
        if "DKIM_PRIVATE_KEY" in os.environ:
            # Implementation would go here - requires additional setup
            pass

        html = f"""\
        <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
            </head>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333333;">
                {message_body}
                <br>
                <p>Best regards,</p>
                <p>The MAIA Admin Team</p>
                <hr>
                <p style="font-size: 12px; color: #666666;">
                    This is an automated message from the MAIA Platform. 
                    If you believe you received this in error, please contact support.
                </p>
            </body>
        </html>
        """

        # Create plain text version
        text = BeautifulSoup(html, "html.parser").get_text()

        # Attach both plain text and HTML versions
        part1 = MIMEText(text, "plain")
        part2 = MIMEText(html, "html")
        message.attach(part1)
        message.attach(part2)

        port = 465  # For SSL
        password = os.environ["email_password"]
        smtp_server = os.environ["email_smtp_server"]

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.send_message(message)  # Using send_message instead of sendmail

        return True

    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False
