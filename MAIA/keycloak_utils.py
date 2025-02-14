from keycloak import KeycloakAdmin
from keycloak import KeycloakOpenIDConnection
import json
import os
import pandas as pd
import sqlite3
from sqlalchemy import create_engine



def get_user_ids(settings):
    """
    Retrieve user IDs and their associated MAIA groups from Keycloak.
    
    Parameters
    ----------
    settings : object
        An object containing the Keycloak server settings. It should have the following attributes:
        - OIDC_SERVER_URL: str, the URL of the Keycloak server.
        - OIDC_USERNAME: str, the username for Keycloak authentication.
        - OIDC_REALM_NAME: str, the realm name in Keycloak.
        - OIDC_RP_CLIENT_ID: str, the client ID for Keycloak.
        - OIDC_RP_CLIENT_SECRET: str, the client secret for Keycloak.
    
    Returns
    -------
    dict
        A dictionary where the keys are user email addresses and the values are lists of MAIA groups the user belongs to.
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

    users = keycloak_admin.get_users()
    
    
    user_list = {}
    groups = keycloak_admin.get_groups()
    maia_groups = {group['id']:group['name'][len("MAIA:"):] for group in groups if group['name'].startswith("MAIA:")}

    delete_groups = ["Brain-Aging","Brain-Diffusion","Demo"]
    
    for maia_group in maia_groups:
        if maia_groups[maia_group] in delete_groups:
            keycloak_admin.delete_group(group_id=maia_group)
            continue
        users = keycloak_admin.get_group_members(group_id=maia_group)
        emails = [user['email'] for user in users if 'email' in user]
        for email in emails:
            if email in user_list:
                user_list[email] += [maia_groups[maia_group]]
            else:
                user_list[email] = [maia_groups[maia_group]]
    
    
    return user_list


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
    if "email_account" in os.environ and "email_password" in os.environ and "email_smtp_server" in os.environ:
        from MAIA.dashboard_utils import send_approved_registration_email
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