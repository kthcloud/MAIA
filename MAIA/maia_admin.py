from omegaconf import OmegaConf
from pathlib import Path
from MAIA.maia_fn import encode_docker_registry_secret, convert_username_to_jupyterhub_username, get_ssh_ports
import argocd_client
from argocd_client.rest import ApiException
from pyhelm3 import Client
from pprint import pprint
import os
import yaml
from secrets import token_urlsafe
import base64


def generate_minio_configs():
    """
    Generate configuration settings for MinIO.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - access_key (str): The access key for MinIO.
        - secret_key (str): A randomly generated secret key for MinIO.
        - console_access_key (str): A base64 encoded access key for console access.
        - console_secret_key (str): A base64 encoded secret key for console access.
    """
    minio_configs = {
        "access_key": "admin",
        "secret_key": token_urlsafe(16).replace("-", "_"),
        "console_access_key": base64.b64encode(token_urlsafe(16).replace("-", "_").encode("ascii")).decode("ascii"),
        "console_secret_key": base64.b64encode(token_urlsafe(16).replace("-", "_").encode("ascii")).decode("ascii")
    }

    return minio_configs

def generate_mlfow_configs(namespace):
    """
    Generate MLflow configuration dictionary with encoded user and password.

    Parameters
    ----------
    namespace : str
        The namespace to be encoded as the MLflow user.

    Returns
    -------
    dict
        A dictionary containing the encoded MLflow user and password.
    """
    mlfow_configs = {
        "mlflow_user": base64.b64encode(namespace.encode("ascii")).decode("ascii"),
        "mlflow_password": base64.b64encode(token_urlsafe(16).replace("-", "_").encode("ascii")).decode("ascii"),
    }

    return mlfow_configs

def generate_mysql_configs(namespace):
    """
    Generate MySQL configuration dictionary.

    Parameters
    ----------
    namespace : str
        The namespace to be used as the MySQL user.

    Returns
    -------
    dict
        A dictionary containing MySQL user and password.
    """
    mysql_configs = {
        "mysql_user": namespace,
        "mysql_password": token_urlsafe(16),
    }

    return mysql_configs

def create_maia_namespace_values(namespace_config, cluster_config, config_folder, minio_configs=None, mlflow_configs=None):
    """
    Create MAIA namespace values for deployment.

    Parameters
    ----------
    namespace_config : dict
        Configuration for the namespace, including group ID and users.
    cluster_config : dict
        Configuration for the cluster, including SSH port type, port range, and storage class.
    config_folder : str
        Path to the folder where configuration files will be saved.
    minio_configs : dict, optional
        Configuration for MinIO, including access keys and console keys. Defaults to None.
    mlflow_configs : dict, optional
        Configuration for MLflow, including user and password. Defaults to None.

    Returns
    -------
    dict
        A dictionary containing the namespace, release name, chart name, repository URL, chart version, 
        and the path to the generated values file.
    """
    ssh_ports = get_ssh_ports(cluster_config["ssh_port_type"], len(namespace_config["users"]))
    users = []

    if cluster_config["ssh_port_type"] == "LoadBalancer":
        for user in namespace_config["users"]:
            users.append({
                "jupyterhub_username": convert_username_to_jupyterhub_username(user),
                "sshPort": ssh_ports.pop(0)
            })
    else:
        for ssh_port, user in zip(ssh_ports, namespace_config["users"]):
            users.append({
                "jupyterhub_username": convert_username_to_jupyterhub_username(user),
                "sshPort": ssh_port
            })

    maia_namespace_values = {
        "pvc": {
            "pvc_type": cluster_config["shared_storage_class"],
            "access_mode": "ReadWriteMany",
            "size": "10Gi"
        },
        "chart_name": "maia-namespace", 
        "chart_version": "0.1.6", 
        "repo_url": "https://kthcloud.github.io/MAIA/", 
        "namespace": namespace_config["group_ID"].lower().replace("_", "-"),
        "serviceType": cluster_config["ssh_port_type"],
        "users": users,
        "metallbSharedIp": cluster_config.get("metallb_shared_ip", False),
        "metallbIpPool": cluster_config.get("metallb_ip_pool", False),
        "loadBalancerIp": cluster_config.get("maia_metallb_ip", False),
    }

    if "imagePullSecrets" in cluster_config:
        maia_namespace_values["dockerRegistrySecret"] = {
            "enabled": True,
            "dockerRegistrySecretName": cluster_config["imagePullSecrets"],
            "dockerRegistrySecret": encode_docker_registry_secret(
                cluster_config["docker_server"],
                cluster_config["docker_username"],
                cluster_config["docker_password"]
            )
        }

    if minio_configs:
        maia_namespace_values["minio"] = {
            "enabled": True,
            "consoleDomain": "https://{}.{}/minio-console".format(namespace_config["group_subdomain"], cluster_config["domain"]),
            "namespace": namespace_config["group_ID"].lower().replace("_", "-"),
            "storageClassName": cluster_config["storage_class"],
            "storageSize": "10Gi",
            "accessKey": minio_configs["access_key"],
            "secretKey": minio_configs["secret_key"],
            "clientId": cluster_config["keycloak"]["client_id"],
            "clientSecret": cluster_config["keycloak"]["client_secret"],
            "openIdConfigUrl": cluster_config["keycloak"]["issuer_url"] + "/.well-known/openid-configuration",
            "consoleAccessKey": minio_configs["console_access_key"],
            "consoleSecretKey": minio_configs["console_secret_key"]
        }

        if cluster_config["url_type"] == "subpath":
            maia_namespace_values["minio"]["consoleDomain"] = "https://{}/{}-minio-console".format(
                cluster_config["domain"], namespace_config["group_ID"].lower().replace("_", "-"))
 
    if mlflow_configs:
        maia_namespace_values["mlflow"] = {
            "enabled": True,
            "user": mlflow_configs["mlflow_user"],
            "password": mlflow_configs["mlflow_password"],
        }

    namespace_id = namespace_config["group_ID"].lower().replace("_", "-")
    Path(config_folder).joinpath(namespace_id, "maia_namespace_values").mkdir(parents=True, exist_ok=True)
    with open(Path(config_folder).joinpath(namespace_id, "maia_namespace_values", "namespace_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(maia_namespace_values))

    return {
        "namespace": maia_namespace_values["namespace"],
        "release": f"{namespace_id}-namespace",
        "chart": maia_namespace_values["chart_name"],
        "repo": maia_namespace_values["repo_url"],
        "version": maia_namespace_values["chart_version"],  
        "values": str(Path(config_folder).joinpath(namespace_id, "maia_namespace_values", "namespace_values.yaml"))
    }

async def get_maia_toolkit_apps(group_id, token, argo_cd_host):
    """
    Retrieve and print information about a specific project and its associated applications from Argo CD.

    Parameters
    ----------
    group_id : str
        The group identifier used to construct project and application names.
    token : str
        The authorization token for accessing the Argo CD API.
    argo_cd_host : str
        The host URL of the Argo CD server.

    Returns
    -------
    None

    Raises
    ------
    ApiException
        If there is an error when calling the Argo CD API.
    """
    configuration = argocd_client.Configuration(
        host=argo_cd_host
    )
    with argocd_client.ApiClient(configuration) as api_client:
        api_client.default_headers["Authorization"] = f"Bearer {token}"
        api_instance = argocd_client.ProjectServiceApi(api_client)
        api_instance_apps = argocd_client.ApplicationServiceApi(api_client)
        name = group_id.lower().replace("_", "-")

        try:
            api_response = api_instance.get_mixin6(name)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling ProjectServiceApi->get_mixin6: %s\n" % e)

        names = [
            f'{group_id.lower().replace("_", "-")}-namespace',
            f'{group_id.lower().replace("_", "-")}-jupyterhub',
            f'{group_id.lower().replace("_", "-")}-oauth2-proxy',
        ]
        project = [group_id.lower().replace("_", "-")]
        for name in names:
            try:
                api_response = api_instance_apps.get_mixin9(name, project=project)
                pprint(api_response)
            except ApiException as e:
                print("Exception when calling ApplicationServiceApi->get_mixin9: %s\n" % e)

async def install_maia_project(group_id, values_file, argo_cd_namespace, project_chart, project_repo=None, project_version=None):
    """
    Installs or upgrades a MAIA project using the specified Helm chart and values file.

    Parameters
    ----------
    group_id : str
        The group ID for the project. This will be used as the release name.
    values_file : str
        Path to the YAML file containing the values for the Helm chart.
    argo_cd_namespace : str
        The namespace in which to install the project.
    project_chart : str
        The name of the Helm chart to use for the project.
    project_repo : str, optional
        The repository URL where the Helm chart is located. Defaults to None.
    project_version : str, optional
        The version of the Helm chart to use. Defaults to None.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the values file does not exist.
    yaml.YAMLError
        If there is an error parsing the values file.
    Exception
        If there is an error during the installation or upgrade process.

    Example
    -------
    await install_maia_project(
        group_id="example_group",
        values_file="/path/to/values.yaml",
        argo_cd_namespace="default",
        project_chart="example_chart",
        project_repo="https://example.com/charts",
        project_version="1.0.0"
    )
    """
    client = Client(kubeconfig=os.environ["KUBECONFIG"])

    chart = await client.get_chart(
        project_chart,
        repo=project_repo,
        version=project_version
    )

    with open(values_file) as f:
        values = yaml.safe_load(f)
    
    
    

    
    revision = await client.install_or_upgrade_release(
        group_id.lower().replace("_", "-"),
        chart,
        values,
        namespace=argo_cd_namespace,
        wait=True
    )
    print(
        revision.release.name,
        revision.release.namespace,
        revision.revision,
        str(revision.status)
    )

def create_maia_admin_toolkit_values(config_folder, project_id, cluster_config_dict, maia_config_dict):
    """
    Creates and writes the MAIA admin toolkit values to a YAML file.

    Parameters
    ----------
    config_folder : str
        The path to the configuration folder.
    project_id : str
        The project identifier.
    cluster_config_dict : dict
        Dictionary containing cluster configuration values.
    maia_config_dict : dict
        Dictionary containing MAIA configuration values.

    Returns
    -------
    dict
        A dictionary containing the namespace, release name, chart name, repository URL, chart version, 
        and the path to the generated values YAML file.
    """
    admin_group_ID = maia_config_dict["admin_group_ID"]

    admin_toolkit_values = {
        "namespace": "maia-admin-toolkit",
        "repo_url": "https://kthcloud.github.io/MAIA/",
        "chart_name": "maia-admin-toolkit",
        "chart_version": "1.1.0",
    }

    admin_toolkit_values.update(
        {
            "argocd": {
                "enabled": True,
                "argocd_namespace": "argocd",
                "argocd_domain": "argocd." + cluster_config_dict["domain"],
                "keycloak_issuer_url": "https://iam." + cluster_config_dict["domain"] + "/realms/maia",
                "keycloak_client_id": "maia",
                "keycloak_client_secret": cluster_config_dict["keycloak_maia_client_secret"],
            },
            "certResolver": cluster_config_dict["traefik_resolver"],
            "admin_group_ID": admin_group_ID,
            "harbor": {
                "enabled": True,
                "values": {
                    "namespace": "harbor",
                    "storageClassName": cluster_config_dict["storage_class"],
                }
            },
            "dashboard": {
                "enabled": True,
                "dashboard_domain": "dashboard." + cluster_config_dict["domain"],
            }
        }
    )

    Path(config_folder).joinpath(project_id, "maia_admin_toolkit_values").mkdir(parents=True, exist_ok=True)

    with open(Path(config_folder).joinpath(project_id, "maia_admin_toolkit_values", "maia_admin_toolkit_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(admin_toolkit_values))

    return {
        "namespace": admin_toolkit_values["namespace"],
        "release": f"{project_id}-toolkit",
        "chart": admin_toolkit_values["chart_name"],
        "repo": admin_toolkit_values["repo_url"],
        "version": admin_toolkit_values["chart_version"],
        "values": str(Path(config_folder).joinpath(project_id, "maia_admin_toolkit_values", "maia_admin_toolkit_values.yaml"))
    }

def create_harbor_values(config_folder, project_id, cluster_config_dict):
    """
    Create and save Harbor values configuration for a given project and cluster configuration.

    Parameters
    ----------
    config_folder : str
        The path to the configuration folder where the Harbor values file will be saved.
    project_id : str
        The unique identifier for the project.
    cluster_config_dict : dict
        A dictionary containing cluster configuration details, including:
            - domain (str): The domain name for the Harbor registry.
            - ingress_class (str): The ingress class to be used (e.g., "maia-core-traefik", "nginx").
            - traefik_resolver (str, optional): The Traefik resolver to be used if ingress_class is "maia-core-traefik".

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - namespace (str): The Kubernetes namespace for Harbor.
        - release (str): The release name for the Harbor Helm chart.
        - chart (str): The name of the Harbor Helm chart.
        - repo (str): The URL of the Harbor Helm chart repository.
        - version (str): The version of the Harbor Helm chart.
        - values (str): The path to the generated Harbor values YAML file.
    """
    DOMAIN = cluster_config_dict["domain"]
    harbor_values = {
        "namespace": "harbor",
        "repo_url": "https://helm.goharbor.io",
        "chart_name": "harbor",
        "chart_version": "1.16.0",
    }

    harbor_values.update({
        "expose": {
            "type": "ingress",
            "tls": {
                "enabled": True,
            },
            "ingress": {
                "hosts": {
                    "core": f"registry.{DOMAIN}"
                },
                "annotations": {
                },
                "controller": "default",
                "className": cluster_config_dict["ingress_class"]
            }
        },
        "externalURL": f"https://registry.{DOMAIN}",
        "persistence": {
            "enabled": True,
            "resourcePolicy": "keep",
            "persistentVolumeClaim": {
                "registry": {
                    "existingClaim": "pvc-harbor",
                    "subPath": "registry",
                    "storageClass": cluster_config_dict["ingress_class"],
                    "accessMode": "ReadWriteMany"
                },
                "jobservice": {
                    "jobLog": {
                        "existingClaim": "pvc-harbor",
                        "subPath": "job_logs",
                        "storageClass": cluster_config_dict["ingress_class"],
                        "accessMode": "ReadWriteMany"
                    }
                },
                "database": {
                    "existingClaim": "pvc-harbor",
                    "subPath": "database",
                    "storageClass": cluster_config_dict["ingress_class"],
                    "accessMode": "ReadWriteMany"
                },
                "redis": {
                    "existingClaim": "pvc-harbor",
                    "subPath": "redis",
                    "storageClass": cluster_config_dict["ingress_class"],
                    "accessMode": "ReadWriteMany"
                },
                "trivy": {
                    "existingClaim": "pvc-harbor",
                    "subPath": "trivy",
                    "storageClass": cluster_config_dict["ingress_class"],
                    "accessMode": "ReadWriteMany"
                }
            },
            "imageChartStorage": {
                "type": "filesystem"
            }
        },
        "database": {
            "internal": {
                "password": "harbor"
            }
        },
        "metrics": {
            "enabled": True,
            "core": {
                "path": "/metrics",
                "port": 8001
            },
            "registry": {
                "path": "/metrics",
                "port": 8001
            },
            "jobservice": {
                "path": "/metrics",
                "port": 8001
            },
            "exporter": {
                "path": "/metrics",
                "port": 8001
            }
        }
    })

    if cluster_config_dict["ingress_class"] == "maia-core-traefik":
        harbor_values["expose"]["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.entrypoints"] = "websecure"
        harbor_values["expose"]["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls"] = 'true'
        harbor_values["expose"]["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls.certresolver"] = cluster_config_dict["traefik_resolver"]
    elif cluster_config_dict["ingress_class"] == "nginx":
        harbor_values["expose"]["ingress"]["annotations"]["cert-manager.io/cluster-issuer"] = "cluster-issuer"

    Path(config_folder).joinpath(project_id, "harbor_values").mkdir(parents=True, exist_ok=True)
    with open(Path(config_folder).joinpath(project_id, "harbor_values", "harbor_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(harbor_values))

    return {
        "namespace": harbor_values["namespace"],
        "release": f"{project_id}-harbor",
        "chart": harbor_values["chart_name"],
        "repo": harbor_values["repo_url"],
        "version": harbor_values["chart_version"],
        "values": str(Path(config_folder).joinpath(project_id, "harbor_values", "harbor_values.yaml"))
    }

def create_keycloak_values(config_folder, project_id, cluster_config_dict):
    """
    Generates Keycloak Helm chart values and writes them to a YAML file.

    Parameters
    ----------
    config_folder : str
        The path to the configuration folder where the YAML file will be saved.
    project_id : str
        The project identifier used to create a unique namespace and release name.
    cluster_config_dict : dict
        A dictionary containing cluster configuration details such as domain, ingress class, and traefik resolver.

    Returns
    -------
    dict
        A dictionary containing the namespace, release name, chart name, repository URL, chart version, 
        and the path to the generated values YAML file.
    """
    keycloak_values = {
        "namespace": "keycloak",
        "repo_url": "https://charts.bitnami.com/bitnami",
        "chart_name": "keycloak",
        "chart_version": "24.2.0",
    }

    keycloak_values.update({
        "extraEnvVars": [
            {
                "name": "KEYCLOAK_EXTRA_ARGS",
                "value": "--import-realm"
            },
            {
                "name": "PROXY_ADDRESS_FORWARDING",
                "value": "true"
            },
            {
                "name": "KEYCLOAK_HOSTNAME",
                "value": "iam." + cluster_config_dict["domain"]
            },
        ],
        "proxy": "edge",
        "ingress": {
            "enabled": True,
            "tls": True,
            "ingressClassName": cluster_config_dict["ingress_class"],
            "hostname": "iam." + cluster_config_dict["domain"],
            "annotations": {},
        },
        "extraVolumeMounts": [
            {
                "name": "keycloak-import",
                "mountPath": "/opt/bitnami/keycloak/data/import"
            }
        ],
        "extraVolumes": [
            {
                "name": "keycloak-import",
                "configMap": {
                    "name": "maia-realm-import"
                }
            }
        ]
    })

    if cluster_config_dict["ingress_class"] == "maia-core-traefik":
        keycloak_values["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.entrypoints"] = "websecure"
        keycloak_values["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls"] = 'true'
        keycloak_values["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls.certresolver"] = cluster_config_dict["traefik_resolver"]
    elif cluster_config_dict["ingress_class"] == "nginx":
        keycloak_values["ingress"]["annotations"]["cert-manager.io/cluster-issuer"] = "cluster-issuer"

    Path(config_folder).joinpath(project_id, "keycloak_values").mkdir(parents=True, exist_ok=True)
    with open(Path(config_folder).joinpath(project_id, "keycloak_values", "keycloak_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(keycloak_values))

    return {
        "namespace": keycloak_values["namespace"],
        "release": f"{project_id}-keycloak",
        "chart": keycloak_values["chart_name"],
        "repo": keycloak_values["repo_url"],
        "version": keycloak_values["chart_version"],
        "values": str(Path(config_folder).joinpath(project_id, "keycloak_values", "keycloak_values.yaml"))
    }

def create_loginapp_values(config_folder, project_id, cluster_config_dict):
    """
    Creates and writes the loginapp values configuration file for a given project and cluster configuration.

    Parameters
    ----------
    config_folder : str
        The base directory where the configuration files will be stored.
    project_id : str
        The unique identifier for the project.
    cluster_config_dict : dict
        A dictionary containing cluster configuration details, including:
            - keycloak_maia_client_secret (str): The client secret for Keycloak.
            - domain (str): The domain name for the cluster.
            - ingress_class (str): The ingress class to be used (e.g., "maia-core-traefik" or "nginx").
            - traefik_resolver (str, optional): The Traefik resolver to be used if ingress_class is "maia-core-traefik".

    Returns
    -------
    dict
        A dictionary containing the namespace, release name, chart name, repository URL, chart version, 
        and the path to the generated values file.

    Raises
    ------
    KeyError
        If required keys are missing from the cluster_config_dict.
    OSError
        If there is an error creating directories or writing the configuration file.
    """
    loginapp_values = {
        "namespace": "authentication",
        "repo_url": "https://storage.googleapis.com/loginapp-releases/charts/",
        "chart_name": "loginapp",
        "chart_version": "1.3.0",
    }

    secret = token_urlsafe(16).replace("-", "_")
    client_id = "maia"
    client_secret = cluster_config_dict["keycloak_maia_client_secret"]
    issuer_url = "https://iam." + cluster_config_dict["domain"] + "/realms/maia"
    cluster_server_address = "https://" + cluster_config_dict["domain"] + ":16443"
    ca_file = "/var/snap/microk8s/current/certs/ca.crt"

    loginapp_values.update({
        "env": {
            "LOGINAPP_NAME": "MAIA Login"
        },
        "configOverwrites": {
            "oidc": {
                "scopes": ["openid", "profile", "email"]
            },
            "service": {
                "type": "ClusterIP"
            }
        },
        "ingress": {
            "enabled": True,
            "annotations": {},
            "tls": [
                {
                    "hosts": [
                        "login." + cluster_config_dict["domain"]
                    ]
                }
            ],
            "hosts": [
                {
                    "host": "login." + cluster_config_dict["domain"],
                    "paths": [
                        {
                            "path": "/",
                            "pathType": "Prefix"
                        }
                    ]
                }
            ]
        },
        "config": {
            "tls": {
                "enabled": False
            },
            "issuerInsecureSkipVerify": True,
            "refreshToken": True,
            "clientRedirectURL": f"https://login." + cluster_config_dict["domain"] + "/callback",
            "secret": secret,
            "clientID": client_id,
            "clientSecret": client_secret,
            "issuerURL": issuer_url,
            "clusters": [
                {
                    "server": cluster_server_address,
                    "name": "MAIA",
                    "insecure-skip-tls-verify": True,
                    "certificate-authority": ca_file
                }
            ]
        }
    })

    if cluster_config_dict["ingress_class"] == "maia-core-traefik":
        loginapp_values["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.entrypoints"] = "websecure"
        loginapp_values["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls"] = 'true'
        loginapp_values["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls.certresolver"] = cluster_config_dict["traefik_resolver"]
    elif cluster_config_dict["ingress_class"] == "nginx":
        loginapp_values["ingress"]["annotations"]["cert-manager.io/cluster-issuer"] = "cluster-issuer"
        loginapp_values["ingress"]["tls"][0]["secretName"] = "loginapp." + cluster_config_dict["domain"]

    Path(config_folder).joinpath(project_id, "loginapp_values").mkdir(parents=True, exist_ok=True)
    with open(Path(config_folder).joinpath(project_id, "loginapp_values", "loginapp_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(loginapp_values))

    return {
        "namespace": loginapp_values["namespace"],
        "release": f"{project_id}-loginapp",
        "chart": loginapp_values["chart_name"],
        "repo": loginapp_values["repo_url"],
        "version": loginapp_values["chart_version"],
        "values": str(Path(config_folder).joinpath(project_id, "loginapp_values", "loginapp_values.yaml"))
    }

def create_minio_operator_values(config_folder, project_id, cluster_config_dict):
    """
    Creates and writes MinIO operator values to a YAML file and returns a dictionary with deployment details.

    Parameters
    ----------
    config_folder : str
        The path to the configuration folder.
    project_id : str
        The unique identifier for the project.
    cluster_config_dict : dict
        A dictionary containing cluster configuration details.

    Returns
    -------
    dict
        A dictionary containing the namespace, release name, chart name, repository URL, chart version, 
        and the path to the generated YAML values file.
    """
    minio_operator_values = {
        "namespace": "minio-operator",
        "repo_url": "https://operator.min.io",
        "chart_name": "operator",
        "chart_version": "6.0.4",
    }

    Path(config_folder).joinpath(project_id, "minio_operator_values").mkdir(parents=True, exist_ok=True)
    with open(Path(config_folder).joinpath(project_id, "minio_operator_values", "minio_operator_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(minio_operator_values))

    return {
        "namespace": minio_operator_values["namespace"],
        "release": f"{project_id}-minio-operator",
        "chart": minio_operator_values["chart_name"],
        "repo": minio_operator_values["repo_url"],
        "version": minio_operator_values["chart_version"],
        "values": str(Path(config_folder).joinpath(project_id, "minio_operator_values", "minio_operator_values.yaml"))
    }
