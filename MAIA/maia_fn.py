import base64
import json
import os
import subprocess
from pathlib import Path
from pprint import pprint
from secrets import token_urlsafe
from typing import Dict
from omegaconf import OmegaConf
import kubernetes
import toml
import yaml
from MAIA.helm_values import read_config_dict_and_generate_helm_values_dict
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import random
import string
import nltk
from nltk.corpus import words

def generate_random_password(length=12):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))

def generate_human_memorable_password(length=12):
    nltk.download('words')
    word_list = words.words()
    password = '-'.join(random.choice(word_list) for _ in range(length // 6))
    password += ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length - len(password)))
    return password

def create_config_map_from_data(data: str, config_map_name: str, namespace: str, kubeconfig_dict: Dict,
                                data_key: str = "values.yaml"):
    """
    Create a ConfigMap on a Kubernetes Cluster.

    Parameters
    ----------
    data : str
        String containing the content of the ConfigMap to dump.
    config_map_name : str
        ConfigMap name.
    namespace : str
        Namespace where to create the ConfigMap.
    data_key : str, optional
        Value to use as the filename for the content in the ConfigMap.
    kubeconfig_dict : dict
        Kube Configuration dictionary for Kubernetes cluster authentication.
    """
    config.load_kube_config_from_dict(kubeconfig_dict)
    metadata = kubernetes.client.V1ObjectMeta(
        name=config_map_name,
        namespace=namespace,
    )

    if type(data_key) == list and type(data) == list:
        configmap = kubernetes.client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            data={data_key[i]: data[i] for i in range(len(data))},
            metadata=metadata
        )
    else:
        configmap = kubernetes.client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            data={data_key: data},
            metadata=metadata
        )

    with kubernetes.client.ApiClient() as api_client:
        api_instance = kubernetes.client.CoreV1Api(api_client)

        pretty = 'true'
        try:
            api_response = api_instance.create_namespaced_config_map(namespace, configmap, pretty=pretty)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CoreV1Api->delete_namespaced_config_map: %s\n" % e)

def get_ssh_port_dict(port_type,namespace,port_range, maia_metallb_ip=None):
    """
    Retrieve a dictionary of used SSH ports for services in a Kubernetes cluster.

    Parameters
    ----------
    port_type : str
        The type of port to check ('LoadBalancer' or 'NodePort').
    namespace : str
        The namespace to filter services by.
    port_range : tuple
        A tuple specifying the range of ports to check (start, end).
    maia_metallb_ip : str, optional
        The IP address of the MetalLB load balancer (default is None).

    Returns
    -------
    list of dict
        A list of dictionaries with service names as keys and their corresponding used SSH ports as values.
        Returns None if an exception occurs.
    """
    if not "KUBECONFIG_LOCAL" in os.environ:
        os.environ["KUBECONFIG_LOCAL"] = os.environ["KUBECONFIG"]
    kubeconfig = yaml.safe_load(Path(os.environ["KUBECONFIG_LOCAL"]).read_text())
    config.load_kube_config_from_dict(kubeconfig)

    v1 = client.CoreV1Api()

    try:
        used_port = []
        services = v1.list_service_for_all_namespaces(watch=False)
        for svc in services.items:
            if port_type == 'LoadBalancer':
                if svc.status.load_balancer.ingress is not None:
                    if svc.spec.type == 'LoadBalancer' and svc.status.load_balancer.ingress[0].ip == maia_metallb_ip:
                        for port in svc.spec.ports:
                            if port.name == 'ssh' and svc.metadata.namespace == namespace or port.name == 'orthanc-dicom' and svc.metadata.namespace == namespace:
                                used_port.append({svc.metadata.name:int(port.port)})
            elif port_type == "NodePort":
                if svc.spec.type == 'NodePort' and svc.metadata.namespace == namespace:
                    for port in svc.spec.ports:
                        if port.port >= port_range[0] and port.port <= port_range[1]:
                            used_port.append({svc.metadata.name:int(port.port)})
        return used_port
    except:
        return None

def get_ssh_ports(n_requested_ports, port_type, ip_range, maia_metallb_ip=None):
    """
    Retrieve a list of available SSH ports based on the specified criteria.

    Parameters
    ----------
    n_requested_ports : int
        The number of SSH ports requested.
    port_type : str
        The type of port to search for ('LoadBalancer' or 'NodePort').
    ip_range : tuple
        A tuple specifying the range of IPs to search within (start, end).
    maia_metallb_ip : str, optional
        The specific IP address to match for 'LoadBalancer' type. Defaults to None.

    Returns
    -------
    list
        A list of available SSH ports that meet the specified criteria.
    None
        If an error occurs during the process.
    """
    if not "KUBECONFIG_LOCAL" in os.environ:
        os.environ["KUBECONFIG_LOCAL"] = os.environ["KUBECONFIG"]
    kubeconfig = yaml.safe_load(Path(os.environ["KUBECONFIG_LOCAL"]).read_text())
    config.load_kube_config_from_dict(kubeconfig)

    v1 = client.CoreV1Api()

    print(v1.list_namespace(watch=False))

    try:
        used_port = []
        services = v1.list_service_for_all_namespaces(watch=False)
        for svc in services.items:
            if port_type == 'LoadBalancer':
                if svc.status.load_balancer.ingress is not None:
                    if svc.spec.type == 'LoadBalancer' and svc.status.load_balancer.ingress[0].ip == maia_metallb_ip:
                        for port in svc.spec.ports:
                            if port.name == 'ssh' or port.name == 'orthanc-dicom':
                                used_port.append(int(port.port))
            elif port_type == "NodePort":
                if svc.spec.type == 'NodePort':
                    for port in svc.spec.ports:
                        used_port.append(int(port.port))

        ports = []

        for request in range(n_requested_ports):
            for port in range(ip_range[0], ip_range[1]):
                if port not in used_port:
                    ports.append(port)
                    used_port.append(port)
                    break

        return ports
    except:
        return None


def convert_username_to_jupyterhub_username(username):
    """
    Convert a username to a JupyterHub-compatible username.

    Parameters
    ----------
    username : str
        The original username.

    Returns
    -------
    str
        The JupyterHub-compatible username.
    """
    return username.replace("-", "-2d").replace("@", "-40").replace(".", "-2e")


def encode_docker_registry_secret(docker_server, docker_username, docker_password):
    """
    Encode Docker registry credentials into a base64-encoded string.

    Parameters
    ----------
    docker_server : str
        The Docker registry server.
    docker_username : str
        The Docker registry username.
    docker_password : str
        The Docker registry password.

    Returns
    -------
    str
        The base64-encoded Docker registry credentials.
    """
    auth = base64.b64encode(f"{docker_username}:{docker_password}".encode("utf-8")).decode("utf-8")
    return base64.b64encode(json.dumps({
        "auths": {
            docker_server: {
                "username": docker_username,
                "password": docker_password,
                "auth": auth,
            }
        }
    }).encode("utf-8")).decode("utf-8")


def deploy_oauth2_proxy(cluster_config, user_config, config_folder=None):
    """
    Deploy an OAuth2 Proxy using the provided cluster and user configurations.

    Parameters
    ----------
    cluster_config : dict
        Configuration dictionary for the cluster. Expected keys include:
            - "keycloak": A dictionary with "issuer_url", "client_id", and "client_secret".
            - "domain": The domain name for the cluster.
            - "url_type": The type of URL, either "subpath" or other.
            - "storage_class": The storage class for Redis.
            - "nginx_cluster_issuer" (optional): The cluster issuer for NGINX.
            - "traefik_resolver" (optional): The resolver for Traefik.
    user_config : dict
        Configuration dictionary for the user. Expected keys include:
            - "group_ID": The group ID for the user.
            - "group_subdomain": The subdomain for the user's group.
    config_folder : str, optional
        The folder path where the configuration files will be saved. Defaults to None.

    Returns
    -------
    dict
        A dictionary containing deployment details:
            - "namespace": The namespace for the deployment.
            - "release": The release name for the deployment.
            - "chart": The chart name for the deployment.
            - "repo": The repository URL for the chart.
            - "version": The chart version.
            - "values": The path to the generated values YAML file.
    """
    config_file = {
        "oidc_issuer_url": cluster_config["keycloak"]["issuer_url"],
        "provider": "oidc",
        "upstreams": ["static://202"],
        "http_address": "0.0.0.0:4180",
        "oidc_groups_claim": "groups",
        "skip_jwt_bearer_tokens": True,
        "oidc_email_claim": "email",
        "allowed_groups": ["MAIA:" + user_config["group_ID"], "MAIA:admin"],
        "scope": "openid email profile",
        "redirect_url": "https://{}.{}/oauth2/callback".format(user_config["group_subdomain"], cluster_config["domain"]),
        "email_domains": ["*"],
        "proxy_prefix": "/oauth2",
        "ssl_insecure_skip_verify": True,
        "insecure_oidc_skip_issuer_verification": True,
        "cookie_secure": True,
        "reverse_proxy": True,
        "pass_access_token": True,
        "pass_authorization_header": True,
        "set_authorization_header": True,
        "set_xauthrequest": True,
        "pass_user_headers": True,
        "whitelist_domains": ["*"]
    }

    if cluster_config["url_type"] == "subpath":
        config_file["redirect_url"] = "https://{}/oauth2-{}/callback".format(cluster_config["domain"], user_config["group_subdomain"])
        config_file["proxy_prefix"] = "/oauth2-{}".format(user_config["group_subdomain"])

    oauth2_proxy_config = {
        "config": {
            "clientID": cluster_config["keycloak"]["client_id"],
            "clientSecret": cluster_config["keycloak"]["client_secret"],
            "cookieSecret": token_urlsafe(16),
            "configFile": toml.dumps(config_file)
        },
        "redis": {
            "enabled": True,
            "global": {
                "storageClass": cluster_config["storage_class"]
            }
        },
        "sessionStorage": {
            "type": "redis",
        },
        "image": {
            "repository": "quay.io/oauth2-proxy/oauth2-proxy",
            "tag": "",
            "pullPolicy": "IfNotPresent"
        },
        "service": {
            "type": "ClusterIP",
            "portNumber": 80,
            "appProtocol": "https",
            "annotations": {}
        },
        "serviceAccount": {
            "enabled": True,
            "name": "",
            "automountServiceAccountToken": True,
            "annotations": {}
        },
        "ingress": {
            "enabled": True,
            "path": "/oauth2",
            "pathType": "Prefix",
            "tls": [{"secretName": "{}.{}-tls".format(user_config["group_subdomain"], cluster_config["domain"]), "hosts": ["{}.{}".format(user_config["group_subdomain"], cluster_config["domain"])]}],
            "hosts": ["{}.{}".format(user_config["group_subdomain"], cluster_config["domain"])],
            "annotations": {}
        }
    }

    if cluster_config["url_type"] == "subpath":
        oauth2_proxy_config["ingress"]["hosts"] = [cluster_config["domain"]]
        oauth2_proxy_config["ingress"]["tls"][0]["hosts"] = [cluster_config["domain"]]
        oauth2_proxy_config["ingress"]["path"] = "/oauth2-{}".format(user_config["group_subdomain"])
    if "nginx_cluster_issuer" in cluster_config:
        oauth2_proxy_config["ingress"]["annotations"]["cert-manager.io/cluster-issuer"] = cluster_config["nginx_cluster_issuer"]
    if "traefik_resolver" in cluster_config:
        oauth2_proxy_config["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.entrypoints"] = "websecure"
        oauth2_proxy_config["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls"] = 'true'
        oauth2_proxy_config["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls.certresolver"] = cluster_config["traefik_resolver"]

    oauth2_proxy_config["chart_name"] = "oauth2-proxy"
    oauth2_proxy_config["chart_version"] = "7.7.8"
    oauth2_proxy_config["repo_url"] = "https://oauth2-proxy.github.io/manifests"  # TODO: Change this to updated values

    Path(config_folder).joinpath(user_config["group_ID"], "oauth2_proxy_values").mkdir(parents=True, exist_ok=True)

    with open(Path(config_folder).joinpath(user_config["group_ID"], "oauth2_proxy_values", "oauth2_proxy_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(oauth2_proxy_config))

    return {
        "namespace": user_config["group_ID"].lower().replace("_", "-"),
        "release": user_config["group_ID"].lower().replace("_", "-") + "-oauth2-proxy",
        "chart": oauth2_proxy_config["chart_name"],
        "repo": oauth2_proxy_config["repo_url"],
        "version": oauth2_proxy_config["chart_version"],
        "values": str(Path(config_folder).joinpath(user_config["group_ID"], "oauth2_proxy_values", "oauth2_proxy_values.yaml"))
    }


def deploy_mysql(cluster_config, user_config, config_folder, mysql_configs):
    """
    Deploy a MySQL instance on a Kubernetes cluster using Helm.

    Parameters
    ----------
    cluster_config : dict
        Configuration dictionary for the cluster, including storage class.
    user_config : dict
        Configuration dictionary for the user, including group ID.
    config_folder : str
        Path to the folder where configuration files will be stored.
    mysql_configs : dict
        Configuration dictionary for MySQL, including user, password, and other settings.

    Returns
    -------
    dict
        A dictionary containing deployment details such as namespace, release name, chart name, repository URL, version, and values file path.
    """
    namespace = user_config["group_ID"].lower().replace("_", "-")
    if not "KUBECONFIG_LOCAL" in os.environ:
        os.environ["KUBECONFIG_LOCAL"] = os.environ["KUBECONFIG"]
    kubeconfig = yaml.safe_load(Path(os.environ["KUBECONFIG_LOCAL"]).read_text())

    mysql_config = {
        "namespace": namespace,
        "chart_name": "mysql-db-v1",
        "docker_image": "mysql",
        "tag": "8.0.28",
        "memory_request": "2Gi",
        "cpu_request": "500m",
        "deployment": True,
        "ports": {
            "mysql": [
                3306
            ]
        },
        "persistent_volume": [
            {
                "mountPath": "/var/lib/mysql",
                "size": "20Gi",
                "access_mode": "ReadWriteMany",
                "pvc_type": cluster_config["storage_class"]
            }
        ],
        "env_variables": {
            "MYSQL_ROOT_PASSWORD": mysql_configs.get("mysql_password", "root"),
            "MYSQL_USER": mysql_configs.get("mysql_user", "root"),
            "MYSQL_PASSWORD": mysql_configs.get("mysql_password", "root"),
            "MYSQL_DATABASE": "mysql"
        }
    }  # TODO: Change this to updated values

    mysql_values = read_config_dict_and_generate_helm_values_dict(mysql_config, kubeconfig)

    mysql_values["chart_name"] = "mkg"
    mysql_values["chart_version"] = "1.0.2"
    mysql_values["repo_url"] = "https://kthcloud.github.io/MAIA/"  # TODO: Change this to updated values

    Path(config_folder).joinpath(user_config["group_ID"], "mysql_values").mkdir(parents=True, exist_ok=True)

    with open(Path(config_folder).joinpath(user_config["group_ID"], "mysql_values", "mysql_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(mysql_values))

    return {
        "namespace": user_config["group_ID"].lower().replace("_", "-"),
        "release": user_config["group_ID"].lower().replace("_", "-") + "-mysql",
        "chart": mysql_values["chart_name"],
        "repo": mysql_values["repo_url"],
        "version": mysql_values["chart_version"],
        "values": str(Path(config_folder).joinpath(user_config["group_ID"], "mysql_values", "mysql_values.yaml"))
    }


def deploy_mlflow(cluster_config, user_config, config_folder, mysql_config=None, minio_config=None):
    """
    Deploy an MLflow instance on a Kubernetes cluster using Helm.

    Parameters
    ----------
    cluster_config : dict
        Configuration dictionary for the Kubernetes cluster.
    user_config : dict
        Configuration dictionary for the user, including group_ID.
    config_folder : str
        Path to the folder where configuration files will be stored.
    mysql_config : dict, optional
        Configuration dictionary for MySQL, including mysql_user and mysql_password. Defaults to None.
    minio_config : dict, optional
        Configuration dictionary for MinIO, including console_access_key and console_secret_key. Defaults to None.

    Returns
    -------
    dict
        A dictionary containing deployment details such as namespace, release name, chart name, repository URL, chart version, and path to the values file.
    """
    namespace = user_config["group_ID"].lower().replace("_", "-")
    if not "KUBECONFIG_LOCAL" in os.environ:
        os.environ["KUBECONFIG_LOCAL"] = os.environ["KUBECONFIG"]
    kubeconfig = yaml.safe_load(Path(os.environ["KUBECONFIG_LOCAL"]).read_text())
    config.load_kube_config_from_dict(kubeconfig)

    mlflow_config = {
        "namespace": namespace,
        "chart_name": "mlflow-v1",
        "docker_image": "kthcloud/mlflow",  # TODO: Make this configurable
        "tag": "1.1",
        "memory_request": "2Gi",
        "cpu_request": "500m",
        "allocationTime": "180d",
        "ports": {
            "mlflow": [
                5000
            ]
        },
        "user_secret": [
            namespace
        ],
        "user_secret_params": [
            "user",
            "password"
        ],
        "env_variables": {
            "MYSQL_URL": "{}-mysql-mkg".format(namespace),
            "MYSQL_PASSWORD": mysql_config.get("mysql_password", "root"),
            "MYSQL_USER": mysql_config.get("mysql_user", "root"),
            "BUCKET_NAME": "mlflow",
            "BUCKET_PATH": "mlflow",
            "AWS_ACCESS_KEY_ID": minio_config.get("console_access_key", "minio"),
            "AWS_SECRET_ACCESS_KEY": minio_config.get("console_secret_key", "minio"),
            "MLFLOW_S3_ENDPOINT_URL": "http://minio:80"
        }
    }  # TODO: Change this to updated values
    if "imagePullSecrets" in cluster_config:
        mlflow_config["image_pull_secret"] = cluster_config["imagePullSecrets"]

    mlflow_values = read_config_dict_and_generate_helm_values_dict(mlflow_config, kubeconfig)

    mlflow_values["chart_name"] = "mkg"
    mlflow_values["chart_version"] = "1.0.2"
    mlflow_values["repo_url"] = "https://kthcloud.github.io/MAIA/"  # TODO: Change this to updated values

    Path(config_folder).joinpath(user_config["group_ID"], "mlflow_values").mkdir(parents=True, exist_ok=True)

    with open(Path(config_folder).joinpath(user_config["group_ID"], "mlflow_values", "mlflow_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(mlflow_values))

    return {
        "namespace": user_config["group_ID"].lower().replace("_", "-"),
        "release": user_config["group_ID"].lower().replace("_", "-") + "-mlflow",
        "chart": mlflow_values["chart_name"],
        "repo": mlflow_values["repo_url"],
        "version": mlflow_values["chart_version"],
        "values": str(Path(config_folder).joinpath(user_config["group_ID"], "mlflow_values", "mlflow_values.yaml"))
    }


def deploy_orthanc(cluster_config, user_config, maia_config_dict, config_folder):
    """
    Deploys Orthanc using the provided configuration.
    Parameters
    ----------
    cluster_config : dict
        Dictionary containing the cluster configuration.
    user_config : dict
        Dictionary containing the user configuration.
    maia_config_dict : dict
        Dictionary containing the MAIA configuration.
    config_folder : str or Path
        Path to the configuration folder.
    Returns
    -------
    dict
        A dictionary containing deployment details such as namespace, release, chart, repo, version, and values file path.
    """

    with open(Path(config_folder).joinpath(user_config["group_ID"], "maia_namespace_values", "namespace_values.yaml"), "r") as f:
        namespace_values = yaml.safe_load(f)
        orthanc_port = namespace_values["orthanc"]["port"]

    random_path = generate_random_password(16)
    orthanc_config = {
        "pvc": {
            "pvc_type": cluster_config["storage_class"],
            "access_mode": "ReadWriteMany",
            "size": "10Gi"
        },
        "imagePullSecret": cluster_config["imagePullSecrets"],
        "image": {
            "repository": maia_config_dict["maia_orthanc_image"],
            "tag": maia_config_dict["maia_orthanc_version"],
        },
        "cpu": "1000m",
        "memory": "1Gi",
        "gpu": False,
        "orthanc_dicom_service_annotations": {},
        "ingress_annotations": {},
        "ingress_tls": {
            "host": ""
        },
        "monai_label_path": f"monai-label-{random_path}",
        "orthanc_path": f"orthanc-{random_path}",
        "orthanc_node_port": orthanc_port,
        "serviceType": "NodePort"
    }

    domain = cluster_config["domain"]
    group_subdomain = user_config["group_subdomain"]

    if "url_type" in cluster_config:
        if cluster_config["url_type"] == "subdomain":
            orthanc_address = f"{group_subdomain}.{domain}"
        elif cluster_config["url_type"] == "subpath":
            orthanc_address = domain
        else:
            orthanc_address = None

    if orthanc_address is not None:
        orthanc_config["ingress_tls"]["host"] = orthanc_address

    if cluster_config["ssh_port_type"] == "LoadBalancer":
        orthanc_config["orthanc_dicom_service_annotations"]["metallb.universe.tf/allow-shared-ip"] = cluster_config.get("metallb_shared_ip", False) 
        orthanc_config["orthanc_dicom_service_annotations"]["metallb.universe.tf/ip-allocated-from-pool"] = cluster_config.get("metallb_ip_pool", False)
        orthanc_config["orthanc_node_port"] =  {"loadBalancer": orthanc_port}
        orthanc_config["loadBalancerIp"] = cluster_config.get("maia_metallb_ip", False)

    if cluster_config["ingress_class"] == "maia-core-traefik":
        orthanc_config["ingress_annotations"]["traefik.ingress.kubernetes.io/router.entrypoints"] = "websecure"
        orthanc_config["ingress_annotations"]["traefik.ingress.kubernetes.io/router.tls"] = 'true'
        orthanc_config["ingress_annotations"]["traefik.ingress.kubernetes.io/router.tls.certresolver"] = cluster_config["traefik_resolver"]
    elif cluster_config["ingress_class"] == "nginx":
        orthanc_config["ingress_annotations"]["cert-manager.io/cluster-issuer"] = "cluster-issuer"


    orthanc_config["chart_name"] = "maia-orthanc"
    orthanc_config["chart_version"] = "0.0.3"
    orthanc_config["repo_url"] = "https://kthcloud.github.io/MAIA/"

    Path(config_folder).joinpath(user_config["group_ID"], "orthanc_values").mkdir(parents=True, exist_ok=True)

    with open(Path(config_folder).joinpath(user_config["group_ID"], "orthanc_values", "orthanc_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(orthanc_config))

    return {
        "namespace": user_config["group_ID"].lower().replace("_", "-"),
        "release": user_config["group_ID"].lower().replace("_", "-") + "-oauth2-proxy",
        "chart": orthanc_config["chart_name"],
        "repo": orthanc_config["repo_url"],
        "version": orthanc_config["chart_version"],
        "values": str(Path(config_folder).joinpath(user_config["group_ID"], "orthanc_values", "orthanc_values.yaml"))
    }



def gpu_list_from_nodes():
    """
    Retrieves a list of GPUs from the nodes in a Kubernetes cluster.

    This function loads the Kubernetes configuration from the environment,
    initializes the Kubernetes client, and retrieves the list of nodes.
    It then checks each node to see if it is ready and has GPU labels,
    and constructs a dictionary with the node names as keys and a list
    containing the GPU product and count as values.

    Returns
    -------
    dict
        A dictionary where the keys are node names and the values are lists
        containing the GPU product and GPU count.
    """

    kubeconfig = yaml.safe_load(Path(os.environ["KUBECONFIG"]).read_text())
    config.load_kube_config_from_dict(kubeconfig)

    v1 = client.CoreV1Api()


    nodes = v1.list_node(watch=False)
    gpu_dict = {}
    for node in nodes.items:
        for status in node.status.conditions:
            if status.type == "Ready" and status.status == "True":
                if "nvidia.com/gpu.product" in node.metadata.labels:
                    gpu_dict[node.metadata.name] = [ node.metadata.labels["nvidia.com/gpu.product"], node.metadata.labels["nvidia.com/gpu.count"] ] 
    return gpu_dict