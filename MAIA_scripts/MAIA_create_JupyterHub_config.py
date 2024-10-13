#!/usr/bin/env python

import json
from pathlib import Path

import click
import yaml
from minio import Minio


@click.command()
@click.option("--form", type=str)
@click.option("--cluster-config-file", type=str)
def create_jupyterhub_config(form,cluster_config_file):
    create_jupyterhub_config_api(form,cluster_config_file)

def create_jupyterhub_config_api( form,
                              cluster_config_file,
                                  config_folder = None
                             ):


    with open(cluster_config_file, "r") as f:
        cluster_config = yaml.safe_load(f)

    with open(form, "r") as f:
        user_form = yaml.safe_load(f)

    storage_class = cluster_config["storage_class"]

    traefik_resolver = None
    if "traefik_resolver" in cluster_config:
        traefik_resolver = cluster_config["traefik_resolver"]

    nginx_cluster_issuer = None
    if "nginx_cluster_issuer" in cluster_config:
        nginx_cluster_issuer = cluster_config["nginx_cluster_issuer"]

    hub_storage_class = None
    if "hub_storage_class" in cluster_config:
        hub_storage_class = cluster_config["hub_storage_class"]

    hub_image = None
    hub_tag = None
    if "hub_image" in cluster_config:
        hub_image = cluster_config["hub_image"]
        hub_tag = cluster_config["hub_tag"]

    base_url = None
    if "base_url" in cluster_config:
        base_url = cluster_config["base_url"]

    keycloack = None
    if "keycloack" in cluster_config:
        keycloack = cluster_config["keycloack"]

    namespace = user_form["group_ID"]
    group_subdomain = user_form["group_subdomain"]
    team_id = user_form["group_ID"]
    resources_limits = user_form["resources_limits"]

    gpu_request = False
    if "gpu_request" in user_form:
        gpu_request = user_form["gpu_request"]

    domain = cluster_config["domain"]

    if "url_type" in cluster_config:
        if cluster_config["url_type"] == "subdomain":
            hub_address = f"{group_subdomain}.{domain}"
        elif cluster_config["url_type"] == "subpath":
            hub_address = domain
        else:
            hub_address = None
    admins = cluster_config["admins"]


    ## Used for CIFS mount
    extra_host_volumes = [

    ]


    extra_volumes = [
       {
            "name": "jupyterhub-shared",
            "pvc-name": "shared",
            "mount-path": "/home/maia-user/shared"
        }
    ]

    jh_helm_template = {
      "resource": {
        "helm_release": {
          "jupyterhub": {
            "name": "jupyterhub-{}".format(namespace.lower()),
            "repository": "https://hub.jupyter.org/helm-chart/",
            "chart": "jupyterhub",
            "version": "3.1.0",
            "namespace": namespace.lower(),
            "create_namespace": False

          }
        }
      }
    }

    jh_template = {
        "cull": {
            "enabled": False
        },
        "ingress":  {
            "enabled": True,
            "hosts": [
                hub_address
            ],
            "annotations": {

            },
            "tls": [
                {
                    "hosts": [
                        hub_address
                    ]

                }
            ]
        },
        "hub":{
            # activeServerLimit
            # concurrentSpawnLimit
        #"loadRoles": {
       #
        #    "user": {
        #    "description": "Allow users to access the shared server in addition to default perms",
        #    "scopes": ["self", "access:servers!user=<USER_EMAIL>"],
        #        "users": [""],
        #        "groups": [""]
        #}
        #                        },
            "config":{
                "GenericOAuthenticator":{

                    "login_service": "MAIA Account",
                    "username_claim": "preferred_username",
                    "scope": [
                        "openid",
                        "profile",
                        "email"
                    ],
                    "userdata_params": {
                        "state": "state"
                    },
                    "claim_groups_key": "groups",
                    "allowed_groups": [
                        f"MAIA:{team_id}"
                    ],
                    "admin_groups": [
                        "MAIA:admin"
                    ],

                },
                "JupyterHub":{
                    "admin_access": True,
                    "authenticator_class": "generic-oauth"
                },
                "Authenticator":{
                    "admin_users": admins,
                    "allowed_users": user_form["users"]
            }
        },


    },
        "singleuser":{
                "allowPrivilegeEscalation": True,
                "uid": 1000,
                "networkPolicy":{
                "enabled": False},
                "defaultUrl": "/lab/tree/Welcome.ipynb",
                "extraEnv":{
                    "GRANT_SUDO": "yes",
                    "SHELL": "/usr/bin/bash",
                    "TZ": "UTC",
                    "SIZEW": "1920",
                    "SIZEH": "1080",
                    "REFRESH": "60",
                    "DPI": "96",
                    "CDEPTH": "24",
                    "PASSWD": "maia",
                    "WEBRTC_ENCODER": "nvh264enc",
                    "BASIC_AUTH_PASSWORD": "maia",
                    "NOVNC_ENABLE": "true",
                    "ssh_publickey": "NOKEY",
                    "NB_USER": "maia-user",
                    "MINIO_ACCESS_KEY": user_form.get("minio_access_key", "N/A"),
                    "MINIO_SECRET_KEY": user_form.get("minio_secret_key", "N/A"),
                    "MLFLOW_TRACKING_URI": f"https://{hub_address}/mlflow",
                    "HOSTNAME": cluster_config["ssh_hostname"],
                    "NAMESPACE": namespace.lower()

                }
            }
    }

    if "minio_env_name" in user_form:
        minio_env_name = user_form["minio_env_name"]
        client = Minio(cluster_config["minio_url"],
                       access_key=cluster_config["minio_access_key"],
                       secret_key=cluster_config["minio_secret_key"],
                       secure=True)
        client.fget_object(cluster_config["bucket_name"], minio_env_name, minio_env_name)
        with open(minio_env_name, "r") as f:
            file_string = f.read()
            if file_string.startswith("name:"):
                jh_template["singleuser"]["extraEnv"]["CONDA_ENV"] = str(file_string)
            else:
                jh_template["singleuser"]["extraEnv"]["PIP_ENV"] = str(file_string)


    if "url_type" in cluster_config:
        if cluster_config["url_type"] == "subpath":
            jh_template["hub"]["baseUrl"] = f"/{group_subdomain}-hub"


    if keycloack is not None:
        jh_template["hub"]["config"]["GenericOAuthenticator"]["client_id"] = keycloack["client_id"]
        jh_template["hub"]["config"]["GenericOAuthenticator"]["client_secret"] = keycloack["client_secret"]
        jh_template["hub"]["config"]["GenericOAuthenticator"]["authorize_url"] = keycloack["authorize_url"]
        jh_template["hub"]["config"]["GenericOAuthenticator"]["token_url"] = keycloack["token_url"]
        jh_template["hub"]["config"]["GenericOAuthenticator"]["userdata_url"] = keycloack["userdata_url"]
        if "url_type" in cluster_config:
            if cluster_config["url_type"]  == "subdomain" :
                jh_template["hub"]["config"]["GenericOAuthenticator"]["oauth_callback_url"] = f"https://{hub_address}/hub/oauth_callback"
                # print("Register Callback: ")
                # print(f"https://{hub_address}/hub/oauth_callback")
            elif cluster_config["url_type"]  == "subpath":
                jh_template["hub"]["config"]["GenericOAuthenticator"][
                    "oauth_callback_url"] = f"https://{hub_address}/{group_subdomain}-hub/oauth_callback"
                # print("Register Callback: ")
                #print(f"https://{hub_address}/{group_subdomain}-hub/oauth_callback")


    if "custom_hub" in user_form:

        jh_template["hub"]["image"] = {

                    "name": "registry.maia.cloud.cbh.kth.se/jupyterhub", #TODO
                    "tag": "1.1"

        }

    if not gpu_request:
        jh_template["singleuser"]["extraEnv"]["NVIDIA_VISIBLE_DEVICES"] = ""


    if "ssh_users" in user_form:
        for ssh_port in user_form["ssh_users"]:
            username = ssh_port["username"].replace("@","__at__")
            jh_template["singleuser"]["extraEnv"][f"SSH_PORT_{username}"] = str(ssh_port["ssh_port"])


    if traefik_resolver is not None:
        jh_template["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls.certresolver"] = traefik_resolver
        jh_template["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.entrypoints"]= "websecure"
        jh_template["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls"]= "true"

    if nginx_cluster_issuer is not None:
        jh_template["ingress"]["annotations"]["nginx.ingress.kubernetes.io/proxy-body-size"]= "2g"
        jh_template["ingress"]["annotations"]["cert-manager.io/cluster-issuer"] = nginx_cluster_issuer
        jh_template["ingress"]["tls"][0]["secretName"] = "jupyterhub-{}-tls".format(namespace.lower())

    if hub_storage_class is not None:
        jh_template["hub"]["db"] = {"pvc":{"storageClassName": hub_storage_class}}


    if hub_image is not None:
        jh_template["hub"]["image"] = {
            "name": hub_image,
            "tag": hub_tag
        }

    if base_url is not None:
        jh_template["hub"]["base_url"] = base_url


    jh_template["singleuser"]["storage"] = {
        "homeMountPath": "/home/maia-user",
        "dynamic": {
            "storageClass": storage_class
        },
        "extraVolumes": [
            {
                "name": "shm-volume",
                "emptyDir": {
                    "medium": "Memory"
                }
            }
        ],
        "extraVolumeMounts": [
            {
                "name": "shm-volume",
                "mountPath": "/dev/shm"
            }
        ]
    }


    jh_template["singleuser"]["memory"] = {
        "limit": resources_limits["memory"][1],
        "guarantee": resources_limits["memory"][0]
    }

    jh_template["singleuser"]["cpu"] = {
        "limit": resources_limits["cpu"][1],
        "guarantee": resources_limits["cpu"][0]
    }

    for extra_volume in extra_volumes:
        jh_template["singleuser"]["storage"]["extraVolumes"].append({
            "name": extra_volume["name"],
            "persistentVolumeClaim": {
                "claimName": extra_volume["pvc-name"]
            }
        })
        jh_template["singleuser"]["storage"]["extraVolumeMounts"].append({
            "name": extra_volume["name"],
            "mountPath": extra_volume["mount-path"]
        })
    
    for extra_host_volume in extra_host_volumes:
        jh_template["singleuser"]["storage"]["extraVolumes"].append({
            "name": extra_host_volume["name"],
            "hostPath": {
                "path": extra_host_volume["host-path"]
            }
        })
        jh_template["singleuser"]["storage"]["extraVolumeMounts"].append({
            "name": extra_host_volume["name"],
            "mountPath": extra_host_volume["mount-path"]
        })

    jh_template["singleuser"]["image"] = {
        "name": "jupyter/datascience-notebook",
        "tag": "latest",
        "pullSecrets": [
            cluster_config["imagePullSecrets"]
        ]
    }
    maia_workspace_version = user_form["maia_workspace_version"]
    jh_template["singleuser"]["profileList"] = [
        {"display_name": f"MAIA Workspace v{maia_workspace_version}",
         "description": "MAIA Workspace with Python 3.10, Anaconda, MatLab, RStudio, VSCode and SSH Connection",
         "default": True,
        "kubespawner_override":{"image": f"kthcloud/maia-workspace-ssh-addons:{maia_workspace_version}",
                                #"image": f"registry.cloud.cbh.kth.se/maia/maia-workspace-ssh-addons:{maia_workspace_version}",  #TODO
                                "start_timeout": 3600,
                                "http_timeout": 3600,
                                #mem_limit
                                #cpu_limit
                                #mem_guarantee
                                #cpu_guarantee
                               "extra_resource_limits": {
                               },

                                "container_security_context": {
                                    "privileged": True,
                                    "procMount": "unmasked",
                                    "seccompProfile": {
                                        "type": "Unconfined"
                                    }
                                }
                                }
         }

    ]

    if gpu_request:
        jh_template["singleuser"]["profileList"][0]["kubespawner_override"]["extra_resource_limits"] = {
            "nvidia.com/gpu": "1"
        }




    jh_helm_template["resource"]["helm_release"]["jupyterhub"]["values"] = [
        yaml.dump(jh_template)
    ]

    if config_folder is None:
        config_folder = "."

    Path(config_folder).joinpath(namespace).mkdir(parents=True, exist_ok=True)
    with open(Path(config_folder).joinpath(namespace,f"{namespace}_jupyterhub.tf.json"), "w") as f:
        json.dump(jh_helm_template, f, indent=2)

    with open(Path(config_folder).joinpath(namespace,f"{namespace}_jupyterhub_values.yaml"), "w") as f:
        print(jh_helm_template["resource"]["helm_release"]["jupyterhub"]["values"][0],file=f)


    helm_namespace = jh_helm_template["resource"]["helm_release"]["jupyterhub"]["namespace"]
    helm_chart = jh_helm_template["resource"]["helm_release"]["jupyterhub"]["chart"]
    helm_name = jh_helm_template["resource"]["helm_release"]["jupyterhub"]["name"]
    helm_repo = jh_helm_template["resource"]["helm_release"]["jupyterhub"]["repository"]
    helm_repo_version = jh_helm_template["resource"]["helm_release"]["jupyterhub"]["version"]

    config_path = Path(config_folder).joinpath(namespace,f"{namespace}_jupyterhub_values.yaml")
    cmds = [
        #"Run the following command to deploy JupyterHub: ",
            f"helm repo add jupyterhub {helm_repo}",
            f"helm repo update",
            f"helm upgrade --install -n {helm_namespace} {helm_name} jupyterhub/{helm_chart} --values {config_path} --version={helm_repo_version}"]

    print("\n".join(cmds))

    return cmds


def main():
    create_jupyterhub_config()

if __name__ == "__main__":
    main()
