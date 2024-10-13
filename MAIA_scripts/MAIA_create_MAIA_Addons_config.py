#!/usr/bin/env python

import json
from pathlib import Path

import click
import yaml

from MAIA.maia_fn import get_ssh_ports


@click.command()
@click.option("--form", type=str)
@click.option("--cluster-config-file", type=str)
def create_maia_addons_config(form,cluster_config_file):
    create_maia_addons_config_api(form,cluster_config_file)

def create_maia_addons_config_api( form,
                              cluster_config_file,
                                   config_folder = None
                             ):

    with open(cluster_config_file,"r") as f:
        cluster_config = yaml.safe_load(f)


    with open(form,"r") as f:
        user_form = yaml.safe_load(f)

    group_subdomain = user_form["group_subdomain"]
    namespace = user_form["group_ID"]


    users = user_form["users"]

    jupyterhub_users = []

    oauth_url = user_form["oauth_url"]

    for user in users:
        jupyterhub_users.append(user.replace("-", "-2d").replace("@", "-40").replace(".", "-2e"))

    domain = cluster_config["domain"]

    if "url_type" in cluster_config:
        if cluster_config["url_type"] == "subdomain":
            hub_address = f"{group_subdomain}.{domain}"
        elif cluster_config["url_type"] == "subpath":
            hub_address = domain
        else:
            hub_address = None

    maia_addons_values_template = {
        "namespace": namespace.lower(),
        "oauth_url": oauth_url,
        "ingress_host": hub_address,
        "metallb": {

        },
        "ssh_port_type": cluster_config["ssh_port_type"],
        "users": []
    }
    if "metallb_shared_ip" in cluster_config:
        maia_addons_values_template["metallb"]["shared_ip"] = cluster_config["metallb_shared_ip"]
    if "metallb_ip_pool" in cluster_config:
        maia_addons_values_template["metallb"]["ip_pool"] = cluster_config["metallb_ip_pool"]

    metallb_shared_ip = cluster_config["maia_metallb_ip"] if "maia_metallb_ip" in cluster_config else None
    maia_addons_values_template["metallb"]["load_balancer_ip"] = metallb_shared_ip
    ssh_ports = get_ssh_ports(len(users),cluster_config["ssh_port_type"],cluster_config["port_range"],maia_metallb_ip=metallb_shared_ip)
    for user, jupyterhub_user,ssh_port in zip(users,jupyterhub_users, ssh_ports):
        maia_addons_values_template["users"].append(
            {"jupyterhub_username": jupyterhub_user,
             "username": user,
             "ssh_port": ssh_port
             })


    if "traefik_resolver" in cluster_config:
        maia_addons_values_template["traefik"] = {
            "enabled": True,
            "traefik_resolver": cluster_config["traefik_resolver"]
        }


    if "nginx_cluster_issuer" in cluster_config:
        maia_addons_values_template["nginx"] = {
            "enabled": True,
            "cluster_issuer": cluster_config["nginx_cluster_issuer"]
        }

    if "minio_console_service" in user_form:
        maia_addons_values_template["proxy_nginx"] = {
            "enabled": True,
            "imagePullSecrets": cluster_config["imagePullSecrets"],
            "image": cluster_config["nginx_proxy_image"],
            "console_service": user_form["minio_console_service"],
            #"cluster_issuer": cluster_config["nginx_cluster_issuer"],
            "mlflow": user_form["mlflow_service"],
            "console_service_path": "minio-console",
            "mlflow_path": "mlflow",
            "label_studio_path": "label-studio",
            "kubeflow_path": "kubeflow",
            "minio_path": "minio"
        }
        if cluster_config["url_type"] == "subpath":
            maia_addons_values_template["proxy_nginx"]["console_service_path"] = f"{namespace}-minio-console"
            maia_addons_values_template["proxy_nginx"]["mlflow_path"] = f"{namespace}-mlflow"
            maia_addons_values_template["proxy_nginx"]["label_studio_path"] = f"{namespace}-label-studio"
            maia_addons_values_template["proxy_nginx"]["kubeflow_path"] = f"{namespace}-kubeflow"
            maia_addons_values_template["proxy_nginx"]["minio_path"] = f"{namespace}-minio"

    maia_addons_template = {
        "resource": {
            "helm_release": {
                "maia-addons": {
                    "name": "maia-addons-{}".format(namespace.lower()),
                    "repository": "https://kthcloud.github.io/MAIA/",
                    "chart": "maia-addons",
                    "version": "0.1.4",
                    "namespace": namespace.lower(),
                    "create_namespace": False,
                    "values": [
                        yaml.dump(maia_addons_values_template)
                    ]

                }
            }
        }
    }

    if config_folder is None:
        config_folder = "."

    Path(config_folder).joinpath(namespace).mkdir(parents=True, exist_ok=True)
    with open(Path(config_folder).joinpath(namespace,f"{namespace}_maia_addons.tf.json"), "w") as f:
        json.dump(maia_addons_template, f, indent=2)

    with open(Path(config_folder).joinpath(namespace,f"{namespace}_maia_addons_values.yaml"), "w") as f:
        print(maia_addons_template["resource"]["helm_release"]["maia-addons"]["values"][0], file=f)

    helm_namespace = maia_addons_template["resource"]["helm_release"]["maia-addons"]["namespace"]
    helm_name = maia_addons_template["resource"]["helm_release"]["maia-addons"]["name"]
    helm_chart = maia_addons_template["resource"]["helm_release"]["maia-addons"]["chart"]
    helm_repo = maia_addons_template["resource"]["helm_release"]["maia-addons"]["repository"]
    helm_repo_version = maia_addons_template["resource"]["helm_release"]["maia-addons"]["version"]

    config_path = Path(config_folder).joinpath(namespace,f"{namespace}_maia_addons_values.yaml")
    cmds =[
        # "Run the following command to deploy JupyterHub: ",
        f"helm repo add maia {helm_repo}",
        f"helm repo update",
        f"helm upgrade --install -n {helm_namespace} {helm_name} maia/{helm_chart} --values {config_path} --version={helm_repo_version}"

    ]
    print("\n".join(cmds))

    return cmds


def main():
    create_maia_addons_config()


if __name__ == "__main__":
    main()
