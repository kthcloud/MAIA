#!/usr/bin/env python

import json
from pathlib import Path

import click
import yaml

import datetime
import json

import MAIA
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent
from MAIA.maia_fn import convert_username_to_jupyterhub_username
version = MAIA.__version__

TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())
from MAIA.maia_fn import get_ssh_ports


DESC = dedent(
    """
    Script to deploy the MAIA-Addons helm chart to a Kubernetes cluster. The target cluster is specified by setting the corresponding ``--cluster--config-file``,
    while the namespace-related configuration is specified with ``--form``.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename}  --form /PATH/TO/form.yaml --cluster-config-file /PATH/TO/cluster.yaml
    """.format(  # noqa: E501
        filename=Path(__file__).stem
    )
)

def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--form",
        type=str,
        required=True,
        help="YAML configuration file used to extract the namespace configuration.",
    )
    pars.add_argument(
        "--maia-config-file",
        type=str,
        required=True,
        help="YAML configuration file used to extract the MAIA configuration.",
    )
    pars.add_argument(
        "--cluster-config-file",
        type=str,
        required=True,
        help="YAML configuration file used to extract the cluster configuration.",
    )

    pars.add_argument('-v', '--version', action='version', version='%(prog)s ' + version)

    return pars

@click.command()
@click.option("--form", type=str)
@click.option("--maia-config-file", type=str)
@click.option("--cluster-config-file", type=str)
def create_maia_addons_config(form,maia_config_file,cluster_config_file):
    create_maia_addons_config_api(form,maia_config_file,cluster_config_file)

def create_maia_addons_config_api( form,
                                  maia_config_file,
                              cluster_config_file,
                                   config_folder = None
                             ):

    with open(maia_config_file, "r") as f:
        maia_form = yaml.safe_load(f)

    with open(cluster_config_file,"r") as f:
        cluster_config = yaml.safe_load(f)


    with open(form,"r") as f:
        user_form = yaml.safe_load(f)

    group_subdomain = user_form["group_subdomain"]
    namespace = user_form["group_ID"].lower().replace("_", "-")


    users = user_form["users"]

    jupyterhub_users = []

    oauth_url = user_form["oauth_url"]

    for user in users:
        jupyterhub_users.append(convert_username_to_jupyterhub_username(user))

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
       
        cluster_config["nginx_proxy_image"] = maia_form["nginx_proxy_image"]
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
                    "version": maia_form["maia_addons_version"],
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
    
    group_id = user_form["group_ID"]
    Path(config_folder).joinpath(user_form["group_ID"]).mkdir(parents=True, exist_ok=True)
    with open(Path(config_folder).joinpath(user_form["group_ID"],f"{group_id}_maia_addons.tf.json"), "w") as f:
        json.dump(maia_addons_template, f, indent=2)

    with open(Path(config_folder).joinpath(user_form["group_ID"],f"{group_id}_maia_addons_values.yaml"), "w") as f:
        print(maia_addons_template["resource"]["helm_release"]["maia-addons"]["values"][0], file=f)

    helm_namespace = maia_addons_template["resource"]["helm_release"]["maia-addons"]["namespace"]
    helm_name = maia_addons_template["resource"]["helm_release"]["maia-addons"]["name"]
    helm_chart = maia_addons_template["resource"]["helm_release"]["maia-addons"]["chart"]
    helm_repo = maia_addons_template["resource"]["helm_release"]["maia-addons"]["repository"]
    helm_repo_version = maia_addons_template["resource"]["helm_release"]["maia-addons"]["version"]

    config_path = Path(config_folder).joinpath(user_form["group_ID"],f"{group_id}_maia_addons_values.yaml")
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
