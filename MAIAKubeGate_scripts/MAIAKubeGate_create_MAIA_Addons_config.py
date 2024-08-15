#!/usr/bin/env python

import json
import yaml
import click
from kubernetes import client, config
from pathlib import Path
def get_ssh_ports(n_requested_ports, port_type, ip_range, maia_metallb_ip=None):

    config.load_kube_config()

    v1 = client.CoreV1Api()


    used_port = []
    services = v1.list_service_for_all_namespaces(watch=False)
    for svc in services.items:
        if port_type == 'LoadBalancer':
            if svc.status.load_balancer.ingress is not None:

                if svc.spec.type == 'LoadBalancer' and svc.status.load_balancer.ingress[0].ip == maia_metallb_ip:
                    for port in svc.spec.ports:
                        if port.name == 'ssh':
                            used_port.append(int(port.port))
        elif port_type == "NodePort":
            if svc.spec.type == 'NodePort':
                for port in svc.spec.ports:
                    used_port.append(int(port.port))

    ports = []
    print(used_port)
    for request in range(n_requested_ports):
        for port in range(ip_range[0], ip_range[1]):
            if port not in used_port:
                ports.append(port)
                used_port.append(port)
                break
    print(ports)
    return ports


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
            "mlflow": user_form["mlflow_service"]
        }

    maia_addons_template = {
        "resource": {
            "helm_release": {
                "maia-addons": {
                    "name": "maia-addons-{}".format(namespace.lower()),
                    "repository": "https://simonebendazzoli93.github.io/MAIAKubeGate/",
                    "chart": "maia-addons",
                    "version": "0.1.0",
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

    with open(Path(config_folder).joinpath(f"{namespace}_maia_addons.tf.json"), "w") as f:
        json.dump(maia_addons_template, f, indent=2)

    with open(Path(config_folder).joinpath(f"{namespace}_maia_addons_values.yaml"), "w") as f:
        print(maia_addons_template["resource"]["helm_release"]["maia-addons"]["values"][0], file=f)

    helm_namespace = maia_addons_template["resource"]["helm_release"]["maia-addons"]["namespace"]
    helm_name = maia_addons_template["resource"]["helm_release"]["maia-addons"]["name"]
    helm_chart = maia_addons_template["resource"]["helm_release"]["maia-addons"]["chart"]
    helm_repo = maia_addons_template["resource"]["helm_release"]["maia-addons"]["repository"]
    helm_repo_version = maia_addons_template["resource"]["helm_release"]["maia-addons"]["version"]

    config_path = Path(config_folder).joinpath(f"{namespace}_maia_addons_values.yaml")
    cmds =[
        "Run the following command to deploy JupyterHub: ",
        f"helm repo add maiakubegate {helm_repo}",
        f"helm upgrade --install -n {helm_namespace} {helm_name} maiakubegate/{helm_chart} --values {config_path} --version={helm_repo_version}"

    ]
    print("\n".join(cmds))

    return cmds


if __name__ == "__main__":
    create_maia_addons_config(
    )