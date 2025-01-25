#!/usr/bin/env python
from pathlib import Path
import click
import yaml
import datetime
import json

import os
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import MAIA
import random
import string
version = MAIA.__version__


TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Script to initialize the MAIA Cluster configuration. The cluster configuration is specified by the ``--cluster-config`` option,
    and the configuration folder where the resolved configuration files will be saved is specified by the ``--config-folder`` option.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --cluster-config /PATH/TO/cluster_config.yaml --config-folder /PATH/TO/config_folder
    """.format(  # noqa: E501
        filename=Path(__file__).stem
    )
)

def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--cluster-config",
        type=str,
        required=True,
        help="YAML configuration file used to extract the cluster configuration.",
    )

    pars.add_argument(
        "--config-folder",
        type=str,
        required=True,
        help="Configuration Folder where to save the resolved MAIA Cluster configuration files.",
    )

    pars.add_argument('-v', '--version', action='version', version='%(prog)s ' + version)

    return pars


def generate_random_password(length=12):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))


@click.command()
@click.option("--cluster-config", type=str)
@click.option("--config-folder", type=str)
def main(cluster_config, config_folder):
    create_configuration(cluster_config, config_folder)

def create_configuration(cluster_config, config_folder):

   
    cluster_config_dict = yaml.safe_load(Path(cluster_config).read_text())



    cluster_config_dict["traefik_dashboard_password"] = generate_random_password(12)
    cluster_config_dict["rancher_password"] = generate_random_password(12)
    cluster_config_dict["ingress_class"] = "maia-core-traefik" #Alternative: "nginx"
    cluster_config_dict["traefik_resolver"] = "maiaresolver"
    cluster_config_dict["ssh_port_type"] = "NodePort"
    cluster_config_dict["port_range"] = [30000, 32767]
    cluster_config_dict["shared_storage_class"] = "microk8s-hostpath"
    cluster_config_dict["storage_class"] = "microk8s-hostpath"
    cluster_config_dict["url_type"] = "subdomain"
    cluster_config_dict["ssh_hostname"] = cluster_config_dict["domain"]
    cluster_config_dict["keycloak_maia_client_secret"] = generate_random_password(16)
    cluster_config_dict["keycloak"] = {
        "client_id": "maia",
        "issuer_url": f"https://iam.{cluster_config_dict['domain']}/realms/maia",
        "client_secret": cluster_config_dict["keycloak_maia_client_secret"],
        "authorize_url": f"https://iam.{cluster_config_dict['domain']}/realms/maia/protocol/openid-connect/auth",
        "token_url": f"https://iam.{cluster_config_dict['domain']}/realms/maia/protocol/openid-connect/token",
        "userdata_url": f"https://iam.{cluster_config_dict['domain']}/realms/maia/protocol/openid-connect/userinfo"
    }
    cluster_config_dict["argocd_destination_cluster_address"] = f"https://mgmt.{cluster_config_dict['domain']}/k8s/clusters/local"
    cluster_config_dict["services"] = {
        "dashboard": f"https://dashboard.{cluster_config_dict['domain']}",
        "login":  f"https://login.{cluster_config_dict['domain']}",
        "grafana": f"https://grafana.{cluster_config_dict['domain']}",
        "rancher": f"https://mgmt.{cluster_config_dict['domain']}",
        "argocd": f"https://argocd.{cluster_config_dict['domain']}",
        "registry": f"https://registry.{cluster_config_dict['domain']}",
        "keycloak": f"https://iam.{cluster_config_dict['domain']}",
        "traefik": f"https://traefik.{cluster_config_dict['domain']}"

    }

    cluster_config_dict["maia_dashboard"] = {
        "enabled": True,
        "token": ""
    }


    with open(os.path.join(config_folder,"MAIA_realm_template.json"), "r") as f:
        realm = json.load(f)
        clients = realm["clients"]
        for idx,client in enumerate(clients):
            if client['clientId'] == 'maia':
                clients[idx]['secret'] = cluster_config_dict["keycloak_maia_client_secret"]
    
    with open(os.path.join(config_folder,"MAIA_realm.json"), "w") as f:
        json.dump(realm, f)

    

    with open(os.path.join(config_folder, f"{cluster_config_dict['cluster_name']}.yaml"), "w") as f:
        yaml.dump(cluster_config_dict, f)

if __name__ == "__main__":
    main()