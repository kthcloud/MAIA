#!/usr/bin/env python
from pathlib import Path
import click
import yaml
from omegaconf import OmegaConf
from MAIA.maia_admin import install_maia_project
from hydra import initialize, initialize_config_dir
from hydra import compose as hydra_compose
import datetime

import os
import asyncio
from pyhelm3 import Client
from MAIA.maia_admin import create_loginapp_values, create_minio_operator_values
from MAIA.maia_core import create_prometheus_values, create_tempo_values, create_loki_values, create_core_toolkit_values, create_traefik_values, create_metallb_values, create_cert_manager_values, create_gpu_booking_values, create_gpu_operator_values, create_ingress_nginx_values, create_nfs_server_provisioner_values
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import MAIA
version = MAIA.__version__


TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Script to Install MAIA Core Toolkit to a Kubernetes cluster from ArgoCD. The specific MAIA configuration is specified by setting the corresponding ``--maia-config-file``,
    and the cluster configuration is specified by setting the corresponding ``--cluster-config``.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --maia-config-file /PATH/TO/maia_config.yaml --cluster-config /PATH/TO/clusyer_config.yaml --config-folder /PATH/TO/config_folder
    """.format(  # noqa: E501
        filename=Path(__file__).stem
    )
)

def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)


    pars.add_argument(
        "--maia-config-file",
        type=str,
        required=True,
        help="YAML configuration file used to extract MAIA configuration.",
    )

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
        help="Configuration Folder where to locate (and temporarily store) the MAIA configuration files.",
    )

    pars.add_argument('-v', '--version', action='version', version='%(prog)s ' + version)

    return pars


async def verify_installed_maia_core_toolkit(project_id, namespace):

    print(os.environ["KUBECONFIG"])
    client = Client(kubeconfig = os.environ["KUBECONFIG"])


    try:
        revision = await client.get_current_revision(project_id, namespace = namespace)
    except:
        print("Project not found")
        return -1
    chart_metadata = await revision.chart_metadata()
    print(
        revision.release.name,
        revision.release.namespace,
        revision.revision,
        str(revision.status),
        chart_metadata.name,
        chart_metadata.version
    )
    return revision.revision

@click.command()
@click.option("--maia-config-file", type=str)
@click.option("--cluster-config", type=str)
@click.option("--config-folder", type=str)
def main(maia_config_file, cluster_config, config_folder):
    install_maia_core_toolkit(maia_config_file, cluster_config, config_folder)

def install_maia_core_toolkit(maia_config_file, cluster_config, config_folder):

    maia_config_dict = yaml.safe_load(Path(maia_config_file).read_text())

    cluster_config_dict = yaml.safe_load(Path(cluster_config).read_text())

    admin_group_ID = maia_config_dict["admin_group_ID"]
    project_id = "maia-core"

    if "argocd_destination_cluster_address" in cluster_config_dict and not cluster_config_dict["argocd_destination_cluster_address"].endswith("/k8s/clusters/local"):
        cluster_address = cluster_config_dict["argocd_destination_cluster_address"]
        if cluster_config_dict["argocd_destination_cluster_address"] != "https://kubernetes.default.svc":
            project_id += f"-{cluster_config_dict['cluster_name']}"
    else:
        cluster_address = "https://kubernetes.default.svc"

    helm_commands = []
    helm_commands.append(create_prometheus_values(config_folder, project_id, cluster_config_dict, maia_config_dict))
    helm_commands.append(create_loki_values(config_folder, project_id))
    helm_commands.append(create_tempo_values(config_folder, project_id))
    helm_commands.append(create_core_toolkit_values(config_folder, project_id, cluster_config_dict))

    # Allow either traefik or nginx ingress controller
    helm_commands.append(create_traefik_values(config_folder, project_id, cluster_config_dict))
    helm_commands.append(create_ingress_nginx_values(config_folder, project_id))
    
    helm_commands.append(create_metallb_values(config_folder, project_id))
    helm_commands.append(create_cert_manager_values(config_folder, project_id))
    helm_commands.append(create_gpu_operator_values(config_folder, project_id, cluster_config_dict))
    helm_commands.append(create_nfs_server_provisioner_values(config_folder, project_id, cluster_config_dict))
    
    helm_commands.append(create_loginapp_values(config_folder, project_id,cluster_config_dict))
    helm_commands.append(create_minio_operator_values(config_folder, project_id,cluster_config_dict))
    

    for helm_command in helm_commands:
        cmd = [
            "helm",
            "upgrade",
            "--install",
            "--wait",
            "-n",
            helm_command["namespace"],
            helm_command["release"],
            helm_command["chart"],
            "--repo",
            helm_command["repo"],
            "--version",
            helm_command["version"],
            "--values",
            helm_command["values"],
        ]
        print(" ".join(cmd))


    values = {
        "defaults": [
              "_self_",
              {"prometheus_values": "prometheus_values"},
                {"loki_values": "loki_values"},
                {"tempo_values": "tempo_values"},
                {"core_toolkit_values": "core_toolkit_values"},
                {"traefik_values": "traefik_values"},
                {"metallb_values": "metallb_values"},
                {"cert_manager_values": "cert_manager_values"},
                
                {"loginapp_values": "loginapp_values"},
                {"minio_operator_values": "minio_operator_values"},
                {"gpu_operator_values": "gpu_operator_values"},
                {"ingress_nginx_values": "ingress_nginx_values"},
                {"nfs_provisioner_values": "nfs_provisioner_values"},
                {"cert_manager_chart_info": "cert_manager_chart_info"},
                
         ],
        "argo_namespace": maia_config_dict["argocd_namespace"],
        "admin_group_ID": admin_group_ID,
        "destination_server": f"{cluster_address}",
        "sourceRepos": [
                "https://prometheus-community.github.io/helm-charts",
                "https://grafana.github.io/helm-charts",
                "https://kthcloud.github.io/MAIA/",
                "https://traefik.github.io/charts",
                "https://metallb.github.io/metallb",
                "https://charts.jetstack.io",
                "https://helm.ngc.nvidia.com/nvidia",
                "https://kubernetes-sigs.github.io/nfs-subdir-external-provisioner/",
                "https://storage.googleapis.com/loginapp-releases/charts/",
                "https://operator.min.io"
                #"https://kubernetes.github.io/ingress-nginx"
        ]
    }
    Path(config_folder).joinpath(project_id).mkdir(parents=True, exist_ok=True)

    with open(Path(config_folder).joinpath(project_id,"values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(values))

    initialize_config_dir(config_dir=str(Path(config_folder).joinpath(project_id)), job_name=project_id)
    cfg = hydra_compose("values.yaml")
    OmegaConf.save(cfg, str(Path(config_folder).joinpath(project_id, f"{project_id}_values.yaml")), resolve=True)

    

    revision = asyncio.run(verify_installed_maia_core_toolkit(project_id, maia_config_dict["argocd_namespace"]))



    if revision == -1:
        print("Installing MAIA Core Toolkit")
    
    
        project_chart = maia_config_dict["core_project_chart"]
        project_repo = maia_config_dict["core_project_repo"]
        project_version = maia_config_dict["core_project_version"]
        asyncio.run(install_maia_project(project_id, Path(config_folder).joinpath(project_id, f"{project_id}_values.yaml"),maia_config_dict["argocd_namespace"], project_chart, project_repo=project_repo, project_version=project_version))
    else:
        print("Upgrading MAIA Core Toolkit")
    
    
        project_chart = maia_config_dict["core_project_chart"]
        project_repo = maia_config_dict["core_project_repo"]
        project_version = maia_config_dict["core_project_version"]
        asyncio.run(install_maia_project(project_id, Path(config_folder).joinpath(project_id, f"{project_id}_values.yaml"),maia_config_dict["argocd_namespace"], project_chart, project_repo=project_repo, project_version=project_version))


if __name__ == "__main__":
    main()