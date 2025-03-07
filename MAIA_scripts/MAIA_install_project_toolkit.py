#!/usr/bin/env python
import subprocess
from pathlib import Path
from hydra import initialize, initialize_config_dir
from hydra import compose as hydra_compose
import click
import hydra
import yaml
from omegaconf import OmegaConf
from MAIA.maia_fn import deploy_oauth2_proxy, deploy_mysql, deploy_mlflow, deploy_orthanc
from MAIA.maia_admin import create_maia_namespace_values, install_maia_project, get_maia_toolkit_apps, generate_minio_configs, generate_mlflow_configs, generate_mysql_configs
from MAIA_scripts.MAIA_create_JupyterHub_config import create_jupyterhub_config_api
import datetime
import argparse
import json
import os
import asyncio
from pyhelm3 import Client
import subprocess
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent

import MAIA
version = MAIA.__version__


TIMESTAMP = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

DESC = dedent(
    """
    Script to deploy a MAIA Project Toolkit to a Kubernetes cluster. The target cluster is specified by setting the corresponding ``--cluster-config``,
    while the project-related configuration is specified with ``--project-config-file`` and ``--maia-config-file``.
    The necessary MAIA Configuration files should be found in ``--config-folder``.
    """  # noqa: E501
)
EPILOG = dedent(
    """
    Example call:
    ::
        {filename} --project-config-file /PATH/TO/form.yaml --cluster-config /PATH/TO/cluster.yaml  --config-folder /PATH/TO/config_folder --maia-config-file /PATH/TO/maia_config.yaml
    """.format(  # noqa: E501
        filename=Path(__file__).stem
    )
)

def get_arg_parser():
    pars = ArgumentParser(description=DESC, epilog=EPILOG, formatter_class=RawTextHelpFormatter)

    pars.add_argument(
        "--project-config-file",
        type=str,
        required=True,
        help="YAML configuration file used to extract the project configuration.",
    )

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


    pars.add_argument(
        "--minimal",
        type=str,
        required=False,
        help="Optional flag to only deploy JupyterHub in the MAIA namespace.",
    )

    pars.add_argument('-v', '--version', action='version', version='%(prog)s ' + version)

    return pars


async def verify_installed_maia_toolkit(project_id, namespace, get_chart_metadata = True):

    
    client = Client(kubeconfig = os.environ["KUBECONFIG"])


    try:
        revision = await client.get_current_revision(project_id, namespace = namespace)
    except:
        print("Project not found")
        return -1
    if get_chart_metadata:
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
    return {
        "revision": revision.revision,
        "release_name": revision.release.name,
        "release_namespace": revision.release.namespace,
        "status": str(revision.status)
    }

@click.command()
@click.option("--project-config-file", type=str)
@click.option("--maia-config-file", type=str)
@click.option("--cluster-config", type=str)
@click.option('--minimal', is_flag=True)
@click.option("--config-folder", type=str)
def main(project_config_file,maia_config_file, cluster_config, config_folder, minimal=False):
    deploy_maia_toolkit(project_config_file, maia_config_file, cluster_config, config_folder, minimal)

def deploy_maia_toolkit(project_config_file, maia_config_file, cluster_config, config_folder, minimal=False):
    project_form_dict = yaml.safe_load(Path(project_config_file).read_text())
    
    cluster_config_dict = yaml.safe_load(Path(cluster_config).read_text())
    maia_config_dict = yaml.safe_load(Path(maia_config_file).read_text())

    deploy_maia_toolkit_api(project_form_dict, maia_config_dict, cluster_config_dict, config_folder, minimal)

def deploy_maia_toolkit_api(project_form_dict, maia_config_dict, cluster_config_dict, config_folder,minimal=False, redeploy_enabled = True):
    group_id = project_form_dict["group_ID"]
    Path(config_folder).joinpath(project_form_dict["group_ID"]).mkdir(parents=True, exist_ok=True)

    

    namespace = project_form_dict["group_ID"].lower().replace("_", "-")



    helm_commands = []


    minio_configs = generate_minio_configs(namespace=group_id.lower().replace("_", "-"))
    mlflow_configs = generate_mlflow_configs(namespace=group_id.lower().replace("_", "-"))
    mysql_configs = generate_mysql_configs(namespace=group_id.lower().replace("_", "-"))

    project_form_dict["minio_access_key"] = minio_configs["console_access_key"]
    project_form_dict["minio_secret_key"] = minio_configs["console_secret_key"]

    helm_commands.append(create_maia_namespace_values(project_form_dict, cluster_config_dict, config_folder, minio_configs=minio_configs, mlflow_configs=mlflow_configs))


    with open(Path(config_folder).joinpath(group_id,"maia_namespace_values","namespace_values.yaml")) as f:
        maia_namespace_values = yaml.safe_load(f)
        project_form_dict["ssh_users"] = []
        for user in maia_namespace_values["users"]:
            project_form_dict["ssh_users"].append(
                {
                    "username": user["jupyterhub_username"].replace("-2d", "-").replace("-40", "@").replace("-2e", "."),
                    "ssh_port": user["sshPort"]
                }
            )

    helm_commands.append(create_jupyterhub_config_api(
        project_form_dict,
        maia_config_dict,
        cluster_config_dict,
        config_folder
    ))


    helm_commands.append(deploy_oauth2_proxy(cluster_config_dict, project_form_dict, config_folder))

    helm_commands.append(deploy_mysql(cluster_config_dict, project_form_dict, config_folder, mysql_configs=mysql_configs))
    helm_commands.append(deploy_mlflow(cluster_config_dict, project_form_dict, config_folder,mysql_config=mysql_configs, minio_config=minio_configs))

    helm_commands.append(deploy_orthanc(cluster_config_dict, project_form_dict, maia_config_dict, config_folder))


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

    destination_cluster_address = cluster_config_dict["argocd_destination_cluster_address"]

    values = {
       "defaults": [
              "_self_",
              {"maia_namespace_values": "namespace_values"},
              {"jupyterhub_values": "jupyterhub_values"},
            {"oauth2_proxy_values": "oauth2_proxy_values"},
            {"mysql_values": "mysql_values"},
            {"mlflow_values": "mlflow_values"},
            {"jupyterhub_chart_info": "jupyterhub_chart_info"},
            {"orthanc_values": "orthanc_values"}
         ],
        "argo_namespace": maia_config_dict["argocd_namespace"],
        "group_ID": f"MAIA:{group_id}",
        "destination_server": f"{destination_cluster_address}",
        "sourceRepos": [
            "https://kthcloud.github.io/MAIA/",
            "https://hub.jupyter.org/helm-chart/",
            "https://oauth2-proxy.github.io/manifests"
        ]
    }
    
    with open(Path(config_folder).joinpath(group_id,"values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(values))

    try:
        initialize_config_dir(config_dir=str(Path(config_folder).joinpath(group_id)), job_name=group_id)
    except:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=str(Path(config_folder).joinpath(group_id)), job_name=group_id)
    cfg = hydra_compose("values.yaml")
    OmegaConf.save(cfg, str(Path(config_folder).joinpath(group_id, f"{group_id}_values.yaml")), resolve=True)

    project_id = namespace
    
    

    revision = asyncio.run(verify_installed_maia_toolkit(project_id, maia_config_dict["argocd_namespace"]))

    if revision == -1 or redeploy_enabled:
        print("Installing MAIA Workspace")
    
        
        project_chart = maia_config_dict["maia_project_chart"]
        project_repo = maia_config_dict["maia_project_repo"]
        project_version = maia_config_dict["maia_project_version"]
        asyncio.run(install_maia_project(group_id, Path(config_folder).joinpath(group_id, f"{group_id}_values.yaml"),maia_config_dict["argocd_namespace"], project_chart, project_repo=project_repo, project_version=project_version))
    else:
        argocd_host = maia_config_dict["argocd_host"]
        token = maia_config_dict["argocd_token"]
        print("MAIA Workspace already installed")
        asyncio.run(get_maia_toolkit_apps(group_id, token, argocd_host))
if __name__ == "__main__":
    main()
