import subprocess
from pathlib import Path

import click
import yaml

from MAIA.maia_fn import configure_minio, create_share_pvc, create_namespace, create_docker_registry_secret, \
    deploy_kubeflow, deploy_orthanc_ohif, deploy_mysql, deploy_mlflow, deploy_label_studio, get_ssh_ports, \
    create_ssh_service, deploy_minio_tenant, deploy_oauth2_proxy
from MAIA_scripts.MAIA_create_JupyterHub_config import create_jupyterhub_config_api
from MAIA_scripts.MAIA_create_MAIA_Addons_config import create_maia_addons_config_api


@click.command()
@click.option("--namespace-config-file", type=str)
@click.option("--cluster-config", type=str)
@click.option('--create-script', is_flag=True)
@click.option('--minimal', is_flag=True)
@click.option("--config-folder", type=str)
def main(namespace_config_file, cluster_config, config_folder, create_script, minimal):
    user_form_dict = yaml.safe_load(Path(namespace_config_file).read_text())

    Path(config_folder).joinpath(user_form_dict["group_ID"]).mkdir(parents=True, exist_ok=True)

    cluster_config_dict = yaml.safe_load(Path(cluster_config).read_text())

    namespace = user_form_dict["group_ID"].lower()

    script = ["#!/bin/bash"]

    script.append(create_namespace(namespace, create_script=create_script))
    script.append(create_docker_registry_secret(namespace,
                                                cluster_config_dict["imagePullSecrets"],
                                                cluster_config_dict["docker_server"],
                                                cluster_config_dict["docker_username"],
                                                cluster_config_dict["docker_password"],
                                                create_script=create_script))
    if "shared_storage_class" in cluster_config_dict:
        script.append(create_share_pvc(namespace, "shared", cluster_config_dict["shared_storage_class"], "10Gi",
                                       create_script=create_script))
    else:
        script.append(create_share_pvc(namespace, "shared", cluster_config_dict["storage_class"], "10Gi",
                                       create_script=create_script))

    if not minimal:
        oauth_url, cmds = deploy_oauth2_proxy(cluster_config_dict, user_form_dict, config_folder,
                                              create_script=create_script)
        user_form_dict.update(oauth_url)
        script.extend(cmds)

        minio_conf, cmds = deploy_minio_tenant(namespace, config_folder, user_form_dict, cluster_config_dict,
                                               create_script=create_script)
        user_form_dict.update(minio_conf)
        script.extend(cmds)

        mysql_conf, cmds = deploy_mysql(namespace, cluster_config_dict, user_form_dict, config_folder,
                                        create_script=create_script)
        user_form_dict.update(mysql_conf)
        script.extend(cmds)

        mlflow_conf, cmds = deploy_mlflow(namespace, cluster_config_dict, user_form_dict, config_folder,
                                          create_script=create_script)
        user_form_dict.update(mlflow_conf)
        script.extend(cmds)

        cmds = deploy_orthanc_ohif(namespace, cluster_config_dict, user_form_dict, config_folder,
                                   create_script=create_script)
        script.extend(cmds)

        label_studio_conf, cmds = deploy_label_studio(namespace, cluster_config_dict, user_form_dict, config_folder,
                                                      create_script=create_script)
        user_form_dict.update(label_studio_conf)
        script.extend(cmds)

        cmds = deploy_kubeflow(namespace, user_form_dict, cluster_config_dict, config_folder,
                               Path(config_folder).joinpath("pipelines"),
                               create_script=create_script)
        script.extend(cmds)

        #user_form_dict["nginx_proxy_image"] = "registry.cloud.cbh.kth.se/maia/nginx-proxy:1.4" #TODO
        user_form_dict["nginx_proxy_image"] = "kthcloud/nginx-proxy:1.4"
        with open(Path(config_folder).joinpath(user_form_dict["group_ID"],
                                               "{}_user_config.yaml".format(user_form_dict["group_ID"])), "w") as f:
            yaml.dump(user_form_dict, f)

        cmds = create_maia_addons_config_api(Path(config_folder).joinpath(user_form_dict["group_ID"],
                                                                          "{}_user_config.yaml".format(
                                                                              user_form_dict["group_ID"])),
                                             cluster_config, config_folder)

        for cmd in cmds:
            try:
                if not create_script:
                    subprocess.run(cmd.split(' '))
                else:
                    script.append(cmd)
            except:
                print(cmd.split(' '))

        cmds = configure_minio(namespace, user_form_dict, cluster_config_dict, config_folder,
                               create_script=create_script)
        script.extend(cmds)

    user_form_dict["maia_workspace_version"] = "1.3"

    if minimal:
        user_form_dict["ssh_users"] = []
        users = user_form_dict["users"]
        load_balancer_ip = cluster_config_dict["maia_metallb_ip"] if "maia_metallb_ip" in cluster_config_dict else None
        metallb_ip_pool = cluster_config_dict["metallb_ip_pool"] if "metallb_ip_pool" in cluster_config_dict else None
        metallb_shared_ip = cluster_config_dict[
            "metallb_shared_ip"] if "metallb_shared_ip" in cluster_config_dict else None

        jupyterhub_users = []
        for user in users:
            jupyterhub_users.append(user.replace("-", "-2d").replace("@", "-40").replace(".", "-2e"))
        ssh_ports = get_ssh_ports(len(users), cluster_config_dict["ssh_port_type"], cluster_config_dict["port_range"],
                                  maia_metallb_ip=load_balancer_ip)
        for user, jupyterhub_user, ssh_port in zip(users, jupyterhub_users, ssh_ports):
            user_form_dict["ssh_users"].append(
                {"jupyterhub_username": jupyterhub_user,
                 "username": user,
                 "ssh_port": ssh_port
                 })
        cmds = create_ssh_service(namespace, user_form_dict["ssh_users"], cluster_config_dict["ssh_port_type"],
                                  create_script=create_script, metallb_shared_ip=metallb_shared_ip,
                                  metallb_ip_pool=metallb_ip_pool, load_balancer_ip=load_balancer_ip)
        script.extend(cmds)
    else:
        with open(Path(config_folder).joinpath(user_form_dict["group_ID"],
                                               "{}_maia_addons_values.yaml".format(user_form_dict["group_ID"])),
                  "r") as f:
            maia_addons_dict = yaml.safe_load(f)
        user_form_dict["ssh_users"] = maia_addons_dict["users"]

    with open(Path(config_folder).joinpath(user_form_dict["group_ID"],
                                           "{}_user_jupyterhub_config.yaml".format(user_form_dict["group_ID"])),
              "w") as f:
        yaml.dump(user_form_dict, f)

    cmds = create_jupyterhub_config_api(Path(config_folder).joinpath(user_form_dict["group_ID"],
                                                                     "{}_user_jupyterhub_config.yaml".format(
                                                                         user_form_dict["group_ID"])), cluster_config,
                                        config_folder)

    for cmd in cmds:
        try:
            if not create_script:
                subprocess.run(cmd.split(' '))
            else:
                script.append(cmd)
        except:
            print(cmd.split(' '))

    if create_script:
        with open(Path(config_folder).joinpath(user_form_dict["group_ID"],
                                               "{}_namespace.sh".format(user_form_dict["group_ID"])), "w") as f:
            f.write("\n".join(script))


if __name__ == "__main__":
    main()

    # Deploy MONAI Label for model

    # Binaries needed Kustomize MC
    # curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh"  | bash
