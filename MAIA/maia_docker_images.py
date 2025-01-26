from pathlib import Path
import yaml

def deploy_maia_kaniko(namespace, config_folder, cluster_config_dict, release_name, project_id, registry_url, registry_secret_name, image_name, image_tag, subpath, build_args=None):
    """
    Deploys a Kaniko job for building and pushing Docker images to a specified registry.

    Parameters
    ----------
    namespace : str
        The Kubernetes namespace where the Kaniko job will be deployed.
    config_folder : str
        The folder path where the configuration files will be stored.
    cluster_config_dict : dict
        Dictionary containing cluster configuration details, including storage class.
    release_name : str
        The release name for the Kaniko job.
    project_id : str
        The project identifier.
    registry_url : str
        The URL of the Docker registry where the image will be pushed.
    registry_secret_name : str
        The name of the Kubernetes secret for accessing the Docker registry.
    image_name : str
        The name of the Docker image to be built.
    image_tag : str
        The tag of the Docker image to be built.
    subpath : str
        The subpath of the repository where the Dockerfile is located.
    build_args : list, optional
        A list of build arguments to be passed to the Kaniko job.

    Returns
    -------
    dict
        A dictionary containing deployment details including namespace, release name, chart name, repo URL, chart version, and values file path.
    """
    
    kaniko_values = {
        "chart_name": "mkg-kaniko",
        "repo_url": "https://kthcloud.github.io/MAIA/",
        "chart_version": "1.0.3",
        "namespace": "mkg-kaniko",
    }
    
    
    kaniko_values.update(
        {
            "docker_registry_secret": registry_secret_name,
            "pvc": {
                "pvc_type": cluster_config_dict["storage_class"],
                "size": "10Gi",
            },
            "args": [
                "--dockerfile=Dockerfile",
                "--context=git://github.com/kthcloud/MAIA.git", #  #refs/heads/mybranch
                "--context-sub-path="+subpath,
                "--destination={}/{}:{}".format(registry_url, image_name, image_tag),
                "--cache=true",
                "--cache-dir=/cache",
            ],
        }
    )
    
    if build_args:
        for build_arg in build_args:
            kaniko_values["args"].append(f"--build-arg={build_arg}")
    
    
    release_name_values = release_name.replace("-", "_")
    Path(config_folder).joinpath(project_id, f"{release_name_values}_values").mkdir(parents=True, exist_ok=True)
    with open(Path(config_folder).joinpath(project_id, f"{release_name_values}_values", f"{release_name_values}_values.yaml"), "w") as f:
        yaml.dump(kaniko_values, f)
    
    
    return {
        "namespace": namespace,
        "release":  f"{project_id}-{release_name}",
        "chart": kaniko_values["chart_name"],
        "repo": kaniko_values["repo_url"],
        "version": kaniko_values["chart_version"],
        "values": str(Path(config_folder).joinpath(project_id, f"{release_name_values}_values", f"{release_name_values}_values.yaml"))
    }