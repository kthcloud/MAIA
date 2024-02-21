import subprocess
from pprint import pprint
from typing import Dict

import kubernetes
import yaml
from kubernetes import config
from kubernetes.client.rest import ApiException


def create_config_map_from_data(data: str, config_map_name: str, namespace: str, kubeconfig_dict: Dict,
                                data_key: str = "values.yaml"):
    """
    Function to create a ConfigMap on a Kubernetes Cluster.

    Parameters
    ----------
    data :
        String containing the content of the ConfigMap to dump.
    config_map_name :
        ConfigMap name.
    namespace   :
        Namespace where to create the ConfigMap
    data_key    :
        value to use as the filename for the content in the ConfigMap.
    kubeconfig_dict :
        Kube Configuration dictionary for Kubernetes cluster authentication.
    """
    config.load_kube_config_from_dict(kubeconfig_dict)
    metadata = kubernetes.client.V1ObjectMeta(

        name=config_map_name,
        namespace=namespace,
    )

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


def build_and_upload_image_on_docker_registry(config_dict: Dict, kubeconfig_dict: Dict, private_registry: str,
                                              interactive: bool = False):
    """
    Function to create and start a Kaniko-based Hive workflow to build and upload a Docker image on a Docker Registry.

    The workflow is structured as follows:
        1.  A ConfigMap for the maiakubegate-kaniko chart is created from the given configuration.
        2.  The maiakubegate-kaniko chart is deployed on the cluster through the hive-deploy chart.
        3. The Docker Image is built and pushed on the registry from the Kaniko environment.
        4. After the Docker image is successfully pushed to the registry, the hive-deploy chart and the ConfigMap are deleted.

    Parameters
    ----------
    interactive :
        Flag to enable interactive mode. Waits for the process to finish before exiting.
    config_dict :
        Configuration dictionary for building and pushing the Docker image.
    kubeconfig_dict :
        Kube Configuration dictionary for Kubernetes cluster authentication.
    private_registry    :
        Docker private registry to push the Docker Image.

    Returns
    -------
    Docker Image URL on the private registry.
    """
    api_server = kubeconfig_dict["clusters"][0]["cluster"]["server"]
    id_token = kubeconfig_dict["users"][0]["user"]["auth-provider"]["config"]["id-token"]
    # ca_cert = base64.b64decode(kubeconfig_dict["clusters"][0]["cluster"]["certificate-authority-data"])

    config.load_kube_config_from_dict(kubeconfig_dict)
    helm_values_kaniko = [
        "docker_registry_secret=docker-registry-secret",
    ]  # TODO

    values = {}
    for helm_value in helm_values_kaniko:
        key, val = helm_value.split("=")
        values[key] = val

    values["kaniko_args"] = [
        '--dockerfile=Dockerfile',
        '--context=git://{}'.format(config_dict["git_path"]),
        '--context-sub-path={}'.format(config_dict["git_subpath"]),
        '--destination={}/{}:{}'.format(private_registry, config_dict["docker_image"].lower(),
                                        config_dict["tag"])
    ]
    if "build_args" in config_dict:
        for build_arg in config_dict["build_args"]:
            values["kaniko_args"].append("--build-arg={}={}".format(build_arg, config_dict["build_args"][build_arg]))

    if 'memory_request' in config_dict:
        values["resources"] = {}
        values["resources"]["requests"] = {}
        values["resources"]["limits"] = {}
        values["resources"]["requests"]["memory"] = config_dict['memory_request']
        values["resources"]["limits"]["memory"] = config_dict['memory_request']
    if 'cpu_request' in config_dict:
        if "resources" not in values:
            values["resources"] = {}
            values["resources"]["requests"] = {}
            values["resources"]["limits"] = {}
        values["resources"]["requests"]["cpu"] = config_dict['cpu_request']
        values["resources"]["limits"]["cpu"] = config_dict['cpu_request']

    # if "GIT_USERNAME" not in os.environ or "GIT_TOKEN" not in os.environ:
    #    print("Process aborted! Please set GIT_USERNAME and GIT_TOKEN before starting again.")
    #    return

    values["git_username"] = config_dict["GIT_USERNAME"]
    values["git_token"] = config_dict["GIT_TOKEN"]

    set_cluster = ""
    for idx, cluster in enumerate(config_dict["clusters"]):
        set_cluster += ",clusters[{}]={}".format(idx, cluster)

    create_config_map_from_data(yaml.dump(values),
                                "{}-values".format(config_dict["docker_image"].lower()),
                                config_dict["namespace"], kubeconfig_dict=kubeconfig_dict)

    chart_url = "https://simonebendazzoli93.github.io/MAIAKubeGate/"  # TODO
    chart_name = "maiakubegate-kaniko"  # TODO
    chart_version = "1.0.0"  # TODO
    sshProcess = subprocess.Popen(["sh"],
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  universal_newlines=True,
                                  bufsize=0,
                                  encoding="utf-8"
                                  )

    # temp_ca_file = tempfile.NamedTemporaryFile()

    # with open(temp_ca_file.name+".crt","wb") as f:
    # f.write(ca_cert)

    sshProcess.stdin.write(
        "helm upgrade --install {} --namespace={} maiakubegate/maiakubegate-deploy  --kube-insecure-skip-tls-verify --kube-apiserver {} --kube-token {} --set chart_url={},chart_name={},chart_version={},config_map={}{}\n".format(
            config_dict["chart_name"] + "-kaniko",
            config_dict["namespace"],
            api_server,
            # temp_ca_file.name+".crt",
            id_token,
            chart_url,
            chart_name,
            chart_version,
            "{}-values".format(config_dict["docker_image"].lower(),
                               ),
            set_cluster
        ))

    if not interactive:
        sshProcess.stdin.close()
        for line in sshProcess.stdout:
            if line == "END\n":
                break
            print(line, end="")
        return "{}/{}".format(private_registry,
                              config_dict["docker_image"].lower())
    sshProcess.stdin.write(
        "sleep 10\n")
    sshProcess.stdin.write(
        "kubectl logs job/{}-kaniko-maiakubegate-deploy-maiakubegate-kaniko -n {} -f\n".format(
            config_dict["chart_name"],
            config_dict["namespace"]))
    sshProcess.stdin.close()
    for line in sshProcess.stdout:
        if line == "END\n":
            break
        print(line, end="")

    sshProcess = subprocess.Popen(["sh"],
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  universal_newlines=True,
                                  encoding="utf-8",
                                  bufsize=0)

    with kubernetes.client.ApiClient() as api_client:
        api_instance = kubernetes.client.CoreV1Api(api_client)
        name = "{}-values".format(config_dict["docker_image"].lower())
        pretty = 'true'
        try:
            api_response = api_instance.delete_namespaced_config_map(name, config_dict["namespace"], pretty=pretty)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CoreV1Api->delete_namespaced_config_map: %s\n" % e)

    sshProcess.stdin.write(
        "helm delete {} --kube-insecure-skip-tls-verify --kube-insecure-skip-tls-verify --kube-apiserver {} --kube-token {} --namespace={}\n".format(
            config_dict["chart_name"] + "-kaniko", api_server,

            id_token, config_dict["namespace"]))
    sshProcess.stdin.close()
    for line in sshProcess.stdout:
        if line == "END\n":
            break
        print(line, end="")

    return "{}/{}".format(private_registry,
                          config_dict["docker_image"].lower())
