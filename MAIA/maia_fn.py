import base64
import json
import os
import subprocess
from pathlib import Path
from pprint import pprint
from secrets import token_urlsafe
from typing import Dict

import kubernetes
import toml
import yaml
from kubernetes import client, config
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


def create_ssh_service(namespace, users, service_type, create_script=False, metallb_shared_ip=None,
                       metallb_ip_pool=None, load_balancer_ip=None):
    kubeconfig = yaml.safe_load(Path(os.environ["KUBECONFIG"]).read_text())
    config.load_kube_config_from_dict(kubeconfig)

    cmds = []
    for user in users:
        jupyterhub_username = user['jupyterhub_username']
        ssh_port = user['ssh_port']
        if metallb_shared_ip is not None and metallb_ip_pool is not None and load_balancer_ip is not None:
            cmds.append(
                f"kubectl create -f - <<EOF\napiVersion: v1\nkind: Service\nmetadata:\n  name: {jupyterhub_username}\n  namespace: {namespace}\n  annotations:\n    metallb.universe.tf/allow-shared-ip: {metallb_shared_ip}\n    metallb.universe.tf/ip-allocated-from-pool: {metallb_ip_pool}\nspec:\n  loadBalancerIP: {load_balancer_ip}\n  ports:\n    - port: {ssh_port}\n      targetPort: 2022\n      name: ssh\n      protocol: TCP\n      {f'nodePort: {ssh_port}' if service_type == 'NodePort' else ''}\n  selector:\n    hub.jupyter.org/username: {jupyterhub_username}\n  type: {service_type}\nEOF")

        else:
            cmds.append(
            f"kubectl create -f - <<EOF\napiVersion: v1\nkind: Service\nmetadata:\n  name: {jupyterhub_username}\n  namespace: {namespace}\nspec:\n  ports:\n    - port: {ssh_port}\n      targetPort: 2022\n      name: ssh\n      protocol: TCP\n      {f'nodePort: {ssh_port}' if service_type == 'NodePort' else ''}\n  selector:\n    hub.jupyter.org/username: {jupyterhub_username}\n  type: {service_type}\nEOF")
    if not create_script:
        with kubernetes.client.ApiClient() as api_client:
            for user in users:
                api_instance = kubernetes.client.CoreV1Api(api_client)
                body = kubernetes.client.V1Service()

                metadata = kubernetes.client.V1ObjectMeta()
                metadata.name = user["jupyterhub_username"]

                body.metadata = metadata

                # Creating spec
                spec = kubernetes.client.V1ServiceSpec()

                # Creating Port object
                port = kubernetes.client.V1ServicePort(port=user["ssh_port"], target_port=2022, name="ssh",
                                                       protocol="TCP")
                if service_type == "NodePort":
                    port.node_port = user["ssh_port"]

                spec.ports = [port]
                spec.type = service_type
                spec.selector = {"hub.jupyter.org/username": user["jupyterhub_username"]}

                body.spec = spec
                try:
                    api_response = api_instance.create_namespaced_service(namespace, body)

                    print(api_response)
                except ApiException as e:
                    print("Exception when calling CoreV1Api->create_namespaced_secret: %s\n" % e)

    return cmds

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

    for request in range(n_requested_ports):
        for port in range(ip_range[0], ip_range[1]):
            if port not in used_port:
                ports.append(port)
                used_port.append(port)
                break

    return ports


def create_namespace(namespace, create_script=False):
    if create_script:
        return f"kubectl create namespace {namespace}"
    kubeconfig = yaml.safe_load(Path(os.environ["KUBECONFIG"]).read_text())
    config.load_kube_config_from_dict(kubeconfig)
    with kubernetes.client.ApiClient() as api_client:
        # Create an instance of the API class
        api_instance = kubernetes.client.CoreV1Api(api_client)
        body = kubernetes.client.V1Namespace(metadata=kubernetes.client.V1ObjectMeta(name=namespace))
        try:
            api_response = api_instance.create_namespace(body)
            print(api_response)
        except ApiException as e:
            print("Exception when calling CoreV1Api->create_namespace: %s\n" % e)

    return f"kubectl create namespace {namespace}"


def create_docker_registry_secret(namespace, secret_name, docker_server, docker_username, docker_password,
                                  create_script=False):
    if create_script:
        return f"kubectl create secret docker-registry {secret_name} --docker-server={docker_server} --docker-username='{docker_username}' --docker-password={docker_password} --namespace {namespace}"
    kubeconfig = yaml.safe_load(Path(os.environ["KUBECONFIG"]).read_text())
    config.load_kube_config_from_dict(kubeconfig)
    with kubernetes.client.ApiClient() as api_client:
        # Create an instance of the API class
        api_instance = kubernetes.client.CoreV1Api(api_client)

        auth = base64.b64encode(f"{docker_username}:{docker_password}".encode("utf-8")).decode("utf-8")
        body = kubernetes.client.V1Secret(
            metadata=kubernetes.client.V1ObjectMeta(name=secret_name),
            type="kubernetes.io/dockerconfigjson",
            data={
                ".dockerconfigjson":
                    base64.b64encode(json.dumps({
                        "auths": {
                            docker_server: {
                                "username": docker_username,
                                "password": docker_password,
                                "auth": auth,
                            }
                        }
                    }).encode("utf-8")).decode("utf-8")

            }
        )


        try:
            api_response = api_instance.create_namespaced_secret(namespace, body)
            print(api_response)
        except ApiException as e:
            print("Exception when calling CoreV1Api->create_namespaced_secret: %s\n" % e)

    return f"kubectl create secret docker-registry {secret_name} --docker-server={docker_server} --docker-username={docker_username} --docker-password={docker_password} --namespace {namespace}"


def create_share_pvc(namespace, pvc_name, storage_class, storage_size, create_script=False):
    if create_script:
        return f"kubectl create -f - <<EOF\napiVersion: v1\nkind: PersistentVolumeClaim\nmetadata:\n  name: {pvc_name}\n  namespace: {namespace}\nspec:\n  accessModes:\n    - ReadWriteMany\n  storageClassName: {storage_class}\n  resources:\n    requests:\n      storage: {storage_size}\nEOF"
    kubeconfig = yaml.safe_load(Path(os.environ["KUBECONFIG"]).read_text())
    config.load_kube_config_from_dict(kubeconfig)
    with kubernetes.client.ApiClient() as api_client:
        # Create an instance of the API class
        api_instance = kubernetes.client.CoreV1Api(api_client)

        body = kubernetes.client.V1PersistentVolumeClaim(
            metadata=kubernetes.client.V1ObjectMeta(name=pvc_name),
            spec=kubernetes.client.V1PersistentVolumeClaimSpec(
                access_modes=["ReadWriteMany"],
                storage_class_name=storage_class,
                resources=kubernetes.client.V1ResourceRequirements(requests={"storage": storage_size})
            )
        )

        try:
            api_response = api_instance.create_namespaced_persistent_volume_claim(namespace, body)
            print(api_response)
        except ApiException as e:
            print("Exception when calling CoreV1Api->create_namespaced_persistent_volume_claim: %s\n" % e)
    return f"kubectl create -f - <<EOF\napiVersion: v1\nkind: PersistentVolumeClaim\nmetadata:\n  name: {pvc_name}\n  namespace: {namespace}\nspec:\n  accessModes:\n    - ReadWriteMany\n  storageClassName: {storage_class}\n  resources:\n    requests:\n      storage: {storage_size}\nEOF"


def deploy_oauth2_proxy(cluster_config, user_config, config_folder, create_script=False):

    config_file = {
        "oidc_issuer_url": cluster_config["keycloack"]["issuer_url"],
        "provider": "oidc",
        "upstreams": ["static://202"],
        "http_address": "0.0.0.0:4180",
        "oidc_groups_claim": "groups",
        "skip_jwt_bearer_tokens": True,
        "oidc_email_claim": "email",
        "allowed_groups": ["MAIA:"+user_config["group_ID"],"MAIA:admin"],
        "scope": "openid email profile",
        "redirect_url": "https://{}.{}/oauth2/callback".format(user_config["group_subdomain"],cluster_config["domain"]),
        "email_domains": ["*"],
        "proxy_prefix": "/oauth2",
        "ssl_insecure_skip_verify": True,
        "insecure_oidc_skip_issuer_verification": True,
        "cookie_secure": True,
        "reverse_proxy": True,
        "pass_access_token": True,
        "pass_authorization_header": True,
        "set_authorization_header": True,
        "set_xauthrequest": True,
        "pass_user_headers": True,
        "whitelist_domains": ["*"]
    }
    #proxy_prefix = "/oauth2-demo"
    #upstreams = [ "file:///dev/null" ]


    if cluster_config["url_type"] == "subpath":
        config_file["redirect_url"] = "https://{}/oauth2-{}/callback".format(cluster_config["domain"],user_config["group_subdomain"])
        config_file["proxy_prefix"] = "/oauth2-{}".format(user_config["group_subdomain"])

    oauth2_proxy_config = {
        "config": {
            "clientID": cluster_config["keycloack"]["client_id"],
            "clientSecret": cluster_config["keycloack"]["client_secret"],
            "cookieSecret": token_urlsafe(16),
            "configFile": toml.dumps(config_file)

        },
        "redis":{
            "enabled": True,
            "global": {
                "storageClass": cluster_config["storage_class"]
            }

        },
        "sessionStorage": {
            "type": "redis",
        },
        "image": {
            "repository": "quay.io/oauth2-proxy/oauth2-proxy",
            "tag": "",
            "pullPolicy": "IfNotPresent"
        },
        "service": {
            "type": "ClusterIP",
            "portNumber": 80,
            "appProtocol": "https",
            "annotations": {

            }
        },
        "serviceAccount": {
            "enabled": True,
            "name": "",
            "automountServiceAccountToken": True,
            "annotations": {

            }},
        "ingress": {
            "enabled": True,
            "path": "/oauth2",
            "pathType": "Prefix",
            "tls":[{"secretName":"{}.{}-tls".format(user_config["group_subdomain"],cluster_config["domain"]),"hosts": ["{}.{}".format(user_config["group_subdomain"],cluster_config["domain"])]}],
            "hosts": ["{}.{}".format(user_config["group_subdomain"],cluster_config["domain"])],
            "annotations": {

            }
        }
    }
    if cluster_config["url_type"] == "subpath":
        oauth2_proxy_config["ingress"]["hosts"] = [cluster_config["domain"]]
        oauth2_proxy_config["ingress"]["tls"][0]["hosts"] = [cluster_config["domain"]]
        oauth2_proxy_config["ingress"]["path"] = "oauth2-{}".format(user_config["group_subdomain"])
    if "nginx_cluster_issuer" in cluster_config:
        oauth2_proxy_config["ingress"]["annotations"]["cert-manager.io/cluster-issuer"] = cluster_config["nginx_cluster_issuer"]
    if "traefik_resolver" in cluster_config:
        oauth2_proxy_config["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.entrypoints"] = "websecure"
        oauth2_proxy_config["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls.certresolver"] = cluster_config["traefik_resolver"]

    with open(Path(config_folder).joinpath(user_config["group_ID"],"{}_oauth2_proxy_values.yaml".format(user_config["group_ID"])), "w") as f:
        yaml.dump(oauth2_proxy_config, f)

    cmds = [
        "helm repo add oauth2-proxy https://oauth2-proxy.github.io/manifests",
        "helm repo update",
        "helm upgrade --install oauth2-proxy oauth2-proxy/oauth2-proxy -f {} --namespace {}".format(
            Path(config_folder).joinpath(user_config["group_ID"],
                                         "{}_oauth2_proxy_values.yaml".format(user_config["group_ID"])),
            user_config["group_ID"].lower())
    ]

    if create_script:

        return {"oauth_url": "{}.{}".format(user_config["group_subdomain"], cluster_config["domain"])}, cmds
    else:
        for cmd in cmds:
            subprocess.run(cmd.split(' '))

    return {"oauth_url": "{}.{}".format(user_config["group_subdomain"], cluster_config["domain"])}, cmds


def deploy_minio_tenant(namespace, config_folder, user_config, cluster_config, create_script=False):
    #kubectl kustomize https://github.com/minio/operator/examples/kustomization/base/
    with open(Path(config_folder).joinpath("tenant-base.yaml"), "r") as f:
        tenant = yaml.safe_load(f)

    tenant["metadata"]["name"] = namespace
    tenant["metadata"]["namespace"] = namespace

    tenant["spec"]["pools"][0]["servers"] = 1
    tenant["spec"]["pools"][0]["volumesPerServer"] = 1

    tenant["spec"]["pools"][0]["volumeClaimTemplate"]["spec"]["storageClassName"] = cluster_config["storage_class"]
    tenant["spec"]["pools"][0]["volumeClaimTemplate"]["spec"]["resources"]["requests"]["storage"] = "100Gi"
    tenant["spec"]["requestAutoCert"] = False
    tenant["spec"]["features"]["domains"]["console"] = "https://{}.{}/minio-console".format(user_config["group_subdomain"],cluster_config["domain"])


    with open(Path(config_folder).joinpath(user_config["group_ID"],"{}_minio_tenant.yaml".format(user_config["group_ID"])), "w") as f:
        yaml.dump(tenant, f)

    with open(Path(config_folder).joinpath("tenant-secret-configuration.yaml"), "r") as f:
        minio_conf = yaml.safe_load(f)

    minio_conf["metadata"]["namespace"] = namespace

    root_user = "admin"
    root_password = token_urlsafe(16)
    root_password = root_password.replace("-","_")
    minio_conf["stringData"][
        "config.env"] = "export MINIO_BROWSER=on\nexport MINIO_IDENTITY_OPENID_CLIENT_SECRET={}\nexport MINIO_IDENTITY_OPENID_CLAIM_NAME=groups\nexport MINIO_IDENTITY_OPENID_SCOPES=email,openid,profile\nexport MINIO_ROOT_USER={}\nexport MINIO_ROOT_PASSWORD={}\nexport MINIO_IDENTITY_OPENID_CONFIG_URL={}\nexport MINIO_IDENTITY_OPENID_CLIENT_ID={}\nexport MINIO_IDENTITY_OPENID_DISPLAY_NAME=MAIA".format(
        cluster_config["keycloack"]["client_secret"], root_user, root_password,
        cluster_config["keycloack"]["issuer_url"] + "/.well-known/openid-configuration",
        cluster_config["keycloack"]["client_id"])


    with open(Path(config_folder).joinpath(user_config["group_ID"],"{}_minio_tenant_secret_configuration.yaml".format(user_config["group_ID"])), "w") as f:
        yaml.dump(minio_conf, f)

    with open(Path(config_folder).joinpath("tenant-secret-user.yaml"), "r") as f:
        minio_user = yaml.safe_load(f)

    access_key = token_urlsafe(16)
    access_key = access_key.replace("-", "_")
    secret_key = token_urlsafe(16)
    secret_key = secret_key.replace("-", "_")
    minio_user["metadata"]["namespace"] = namespace
    minio_user["data"]["CONSOLE_ACCESS_KEY"] = base64.b64encode(access_key.encode("ascii")).decode("ascii")
    minio_user["data"]["CONSOLE_SECRET_KEY"] = base64.b64encode(secret_key.encode("ascii")).decode("ascii")

    with open(Path(config_folder).joinpath(user_config["group_ID"],"{}_minio_tenant_secret_user.yaml".format(user_config["group_ID"])), "w") as f:
        yaml.dump(minio_user, f)

    cmds = [
        "kubectl apply -f {}".format(Path(config_folder).joinpath(user_config["group_ID"],
                                                                  "{}_minio_tenant.yaml".format(
                                                                      user_config["group_ID"]))),
        "kubectl apply -f {}".format(Path(config_folder).joinpath(user_config["group_ID"],
                                                                  "{}_minio_tenant_secret_configuration.yaml".format(
                                                                      user_config["group_ID"]))),
        "kubectl apply -f {}".format(Path(config_folder).joinpath(user_config["group_ID"],
                                                                  "{}_minio_tenant_secret_user.yaml".format(
                                                                      user_config["group_ID"])))
    ]

    if create_script:
        return {
            "minio_console_service": "{}-console".format(namespace),
            "minio_access_key": access_key,
            "minio_secret_key": secret_key,
            "minio_root_user": "admin",
            "minio_root_password": root_password
        }, cmds
    else:
        for cmd in cmds:
            subprocess.run(cmd.split(' '))

    return {
        "minio_console_service": "{}-console".format(namespace),
        "minio_access_key": access_key,
        "minio_secret_key": secret_key,
        "minio_root_user": "admin",
        "minio_root_password": root_password
    }


def deploy_mysql(namespace, cluster_config, user_config, config_folder, create_script=False):
    mysql_pw = token_urlsafe(16)
    mysql_config = {
        "namespace": namespace,
        "chart_name": "mysql-db-v1",
        "docker_image": "mysql",
        "tag": "8.0.28",
        "memory_request": "2Gi",
        "cpu_request": "500m",
        "deployment": True,
        "ports": {
            "mysql": [
                3306
            ]
        },
        "persistent_volume": [
            {
                "mountPath": "/var/lib/mysql",
                "size": "20Gi",
                "access_mode": "ReadWriteMany",
                "pvc_type": cluster_config["storage_class"]
            }
        ],
        "env_variables": {
            "MYSQL_ROOT_PASSWORD": mysql_pw ,
            "MYSQL_USER": namespace,
            "MYSQL_PASSWORD": mysql_pw,
            "MYSQL_DATABASE": "mysql"
        }
    }

    with open(Path(config_folder).joinpath(user_config["group_ID"],"{}_mysql_values.json".format(user_config["group_ID"])), "w") as f:
        json.dump(mysql_config, f)

    if create_script:
        return {"MYSQL_PASSWORD": mysql_pw, "MYSQL_USERNAME": namespace}, [
            "MAIA_deploy_helm_chart --config-file " + str(Path(config_folder).joinpath(user_config["group_ID"],
                                                                                               "{}_mysql_values.json".format(
                                                                                                   user_config[
                                                                                                       "group_ID"])))]
    else:
        subprocess.run(["MAIA_deploy_helm_chart", "--config-file",
                        Path(config_folder).joinpath(user_config["group_ID"],
                                                     "{}_mysql_values.json".format(user_config["group_ID"]))])

    return {"MYSQL_PASSWORD": mysql_pw, "MYSQL_USERNAME": namespace}


def deploy_mlflow(namespace, cluster_config, user_config, config_folder, create_script=False):
    kubeconfig = yaml.safe_load(Path(os.environ["KUBECONFIG"]).read_text())
    config.load_kube_config_from_dict(kubeconfig)

    script = []

    user_pw = token_urlsafe(16)
    if create_script:
        script.append(
            f"kubectl create secret generic {namespace} --from-literal=user={namespace} --from-literal=password={user_pw} --namespace {namespace}")
    else:
        with kubernetes.client.ApiClient() as api_client:
            # Create an instance of the API class
            api_instance = kubernetes.client.CoreV1Api(api_client)

            body = kubernetes.client.V1Secret(
                metadata=kubernetes.client.V1ObjectMeta(name=namespace),
                type="Opaque",
                data=
                {"user": base64.b64encode(namespace.encode("ascii")).decode("ascii"),
                 "password": base64.b64encode(user_pw.encode("ascii")).decode("ascii")
                 }
            )

            try:
                api_response = api_instance.create_namespaced_secret(namespace, body)
                print(api_response)
            except ApiException as e:
                print("Exception when calling CoreV1Api->create_namespaced_secret: %s\n" % e)

    mlflow_config = {
  "namespace": namespace,
  "chart_name": "mlflow-v1",
        "docker_image": "kthcloud/mlflow",
      #"docker_image": "registry.cloud.cbh.kth.se/maia/mlflow",  #TODO
      "tag": "1.1",
  "memory_request": "2Gi",
  "cpu_request": "500m",
  "allocationTime": "180d",
      "ports": {
      "mlflow": [
        5000
      ]
    },
  "user_secret": [
    namespace
  ],
  "user_secret_params": [
    "user",
    "password"
  ],
    "image_pull_secret":    cluster_config["imagePullSecrets"],
  "env_variables": {
    "MYSQL_URL": "mysql-db-v1-mkg",
    "MYSQL_PASSWORD": user_config["MYSQL_PASSWORD"],
    "MYSQL_USER": user_config["MYSQL_USERNAME"],
    "BUCKET_NAME": "mlflow",
    "BUCKET_PATH": "mlflow",
    "AWS_ACCESS_KEY_ID": user_config["minio_access_key"],
    "AWS_SECRET_ACCESS_KEY": user_config["minio_secret_key"],
    "MLFLOW_S3_ENDPOINT_URL": "http://minio:80"
  }
}
    with open(Path(config_folder).joinpath(user_config["group_ID"],"{}_mlflow_values.json".format(user_config["group_ID"])), "w") as f:
        json.dump(mlflow_config, f)

    if create_script:
        script.append(f"MAIA_deploy_helm_chart --config-file " + str(
            Path(config_folder).joinpath(user_config["group_ID"],
                                         "{}_mlflow_values.json".format(user_config["group_ID"]))))
        return {"mlflow_service": "mlflow-v1-mkg"}, script
    else:
        subprocess.run(["MAIA_deploy_helm_chart", "--config-file",
                        Path(config_folder).joinpath(user_config["group_ID"],
                                                     "{}_mlflow_values.json".format(user_config["group_ID"]))])

    return {"mlflow_service": "mlflow-v1-mkg"}


def deploy_orthanc_ohif(namespace, cluster_config, user_config, config_folder, create_script=False):
    orthanc_ohif_config = {
        "pvc" : {
            "pvc_type": cluster_config["storage_class"],
            "access_mode": "ReadWriteMany",
            "size": "10Gi"

        },
        "imagePullSecret": cluster_config["imagePullSecrets"],
        "namespace": namespace,
        "image":{
            #"repository": "registry.cloud.cbh.kth.se/maia/monai-label-ohif", #TODO
            "repository": "kthcloud/monai-label-ohif",
            "tag": "1.14"
        },

        "hostname": "{}.{}".format(user_config["group_subdomain"],cluster_config["domain"]),
        "oauth_url": "{}.{}".format(user_config["group_subdomain"],cluster_config["domain"]),
    }

    with open(Path(config_folder).joinpath(user_config["group_ID"],"{}_orthanc_ohif_values.yaml".format(user_config["group_ID"])), "w") as f:
        yaml.dump(orthanc_ohif_config, f)

    cmds = [
        "helm repo add maia https://kthcloud.github.io/MAIA/",
        "helm repo update",
        "helm upgrade --install {} maia/monai-label-ohif-maia -f {} --namespace {}".format(namespace, Path(
            config_folder).joinpath(user_config["group_ID"],
                                    "{}_orthanc_ohif_values.yaml".format(user_config["group_ID"])), namespace)
    ]

    if create_script:
        return cmds

    for cmd in cmds:
        subprocess.run(cmd.split(' '))

    return cmds


def deploy_label_studio(namespace, cluster_config, user_config, config_folder, create_script=False):
    if "nginx_cluster_issuer" not in cluster_config:
        cluster_config["nginx_cluster_issuer"] = "N/A"
    label_studio_pw = token_urlsafe(16)
    label_studio_config = {
        "app":{
            "ingress":{
                "annotations": {
                    "cert-manager.io/cluster-issuer": cluster_config["nginx_cluster_issuer"]
        },
                "enabled": True,
                "host": "label-studio.{}.{}".format(user_config["group_subdomain"],cluster_config["domain"]),
                "tls": [{"secretName": "label-studio-{}-tls".format(namespace), "hosts": ["label-studio.{}.{}".format(user_config["group_subdomain"],cluster_config["domain"])]}]

    }
        },
    "global":{
        "extraEnvironmentVars": {
            "LABEL_STUDIO_HOST": "https://label-studio.{}.{}".format(user_config["group_subdomain"],cluster_config["domain"]),
            "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": "/label-studio/data/LabelStudio",
            "LABEL_STUDIO_PASSWORD": label_studio_pw,
            "LABEL_STUDIO_USERNAME": user_config["users"][0],
            "LOCAL_FILES_SERVING_ENABLED": "true",
    },
        "persistence":{
            "config": {
                "volume": {
                    "existingClaim": "shared",
                    "storageClass": cluster_config["storage_class"]
                }
        }
    }
    },
        "postgresql":{
            "global":{

                "storageClass": cluster_config["storage_class"]
            },

        }
    }

    if "traefik_resolver" in cluster_config:
        label_studio_config["app"]["ingress"]["annotations"][
            "traefik.ingress.kubernetes.io/router.entrypoints"] = "websecure"
        label_studio_config["app"]["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls"] = "true"
        label_studio_config["app"]["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls.certresolver"] = \
        cluster_config["traefik_resolver"]

    with open(Path(config_folder).joinpath(user_config["group_ID"],"{}_label_studio_values.yaml".format(user_config["group_ID"])), "w") as f:
        yaml.dump(label_studio_config, f)

    cmds = [
        "helm repo add heartrex https://charts.heartex.com/",
        "helm repo update",
        "helm upgrade --install label-studio heartrex/label-studio -f {} -n {} --namespace {}".format(
            Path(config_folder).joinpath(user_config["group_ID"],
                                         "{}_label_studio_values.yaml".format(user_config["group_ID"])), namespace,
            namespace)
    ]
    if create_script:
        return {"label_studio_password": label_studio_pw}, cmds

    for cmd in cmds:
        subprocess.run(cmd.split(' '))

    return {"label_studio_password": label_studio_pw}, cmds


def deploy_kubeflow(namespace, user_config, cluster_config, config_folder, manifest_folder, create_script=False):
    os.environ["namespace"] = namespace
    os.environ["storageClassName"] = cluster_config["storage_class"]
    script = []

    script.append("export namespace={}".format(namespace))
    script.append("export storageClassName={}".format(cluster_config["storage_class"]))
    script.append("kustomize build {} | envsubst | kubectl apply -f -".format(
        Path(manifest_folder).joinpath("kustomize/cluster-scoped-resources")))

    if not create_script:
        ps = subprocess.Popen(
            ("kustomize", "build", Path(manifest_folder).joinpath("kustomize/cluster-scoped-resources")),
            stdout=subprocess.PIPE)
        output = subprocess.check_output(("envsubst",), stdin=ps.stdout)

        ps.wait()

        with open(Path(config_folder).joinpath(user_config["group_ID"],
                                               "{}_kf_custom_resources.yaml".format(user_config["group_ID"])),
                  "w") as f:
            f.write(output.decode("utf-8"))

        subprocess.run(["kubectl", "apply", "-f", Path(config_folder).joinpath(user_config["group_ID"],
                                                                               "{}_kf_custom_resources.yaml".format(
                                                                                   user_config["group_ID"]))])

    script.append("kustomize build {}| envsubst | kubectl apply -f -".format(
        Path(manifest_folder).joinpath("kustomize/env/dev")))
    if not create_script:
        ps = subprocess.Popen(("kustomize", "build", Path(manifest_folder).joinpath("kustomize/env/dev")),
                              stdout=subprocess.PIPE)
        output = subprocess.check_output(("envsubst",), stdin=ps.stdout)

        ps.wait()
        with open(Path(config_folder).joinpath(user_config["group_ID"], "{}_kf.yaml".format(user_config["group_ID"])),
                  "w") as f:
            f.write(output.decode("utf-8"))

        subprocess.run(["kubectl", "apply", "-f", Path(config_folder).joinpath(user_config["group_ID"],
                                                                               "{}_kf.yaml".format(
                                                                                   user_config["group_ID"]))])

    return script


def configure_minio(namespace, user_config, cluster_config, config_folder, create_script=False):
    alias_cmd = "mc alias set {} https://minio.{}.{} {} {}".format(namespace,
                                                                                               user_config["group_subdomain"],
                                                                                               cluster_config["domain"],
                                                                                               user_config["minio_root_user"],
                                                                                               user_config["minio_root_password"])
    cmds = [alias_cmd]
    if not create_script:
        subprocess.run(alias_cmd.split(' '))

    add_user_cmd = "mc admin user add {} {} {}".format(namespace, user_config["minio_access_key"], user_config["minio_secret_key"])

    cmds.append(add_user_cmd)
    if not create_script:
        subprocess.run(add_user_cmd.split(' '))

    attach_policy_cmd = "mc admin policy attach {} readwrite --user {}".format(namespace,user_config["minio_access_key"])
    cmds.append(attach_policy_cmd)

    if not create_script:
        subprocess.run(attach_policy_cmd.split(' '))

    add_maia_policy = "mc admin policy create {} MAIA:{} {}/readwrite.json".format(namespace,user_config["group_ID"],config_folder)
    cmds.append(add_maia_policy)

    if not create_script:
        subprocess.run(add_maia_policy.split(' '))

    add_maia_admin_policy = "mc admin policy create {} MAIA:admin {}/admin.json".format(namespace,
                                                                                   config_folder)
    cmds.append(add_maia_admin_policy)

    if not create_script:
        subprocess.run(add_maia_admin_policy.split(' '))

    return cmds
