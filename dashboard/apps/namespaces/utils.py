import requests
import json
import numpy as np
from django.conf import settings

def get_namespaces(id_token):
    namespace_list = []
    for API_URL in settings.API_URL:
        if API_URL in settings.PRIVATE_CLUSTERS:
            token = settings.PRIVATE_CLUSTERS[API_URL]
            try:
                response = requests.get(
                API_URL + "/api/v1/namespaces",
                headers={"Authorization": "Bearer {}".format(token)}, verify=False)
            except:
                continue
        else:
            try:
                response = requests.get(
            API_URL + "/api/v1/namespaces",
            headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
            except:
                continue
        namespaces = json.loads(response.text)

        if 'code' in namespaces:
            if namespaces['code'] == 403:
                continue

        for namespace in namespaces['items']:
            namespace_list.append(namespace["metadata"]["name"])
    namespace_list = list(np.unique(namespace_list))
    return namespace_list

def get_pods_for_namespace(id_token, namespace):
    pods_dict = []
    for API_URL in settings.API_URL:
        if API_URL in settings.PRIVATE_CLUSTERS:
            token = settings.PRIVATE_CLUSTERS[API_URL]
            response = requests.get(API_URL + "/api/v1/namespaces/{}/pods".format(namespace),
                                    headers={"Authorization": "Bearer {}".format(token)}, verify=False)
        else:
            response = requests.get(API_URL + "/api/v1/namespaces/{}/pods".format(namespace),
                                headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)

        pods = json.loads(response.text)

        if 'code' in pods:
            if pods['code'] == 403:
                continue
        for pod in pods['items']:
            if API_URL in settings.PRIVATE_CLUSTERS:
                token = settings.PRIVATE_CLUSTERS[API_URL]
                response = requests.get(
                    API_URL + "/api/v1/namespaces/{}/pods/{}/log".format(namespace, pod["metadata"]["name"]),
                    headers={"Authorization": "Bearer {}".format(token)}, verify=False)
            else:
                response = requests.get(API_URL + "/api/v1/namespaces/{}/pods/{}/log".format(namespace,pod["metadata"]["name"]),
                                    headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
            logs = response.text
            logs = logs.split("\n")
            pod["logs"] = ["\n".join(logs[-100:])]
            pods_dict.append(pod)

    return pods_dict


def get_svc_for_namespace(id_token, namespace, user_id, is_admin=False):
    ingress_dict = {}
    services_dict = []
    maia_workspace_ingress = {"models": {}}
    remote_desktop_ingress = {}
    orthanc_list = []

    for API_URL in settings.API_URL:
        if API_URL in settings.PRIVATE_CLUSTERS:
            token = settings.PRIVATE_CLUSTERS[API_URL]
            try:
                response = requests.get(API_URL + "/api/v1/namespaces/{}/services".format(namespace),
                                    headers={"Authorization": "Bearer {}".format(token)}, verify=False)
            except:
                continue
        else:
            try:
                response = requests.get(API_URL + "/api/v1/namespaces/{}/services".format(namespace),
                                headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
            except:
                continue
        services = json.loads(response.text)

        if 'code' in services:
            if services['code'] == 403:
                continue

        for service in services['items']:
            services_dict.append(service)

        if API_URL in settings.PRIVATE_CLUSTERS:
            token = settings.PRIVATE_CLUSTERS[API_URL]
            response = requests.get(API_URL + "/apis/networking.k8s.io/v1/namespaces/{}/ingresses".format(namespace),
                                    headers={"Authorization": "Bearer {}".format(token)}, verify=False)
        else:
            response = requests.get(API_URL + "/apis/networking.k8s.io/v1/namespaces/{}/ingresses".format(namespace),
                                headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
        ingresses = json.loads(response.text)


        for ingress in ingresses['items']:
            for rule in ingress['spec']['rules']:
                for path in rule['http']['paths']:
                    if 'host' not in rule:
                        rule['host'] = settings.DEFAULT_INGRESS_HOST
                    if 'number' in path['backend']['service']['port']:
                        if path['backend']['service']['name']+":"+str(path['backend']['service']['port']['number']) in ingress_dict:
                            ingress_dict[path['backend']['service']['name']+":"+str(path['backend']['service']['port']['number'])].append("https://"+rule['host']+path['path'])
                        else:
                            ingress_dict[path['backend']['service']['name'] + ":" + str(
                                path['backend']['service']['port']['number'])] = ["https://" + rule['host'] + path[
                                'path']]

                    else:
                        if path['backend']['service']['name'] + ":"+path['backend']['service']['port']['name'] in ingress_dict:
                            ingress_dict[path['backend']['service']['name'] + ":"+path['backend']['service']['port']['name']].append("https://" + rule['host'] + path['path'])
                        else:
                            ingress_dict[path['backend']['service']['name'] + ":" + path['backend']['service']['port'][
                                'name']] = ["https://" + rule['host'] + path['path']]
        for ingress_entry in ingress_dict:
            if ingress_entry == 'proxy-public:http':
                maia_workspace_ingress['hub'] = ingress_dict[ingress_entry][0]


        for service in services['items']:
            for port in service['spec']['ports']:
                if 'name' in port and port['name'] == 'remote-desktop-port':
                    hub_url = maia_workspace_ingress['hub']
                    user = service["metadata"]["name"][len("jupyter-"):].replace("-2d", "-").replace("-40", "@").replace("-2e", ".")
                    url = f"{hub_url}/user/{user}/proxy/80/desktop/{user}/"
                    ingress_dict[service['metadata']['name'] + ":" + port['name']] = [url]
                    if user_id == user or is_admin:
                        remote_desktop_ingress[user] = url

        ingress_names = [ingress["metadata"]["name"] for ingress in ingresses["items"]]

        if "mlflow" in ingress_names:
            ingress = maia_workspace_ingress["hub"]
            if len([x for x in ingress.split("/") if x]) == 2:
                maia_workspace_ingress["mlflow"] = maia_workspace_ingress["hub"] + f"mlflow"
            else:
                maia_workspace_ingress["mlflow"] = "https://" + settings.DEFAULT_INGRESS_HOST + f"/{namespace}-mlflow"

        if "minio-console" in ingress_names:
            ingress = maia_workspace_ingress["hub"]
            if len([x for x in ingress.split("/") if x]) == 2:
                maia_workspace_ingress["minio_console"] = maia_workspace_ingress["hub"] + f"minio-console"
            else:
                maia_workspace_ingress[
                    "minio_console"] = "https://" + settings.DEFAULT_INGRESS_HOST + f"/{namespace}-minio-console"

        if "kubeflow" in ingress_names:
            ingress = maia_workspace_ingress["hub"]
            if len([x for x in ingress.split("/") if x]) == 2:
                maia_workspace_ingress["kubeflow"] = maia_workspace_ingress["hub"] + f"kubeflow"
            else:
                maia_workspace_ingress[
                    "kubeflow"] = "https://" + settings.DEFAULT_INGRESS_HOST + f"/{namespace}-kubeflow"


    for ingress_entry in ingress_dict:
        if ingress_entry == 'label-studio-ls-app:80':
            maia_workspace_ingress["label_studio"] = ingress_dict[ingress_entry][0]

        if ingress_entry.endswith("pt-orthanc"):
            if ingress_entry == f"{namespace}:pt-orthanc":
                hub_url = maia_workspace_ingress['hub']
                user = user_id
                maia_workspace_ingress['orthanc'] = f"{hub_url}/user/{user}/proxy/80/{namespace}/" #ingress_dict[ingress_entry][0]
                orthanc_list.append({"name": "Orthanc",
                                      "internal_url": f"http://{namespace}.{namespace}:81/{namespace}/dicom-web",
                                      "url": f"{hub_url}/user/{user}/proxy/80/{namespace}/dicom-web/"})
            else:
                model_name = ingress_entry.split(":")[0]
                if model_name not in maia_workspace_ingress["models"]:
                    maia_workspace_ingress["models"][model_name] = {}
                maia_workspace_ingress["models"][model_name]["orthanc"] = ingress_dict[ingress_entry][0]
                orthanc_list.append({"name": f"Orthanc-{model_name}",
                                      "internal_url": f"http://{model_name}.{namespace}:81/{model_name}/dicom-web",
                                      "url": ingress_dict[ingress_entry][0]+"/dicom-web/"})

        if ingress_entry.endswith("pt-monai-label"):
            if ingress_entry == f"{namespace}:pt-monai-label":
                maia_workspace_ingress['monai_label'] = ingress_dict[ingress_entry][0]
            else:
                model_name = ingress_entry.split(":")[0]
                if model_name not in maia_workspace_ingress["models"]:
                    maia_workspace_ingress["models"][model_name] = {}
                maia_workspace_ingress["models"][model_name]["monai_label"] = ingress_dict[ingress_entry][0]



    if "hub" not in maia_workspace_ingress:
        maia_workspace_ingress["hub"] = "N/A"
    if "orthanc" not in maia_workspace_ingress:
        maia_workspace_ingress["orthanc"] = "N/A"
    if "monai_label" not in maia_workspace_ingress:
        maia_workspace_ingress["monai_label"] = "N/A"
    if "label_studio" not in maia_workspace_ingress:
        maia_workspace_ingress["label_studio"] = "N/A"
    if "kubeflow" not in maia_workspace_ingress:
        maia_workspace_ingress["kubeflow"] = "N/A"
    if "mlflow" not in maia_workspace_ingress:
        maia_workspace_ingress["mlflow"] = "N/A"
    if "minio_console" not in maia_workspace_ingress:
        maia_workspace_ingress["minio_console"] = "N/A"




    return services_dict, ingress_dict, maia_workspace_ingress, remote_desktop_ingress, orthanc_list
def get_nodes(id_token):
    nodes_dict = {}
    for API_URL in settings.API_URL:
        if API_URL in settings.PRIVATE_CLUSTERS:
            token = settings.PRIVATE_CLUSTERS[API_URL]
            response = requests.get(API_URL + "/api/v1/nodes",
                                    headers={"Authorization": "Bearer {}".format(token)}, verify=False)
        else:
            response = requests.get(API_URL + "/api/v1/nodes",
                                headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)

        nodes = json.loads(response.text)

        for node in nodes['items']:
            nodes_dict[node["metadata"]["name"]]= node

    return nodes_dict


def get_jobs(id_token, namespace):
    job_list = []
    for API_URL in settings.API_URL:
        if API_URL in settings.PRIVATE_CLUSTERS:
            token = settings.PRIVATE_CLUSTERS[API_URL]
            response = requests.get(
                API_URL + "/apis/batch/v1/namespaces/{}/jobs".format(
                    namespace),
                headers={"Authorization": "Bearer {}".format(token)}, verify=False)
        else:
            response = requests.get(
            API_URL + "/apis/batch/v1/namespaces/{}/jobs".format(
                namespace),
            headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
        jobs = json.loads(response.text)

        if 'code' in jobs:
            if jobs['code'] == 403:
                continue

        for job in jobs['items']:
            job_list.append(job)

    return job_list

def get_pvclaims(id_token, namespace):
    pvc_list = []
    for API_URL in settings.API_URL:
        if API_URL in settings.PRIVATE_CLUSTERS:
            token = settings.PRIVATE_CLUSTERS[API_URL]
            response = requests.get(
                API_URL + "/api/v1/namespaces/{}/persistentvolumeclaims".format(
                    namespace),
                headers={"Authorization": "Bearer {}".format(token)}, verify=False)
        else:
            response = requests.get(
            API_URL + "/api/v1/namespaces/{}/persistentvolumeclaims".format(
                namespace),
            headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
        pvcs = json.loads(response.text)

        if 'code' in pvcs:
            if pvcs['code'] == 403:
                continue

        for pvc in pvcs['items']:
            if pvc["spec"]["storageClassName"] == "microk8s-hostpath":
                pvc["assigned_node"] = pvc["metadata"]["annotations"]["volume.kubernetes.io/selected-node"]
            else:
                pvc["assigned_node"] = "N/A"
            pvc_list.append(pvc)

    return pvc_list


