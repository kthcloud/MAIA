from django.conf import settings
import requests
import json
import numpy as np
def get_namespaces(id_token):
    namespace_list = []
    for API_URL in settings.API_URL:
        if API_URL in settings.PRIVATE_CLUSTERS:
            token = settings.PRIVATE_CLUSTERS[API_URL]
            try:
                response = requests.get(API_URL + "/api/v1/namespaces",
                                    headers={"Authorization": "Bearer {}".format(token)}, verify=False)
            except:
                continue
        #response = requests.get(API_URL+"/apis/search.karmada.io/v1alpha1/proxying/karmada/proxy/api/v1/namespaces",
        #                        headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
        else:
            try:
                response = requests.get(API_URL+"/api/v1/namespaces",
                                headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
            except:
                continue
        namespaces = json.loads(response.text)

        if 'items' not in namespaces:
            continue
        for namespace in namespaces['items']:
            namespace_list.append(namespace["metadata"]["name"])

    namespace_list = list(np.unique(namespace_list))

    return namespace_list

def get_cluster_status(id_token):
    cluster_dict = {}
    node_status_dict = {}
    for API_URL in settings.API_URL:

    #response = requests.get(API_URL+"/apis/search.karmada.io/v1alpha1/proxying/karmada/proxy/api/v1/nodes",
    #                        headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
        if API_URL in settings.PRIVATE_CLUSTERS:
            token = settings.PRIVATE_CLUSTERS[API_URL]
            try:
                response = requests.get(API_URL + "/api/v1/nodes",
                                    headers={"Authorization": "Bearer {}".format(token)}, verify=False)
            except:
                cluster = settings.CLUSTER_NAMES[API_URL]
                cluster_dict[cluster] = ['Cluster API Not Reachable']
                node_status_dict['Cluster API Not Reachable'] = ['API']
                continue
        else:
            try:
                response = requests.get(API_URL + "/api/v1/nodes",
                            headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
            except:
                cluster = settings.CLUSTER_NAMES[API_URL]
                cluster_dict[cluster] = ['Cluster API Not Reachable']
                node_status_dict['Cluster API Not Reachable'] = ['API']
                continue
        nodes = json.loads(response.text)


        if 'items' not in nodes:
            print(nodes)
            cluster = settings.CLUSTER_NAMES[API_URL]
            cluster_dict[cluster] = ['Cluster API Not Reachable']
            node_status_dict['Cluster API Not Reachable'] = ['API']
            continue
        for node in nodes['items']:
            node_name = node['metadata']['name']
            node_status_dict[node_name] = []
            cluster = settings.CLUSTER_NAMES[API_URL]
            if cluster in cluster_dict:
                cluster_dict[cluster].append(node_name)
            else:
                cluster_dict[cluster] = [node_name]
            for condition in node['status']['conditions']:
                if condition["type"] == "Ready":
                    node_status_dict[node_name].append(condition["status"])

            if 'unschedulable' in node['spec']:
                node_status_dict[node_name].append(node['spec']['unschedulable'])
            else:
                node_status_dict[node_name].append(False)
    return node_status_dict, cluster_dict

