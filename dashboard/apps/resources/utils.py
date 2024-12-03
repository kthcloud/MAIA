import requests
import json
from django.conf import settings


def get_available_resources(id_token):

    gpu_dict = {}
    cpu_dict = {}
    ram_dict = {}
    gpu_allocations = {}


    for API_URL in settings.API_URL:
        cluster_name = settings.CLUSTER_NAMES[API_URL]
        if API_URL in settings.PRIVATE_CLUSTERS:
            token = settings.PRIVATE_CLUSTERS[API_URL]
            try:
                response = requests.get(API_URL + "/api/v1/pods",
                                        headers={"Authorization": "Bearer {}".format(token)}, verify=False)
                pods = json.loads(response.text)
                response = requests.get(API_URL + "/api/v1/nodes",
                                        headers={"Authorization": "Bearer {}".format(token)}, verify=False)

                nodes = json.loads(response.text)

            except:
                continue
            # response = requests.get(API_URL+"/apis/search.karmada.io/v1alpha1/proxying/karmada/proxy/api/v1/namespaces",
            #                        headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
        else:
            try:
                response = requests.get(API_URL + "/api/v1/pods",
                                        headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)
                pods = json.loads(response.text)
                response = requests.get(API_URL + "/api/v1/nodes",
                                        headers={"Authorization": "Bearer {}".format(id_token)}, verify=False)

                nodes = json.loads(response.text)
            except:
                continue




        node_status_dict = {}

        for node in nodes['items']:

            node_name = "{}/{}".format(cluster_name,node['metadata']['name'])

            node_status_dict[node_name] = []

            for condition in node['status']['conditions']:
                if condition["type"] == "Ready":
                    node_status_dict[node_name].append(condition["status"])

            if 'unschedulable' in node['spec']:
                node_status_dict[node_name].append(node['spec']['unschedulable'])
            else:
                node_status_dict[node_name].append(False)


        for node in nodes['items']:

            node_name = "{}/{}".format(cluster_name, node['metadata']['name'])

            if 'nvidia.com/gpu.product' in node['metadata']['labels']:
                gpu_name = node['metadata']['labels']['nvidia.com/gpu.product']
            else:
                gpu_name = "N/A"

            if 'nvidia.com/gpu.memory' in node['metadata']['labels']:
                gpu_size = str(round(int(node['metadata']['labels']['nvidia.com/gpu.memory'])/1024))+" Gi"
            else:
                gpu_size = "N/A"

            if 'nvidia.com/gpu' in node['status']['allocatable']:
                n_gpu_allocatable = int(node['status']['allocatable']['nvidia.com/gpu'])
            else:
                n_gpu_allocatable = 0
            n_cpu_allocatable = int(node['status']['allocatable']['cpu'])
            ram_allocatable = float(int(node['status']['allocatable']['memory'][:-2])/(1024.*1024.))

            n_gpu_requested = 0
            n_cpus_requested = 0
            ram_requested = 0
            for pod in pods['items']:
                if 'nodeName' not in pod['spec']:
                    continue
                if pod['spec']['nodeName'] != node_name.split("/")[1]:
                    continue
                cointainers = pod['spec']['containers']
                for container in cointainers:
                    resources = container['resources']

                    cpu = 0
                    ram = 0
                    if 'requests' in resources:
                        req = resources['requests']
                        if 'nvidia.com/gpu' in req:

                            pod_name = pod['metadata']['name']
                            if pod_name.startswith("jupyter"):
                                pod_name = pod_name.replace("-2d", "-").replace("-40", "@").replace("-2e", ".")[len("jupyter-"):]
                            gpu_allocations[pod_name] = {
                                'node': node_name.split("/")[1],
                                'cluster': cluster_name,
                                'namespace': pod['metadata']['namespace'],
                                'gpu': req['nvidia.com/gpu'],
                                'gpu_name': gpu_name,
                                'gpu_size': gpu_size
                            }

                            n_gpu_requested += int(req['nvidia.com/gpu'])
                        if 'cpu' in req:
                            container_cpu = req['cpu']
                            if container_cpu[-1] == 'm':
                                container_cpu = container_cpu[:-1]
                                container_cpu = int(container_cpu) / 1000.
                            else:
                                container_cpu = int(container_cpu)
                            if container_cpu > cpu:
                                cpu = container_cpu
                        if 'memory' in req:
                            if req['memory'][-2:] == "Mi":
                                container_memory = int(req['memory'][:-2])/1024.
                            elif req['memory'][-2:] == "Gi":
                                container_memory = int(req['memory'][:-2])

                            elif req['memory'][-1:] == "M":
                                container_memory = int(req['memory'][:-1])/1024.
                            else:

                                container_memory = int(req['memory'])/(1024*1024*1024)
                            if container_memory > ram:
                                ram = container_memory

                    #if 'limits' in resources:
                    #    lim = resources['limits']
                    #    if 'cpu' in lim:
                    #        container_cpu = lim['cpu']
                    #        if container_cpu[-1] == 'm':
                    #            container_cpu = container_cpu[:-1]
                    #            container_cpu = int(container_cpu)/1000.
                    #        else:
                    #            container_cpu = int(container_cpu)
                    #        if container_cpu > cpu:
                    #            cpu = container_cpu
                    #    if 'memory' in lim:
                    #        if lim['memory'][-2:] == "Mi":
                    #            container_memory = int(lim['memory'][:-2]) / 1024.
                    #        elif lim['memory'][-2:] == "Gi":
                    #            container_memory = int(lim['memory'][:-2])
                    #        if container_memory > ram:
                    #            ram = container_memory
                    n_cpus_requested += cpu
                    ram_requested += ram

            gpu_dict[node_name] = []

            if node_status_dict[node_name][0] != "True":
                gpu_dict[node_name].append("NA")
                gpu_dict[node_name].append("NA")
            elif node_status_dict[node_name][1]:
                gpu_dict[node_name].append("NA")
                gpu_dict[node_name].append("NA")
            else:
                gpu_dict[node_name].append(n_gpu_allocatable-n_gpu_requested)
                gpu_dict[node_name].append(n_gpu_allocatable)
            gpu_dict[node_name].append("{}, {}".format(gpu_name, gpu_size))

            cpu_dict[node_name] = []
            cpu_dict[node_name].append(n_cpu_allocatable- n_cpus_requested)
            cpu_dict[node_name].append(n_cpu_allocatable)
            cpu_dict[node_name].append((n_cpu_allocatable- n_cpus_requested)*100/n_cpu_allocatable)

            ram_dict[node_name] = []
            ram_dict[node_name].append(ram_allocatable - ram_requested)
            ram_dict[node_name].append(ram_allocatable)
            ram_dict[node_name].append((ram_allocatable - ram_requested) * 100 / ram_allocatable)
    return gpu_dict, cpu_dict, ram_dict, gpu_allocations


def get_filtered_available_nodes(gpu_dict, cpu_dict, ram_dict, gpu_request, cpu_request, memory_request):


    filtered_nodes = []
    for node in gpu_dict:
        if int(gpu_dict[node][0]) >= gpu_request and float(cpu_dict[node][0]) >= cpu_request and float(ram_dict[node][0]) >= memory_request:
            filtered_nodes.append(node)

    return {node: gpu_dict[node] for node in filtered_nodes},{node: cpu_dict[node] for node in filtered_nodes},{node: ram_dict[node] for node in filtered_nodes}


