from pathlib import Path
from omegaconf import OmegaConf
from kubernetes import client, config

def create_prometheus_values(config_folder, project_id, cluster_config_dict, maia_config_dict):
    """
    Generates Prometheus values configuration for a Kubernetes cluster and writes it to a YAML file.

    Parameters
    ----------
    config_folder : str
        The folder where the configuration files will be stored.
    project_id : str
        The project identifier.
    cluster_config_dict : dict
        Dictionary containing cluster configuration details.
    maia_config_dict : dict
        Dictionary containing MAIA configuration details.

    Returns
    -------
    dict
        A dictionary containing the namespace, repository URL, chart version, path to the values file, release name, and chart name.
    """

    config.load_kube_config() 

    # Create a CoreV1Api instance
    v1 = client.CoreV1Api()

    # Get the list of nodes
    nodes = v1.list_node()

    # Extract InternalIP for each node
    internal_ips = []
    for node in nodes.items:
        for address in node.status.addresses:
            if address.type == "InternalIP":
                internal_ips.append(address.address)

    prometheus_values = {
        "namespace": "observability",
        "chart_version": "45.5.0",
        "repo_url": "https://prometheus-community.github.io/helm-charts",
        "chart_name": "kube-prometheus-stack",
    } # TODO: Change this to updated values
    
    admin_group_ID = maia_config_dict["admin_group_ID"]
    domain = cluster_config_dict["domain"]
    prometheus_values.update({
        "kubeControllerManager": {
            "endpoints": internal_ips
        },
        "kubeScheduler": {
            "endpoints": internal_ips
        },
        "grafana": {

            "grafana.ini": {
                "server": {
                    "root_url": "https://grafana."+cluster_config_dict["domain"],
                    "auth.generic_oauth": {
                        "api_url": f"https://iam.{domain}/realms/maia/protocol/openid-connect/userinfo",
                        "auth_url": f"https://iam.{domain}/realms/maia/protocol/openid-connect/auth",
                        "client_id": "maia",
                        "client_secret": cluster_config_dict["keycloak_maia_client_secret"],
                        "enabled": True,
                        "name": "OAuth",
                        "empty_scopes": False,
                        "role_attribute_path": f"contains(groups[*], '{admin_group_ID}') && 'Admin' || 'Viewer'",
                        "scopes": "openid profile email",
                        "team_ids": admin_group_ID,
                        "team_ids_attribute_path": "groups[*]",
                        "teams_url": f"https://iam.{domain}/realms/maia/protocol/openid-connect/userinfo",
                        "token_url": f"https://iam.{domain}/realms/maia/protocol/openid-connect/token"

                }
            }
            },
            "assertNoLeakedSecrets": False,
            "defaultDashboardsEnabled": True,
            "persistence": {
                "enabled": True,
            },
            "ingress": {
                "enabled": True,
                "annotations": {
                },
                "hosts": ["grafana."+cluster_config_dict["domain"]],
                "tls": [
                    {
                        
                        "hosts": ["grafana."+cluster_config_dict["domain"]]
                    }
                ]
            },
            "additionalDataSources": [
                {
                    "name": "loki",
                    "type": "loki",
                    "url": "http://loki-maia-core.observability.svc.cluster.local:3100"
                },
                {
                    "name": "tempo",
                    "type": "tempo",
                    "url": "http://tempo-maia-core.observability.svc.cluster.local:3100"
                }
            ]
        }
    })

    gpu_enabled = True


    if gpu_enabled:
        prometheus_values["prometheus"] = {
            "prometheusSpec": {
                "additionalScrapeConfigs": [
                    {
                        "job_name": "gpu-metrics",
                        "kubernetes_sd_configs": [
                            {
                                "namespaces": {
                                    "names": ["gpu-operator"]
                                },
                                "role": "endpoints"
                            }
                        ],
                        "metrics_path": "/metrics",
                        "relabel_configs": [
                            {
                                "action": "replace",
                                "source_labels": ["__meta_kubernetes_pod_node_name"],
                                "target_label": "kubernetes_node"
                            }
                        ],
                        "scheme": "http",
                        "scrape_interval": "1s"
                    }
                ],
                "serviceMonitorSelectorNilUsesHelmValues": False
            }
        }

    if cluster_config_dict["ingress_class"] == "maia-core-traefik":
        prometheus_values["grafana"]["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.entrypoints"] = "websecure"
        prometheus_values["grafana"]["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls"] = 'true'
        prometheus_values["grafana"]["ingress"]["annotations"]["traefik.ingress.kubernetes.io/router.tls.certresolver"] = cluster_config_dict["traefik_resolver"]
    elif cluster_config_dict["ingress_class"] == "nginx":
        prometheus_values["grafana"]["ingress"]["annotations"]["cert-manager.io/cluster-issuer"] = "cluster-issuer"
        prometheus_values["grafana"]["ingress"]["tls"][0]["secretName"] = "grafana."+cluster_config_dict["domain"]

    Path(config_folder).joinpath(project_id,"prometheus_values").mkdir(parents=True, exist_ok=True)
    with open( Path(config_folder).joinpath(project_id,"prometheus_values","prometheus_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(prometheus_values))

    return {
        "namespace": prometheus_values["namespace"],
        "repo": prometheus_values["repo_url"],
        "version": prometheus_values["chart_version"],
        "values": str( Path(config_folder).joinpath(project_id,"prometheus_values","prometheus_values.yaml")),
        "release": f"{project_id}-prometheus",
        "chart": prometheus_values["chart_name"]
    }

def create_loki_values(config_folder, project_id):
    """
    Creates and writes Loki values configuration to a YAML file and returns deployment details.

    Parameters
    ----------
    config_folder : str
        The path to the configuration folder.
    project_id : str
        The project identifier.

    Returns
    -------
    dict
        A dictionary containing deployment details including namespace, repo URL, chart version, 
        values file path, release name, and chart name.
    """


    loki_values = {
        "namespace": "observability",
        "chart_version": "2.9.9",
        "repo_url": "https://grafana.github.io/helm-charts",
        "chart_name": "loki-stack"
    } # TODO: Change this to updated values

    loki_values.update( {   "grafana": {
            "sidecar": {
                "datasources": {
                    "enabled": False
                }
            }
    }
    })
    

    Path(config_folder).joinpath(project_id,"loki_values").mkdir(parents=True, exist_ok=True)
    with open( Path(config_folder).joinpath(project_id,"loki_values","loki_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(loki_values))

    return {
        "namespace": loki_values["namespace"],
        "repo": loki_values["repo_url"],
        "version": loki_values["chart_version"],
        "values": str( Path(config_folder).joinpath(project_id,"loki_values","loki_values.yaml")),
        "release": f"{project_id}-loki",
        "chart": loki_values["chart_name"]
    }

def create_tempo_values(config_folder, project_id):
    """
    Creates a set of tempo values and writes them to a YAML file in the specified configuration folder.

    Parameters
    ----------
    config_folder : str
        The path to the configuration folder where the tempo values will be stored.
    project_id : str
        The project identifier used to create a subdirectory and name the YAML file.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - namespace (str): The namespace for the tempo values.
        - repo (str): The repository URL for the Helm chart.
        - version (str): The version of the Helm chart.
        - values (str): The path to the generated tempo values YAML file.
        - release (str): The release name for the Helm chart.
        - chart (str): The name of the Helm chart.
    """


    tempo_values = {
        "namespace": "observability",
        "chart_version": "1.0.0",
        "repo_url": "https://grafana.github.io/helm-charts",
        "chart_name": "tempo"
    } # TODO: Change this to updated values
    

    Path(config_folder).joinpath(project_id,"tempo_values").mkdir(parents=True, exist_ok=True)
    with open( Path(config_folder).joinpath(project_id,"tempo_values","tempo_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(tempo_values))

    return {"namespace": tempo_values["namespace"],
     "repo": tempo_values["repo_url"],
     "version": tempo_values["chart_version"],
     "values": str( Path(config_folder).joinpath(project_id,"tempo_values","tempo_values.yaml")),
     "release": f"{project_id}-tempo",
     "chart": tempo_values["chart_name"]
     }

def create_core_toolkit_values(config_folder, project_id, cluster_config_dict):
    """
    Creates and saves the core toolkit values for a Kubernetes cluster.
    This function generates a dictionary of core toolkit values based on the provided
    configuration folder, project ID, and cluster configuration dictionary. It retrieves
    the internal IP addresses of the nodes in the Kubernetes cluster and uses them to
    configure the MetalLB addresses. The generated values are saved to a YAML file.

    Parameters
    ----------
    config_folder : str
        The path to the configuration folder.
    project_id : str
        The project identifier.
    cluster_config_dict : dict
        A dictionary containing cluster configuration details, including 'ingress_class' and 'ingress_resolver_email'.

    Returns
    -------
    dict
        A dictionary containing the namespace, repository URL, chart version, path to the
        values YAML file, release name, and chart name.
    """


    
    config.load_kube_config() 

    # Create a CoreV1Api instance
    v1 = client.CoreV1Api()

    # Get the list of nodes
    nodes = v1.list_node()

    # Extract InternalIP for each node
    internal_ips = []
    for node in nodes.items:
        for address in node.status.addresses:
            if address.type == "InternalIP":
                internal_ips.append(address.address)

    if len(internal_ips) == 1:
        internal_ips.append(internal_ips[0])

    core_toolkit_values = {
        "namespace": "maia-core-toolkit",
        "chart_version": "0.1.7",
        "repo_url": "https://kthcloud.github.io/MAIA/",
        "chart_name": "maia-core-toolkit"
    } # TODO: Change this to updated values

    core_toolkit_values.update({
        "metric_server": {
            "enabled": True
        },

        "dashboard": {
            "enabled": True
        },

        "default_ingress_class": cluster_config_dict["ingress_class"],
        "cert_manager": {
            "enabled": True,
            "email": cluster_config_dict["ingress_resolver_email"]
        },
        "metallb":{
            "enabled": True,
            "addresses": "-".join(internal_ips)
    }
    })
    

    Path(config_folder).joinpath(project_id,"core_toolkit_values").mkdir(parents=True, exist_ok=True)
    with open( Path(config_folder).joinpath(project_id,"core_toolkit_values","core_toolkit_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(core_toolkit_values))


    return {"namespace": core_toolkit_values["namespace"],
     "repo": core_toolkit_values["repo_url"],
     "version": core_toolkit_values["chart_version"],
     "values": str( Path(config_folder).joinpath(project_id,"core_toolkit_values","core_toolkit_values.yaml")),
     "release": f"{project_id}-toolkit",
     "chart": core_toolkit_values["chart_name"]
     }

def create_traefik_values(config_folder, project_id, cluster_config_dict):
    """
    Creates the Traefik values configuration file for a given project and cluster configuration.

    Parameters
    ----------
    config_folder : str
        The path to the configuration folder.
    project_id : str
        The unique identifier for the project.
    cluster_config_dict : dict
        A dictionary containing the cluster configuration.

    Returns
    -------
    dict
        A dictionary containing the namespace, repository URL, chart version, values file path, release name, and chart name for the Traefik deployment.

    Raises
    ------
    OSError
        If there is an error creating the directory or writing the file.

    Example
    -------
    config_folder = "/path/to/config"
    project_id = "my_project"
    cluster_config_dict = {
        "traefik_resolver": "myresolver",
        "ingress_resolver_email": "email@example.com",
        "domain": "example.com",
        "traefik_dashboard_password": "password"
    create_traefik_values(config_folder, project_id, cluster_config_dict)
    """
    Path(config_folder).joinpath(project_id,"traefik_values").mkdir(parents=True, exist_ok=True)

    self_signed_tls = False
    traefik_values = {
        "namespace": "traefik",
        "repo_url": "https://traefik.github.io/charts",
        "chart_name": "traefik",
        "chart_version": "33.2.1"
    } # TODO: Change this to updated values

    resolver = cluster_config_dict["traefik_resolver"]
    acme_email = cluster_config_dict["ingress_resolver_email"]
    traefik_values.update({

        "certificatesResolvers":
        {
            "maiaresolver":
            {
                "acme":
                {
                    "email": cluster_config_dict["ingress_resolver_email"],
                    "httpchallenge": "true",
                    "httpchallenge": {
                        "entryPoint": "web"
                    },
                    "caserver": "https://acme-v02.api.letsencrypt.org/directory",
                    #"caServer": "https://acme-staging-v02.api.letsencrypt.org/directory",
                    "storage": "/data/acme.json"
                }
            }
        },

"ingressRoute": {
    "dashboard": {
        "enabled": True,
        "matchRule": "Host(`{}`)".format("traefik."+cluster_config_dict["domain"]), #&& PathPrefix(`/dashboard`)
        "entryPoints": ["websecure"],
        "tls": {
            "certResolver": cluster_config_dict["traefik_resolver"]
        },
        "middlewares": [
            {"name": "traefik-dashboard-auth"},
            {"name": "traefik-dashboard-replace-path"}
        ]
    }
    },

    "extraObjects": [
    {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": "traefik-dashboard-auth-secret"
        },
        "type": "kubernetes.io/basic-auth",
        "stringData": {
            "username": "admin",
            "password": cluster_config_dict["traefik_dashboard_password"]
        }
    },
    {
        "apiVersion": "traefik.io/v1alpha1",
        "kind": "Middleware",
        "metadata": {
            "name": "traefik-dashboard-replace-path"
        },
        "spec": {
            "replacePathRegex": {
                "regex": "^/dashboard/(.*)",
                "replacement": "/dashboard/$1"
            }

    }
    },
    {
        "apiVersion": "traefik.io/v1alpha1",
        "kind": "Middleware",
        "metadata": {
            "name": "traefik-dashboard-auth"
        },
        "spec": {
            "basicAuth": {
                "secret": "traefik-dashboard-auth-secret"
            }
        }
    }
],

"globalArguments": [
    "--entrypoints.web.http.redirections.entryPoint.to=:443",
    #"--entrypoints.web.http.redirections.entrypoint.to=websecure",
    "--entrypoints.web.http.redirections.entrypoint.scheme=https",
    "--global.checknewversion",
    "--global.sendanonymoususage",
    #"--entrypoints.metrics.address=:9100/tcp",
    #"--entrypoints.traefik.address=:9000/tcp",
    "--api.dashboard=true",
    "--ping=true",
    "--metrics.prometheus=true",
    "--metrics.prometheus.entrypoint=metrics",
    "--providers.kubernetescrd",
    "--providers.kubernetesingress",
    #"--entrypoints.websecure.http.tls=true",
    "--api.insecure",
    "--accesslog",
    #"--entrypoints.web.Address=:80",
    "--entryPoints.websecure.forwardedHeaders.insecure",
    "--entryPoints.web.forwardedHeaders.insecure",
    #"--entrypoints.websecure.Address=:443",
    #f"--certificatesresolvers.{resolver}.acme.httpchallenge=true",
    #f"--certificatesresolvers.{resolver}).acme.httpchallenge.entrypoint=web",
    #f"--certificatesresolvers.{resolver}.acme.email={acme_email}",
    #f"--certificatesresolvers.{resolver}.acme.storage=/data/acme.json",
    #f"--certificatesresolvers.{resolver}.acme.caserver=https://acme-v02.api.letsencrypt.org/directory"
    ]

    })


    if self_signed_tls:
        traefik_values.update({
            "tlsStore": {
                "default": {
                    "defaultCertificate": {
                        "secretName": "wildcard-domain-tls"
                    }
                }
            }
        })


    with open( Path(config_folder).joinpath(project_id,"traefik_values","traefik_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(traefik_values))

    return {"namespace": traefik_values["namespace"],
        "repo": traefik_values["repo_url"],
        "version": traefik_values["chart_version"],
        "values": str( Path(config_folder).joinpath(project_id,"traefik_values","traefik_values.yaml")),
        "release": f"{project_id}-traefik",
        "chart": traefik_values["chart_name"]
        }

def create_metallb_values(config_folder, project_id):
    """
    Creates and writes MetalLB Helm chart values to a YAML file and returns a dictionary with deployment details.

    Parameters
    ----------
    config_folder : str
        The path to the configuration folder where the YAML file will be created.
    project_id : str
        The project identifier used to create a unique directory and release name.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - namespace (str): The Kubernetes namespace for MetalLB.
        - repo (str): The URL of the MetalLB Helm chart repository.
        - version (str): The version of the MetalLB Helm chart.
        - values (str): The file path to the generated YAML values file.
        - release (str): The release name for the MetalLB deployment.
        - chart (str): The name of the MetalLB Helm chart.
    """

    metallb_values = {
        "namespace": "metallb-system",
        "chart_version": "0.14.9",
        "repo_url": "https://metallb.github.io/metallb",
        "chart_name": "metallb"
    } # TODO: Change this to updated values

    Path(config_folder).joinpath(project_id,"metallb_values").mkdir(parents=True, exist_ok=True)
    with open( Path(config_folder).joinpath(project_id,"metallb_values","metallb_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(metallb_values))

    return {"namespace": metallb_values["namespace"],
        "repo": metallb_values["repo_url"],
        "version": metallb_values["chart_version"],
        "values": str( Path(config_folder).joinpath(project_id,"metallb_values","metallb_values.yaml")),
        "release": f"{project_id}-metallb",
        "chart": metallb_values["chart_name"]
        }

def create_cert_manager_values(config_folder, project_id):
    """
    Creates a dictionary of values for configuring cert-manager and writes it to a YAML file.

    Parameters
    ----------
    config_folder : str
        The path to the configuration folder.
    project_id : str
        The project identifier.

    Returns
    -------
    dict
        A dictionary containing the namespace, repository URL, chart version, path to the values file, release name, and chart name.
    """

    cert_manager_chart_info = { 
        "namespace": "cert-manager",
        "chart_version": "1.16.2",
        "repo_url": "https://charts.jetstack.io",
        "chart_name": "cert-manager"
    }

    cert_manager_values = {}
    cert_manager_values.update({
        "crds": {
            "enabled": True
        },
    })

    Path(config_folder).joinpath(project_id,"cert_manager_values").mkdir(parents=True, exist_ok=True)
    Path(config_folder).joinpath(project_id,"cert_manager_chart_info").mkdir(parents=True, exist_ok=True)

    with open( Path(config_folder).joinpath(project_id,"cert_manager_values","cert_manager_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cert_manager_values))
    
    with open( Path(config_folder).joinpath(project_id,"cert_manager_chart_info","cert_manager_chart_info.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cert_manager_chart_info))

    return {"namespace": cert_manager_chart_info["namespace"],
        "repo": cert_manager_chart_info["repo_url"],
        "version": cert_manager_chart_info["chart_version"],
        "values": str( Path(config_folder).joinpath(project_id,"cert_manager_values","cert_manager_values.yaml")),
        "release": f"{project_id}-cert-manager",
        "chart": cert_manager_chart_info["chart_name"]
        }
def create_rancher_values(config_folder, project_id, cluster_config_dict):
    """
    Generates Rancher values configuration and writes it to a YAML file.

    Parameters
    ----------
    config_folder : str
        The path to the configuration folder.
    project_id : str
        The project identifier.
    cluster_config_dict : dict
        A dictionary containing cluster configuration details.

    Returns
    -------
    dict
        A dictionary containing Rancher deployment details including namespace, repo URL, chart version, values file path, release name, and chart name.

    Example
    -------
    config_folder = "/path/to/config"
    project_id = "my_project"
    cluster_config_dict = {
        "traefik_resolver": "myresolver",
        "ingress_resolver_email": "email@example.com",
        "domain": "example.com",
        "rancher_password": "password"
    }
    create_rancher_values(config_folder, project_id, cluster_config_dict)
    """
    
    rancher_values = {
        "namespace": "cattle-system",
        "repo_url": "https://releases.rancher.com/server-charts/latest",
        "chart_name": "rancher",
        "chart_version": "2.10.1"
    } # TODO: Change this to updated values

    rancher_values.update({
        "hostname": "mgmt."+cluster_config_dict["domain"],
        "ingress": {
            "extraAnnotations": {},
            "tls":
                {   
                    "source": "letsEncrypt",
                
                }
            
        },
        "letsEncrypt": {
            "email": cluster_config_dict["ingress_resolver_email"],
            "ingress": {
                "class": cluster_config_dict["ingress_class"],
                
            }
        },
        "bootstrapPassword": cluster_config_dict["rancher_password"],
    })

    if cluster_config_dict["ingress_class"] == "maia-core-traefik":
        rancher_values["ingress"]["extraAnnotations"]["traefik.ingress.kubernetes.io/router.entrypoints"] = "websecure"
        rancher_values["ingress"]["extraAnnotations"]["traefik.ingress.kubernetes.io/router.tls"] = 'true'
        rancher_values["ingress"]["extraAnnotations"]["traefik.ingress.kubernetes.io/router.tls.certresolver"] = cluster_config_dict["traefik_resolver"]
    elif cluster_config_dict["ingress_class"] == "nginx":
        rancher_values["ingress"]["extraAnnotations"]["cert-manager.io/cluster-issuer"] = "cluster-issuer"

    Path(config_folder).joinpath(project_id,"rancher_values").mkdir(parents=True, exist_ok=True)

    with open( Path(config_folder).joinpath(project_id,"rancher_values","rancher_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(rancher_values))

    return {"namespace": rancher_values["namespace"],
        "repo": rancher_values["repo_url"],
        "version": rancher_values["chart_version"],
        "values": str( Path(config_folder).joinpath(project_id,"rancher_values","rancher_values.yaml")),
        "release": f"{project_id}-rancher",
        "chart": rancher_values["chart_name"]
        }

def create_gpu_operator_values(config_folder, project_id, cluster_config_dict):
    """
    Creates GPU operator values configuration for a Kubernetes cluster and writes it to a YAML file.

    Parameters
    ----------
    config_folder : str
        The folder path where the configuration will be saved.
    project_id : str
        The project identifier used to create a unique directory for the configuration.
    cluster_config_dict : dict
        A dictionary containing cluster configuration details, including the Kubernetes distribution.

    Returns
    -------
    dict
        A dictionary containing the namespace, repository URL, chart version, path to the values file, release name, and chart name.
    """

    gpu_operator_values = {
        "namespace": "gpu-operator",
        "chart_version": "24.3.0",
        "repo_url": "https://helm.ngc.nvidia.com/nvidia",
        "chart_name": "gpu-operator"
    } # TODO: Change this to updated values
    
    if cluster_config_dict["k8s_distribution"] == "microk8s":
        gpu_operator_values["toolkit"] = {
            "env": [
                {
                    "name": "CONTAINERD_CONFIG",
                    "value": "/var/snap/microk8s/current/args/containerd-template.toml"
                },
                {
                    "name": "CONTAINERD_SOCKET",
                    "value": "/var/snap/microk8s/common/run/containerd.sock"
                },
                {
                    "name": "CONTAINERD_RUNTIME_CLASS",
                    "value": "nvidia"
                },
                {
                    "name": "CONTAINERD_SET_AS_DEFAULT",
                    "value": "true"
                }
            ]
        }
        

    elif cluster_config_dict["k8s_distribution"] == "rke2":
        gpu_operator_values["toolkit"] = {
            "driver": {
                "enabled": False
            },
            "env": [
                {
                    "name": "CONTAINERD_SOCKET",
                    "value": "/run/k3s/containerd/containerd.sock"
                }
                ,
                {
                    "name": "CONTAINERD_CONFIG",
                    "value": "/var/lib/rancher/rke2/agent/etc/containerd/config.toml.tmpl"
                },
                {
                    "name": "CONTAINERD_RUNTIME_CLASS",
                    "value": "nvidia"
                },
                {
                    "name": "CONTAINERD_SET_AS_DEFAULT",
                    "value": "true"
                }
            ]
        }

    Path(config_folder).joinpath(project_id,"gpu_operator_values").mkdir(parents=True, exist_ok=True)

    with open( Path(config_folder).joinpath(project_id,"gpu_operator_values","gpu_operator_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(gpu_operator_values))

    return {"namespace": gpu_operator_values["namespace"],
        "repo": gpu_operator_values["repo_url"],
        "version": gpu_operator_values["chart_version"],
        "values": str( Path(config_folder).joinpath(project_id,"gpu_operator_values","gpu_operator_values.yaml")),
        "release": f"{project_id}-gpu-operator",
        "chart": gpu_operator_values["chart_name"]
        }

def create_ingress_nginx_values(config_folder, project_id):
    """
    Creates and writes the ingress-nginx Helm chart values to a YAML file.

    Parameters
    ----------
    config_folder : str
        The path to the configuration folder.
    project_id : str
        The unique identifier for the project.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - namespace (str): The namespace for the ingress-nginx.
        - repo (str): The repository URL for the ingress-nginx chart.
        - version (str): The version of the ingress-nginx chart.
        - values (str): The file path to the generated ingress-nginx values YAML file.
        - release (str): The release name for the ingress-nginx chart.
        - chart (str): The name of the ingress-nginx chart.
    """

    ingress_nginx_values = {
        "namespace": "ingress-nginx",
        "repo_url": "https://kubernetes.github.io/ingress-nginx",
        "chart_name": "ingress-nginx",
        "chart_version": "4.11.3"

    } # TODO: Change this to updated values

    Path(config_folder).joinpath(project_id,"ingress_nginx_values").mkdir(parents=True, exist_ok=True)

    with open( Path(config_folder).joinpath(project_id,"ingress_nginx_values","ingress_nginx_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(ingress_nginx_values))

    return {"namespace": ingress_nginx_values["namespace"],
        "repo": ingress_nginx_values["repo_url"],
        "version": ingress_nginx_values["chart_version"],
        "values": str( Path(config_folder).joinpath(project_id,"ingress_nginx_values","ingress_nginx_values.yaml")),
        "release": f"{project_id}-ingress-nginx",
        "chart": ingress_nginx_values["chart_name"]
        }
    
def create_nfs_server_provisioner_values(config_folder, project_id, cluster_config_dict):
    """
    Creates and writes the NFS server provisioner Helm chart values to a YAML file.

    Parameters
    ----------
    config_folder : str
        The path to the configuration folder.
    project_id : str
        The unique identifier for the project.
    cluster_config_dict : dict
        A dictionary containing cluster configuration details, including the NFS server and path.
    
    Returns
    -------
    dict
        A dictionary containing the following keys:
        - namespace (str): The namespace for the NFS server provisioner.
        - repo (str): The repository URL for the NFS server provisioner chart.
        - version (str): The version of the NFS server provisioner chart.
        - values (str): The file path to the generated NFS server provisioner values YAML file.
        - release (str): The release name for the NFS server provisioner chart.
        - chart (str): The name of the NFS server provisioner chart.
    """

    nfs_server_provisioner_values = {
        "namespace": "nfs-server-provisioner",
        "repo_url": "https://kubernetes-sigs.github.io/nfs-subdir-external-provisioner/",
        "chart_name": "nfs-subdir-external-provisioner",
        "chart_version": "4.0.18"
    }
    
    if "nfs_server" not in cluster_config_dict or "nfs_path" not in cluster_config_dict:
        nfs_server_provisioner_values.update({
            "nfs": {
                "server": "nfs-server.default.svc.cluster.local",
                "path": "/exports"
            }
        })
    else:
        nfs_server_provisioner_values.update({
            "nfs": {
                "server": cluster_config_dict["nfs_server"],
                "path": cluster_config_dict["nfs_path"]
            }
            
        })
    
    Path(config_folder).joinpath(project_id,"nfs_provisioner_values").mkdir(parents=True, exist_ok=True)
    
    with open( Path(config_folder).joinpath(project_id,"nfs_provisioner_values","nfs_provisioner_values.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(nfs_server_provisioner_values))
    
    return {"namespace": nfs_server_provisioner_values["namespace"],
        "repo": nfs_server_provisioner_values["repo_url"],
        "version": nfs_server_provisioner_values["chart_version"],
        "values": str( Path(config_folder).joinpath(project_id,"nfs_provisioner_values","nfs_provisioner_values.yaml")),
        "release": f"{project_id}-nfs-server-provisioner",
        "chart": nfs_server_provisioner_values["chart_name"]
        }