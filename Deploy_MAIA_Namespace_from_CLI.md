# MAIA Namespace Deployment from CLI

The main way to deploy the MAIA Namespace is through the MAIA Dashboard. However, in some cases, you may want to deploy the MAIA Namespace from the command line. This guide will show you how to do that.

## Prerequisites

Before you begin, you need to have the following configuration files:

- `maia-namespace.yaml`: This file contains the configuration for the MAIA Namespace.
```yaml
group_ID: group_id # Needs to match the group ID in Keycloak
group_subdomain: group-subdomain # The subdomain for the group
users:
- list
- of
- user
- emails
resources_limits:
  memory:
  - 16 Gi
  - 32 Gi
  cpu:
  - 4
  - 8
gpu_request: 1 # Omit if no GPU is needed
```
- `maia-cluster.yaml`: This file contains the configuration for the Cluster to deploy the MAIA Namespace.
```yaml
api: cluster_api_address
argocd_destination_cluster_address: cluster_destination_address_in_argocd
cluster_name: cluster_name
docker_email: admin@maia.se
docker_password: docker-password
docker_server: docker-registry-url
docker_username: docker-username # Robot Account from Harbor
domain: cluster.domain
imagePullSecrets: docker-registry-secret-name
ingress_class: traefik-or-nginx
maia_metallb_ip: "" # Optional, MetalLB IP address for the cluster, if using MetalLB and Services of type LoadBalancer
metallb_shared_ip: "" #Optional, MetalLB Shared IP for the cluster, if using MetalLB and Services of type LoadBalancer
metallb_ip_pool: "" # Optional, MetalLB IP Pool for the cluster, if using MetalLB and Services of type LoadBalancer
keycloak:
  authorize_url: https://<keycloak_url>/realms/<realm_name>/protocol/openid-connect/auth
  client_id: keycloak-client-id
  client_secret: keycloak-client-secret
  issuer_url: https://<keycloak_url>/realms/<realm_name>
  token_url: https://<keycloak_url>/realms/<realm_name>/protocol/openid-connect/token
  userdata_url: https://<keycloak_url>/realms/<realm_name>/protocol/openid-connect/userinfo
port_range: # LoadBalancer or NodePorts range assigned to the cluster
- 2022
- 2122
shared_storage_class: nfs # StorageClass for shared storage
ssh_hostname:  cluster.domain # Domain for SSH access
ssh_port_type: NodePort # LoadBalancer or NodePort
storage_class: microk8s-hostpath # StorageClass for local storage
traefik_resolver: maiamediumresolver # Traefik resolver
nginx_cluster_issuer: cluster-issuer # If using Nginx Ingress Controller
url_type: subdomain # Subpath or Subdomain
```
- `maia-config.yaml`: This file contains the MAIA configuration about the different Docker images and the MAIA components.
```yaml
admin_group_ID: MAIA:admin
argocd_namespace: argocd
argocd_host
argocd_token:

maia_orthanc_image
maia_orthanc_version

maia_monai_toolkit_image: <registry.domain>/maia/monai-toolkit:1.0  # Optional, Docker image for MONAI Toolkit
maia_project_chart: maia-project
maia_project_repo: https://kthcloud.github.io/MAIA/
maia_project_version: X.Y.Z
maia_workspace_image: maia-workspace-image
maia_workspace_version: 'X.Y.Z'
maia_workspace_pro_image: maia-workspace-image
maia_workspace_pro_version: 'X.Y.Z'
```

## Deploy the MAIA Namespace


```bash
export KUBECONFIG=~/path/to/cluster/kubeconfig
kubectl create namespace <project-name>

export KUBECONFIG=~/path/to/your/kubeconfig/where/argocd/is/installed
MAIA_install_project_toolkit --project-config-file maia-namespace.yaml --cluster-config maia-cluster.yaml  --config-folder /PATH/TO/config_folder --maia-config-file maia-config.yaml
```

If the target cluster is not available from ArgoCD, the command will fail. You can still deploy the MAIA Namespace by using the `--no-argocd` flag. The command will then print to stdout the Helm commands to run to deploy the different components of the MAIA Namespace. You can then run these commands in the target cluster to deploy the MAIA Namespace.




