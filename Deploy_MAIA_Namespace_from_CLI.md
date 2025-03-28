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
ingress_resolver_email: admin@maia.se
k8s_distribution: microk8s-or-rke
keycloak:
  authorize_url: https://<keycloak_url>/realms/<realm_name>/protocol/openid-connect/auth
  client_id: keycloak-client-id
  client_secret: keycloak-client-secret
  issuer_url: https://<keycloak_url>/realms/<realm_name>
  token_url: https://<keycloak_url>/realms/<realm_name>/protocol/openid-connect/token
  userdata_url: https://<keycloak_url>/realms/<realm_name>/protocol/openid-connect/userinfo
keycloak_maia_client_secret: keycloak-maia-client-secret # Keycloak client secret for MAIA client in Core Cluster
maia_dashboard:
  enabled: true
  token: '' # Optional Rancher token for the MAIA Dashboard, to be uses if RBAC is disabled
port_range: # LoadBalancer or NodePorts range assigned to the cluster
- 2022
- 2122
rancher_password: rancher-admin-password
services: # Dashboard links to services
  argocd: https://argocd.maia.se
  dashboard: https://dashboard.maia.se
  grafana: https://grafana.maia.se
  keycloak: https://iam.maia.se
  login: https://login.maia.se
  rancher: https://mgmt.maia.se
  registry: https://registry.maia.se
  traefik: https://traefik.maia.se
shared_storage_class: nfs # StorageClass for shared storage
ssh_hostname:  cluster.domain # Domain for SSH access
ssh_port_type: NodePort # LoadBalancer or NodePort
storage_class: microk8s-hostpath # StorageClass for local storage
traefik_dashboard_password: traefik # Password for Traefik dashboard
traefik_resolver: maiamediumresolver # Traefik resolver
url_type: subdomain # Subpath or Subdomain
```
- `maia-config.yaml`: This file contains the MAIA configuration about the different Docker images and the MAIA components.
```yaml
admin_group_ID: MAIA:admin
argocd_namespace: argocd

maia_monai_toolkit_image: <registry.domain>/maia/monai-toolkit:1.0  # Optional, Docker image for MONAI Toolkit
maia_project_chart: maia-project
maia_project_repo: https://kthcloud.github.io/MAIA/
maia_project_version: X.Y.Z
maia_workspace_image: maia-workspace-image
maia_workspace_version: 'X.Y.Z'
```

## Deploy the MAIA Namespace


```bash
export KUBECONFIG=~/path/to/cluster/kubeconfig
kubectl create namespace <project-name>

export KUBECONFIG=~/path/to/your/kubeconfig/where/argocd/is/installed
MAIA_install_project_toolkit --project-config-file maia-namespace.yaml --cluster-config maia-cluster.yaml  --config-folder /PATH/TO/config_folder --maia-config-file maia-config.yaml
```

If the target cluster is not available from ArgoCD, the command will fail. You can still deploy the MAIA Namespace by using the `--no-argocd` flag. The command will then print to stdout the Helm commands to run to deploy the different components of the MAIA Namespace. You can then run these commands in the target cluster to deploy the MAIA Namespace.




