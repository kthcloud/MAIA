# Installation with MicroK8s

To install MAIA, we provide a set of Ansible playbooks that automate the installation process. The playbooks are designed to be run on a fresh Ubuntu  installation (we have currently tested it for Ubuntu 20.04, Ubuntu 22.04 and Ubuntu 24.04). The Kubernetes Distribution used is MicroK8s.

## Prerequisites
- OpenSSH server should be installed on all nodes

- For the GPU nodes, the NVIDIA drivers should be installed on the nodes. The NVIDIA drivers can be installed using the following command:
    ```bash
    sudo apt-get install nvidia-driver-*
    ```

- To clean up the disk space on the nodes, we recommend running the following command:
    ```bash
    bleachbit --list | grep -E "[a-z0-9_\-]+\.[a-z0-9_\-]+" | grep -v system.free_disk_space | xargs  sudo bleachbit --clean
    ```

- Optionally, for managing the disk storage on the local nodes, we provide an additional set of instructions to create, mount and extend virtual disk partitions on the nodes. This is useful for managing the disk storage on the local nodes. The instructions are provided in the [Local Disk Partitioning](./Local_Disk_Partitioning.md) file.


## Inventory

The inventory file is located at `Ansible/inventory.ini`. The inventory file should be updated with the IP addresses of the nodes that will be part of the MAIA cluster.

## MAIA Configuration Files
To install MAIA, you need to create a configuration file that specifies the configuration of the MAIA cluster. The configuration file is a YAML file that specifies the following parameters:

```yaml
# Cluster Domain
domain: "maia-cloud.com"
# Your Email, used for Let's Encrypt Certificate
ingress_resolver_email: admin@maia.se
# Cluster Name
cluster_name: demo-cluster
# Kubernetes Distribution (microk8s)
k8s_distribution: microk8s
```



## 1. Install MicroK8s
[Install Microk8s Playbook](Ansible/Playbooks/install_microk8s.yaml)

To install MicroK8s on the control node, run the following command:
```bash
ansible-playbook -i inventory.ini Ansible/Playbooks/install_microk8s.yaml
```
After the installation is complete, the MicroK8s cluster will be up and running on the control node. The ports 16443, 80 and 443 will be open on the control node.
The KubeConfig file to access the MicroK8s cluster will be available at `./MAIA-kubeconfig.yaml`.

You can verify the status of the MicroK8s cluster by running the following command:
```bash
export KUBECONFIG=./MAIA-kubeconfig.yaml
microk8s kubectl get nodes
```
Or, if the Cluster API is not public, you can use the following command:
```bash
ssh -L 16443:localhost:16443 <control_node_name>
```bash
export KUBECONFIG=./MAIA-kubeconfig.yaml
kubectl config set-cluster microk8s-cluster --server=https://127.0.0.1:16443
kubectl get nodes
```

## 2. Enable OIDC Authentication
[Enable OIDC Authentication Playbook](Ansible/Playbooks/enable_oidc_authentication.yaml)

OIDC authentication is needed to enable user authentication through JWT tokens provided by Keycloak. To enable OIDC authentication on the MicroK8s cluster, run the following command:
```bash
ansible-playbook -i inventory.ini Ansible/Playbooks/enable_oidc_authentication.yaml -e cluster_config=/path/to/cluster_config.yaml
```

## 3. Install ArgoCD
[Enable OIDC Authentication Playbook](Ansible/Playbooks/install_argocd.yaml)

ArgoCD is a GitOps continuous delivery tool for Kubernetes. ArgoCD is used to deploy and manage all the MAIA components and applications in the cluster. To install ArgoCD on the MicroK8s cluster, run the following command:
```bash
ansible-playbook -i inventory.ini Ansible/Playbooks/install_argocd.yaml
```
If the installation is successful, you will be able to access the ArgoCD dashboard at `http://localhost:8080` by port forwarding the ArgoCD service:
```bash
export KUBECONFIG=./MAIA-kubeconfig.yaml
kubectl port-forward svc/argocd-server -n argocd 8080:443
```
The default username for ArgoCD is `admin` and the password can be retrieved from the output of the installation playbook.

## 4. Install MAIA Core
[Install MAIA Core Playbook](Ansible/Playbooks/install_maia_core.yaml)

You can then install the MAIA Core components by running the following command:
```bash
ansible-playbook -i inventory.ini Ansible/Playbooks/install_maia_core.yaml -e cluster_config=/path/to/cluster_config.yaml -e config_folder=/path/to/config_folder
```
After successful execution of the playbook, you can access the ArgoCD dashboard and synchronize the MAIA Core applications that you will need for your cluster.
