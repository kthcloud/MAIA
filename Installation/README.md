# Installation

To install MAIA, we provide a set of Ansible playbooks that automate the installation process. The playbooks are designed to be run on a fresh Ubuntu  installation (we have currently tested it for Ubuntu 20.04, Ubuntu 22.04 and Ubuntu 24.04).

## Prerequisites
- OpenSSH server should be installed on all nodes.
- Ansible should be installed locally on the machine from which you will run the playbooks.
- The hosts should have access to the internet to download necessary packages and dependencies.

## Inventory

The inventory folder is used to define the hosts and their roles in the MAIA installation. The inventory file is structured in a way that allows you to specify different groups of hosts, such as `nfs_server`, `nfs_clients`, `k8s_master`, `k8s_worker`, and `k8s_storage`. Each group can have multiple hosts, and you can define variables specific to each host or group.

See an example inventory file in `inventory`.

## Prepare Hosts

As a first step, you need to edit the `Ansible/inventory/hosts` file to define the hosts and their roles. For example, to simply add a group of hosts without any specific roles, you can add them as follows:

```ini
maia-server-0
maia-server-1
maia-server-2
maia-server-3
```

Be sure to replace `maia-server-0`, `maia-server-1`, etc., with the actual hostnames or IP addresses of your servers.
For convenience you can add the hosts to the `.ssh/config` file as aliases, so you can use them in the playbooks without specifying the full hostname or IP address.
Be sure to get SSH access to the hosts through their aliases by running the following command:

```bash
ssh maia-server-0
ssh maia-server-1
ssh maia-server-2
ssh maia-server-3   
```

### NVIDIA Driver Installation

To install the NVIDIA driver on the hosts, you can use the [Ansible/Playbooks/install_nvidia_drivers.yaml](Ansible/Playbooks/install_nvidia_drivers.yaml) playbook. This playbook will install the NVIDIA driver on all hosts defined in the `nvidia_hosts` group in the inventory file. You can run the playbook with the following command:

```bash
ansible-playbook -i Ansible/inventory -kK Ansible/Playbooks/install_nvidia_drivers.yaml -e ansible_user=maia-admin -e nvidia_driver_package=nvidia-driver-570
```
Where `nvidia_driver_package` can be set to the desired NVIDIA driver package version. The default is `nvidia-driver-570`, and  `ansible_user` is the user with sudo privileges on the hosts.




