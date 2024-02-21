# MAIAKubeGate

[![Build](https://github.com/MAIA-KTH/MAIAKubeGate/actions/workflows/build.yaml/badge.svg)](https://github.com/MAIA-KTH/MAIAKubeGate/actions/workflows/build.yaml)

MAIAKubeGate is a python package to interact with a Kubernetes cluster, to create custom environments and deploy
applications in MAIA (including pods, services and ingresses).
The package uses Helm charts to deploy the applications, and it is available as a Helm
chart: [MAIAKubeGate](https://github.com/SimoneBendazzoli93/MAIAKubeGate).

With the **MAIAKubeGate** chart it is possible to deploy any *Docker Image* as a Pod, expose the required ports as
services, mount persistent volumes on the specified locations and optionally create Ingress resources to expose the
application to the external traffic using the HTTPS protocol.

To add the chart to Helm, run:

```
helm repo add maiakubegate https://github.com/SimoneBendazzoli93/MAIAKubeGate
helm repo update
```

## Custom Helm values

A number of custom parameters can be specified for the Helm chart, including the Docker image to deploy, the port to
expose, etc.

The custom configuration is set in a JSON configuration file, following the conventions described below.

### General Configuration

#### Namespace [Required]

Specify the Cluster Namespace where to deploy the resources

```json
{
  "namespace": "NAMESPACE_NAME"
}
```

#### Chart Name [Required]

Specify the Helm Chart Release name

```json
{
  "chart_name": "Helm_Chart_name"
}
```

#### Docker image [Required]

To specify the Docker image to deploy

```json
{
  "docker_image": "DOCKER_IMAGE"
}
```

#### Clusters [Required]

List of Kubernetes clusters in the federation where to deploy the resources

```json
{
  "clusters": [
    "CLUSTER_1",
    "CLUSTER_2"
  ]
}
```

#### Requested Resources

To request resources (RAM,CPU and optionally GPU).

```json
{
  "memory_request": "REQUESTED_RAM_SIZE",
  "cpu_request": "REQUESTED_CPUs"
}
```

Optionally, to request GPU usage:

```json
{
  "gpu_request": "NUMBER_OF_GPUs"
}
```

#### Allocation Time [Required]

Since each environment is deployed as a Job with a fixed allocation time, the user can specify the requested allocation
time (default in days) in the following field:

```json
{
  "allocationTime": "2"
}
```

#### Services

To specify which ports (and corresponding services) can be reached from outside the pod.

```json
{
  "ports": {
    "SERVICE_NAME_1": [
      "PORT_NUMBER"
    ],
    "SERVICE_NAME_2": [
      "PORT_NUMBER"
    ]
  }
}

```

The default *Service Type* is **ClusterIP**. To expose a service as a type **NodePort**:

```json
{
  "service_type": "NodePort",
  "ports": {
    "SERVICE_NAME_1": [
      "PORT_NUMBER",
      "NODE_PORT_NUMBER"
    ],
    "SERVICE_NAME_2": [
      "PORT_NUMBER",
      "NODE_PORT_NUMBER"
    ]
  }
}

```

#### Persistent Volumes

2 different types of persistent volumes are available: **hostPath** (local folder) and **nfs** (shared nfs folder).
For each of these types, it is possible to request a Persistent Volume via a Persistent Volume Claim.

The *"readOnly"* options can be added to specify the mounted folder as read-only.

Request PVC:

```json
{
  "persistent_volume": [
    {
      "mountPath": "/mount/path_1",
      "size": "VOLUME_SIZE",
      "access_mode": "ACCESS_TYPE",
      "pvc_type": "STORAGE_CLASS"
    },
    {
      "mountPath": "/mount/path_2",
      "size": "VOLUME_SIZE",
      "access_mode": "ACCESS_TYPE",
      "pvc_type": "STORAGE_CLASS"
    }
  ]
}
```

**"STORAGE_CLASS"** can be any of the storage classes available on the cluster:

```
kubectl get sc
```

#### Existing Persistent Volumes

Previously created pv can be mounted into multiple pods (ONLY if the *access mode* was previously set to **ReadWriteMany
** )

```json
{
  "existing_persistent_volume": [
    {
      "name": "EXISTING_PVC_NAME",
      "mountPath": "/mount/path"
    }
  ]
}
```

#### Mounted files

Single files can be mounted inside the Pod. First, a ConfigMap including the file is created, and then it is mounted
into the Pod.

```json
{
  "mount_files": {
    "file_name": [
      "/local/file/path",
      "/file/mount/path"
    ]
  }
}
```

#### Node Selection

To optionally select which node in the cluster to use for deploying the application.

```json
{
  "node_selector": "NODE_NAME"
}
```

#### GPU Selection

To optionally select which available GPUs in the cluster to request. `type` and `vram` attribute can be specified (only
one of them is needed, both can be included).
Example: `type: "RTX-2070-Super"`, `vram: "8G"`

```json
{
  "gpu_selector": {
    "type": "GPU_TYPE",
    "vram": "VRAM_SIZE"
  }
}
```

#### Ingress

Used to create an Ingress resources to access the application at the specified port by using an HTTPS address.

IMPORTANT! The specified DNS needs to be active and connected to the cluster DNS (**".maia.cloud.cbh.kth.se"**)

IMPORTANT! *traefik_resolver* should be explicitly specified, since only oauth-based authenticated users can be
authorized
through the ingress.
Contact the MAIA admin to retrieve this information.

```json
{
  "ingress": {
    "host": "SUBDOMAIN.maia.cloud.cbh.kth.se",
    "port": "SERVICE_PORT",
    "oauth_url": "oauth.MY_NAMESPACE"
  }
}
```

#### Environment variables

To add environment variables, used during the creation and deployment of the pod (i.e., environment variables to specify
for the Docker Image).

```json
{
  "env_variables": {
    "KEY_1": "VAL_1",
    "KEY_2": "VAL_2"
  }
}
```

### Hive Docker Configuration

#### User info

When deploying Hive-based applications, it is possible to create single-multiple user account in the environment.
For each of the users, *username*, *password* *email*, and, optionally, an *ssh public key* are required.
This information is stored inside Secrets:

```
USER_1_SECRET:
    user: USER_1
    password: pw
    email: user@email.com
    ssh_publickey [Optional]: "ssh-rsa ..." 
```

To provide the user information to the Pod:

```json
{
  "user_secret": [
    "user-1-secret",
    "user-2-secret"
  ],
  "user_secret_params": [
    "user",
    "password",
    "email",
    "ssh_publickey"
  ]
}
```

## Configuration File Example

```json
{
  "namespace": "machine-learning",
  "chart_name": "jupyterlab-1-v1",
  "docker_image": "jupyter/scipy-notebook",
  "tag": "latest",
  "memory_request": "4Gi",
  "allocationTime": "2",
  "cpu_request": "5000m",
  "ports": {
    "jupyter": [
      8888
    ]
  },
  "persistent_volume": [
    {
      "mountPath": "/home/jovyan",
      "size": "100Gi",
      "access_mode": "ReadWriteOnce",
      "pvc_type": "local-hostpath"
    }
  ]
}
```

# Tools

Install the **MAIAKubeGate** package running:

```
pip install maiakubegate
```

Requirements:

```
kubectl  # Kubernetes CLI
helm     # Kubernetes Package Manager
```

## Deploy Charts

To deploy a Hive Chart, first create a config file according to the specific requirements (as
described [above](#Custom Helm values)).

After creating the config file, run:

```shell
MAIAKubeGate_deploy_helm_chart --config-file <PATH/TO/CONFIG/FILE>
```
