# MAIA Dashboard - Django Version

This project is a fork of the [Soft UI Dashboard Django](https://appseed.us/product/soft-ui-dashboard/django/) project.

The MAIA Dashboard is a web application built using Django that provides an interface for managing various aspects of MAIA. This README will guide you through setting up the environment, starting the project, and understanding the different apps included in the dashboard.

## Environment Setup

Set the environment by saving the following environment variables in a `env.env` file in the root directory of the project:

```yaml
# OpenID Connect Configuration used for authentication and user management
OIDC_RP_CLIENT_ID: OpenID Connect client ID.
OIDC_RP_CLIENT_SECRET: OpenID Connect client secret.
OIDC_USERNAME: OpenID Connect username.
OIDC_ISSUER_URL: OpenID Connect issuer URL.
OIDC_SERVER_URL: OpenID Connect server URL.
OIDC_REALM_NAME: OpenID Connect realm name.
OIDC_OP_AUTHORIZATION_ENDPOINT: OpenID Connect authorization endpoint.
OIDC_OP_TOKEN_ENDPOINT: OpenID Connect token endpoint.
OIDC_OP_USER_ENDPOINT: OpenID Connect user endpoint.
OIDC_OP_JWKS_ENDPOINT: OpenID Connect JWKS endpoint.
OIDC_RP_SIGN_ALGO: OpenID Connect signing algorithm.
OIDC_RP_SCOPES: OpenID Connect scopes.

DEBUG: Debug mode.
# Server URL used for generating full URLs for the API
SERVER: Server URL.

# MinIO Configuration used for storing and reading PIP/Conda environments
MINIO_URL: MinIO server URL.
BUCKET_NAME: MinIO bucket name.
MINIO_ACCESS_KEY: MinIO access key.
MINIO_SECRET_KEY: MinIO secret key.
MINIO_SECURE: MinIO secure connection flag.

# MySQL Configuration used for storing user data
DB_ENGINE: Database engine.
DB_NAME: Database name.
DB_HOST: Database host.
DB_PORT: Database port.
DB_USERNAME: Database username.
DB_PASS: Database password.

MAX_MEMORY: Maximum memory allocation.
MAX_CPU: Maximum CPU allocation.

# Discord Webhook URL used for sending notifications
DISCORD_URL: Discord webhook URL.

# Default Ingress Host, used for generating full URLs for the Kubernetes Ingress
DEFAULT_INGRESS_HOST: Default ingress host.
```

Additionally, set the following environment paths to deploy MAIA Namespaces from the dashboard:

```yaml	
CONFIG_PATH: Path to the configuration folder (--config-folder).
MAIA_CONFIG_PATH: Path to the MAIA configuration file (--maia-config-file).
CLUSTER_CONFIG_PATH: Path to the cluster configuration file (--cluster-config-file).
```

## MAIA Dashboard

![MAIA Dashboard](image/README/Screenshot%202024-11-09%20112028.png)

The MAIA Dashboard includes several Django apps, each responsible for different functionalities:

### User Registration

The User Registration app allows users to request an account on the MAIA API. Users can specify the desired username, email, and password, together with the requested resources (GPUs, CPUs, Memory), the allocation time, and the project name. Optional custom PIP/Conda environments can also be specified. The request is then sent to the administrators, who can approve or reject it.

![User Registration](image/README/Screenshot%202024-11-09%20112132.png)

### Home

The home page displays an overview of the MAIA API, including the associated clusters and nodes together with their status. For each cluster, the user can access the corresponding Kubernetes Dashboard and Monitoring Dashboard (Grafana), if available.

![Home](image/README/Screenshot%202024-11-09%20112227.png)

### Namespaces

Each user can view the namespaces associated with their account, inspecting the MAIA applications and services running in each namespace (JupyterHub, Orthanc, Remote Desktops, MLFlow, etc.).

![Namespaces](image/README/Screenshot%202024-11-09%20112747.png)

### Resources

The *Resources* page allows administrators to manage the resources associated with the MAIA API. This includes inspecting the available resources (GPUs, CPUs, Memory), and the current GPU allocations.

### User Management

The *MAIA Users and Namespaces* page allows administrators to manage the users associated with the MAIA API. This includes viewing the users who requested an account, approving or rejecting the requests, together with the associated namespaces (projects) and requested resources. Administrators can deploy new namespaces for users and manage the resources associated with each namespace.
