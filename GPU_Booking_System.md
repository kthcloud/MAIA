# GPU Booking System

## Motivation

GPU Cards are expensive and are a shared resource in a lab. To make sure that everyone gets a fair share of the GPU resources, a booking system is needed. This booking system will allow users to book a GPU card for a specific time slot. The booking system will also allow users to cancel their booking if they no longer need the GPU card.


![](./GPU-Booking-System.png)

## GPU Booking: Access and Priority Manual

The GPU booking system allows you to reserve a GPU card for a specific time slot and manage your reservations, including cancellations. To ensure equitable resource allocation, the system uses two distinct priority levels and several key booking restrictions.

### 1. Understanding Priority Levels

Access to available GPU resources is governed by two priority tiers:

#### **High Priority**

This status is automatically assigned when you have an **active, currently running booking**. High Priority guarantees your continued access to the reserved GPU for the duration of your slot.

#### **Low Priority**

This status is assigned when you **do not** have an active, running booking.

* **Initial Access:** If GPU cards are available, a Low Priority user will immediately be assigned one.

* **Resource Revocation:** Access for Low Priority users is temporary and contingent on resource availability for High Priority users.

  * If a **High Priority user requests a GPU** and all other cards are busy, your temporarily assigned GPU will be **immediately revoked** and reassigned to the High Priority user.

  * If the High Priority user can be assigned a **different available GPU card**, you will retain your card until all resources are exhausted and another High Priority request occurs.

### 2. Booking Restrictions and Rules

To maintain system integrity and fairness, the following rules apply when attempting to create a new booking:

| Rule | Description | 
 | ----- | ----- | 
| **No Active Overlap** | You **cannot** book a new GPU while you currently have an **active, running booking**. | 
| **Future Booking Limit** | You can only have **one planned (future-dated) booking** in the system at a time. You cannot book a new slot if you already have one scheduled. | 
| **14-Day Cooldown** | After an existing booking ends, you must wait **at least 14 full days** before the start time of any new reservation. The system will calculate and display the earliest valid start date for your next booking. | 
| **Minimum Duration** | The length of your booking must be **at least one day** long. | 
| **Maximum Duration** | The length of your booking **cannot exceed 14 days**. | 

These rules are designed to balance dedicated access for research requiring long runs (High Priority) with opportunistic access for users who need quick computation (Low Priority).

## How the system works
In MAIA there are three component that are interplaying for orchestrating the GPU booking system:
- The GPU Booking System, hosted as a API in the MAIA Dashobard as a database which is responsible for creating and managing the bookings.
- The Pod terminator, which is responsible for terminating the pods with a GPU and recreate them without a GPU, when needed.
- The GPU Booking Admission Controller, which is responsible for the checking if the user has an active booking and assigning the correct priority level to the user pods.

### HIgh priority user requesting a GPU card
- The user request to start a GPU-powered pod
- The GPU Booking Admission Controller will check if the user has an active booking and assign the correct priority level to the user pods.
### Low priority user requesting a GPU card




## GPU Booking Controller Execution Flow

The GPU Booking Admission Controller is implemented as a Kubernetes Mutating Admission Webhook. Its main function is to intercept Pod creation requests, validate them against the external GPU booking system, and mutate the Pod specification to enforce resource and termination policies based on the user's booking status.

---

### **Phase 1: Request Reception and User Identification**

1. **Receive and Validate Request**  
   The controller receives an HTTP POST request at the `/mutate` endpoint. It reads the request body and verifies that it is a valid Kubernetes `AdmissionReview` for a Pod resource.

2. **Identify User**  
   The controller attempts to extract the user's email from the Pod's label:  
   `hub.jupyter.org/username`.

3. **Decode Email**  
   The username is K8s-encoded; the controller decodes it by replacing:
   - `-40` → `@`
   - `-2d` → `-`
   - `-2e` → `.`
   to reconstruct the standard email address.

4. **Bypass Check (No User)**  
   If the required user label is missing, the request is immediately marked as **Allowed** with no mutations, and webhook processing ends.

5. **Load Configuration**  
   The controller loads necessary external API URLs (`API_URL`, `GPU_STATS_URL`) and the authentication token (`API_TOKEN`) from environment variables. The two APIs are referring to the MAIA Dashboard functions for accessing the GPU booking system and the GPU availability monitoring system.

---

### **Phase 2: Booking Status Check (External API Call)**

1. **Query Booking System**  
   The controller sends an authenticated POST request to the external booking service (`API_URL`) with the extracted user email and namespace.

2. **Process Booking Response**  
   The expected JSON response (`ApiResponse`) includes:
   - `Schedulable`: Boolean (true if the user has a current or future booking)
   - `Until`: The end time of the user's reservation
   - `GPU`: The specific GPU type reserved by the user

---

### **Phase 3: Action for Non-Schedulable Users (Low Priority / No Booking)**

If `Schedulable` is **false**, the user is denied guaranteed access.

1. **Flag for Termination**  
   If the Pod requests a GPU resource (`nvidia.com/gpu`), the controller adds a JSON patch to annotate the Pod with `terminate-at` set to a date in the distant past (`1900-01-01`). This signals the scheduler/terminator to reject or terminate the Pod immediately.

2. **Check Cluster Availability**  
   The controller performs a GET request to `GPU_STATS_URL` to determine overall cluster GPU availability (specifically checking the count for `NVIDIA-RTX-A6000` as a default).

3. **Deny GPU Access (If Unavailable)**  
   If the availability check returns zero available GPUs, the controller applies further JSON patches to all containers to:
   - Remove the GPU resource requests and limits (`nvidia.com/gpu`)
   - Add or replace the environment variable `NVIDIA_VISIBLE_DEVICES` with the value `"none"`

4. **Finalize**  
   The request is marked as **Allowed** along with the necessary patches to deny GPU access.

---

### **Phase 4: Action for Schedulable Users (High Priority / Active Booking)**

If `Schedulable` is **true**, the user is authorized.

1. **Check Specific GPU Availability**  
   The controller queries `GPU_STATS_URL` to check the current availability of the specific GPU type (`apiResponse.GPU`) the user has booked.

2. **Trigger Low-Priority Preemption**  
   If the specific booked GPU type is unavailable (i.e., a Low Priority Pod is likely using it), the controller triggers preemption logic by sending a POST request to a dedicated service URL:  
   `http://pod-terminator:8080/random-delete`  
   This action terminates a running low-priority GPU Pod to free a resource for the new, high-priority booking.

3. **Enforce Termination Time**  
   The controller adds a JSON patch to annotate the Pod with `terminate-at`, setting the value to `apiResponse.Until`, ensuring the Pod will automatically terminate when the booking expires.

4. **Finalize**  
   The request is marked as **Allowed** along with the termination-time patch.



## Pod Terminator Service

The **Pod Terminator Service** is a Python-based Flask application that manages the lifecycle of GPU Pods in a Kubernetes cluster, ensuring compliance with GPU booking policies. It enforces booking rules by terminating and rescheduling Pods that have exceeded their reservation or have been pre-empted.

### Core Functions

#### 1. Scheduled Termination of Expired Pods

- The service continuously monitors all running Pods in the cluster.
- It identifies Pods annotated with a `terminate-at` timestamp and currently running a GPU workload.
- If the current UTC time is past the `terminate-at` value, the Pod is deleted.

#### 2. Low-Priority Preemption (`/random-delete` Endpoint)

- This endpoint is invoked by the GPU Booking Controller (see Phase 4 above).
- Upon receiving a request, the service locates all expired Pods (typically low-priority Pods running past their reservation or otherwise flagged).
- It randomly selects one of these expired Pods and deletes it immediately, freeing up a GPU resource for a high-priority booking.

#### 3. Graceful Degraded Reschedule (`recreate_pod`)

- After a Pod is deleted (either due to expiry or preemption), the service attempts to recreate it.
- During recreation:
  - All `nvidia.com/gpu` resource requests and limits are removed.
  - The `NVIDIA_VISIBLE_DEVICES` environment variable is reset.
- This allows the workload to be rescheduled on a standard CPU node (without GPU access), providing graceful degradation instead of a hard crash.

---

In summary, the Pod Terminator Service enforces the `terminate-at` annotation set by the GPU Booking Controller, ensuring that GPU resources are reclaimed promptly and that high-priority bookings are honored. It also ensures that workloads can continue running (albeit without GPU access) after preemption or booking expiry.

---

## Installation

The GPU Booking System is a standalone service that can be installed on any Kubernetes cluster. The installation is performed by deploying the `gpu-booking` Helm chart:

```bash
helm install gpu-booking gpu-booking -n gpu-booking --create-namespace
```

The custom values for the GPU Booking component can be set in the `values.yaml` file. The following values can be customized:
```yaml
apiUrl: "https://maia.app.cloud.cbh.kth.se/maia-api/gpu-schedulability/" # The URL of the GPU Schedulability API service from the MAIA Dashboard
gpuStatsUrl: "https://maia.app.cloud.cbh.kth.se/maia-api/gpu-stats/" # The URL of the GPU Stats API service from the MAIA Dashboard
apiToken: "secret-token" # The API token for the GPU Schedulability API service
namespace: "gpu-booking" # The namespace where the GPU Booking component is deployed
```

## GPU Booking Post Installation Steps


The GPU booking system includes a webhook that validates the GPU requests. The webhook requires a CA certificate to be installed in the Kubernetes cluster. The CA certificate is stored in a secret named `gpu-booking-webhook-tls`. The CA certificate is generated during the installation of the GPU Booking component.

After approving and signing the certificate signing request (CSR) for the webhook, the CA certificate is stored in the secret `gpu-
booking-webhook-tls`. You need to manually extract the CA certificate from the secret and update the webhook configuration to use the CA certificate.


After installing the GPU Booking component as an Helm chart, you need to perform the following steps to complete the installation:

```bash
kubectl patch mutatingwebhookconfiguration gpu-booking-webhook -n gpu-booking --type='json' -p="[{'op': 'replace', 'path': '/webhooks/0/clientConfig/caBundle', 'value': \"$(kubectl get secret gpu-booking-webhook-tls -n gpu-booking -o jsonpath='{.data.ca\.crt}')\"}]"
```