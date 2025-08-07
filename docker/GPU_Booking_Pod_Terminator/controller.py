from random import random
from kubernetes import client, config, watch
import time
from datetime import datetime, timedelta
import logging
from flask import Flask, jsonify
import threading
from flask import request

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Load Kubernetes configuration
config.load_incluster_config()  # Use in-cluster config
# config.load_kube_config()  # Uncomment for local testing

v1 = client.CoreV1Api()


def find_all_expired_pods():
    expired_pods = []
    pods = v1.list_pod_for_all_namespaces(watch=False).items
    for pod in pods:
        annotations = pod.metadata.annotations or {}
        if "terminate-at" in annotations:
            try:
                expiry_time = datetime.strptime(annotations["terminate-at"], "%Y-%m-%dT%H:%M:%SZ")
                if datetime.utcnow() > expiry_time:
                    expired_pods.append(pod)
            except Exception as e:
                logger.error(f"Error processing pod {pod.metadata.name}: {e}")
    return expired_pods

def delete_expired_pod(pod):
    try:
        v1.delete_namespaced_pod(pod.metadata.name, pod.metadata.namespace)
        recreate_pod(pod=pod)
    except Exception as e:
        logger.error(f"Error processing deletion for pod {pod.metadata.name}: {e}")


def delete_expired_pods():
    while True:
        pods = v1.list_pod_for_all_namespaces(watch=False).items
        for pod in pods:
            annotations = pod.metadata.annotations or {}
            if "terminate-at" in annotations:
                try:
                    expiry_time = datetime.strptime(annotations["terminate-at"], "%Y-%m-%dT%H:%M:%SZ")
                    if datetime.utcnow() > expiry_time:
                        logger.info(f"Deleting pod {pod.metadata.name} (expired)")
                        v1.delete_namespaced_pod(pod.metadata.name, pod.metadata.namespace)
                          # Wait 5 seconds before checking the next pod
                        recreate_pod(pod=pod)
                except Exception as e:
                    logger.error(f"Error processing deletion for pod {pod.metadata.name}: {e}")
        time.sleep(30)  # Check every 30 seconds

def recreate_pod(pod):
    # Remove resource limits and requests for nvidia.com/gpu
    if pod.spec.containers:
        for container in pod.spec.containers:
            if container.resources:
                if container.resources.limits and "nvidia.com/gpu" in container.resources.limits:
                    del container.resources.limits["nvidia.com/gpu"]
                if container.resources.requests and "nvidia.com/gpu" in container.resources.requests:
                    del container.resources.requests["nvidia.com/gpu"]
            
            # Modify or add NVIDIA_VISIBLE_DEVICES environment variable
            env_vars = container.env or []
            found = False
            for env in env_vars:
                if env.name == "NVIDIA_VISIBLE_DEVICES":
                    env.value = "None"
                    found = True
                    break
            if not found:
                env_vars.append(client.V1EnvVar(name="NVIDIA_VISIBLE_DEVICES", value="None"))
            container.env = env_vars

    # Remove the "terminate-at" annotation if it exists
    annotations = pod.metadata.annotations or {}
    if "terminate-at" in annotations:
        del annotations["terminate-at"]

    new_pod = client.V1Pod(
        metadata=client.V1ObjectMeta(
            name=pod.metadata.name,
            namespace=pod.metadata.namespace,
            labels=pod.metadata.labels,
            annotations=annotations  # Use the updated annotations
        ),
        spec=pod.spec
    )
    time.sleep(30)
    
    for attempt in range(10):
        try:
            v1.create_namespaced_pod(namespace=pod.metadata.namespace, body=new_pod)
            logger.info(f"Recreated pod {pod.metadata.name}")
            break
        except Exception as e:
            logger.error(f"Error recreating pod {pod.metadata.name} (attempt {attempt + 1}/10): {e}")
            time.sleep(10)  # Wait 5 seconds before retrying
        
    

app = Flask(__name__)

@app.route('/random-delete', methods=['POST'])
def trigger_delete_expired_pods():
    expired_pods = find_all_expired_pods()
    if not expired_pods:
        return jsonify({"status": "no expired pods found"}), 200
    logger.info(f"Found {len(expired_pods)} expired pods, starting deletion process.")
    # Randomly select one pod to delete
    pod_to_delete = random.choice(expired_pods)
    # Run the pod deletion in a background thread to avoid blocking
    logger.info(f"Selected pod {pod_to_delete.metadata.name} from {pod_to_delete.metadata.namespace} for deletion.")
    thread = threading.Thread(target=delete_expired_pod, args=(pod_to_delete,), daemon=True)
    thread.start()
    thread.join()
    return jsonify({"status": "success", "deleted_pod": pod_to_delete.metadata.name}), 200

@app.route('/get-expired-pods', methods=['GET'])
def get_expired_pods():
    expired_pods = find_all_expired_pods()
    if not expired_pods:
        return jsonify({"status": "no expired pods found"}), 200
    pod_names = [pod.metadata.name for pod in expired_pods]
    return jsonify({"status": "success", "expired_pods": pod_names}), 200

@app.route('/delete-expired-pod', methods=['POST'])
def delete_expired_pod_endpoint():
    data = request.json
    pod_name = data.get("pod_name")
    namespace = data.get("namespace")
    if not pod_name or not namespace:
        return jsonify({"status": "error", "message": "pod_name and namespace are required"}), 400

    pod = None
    pods = v1.list_namespaced_pod(namespace=namespace).items
    for p in pods:
        if p.metadata.name == pod_name:
            pod = p
            break
    # Verify if the pod is expired
    expired_pods = find_all_expired_pods()
    if pod not in expired_pods:
        return jsonify({"status": "error", "message": "Pod is not expired"}), 400
    if not pod:
        return jsonify({"status": "error", "message": "Pod not found"}), 404

    logger.info(f"Deleting expired pod {pod.metadata.name} from {pod.metadata.namespace}.")
    delete_expired_pod(pod)
    return jsonify({"status": "success", "deleted_pod": pod.metadata.name}), 200

if __name__ == "__main__":
    #delete_expired_pods()
    app.run(host="0.0.0.0", port=8080)
