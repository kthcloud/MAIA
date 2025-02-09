package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
)

type ApiResponse struct {
	Schedulable bool      `json:"schedulable"`
	Until       time.Time `json:"until"`
}

func handleMutation(w http.ResponseWriter, r *http.Request) {
	var admissionReviewReq admissionv1.AdmissionReview

	// Read request body
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Could not read request", http.StatusBadRequest)
		return
	}

	// Deserialize request
	if err := json.Unmarshal(body, &admissionReviewReq); err != nil {
		http.Error(w, "Could not parse request", http.StatusBadRequest)
		return
	}

	// Ensure it's a Pod
	req := admissionReviewReq.Request
	if req == nil || req.Kind.Kind != "Pod" {
		http.Error(w, "Request is not for a Pod", http.StatusBadRequest)
		return
	}

	// Deserialize Pod spec
	var pod corev1.Pod
	if err := json.Unmarshal(req.Object.Raw, &pod); err != nil {
		http.Error(w, "Could not deserialize Pod object", http.StatusBadRequest)
		return
	}

	// Check if the label "hub.jupyter.org/username" exists
	userEmail, labelExists := pod.Labels["hub.jupyter.org/username"]
	namespace := pod.Namespace
	if !labelExists {
		// Log info about pod name and namespace
		log.Printf("Pod %s in namespace %s does not have the label 'hub.jupyter.org/username'", pod.Name, pod.Namespace)

		// If the label does not exist, allow the request without modification
		admissionResponse := &admissionv1.AdmissionResponse{
			UID:     req.UID,
			Allowed: true,
		}
		admissionReviewRes := admissionv1.AdmissionReview{
			TypeMeta: admissionReviewReq.TypeMeta,
			Response: admissionResponse,
		}
		respBytes, _ := json.Marshal(admissionReviewRes)
		w.Header().Set("Content-Type", "application/json")
		w.Write(respBytes)
		return
	}

	userEmail = strings.ReplaceAll(userEmail, "-2d", "-")
	userEmail = strings.ReplaceAll(userEmail, "-40", "@")
	userEmail = strings.ReplaceAll(userEmail, "-2e", ".")
	if userEmail == "" {
		userEmail = "nouser@maia.se"
	}
	log.Printf("User email: %s", userEmail)

	token := os.Getenv("API_TOKEN")
	if token == "" {
		http.Error(w, "API token not set", http.StatusInternalServerError)
		return
	}
	api_url := os.Getenv("API_URL")
	if api_url == "" {
		http.Error(w, "API URL not set", http.StatusInternalServerError)
		return
	}
	apiRequestBody, _ := json.Marshal(map[string]string{"user_email": userEmail, "token": token, "namespace": namespace})
	apiResp, err := http.Post(api_url, "application/json", bytes.NewBuffer(apiRequestBody))
	var apiResponse ApiResponse
	if err != nil || apiResp.StatusCode != http.StatusOK {
		// Default to schedulable = false if API call fails
		apiResponse = ApiResponse{Schedulable: false}
	} else {
		defer apiResp.Body.Close()
		if err := json.NewDecoder(apiResp.Body).Decode(&apiResponse); err != nil {
			// Default to schedulable = false if response parsing fails
			apiResponse = ApiResponse{Schedulable: false}
		}
	}

	// Log if the pod for the user is schedulable or not
	if apiResponse.Schedulable {
		log.Printf("Pod for user %s is GPU-schedulable", userEmail)
	} else {
		log.Printf("Pod for user %s is not GPU-schedulable", userEmail)
	}

	if !apiResponse.Schedulable {
		// JSON Patch list
		var patches []map[string]interface{}

		// Iterate through containers to remove GPU requests/limits and add ENV variable
		for i, container := range pod.Spec.Containers {
			// Remove GPU requests if they exist
			if _, exists := container.Resources.Requests["nvidia.com/gpu"]; exists {
				patches = append(patches, map[string]interface{}{
					"op":   "remove",
					"path": fmt.Sprintf("/spec/containers/%d/resources/requests/nvidia.com~1gpu", i),
				})
			}
			// Remove GPU limits if they exist
			if _, exists := container.Resources.Limits["nvidia.com/gpu"]; exists {
				patches = append(patches, map[string]interface{}{
					"op":   "remove",
					"path": fmt.Sprintf("/spec/containers/%d/resources/limits/nvidia.com~1gpu", i),
				})
			}

			// Check if NVIDIA_VISIBLE_DEVICES exists
			envVarExists := false
			for j, envVar := range container.Env {
				if envVar.Name == "NVIDIA_VISIBLE_DEVICES" {
					envVarExists = true
					patches = append(patches, map[string]interface{}{
						"op":    "replace",
						"path":  fmt.Sprintf("/spec/containers/%d/env/%d/value", i, j),
						"value": "none",
					})
					break
				}
			}

			// If NVIDIA_VISIBLE_DEVICES does not exist, add it
			if !envVarExists {
				// Ensure the env field exists
				if len(container.Env) == 0 {
					patches = append(patches, map[string]interface{}{
						"op":    "add",
						"path":  fmt.Sprintf("/spec/containers/%d/env", i),
						"value": []corev1.EnvVar{},
					})
				}

				// Add NVIDIA_VISIBLE_DEVICES=NULL environment variable
				envVarPatch := map[string]interface{}{
					"op":    "add",
					"path":  fmt.Sprintf("/spec/containers/%d/env/-", i),
					"value": map[string]string{"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"},
				}
				patches = append(patches, envVarPatch)
			}
		}

		// Create AdmissionResponse
		admissionResponse := &admissionv1.AdmissionResponse{
			UID:     req.UID,
			Allowed: true,
		}

		if len(patches) > 0 {
			// Apply JSON patch
			patchBytes, _ := json.Marshal(patches)
			admissionResponse.Patch = patchBytes
			pt := admissionv1.PatchTypeJSONPatch
			admissionResponse.PatchType = &pt
		}

		// Create response
		admissionReviewRes := admissionv1.AdmissionReview{
			TypeMeta: admissionReviewReq.TypeMeta,
			Response: admissionResponse,
		}

		// Serialize and return response
		respBytes, _ := json.Marshal(admissionReviewRes)
		w.Header().Set("Content-Type", "application/json")
		w.Write(respBytes)
	} else {
		// If schedulable, allow the request without modification
		// Add terminate-at annotation

		// Create JSON Patch to add the annotation
		patches := []map[string]interface{}{
			{
				"op":    "add",
				"path":  "/metadata/annotations/terminate-at",
				"value": apiResponse.Until.Format(time.RFC3339),
			},
		}

		admissionResponse := &admissionv1.AdmissionResponse{
			UID:     req.UID,
			Allowed: true,
		}

		if len(patches) > 0 {
			// Apply JSON patch
			patchBytes, _ := json.Marshal(patches)
			admissionResponse.Patch = patchBytes
			pt := admissionv1.PatchTypeJSONPatch
			admissionResponse.PatchType = &pt
		}

		admissionReviewRes := admissionv1.AdmissionReview{
			TypeMeta: admissionReviewReq.TypeMeta,
			Response: admissionResponse,
		}
		respBytes, _ := json.Marshal(admissionReviewRes)
		w.Header().Set("Content-Type", "application/json")
		w.Write(respBytes)
	}
}

func main() {
	http.HandleFunc("/mutate", handleMutation)
	log.Println("Starting mutating webhook server on :443")
	log.Fatal(http.ListenAndServeTLS(":443", "/etc/webhook/certs/tls.crt", "/etc/webhook/certs/tls.key", nil))
}
