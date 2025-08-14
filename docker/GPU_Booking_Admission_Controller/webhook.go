package main

import (
	"bytes"
	"encoding/json"
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
	GPU         string    `json:"gpu"`
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
	gpu_stats_url := os.Getenv("GPU_STATS_URL")
	if api_url == "" {
		http.Error(w, "API URL not set", http.StatusInternalServerError)
		return
	}
	apiRequestBody, _ := json.Marshal(map[string]string{"user_email": userEmail, "token": token, "namespace": namespace})
	apiResp, err := http.Post(api_url, "application/json", bytes.NewBuffer(apiRequestBody))
	var apiResponse ApiResponse
	if err != nil || apiResp.StatusCode != http.StatusOK {
		// Default to schedulable = false if API call fails
		apiResponse = ApiResponse{Schedulable: false, GPU: "unknown"}
	} else {
		defer apiResp.Body.Close()
		if err := json.NewDecoder(apiResp.Body).Decode(&apiResponse); err != nil {
			// Default to schedulable = false if response parsing fails
			apiResponse = ApiResponse{Schedulable: false, GPU: "unknown"}
		}
	}

	// Log if the pod for the user is schedulable or not
	if apiResponse.Schedulable {
		log.Printf("Pod for user %s is GPU-schedulable", userEmail)
	} else {
		log.Printf("Pod for user %s is not GPU-schedulable", userEmail)
	}

	if !apiResponse.Schedulable {
		patches := []map[string]interface{}{
			{
				"op":    "add",
				"path":  "/metadata/annotations/terminate-at",
				"value": time.Date(1900, 1, 1, 8, 0, 0, 0, time.UTC).Format(time.RFC3339),
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
	} else {
		// If schedulable, allow the request without modification
		// Add terminate-at annotation

		// Create JSON Patch to add the annotation

		// Check if the requested GPU is available
		// Query the GPU status summary endpoint

		gpuReq, err := http.NewRequest("GET", gpu_stats_url, nil)
		if err != nil {
			log.Printf("Failed to create GPU status request: %v", err)
		} else {
			gpuReq.Header.Set("Content-Type", "application/json")
			client := &http.Client{Timeout: 5 * time.Second}
			gpuResp, err := client.Do(gpuReq)
			if err != nil {
				log.Printf("Failed to get GPU status summary: %v", err)
			} else {
				defer gpuResp.Body.Close()
				// Optionally, parse and use the response here if needed
				log.Printf("GPU status summary response code: %d", gpuResp.StatusCode)
				var response map[string]interface{}
				if err := json.NewDecoder(gpuResp.Body).Decode(&response); err != nil {
					log.Printf("Failed to decode GPU status summary: %v", err)
				} else {
					gpuStatus := response["gpu"]
					log.Printf("GPU status summary: %+v", gpuStatus)
					gpu_to_book := apiResponse.GPU
					available_gpus := 0
					if statusMap, ok := gpuStatus.(map[string]interface{}); ok {
						if val, exists := statusMap[gpu_to_book]; exists {
							switch v := val.(type) {
							case float64:
								available_gpus = int(v)
							case int:
								available_gpus = v
							}
						}
					}
					log.Printf("Available GPUs for %s: %d", gpu_to_book, available_gpus)
				}

			}
		}
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
