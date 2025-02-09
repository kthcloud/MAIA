# Use a minimal Go image for building
FROM golang:1.23 AS builder

# Set working directory inside the container
WORKDIR /app

# Copy Go module files and download dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy the webhook source code
COPY webhook.go ./

# Build the webhook server binary
RUN CGO_ENABLED=0 GOOS=linux go build -o webhook webhook.go

# Use a minimal base image for the final container
FROM alpine:latest

WORKDIR /root/

# Copy the compiled binary
COPY --from=builder /app/webhook .

# Copy TLS certificates (Ensure they are generated beforehand)
#COPY certs/tls.crt /etc/webhook/tls.crt
#COPY certs/tls.key /etc/webhook/tls.key

# Expose the webhook port
EXPOSE 443

# Run the webhook server
CMD ["/root/webhook", "--tls-cert=/tls/tls.crt", "--tls-key=/tls/tls.key"]
