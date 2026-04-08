# TritonBench Infra Configuration on Google Cloud Platform

It defines the specification of infrastruture used by TorchBench CI.
The Infra is a Kubernetes cluster built on top of Google Cloud Platform.

## Step 1: Create the cluster and install the ARC Controller

```
# login ghcr.io so that remote can pull the image
docker login ghcr.io

# Get credentials for the cluster so that kubectl could use it
gcloud container clusters get-credentials --location us-east4 meta-triton-h100-runner-cluster

# Install the ARC controller
INSTALLATION_NAME="linux.gcp.h100"
NAMESPACE="arc-systems"
helm install "${INSTALLATION_NAME}" \
    --namespace "${NAMESPACE}" \
    --create-namespace \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller
```

### Maintainence

To uninstall the ARC controller:

```
INSTALLATION_NAME="linux.gcp.h100"
NAMESPACE="arc-systems"
helm uninstall -n "${NAMESPACE}" "${INSTALLATION_NAME}"
```

To inspect the controller installation logs:

```
NAMESPACE="arc-systems"
kubectl get pods -n "${NAMESPACE}"
# get the pod name like linux.gcp.h100-gha-rs-controller-...
kubectl logs -n ${NAMESPACE} linux.gcp.h100-gha-rs-controller-...
```

## Step 2: Create secrets and assign it to the namespace

The secrets need to be added to both `arc-systems` and `arc-runners` namespaces.

```
# Set GitHub App secret
kubectl create secret generic arc-secret \
   --namespace=arc-runners \
   --from-literal=github_app_id=${GITHUB_APP_ID} \
   --from-literal=github_app_installation_id=${GITHUB_APP_INSTALL_ID} \
   --from-file=github_app_private_key=${GITHUB_APP_PRIVKEY_FILE}

# Alternatively, set classic PAT
kubectl create secret generic arc-secret \
   --namespace=arc-runners \
   --from-literal=github_token="<GITHUB_PAT>"
```

To get, delete, or update the secrets:

```
# Get
kubectl get -A secrets
# Delete
kubectl delete secrets -n arc-runners arc-secret
# Update
kubectl edit secrets -n arc-runners arc-secret
```

## Step 3: Install runner scale set

```
INSTALLATION_NAME="linux.gcp.h100"
NAMESPACE="arc-runners"
GITHUB_SECRET_NAME="arc-secret"
helm install "${INSTALLATION_NAME}" \
    --namespace "${NAMESPACE}" \
    --create-namespace \
    -f values.yaml \
    oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set
```

To upgrade or uninstall the runner scale set:

```
# command to upgrade
helm upgrade --install linux.gcp.h100 -n arc-runners -f ./values.yaml oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

# command to uninstall
helm uninstall -n arc-runners linux.gcp.h100
```

To inspect runner sacle set logs:

```
kubectl get pods -n arc-runners
# get arc runner name like linux.gcp.h100-...
# inspect the logs
kubectl logs -n arc-runners linux.gcp.h100-...
```
