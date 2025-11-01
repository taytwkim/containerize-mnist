# Containerize MNIST Training

Containerize MNIST training with Docker for reproducible runs and easy hyperparameter sweeps.

## 🚀 Setup

1. Check if docker daemon is running. If not, install and open Docker Desktop.
```bash!
docker info
```

2. Navigate to working directory.
```bash!
cd docker-mnist
```

3. Build docker image
```bash!
docker build -t mnist .

# check image
docker images
```

4. Run container
```bash!
docker run --rm -v "$PWD/data:/data" mnist

# run with training parameters
docker run --rm -v "$PWD/data:/data" mnist --epochs 5 --batch-size 128 --lr 1.0
```

* `--rm`: removes the container after program execution.
* `-v "$PWD/data:/data"`: mounts local `./data`. Our program downloads the dataset only if it is not already available. To save time in repeated runs, download the dataset once and save it.

5. Clean up
```bash!
# Lists all containers (running + stopped)
docker ps -a

# Remove container
docker rm {container_id or name}

# List all images
docker images

# Remove image
docker rmi {image_id or name}
```
