# Docker Volumn http://dockone.io/article/128
# Basics https://www.gitbook.com/book/philipzheng/docker_practice/details 
# https://zhuanlan.zhihu.com/p/47077536

sudo groupadd docker
sudo gpasswd -a $USER docker

# Build Up Dockerfile
vim Dockerfile
# Build an image from a Dockerfile
docker build -t ${tag_name} folder_path
# Start a container
docker run -it -v local_mount_path:docker_path ${tag_name} --name ${container_name}

# Start a container detached it to run in the background
docker run -it -d -v /:/root tag_name

# Run a command in a running container
docker exec -it PID bash

# List running docker process
docker ps
# list all container
docker ps -a
# newest docker process
docker ps -ql

# Shut down docker
docker stop PID

# Boot docker
docker start PID

# Enter container
docker exec -it PID bash

# Kill
docker kill PID

# Remove Container
docker rm Container ID

# Remove Image
docker image rm 

# Kill weeks ago
docker ps --filter "status=exited" | grep 'weeks ago' | awk '{print $1}' | xargs  docker rm

# Remove all stopped containers
docker container prune
