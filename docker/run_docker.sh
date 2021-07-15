#!/bin/bash
DOCKER_IMAGE='openpcdet2'
DOCKER_IMAGE_TAG='latest'
DOCKER_VOLUME='openpcdet-vol'

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
if [ "$EUID" -ne 0 ]
  then printf "${RED}!!! Please run with sudo !!!${NC}\n"
  exit
fi

# Create volume to be used as home if not created before
if ! (docker volume list | grep -q ${DOCKER_VOLUME}); then
    printf "${YELLOW}Creating ${DOCKER_VOLUME} that does not exist.${NC}\n"
    if docker volume create ${DOCKER_VOLUME} &> /dev/null; then
        printf "${GREEN}${DOCKER_VOLUME} has been created successfully.${NC}\n"
    else
        printf "${RED}!!! Failed to creat docker volume ${DOCKER_VOLUME}. Abort.${NC}\n"
        exit -1
    fi
fi

if [[ $# -lt 1 ]]; then
    echo "[Usage]: ./run_docker.sh PATH_TO_WS [host]"
    exit -1
else
    printf "${GREEN}$1 is mapped to /midea_robot/ in docker.${NC}\n"
    if [[ $# -eq 2 && $2 == "host" ]]; then
        printf "${YELLOW}Attached to host network.\n"
        NETWORK_SETTING="--privileged --net=host"
    else
        NETWORK_SETTING=""
    fi
fi

# Allow root connect to xhost
# xhost +si:localuser:root > /dev/null
docker run -it --rm \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="XAUTHORITY=$XAUTH" \
    --env="QT_LOGGING_RULES=*.debug=false;qt.qpa.*=false" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$XAUTH:$XAUTH" \
    --volume="${DOCKER_VOLUME}:/root" \
    --volume="$1:/midea_robot" \
    --privileged \
    --runtime=nvidia ${NETWORK_SETTING} \
    --workdir="/midea_robot" \
    ${DOCKER_IMAGE}:${DOCKER_IMAGE_TAG} \
    bash

