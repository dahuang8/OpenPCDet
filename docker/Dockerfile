# FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# nvidia runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget

# RUN apt update && apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt update && apt install -y python3.7

RUN apt update && apt install -y python3.7 libboost-all-dev git terminator python3-pip

RUN pip3 install torch==1.2.0 torchvision==0.4.0

RUN apt update && apt install -y libssl-dev && \
    cd /tmp && wget https://github.com/Kitware/CMake/releases/download/v3.15.0/cmake-3.15.0.tar.gz && \
    tar -zxvf cmake-3.15.0.tar.gz && cd cmake-3.15.0 && ./bootstrap && make -j && make install

# RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN cd /tmp && git clone --recursive https://github.com/traveller59/spconv.git && \
    cd spconv && python3 setup.py bdist_wheel && cd ./dist && pip3 install *.whl

# && git checkout 8da6f96

RUN cd /tmp && git clone https://github.com/open-mmlab/OpenPCDet.git && \
    cd OpenPCDet && pip3 install -r requirements.txt

RUN pip3 install vtk && pip3 install mayavi
