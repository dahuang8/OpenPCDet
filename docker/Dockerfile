FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu20.04

# nvidia runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y python3.8 cmake libboost-all-dev git terminator python3-pip

RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN cd /tmp && git clone --recursive https://github.com/traveller59/spconv.git && \
    cd spconv && python3 setup.py bdist_wheel && cd ./dist && pip3 install spconv-1.2.1-cp38-cp38-linux_x86_64.whl

RUN cd /tmp && git clone https://github.com/open-mmlab/OpenPCDet.git && \
    cd OpenPCDet && pip3 install -r requirements.txt

RUN pip3 install vtk && pip3 install mayavi
