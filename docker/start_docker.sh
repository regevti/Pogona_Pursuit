#!/bin/bash

container_name=pogona
docker stop ${container_name} && docker rm ${container_name}

docker run -itd --gpus all --name ${container_name} -p 2350:8888 --restart always --privileged \
      -v /media/sil2/regev/:/app/ \
      -v /etc/udev/:/etc/udev/ \
      -v /dev/bus/usb/:/dev/bus/usb/ \
      pogona_pursuit_yolo4:latest

