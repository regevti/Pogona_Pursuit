#FROM nablascom/cuda-pytorch
FROM pogona_pursuit_yolo4:0.2

USER root

ENV TZ=Asia/Jerusalem
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN echo "alias ll='ls -lah'" >> ~/.bashrc
ENTRYPOINT []
CMD bash -c "jupyter lab --ip 0.0.0.0 --allow-root"
