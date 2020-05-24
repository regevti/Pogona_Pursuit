FROM nablascom/cuda-pytorch

USER root

ENV TZ=Asia/Jerusalem
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN echo "alias ll='ls -lah'" >> ~/.bashrc

CMD bash -c "jupyter lab --ip 0.0.0.0 --allow-root"
