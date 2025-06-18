FROM bitnami/spark:4.0.0

# Custom logging
#COPY log4j2.properties /opt/bitnami/spark/conf/log4j2.properties

ENV YARN_CONF_DIR="/opt/bitnami/spark/yarn"
RUN export YARN_CONF_DIR

# any files/libraries you need on the cluster, install here ie:
RUN pip install pandas
RUN pip install pyarrow