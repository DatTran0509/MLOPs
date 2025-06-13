FROM fluent/fluentd:v1.16-debian-1

USER root

# Tạo thư mục log với quyền đúng
RUN mkdir -p /var/log/fluentd/fastapi \
    && mkdir -p /var/log/fluentd/prometheus \
    && mkdir -p /var/log/fluentd/grafana \
    && mkdir -p /var/log/fluentd/alertmanager \
    && mkdir -p /var/log/fluentd/node-exporter \
    && mkdir -p /var/log/fluentd/webhook-logger \
    && mkdir -p /var/log/fluentd/other \
    && chown -R fluent:fluent /var/log/fluentd

# Install plugins nếu cần
RUN gem install fluent-plugin-prometheus && gem cleanup

USER fluent