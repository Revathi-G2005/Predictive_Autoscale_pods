import pandas as pd
from prometheus_api_client import PrometheusConnect
from kubernetes import client, config
from datetime import datetime
import time
import os

# Prometheus connection
prom = PrometheusConnect(url="http://localhost:9090", disable_ssl=True)

# Kubernetes connection
config.load_kube_config()
v1 = client.AppsV1Api()

namespace = "default"          
deployment_name = "flask-app"  
csv_file = 'data/metrics.csv'

def fetch_metric(metric_name, namespace):
    now = datetime.utcnow()
    start_time = now - pd.Timedelta(seconds=60)
    data = prom.get_metric_range_data(
        metric_name=metric_name,
        start_time=start_time,
        end_time=now,
        step='60s',
        label_config={"namespace": namespace}
    )
    if not data:
        return pd.DataFrame(columns=['timestamp', 'value'])
    df = pd.DataFrame([
        {"timestamp": pd.to_datetime(item['value'][0], unit='s'), "value": float(item['value'][1])}
        for item in data
    ])
    return df

def get_replicas(deployment_name, namespace):
    deployment = v1.read_namespaced_deployment(deployment_name, namespace)
    return deployment.status.replicas or 0

# Continuous loop
while True:
    try:
        timestamp = datetime.utcnow()
        cpu_df = fetch_metric('container_cpu_usage_seconds_total', namespace)
        mem_df = fetch_metric('container_memory_usage_bytes', namespace)
        req_df = fetch_metric('requests_total', namespace)  

        replicas = get_replicas(deployment_name, namespace)

        row = {
            'timestamp': timestamp,
            'cpu_usage': cpu_df['value'].iloc[-1] if not cpu_df.empty else 0,
            'memory_usage': mem_df['value'].iloc[-1] if not mem_df.empty else 0,
            'requests_per_second': req_df['value'].iloc[-1] if not req_df.empty else 0,
            'replicas': replicas
        }

        if os.path.exists(csv_file):
            pd.DataFrame([row]).to_csv(csv_file, mode='a', header=False, index=False)
        else:
            pd.DataFrame([row]).to_csv(csv_file, index=False)

        print(f"✅ Metrics recorded at {timestamp}")
        time.sleep(60)

    except Exception as e:
        print(f"❌ Error: {e}")
        time.sleep(60)
