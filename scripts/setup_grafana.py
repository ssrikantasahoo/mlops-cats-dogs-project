import requests
import json
import time

GRAFANA_URL = "http://localhost:3000"
AUTH = ("admin", "admin")
DATASOURCE_NAME = "Prometheus"

def wait_for_grafana():
    print("Waiting for Grafana to be ready...")
    for _ in range(30):
        try:
            response = requests.get(f"{GRAFANA_URL}/api/health", timeout=5)
            if response.status_code == 200:
                print("Grafana is ready.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    return False

def setup_datasource():
    print("Setting up Prometheus data source...")
    datasource_payload = {
        "name": DATASOURCE_NAME,
        "type": "prometheus",
        "url": "http://prometheus:9090",
        "access": "proxy",
        "basicAuth": False
    }
    
    # Check if exists
    try:
        res = requests.get(f"{GRAFANA_URL}/api/datasources/name/{DATASOURCE_NAME}", auth=AUTH)
        if res.status_code == 200:
            print("Data source already exists.")
            return res.json()['uid']
    except Exception as e:
        print(f"Error checking datasource: {e}")

    # Create
    res = requests.post(f"{GRAFANA_URL}/api/datasources", json=datasource_payload, auth=AUTH)
    if res.status_code == 200:
        print("Data source created.")
        return res.json()['datasource']['uid']
    else:
        print(f"Failed to create data source: {res.text}")
        return None

def create_dashboard(datasource_uid):
    print("Creating MLOps Dashboard...")
    
    dashboard_json = {
        "dashboard": {
            "id": None,
            "uid": "mlops-monitoring",
            "title": "MLOps Model Monitoring",
            "tags": ["mlops", "prometheus"],
            "timezone": "browser",
            "schemaVersion": 16,
            "version": 0,
            "refresh": "5s",
            "panels": [
                {
                    "title": "Total Requests",
                    "type": "stat",
                    "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
                    "targets": [
                        {
                            "expr": "sum(prediction_requests_total)",
                            "legendFormat": "Total",
                            "datasource": {"type": "prometheus", "uid": datasource_uid}
                        }
                    ],
                    "datasource": {"type": "prometheus", "uid": datasource_uid}
                },
                {
                    "title": "Success Rate",
                    "type": "gauge",
                    "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4},
                    "targets": [
                        {
                            "expr": "sum(prediction_requests_total{status='success'}) / sum(prediction_requests_total) * 100",
                            "legendFormat": "Success %",
                            "datasource": {"type": "prometheus", "uid": datasource_uid}
                        }
                    ],
                    "datasource": {"type": "prometheus", "uid": datasource_uid},
                    "fieldConfig": {
                        "defaults": {
                            "min": 0,
                            "max": 100,
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "red", "value": None},
                                    {"color": "yellow", "value": 80},
                                    {"color": "green", "value": 90}
                                ]
                            }
                        }
                    }
                },
                {
                    "title": "Predictions by Class",
                    "type": "bargauge",
                    "gridPos": {"x": 12, "y": 0, "w": 6, "h": 8},
                    "targets": [
                        {
                            "expr": "predictions_total",
                            "legendFormat": "{{predicted_class}}",
                            "datasource": {"type": "prometheus", "uid": datasource_uid}
                        }
                    ],
                    "datasource": {"type": "prometheus", "uid": datasource_uid},
                    "options": {
                        "orientation": "vertical"
                    }
                },
                {
                    "title": "Requests per Second",
                    "type": "timeseries",
                    "gridPos": {"x": 0, "y": 4, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "rate(prediction_requests_total[1m])",
                            "legendFormat": "{{endpoint}} - {{status}}",
                            "datasource": {"type": "prometheus", "uid": datasource_uid}
                        }
                    ],
                    "datasource": {"type": "prometheus", "uid": datasource_uid}
                },
                {
                    "title": "API Latency (s)",
                    "type": "timeseries",
                    "gridPos": {"x": 0, "y": 12, "w": 12, "h": 8},
                    "targets": [
                        {
                            "expr": "rate(prediction_request_latency_seconds_sum[1m]) / rate(prediction_request_latency_seconds_count[1m])",
                            "legendFormat": "{{endpoint}}",
                            "datasource": {"type": "prometheus", "uid": datasource_uid}
                        }
                    ],
                    "datasource": {"type": "prometheus", "uid": datasource_uid},
                    "fieldConfig": {
                        "defaults": {
                            "unit": "s"
                        }
                    }
                }
            ]
        },
        "overwrite": True
    }
    
    res = requests.post(f"{GRAFANA_URL}/api/dashboards/db", json=dashboard_json, auth=AUTH)
    if res.status_code == 200:
        print("Dashboard created successfully!")
        print(f"URL: {GRAFANA_URL}{res.json()['url']}")
    else:
        print(f"Failed to create dashboard: {res.text}")

if __name__ == "__main__":
    if wait_for_grafana():
        uid = setup_datasource()
        if uid:
            create_dashboard(uid)
    else:
        print("Grafana is not accessible. Is Docker running?")
