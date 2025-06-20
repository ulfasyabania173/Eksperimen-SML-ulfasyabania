from prometheus_client import start_http_server, Counter, Summary
import time
import random

# Metrik Prometheus
REQUEST_COUNT = Counter('custom_request_count', 'Total request yang diterima')
SUCCESS_COUNT = Counter('custom_success_count', 'Total request sukses')
REQUEST_LATENCY = Summary('custom_request_latency_seconds', 'Waktu proses request (detik)')

@REQUEST_LATENCY.time()
def process_request(success=True):
    REQUEST_COUNT.inc()
    # Simulasi proses
    time.sleep(random.uniform(0.1, 0.5))
    if success:
        SUCCESS_COUNT.inc()

if __name__ == '__main__':
    # Jalankan HTTP server Prometheus di port 8000
    start_http_server(8000)
    print("Prometheus exporter berjalan di http://localhost:8000/metrics")
    # Simulasi request
    while True:
        process_request(success=random.choice([True, True, False]))
        time.sleep(2)