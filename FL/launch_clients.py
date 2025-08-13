# launch_clients.py
# run: python launch_clients.py --clients 7
import subprocess
import time
import argparse

def launch_clients(num_clients):
    processes = []
    for i in range(num_clients):
        print(f"Launching client {i + 1}")
        p = subprocess.Popen(["python", "federated_client.py", "--id", str(i + 1)])
        processes.append(p)
        time.sleep(0.5)

    for p in processes:
        p.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=3, help="Number of clients to launch")
    args = parser.parse_args()

    launch_clients(args.clients)