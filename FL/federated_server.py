# federated_server.py
# run: python federated_server.py --clients 7 --rounds 10
import socket
import pickle
import threading
import torch
import argparse
from model import SimpleNN

HOST = 'localhost'
PORT = 8000
clients = []

def handle_client(conn, addr):
    print(f"[Connected] {addr}")
    clients.append(conn)

def accept_clients(server_socket, num_clients):
    while len(clients) < num_clients:
        conn, addr = server_socket.accept()
        threading.Thread(target=handle_client, args=(conn, addr)).start()
    print(f"\n[Info] {num_clients} clients connected. Starting training...\n")

def aggregate_weights(weights_list):
    avg_weights = weights_list[0]
    for key in avg_weights.keys():
        for i in range(1, len(weights_list)):
            avg_weights[key] += weights_list[i][key]
        avg_weights[key] /= len(weights_list)
    return avg_weights

def main(num_clients, num_rounds):
    model = SimpleNN()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[Listening] Waiting for {num_clients} clients to connect...")

        accept_clients(s, num_clients)

        for rnd in range(num_rounds):
            print(f"[Round {rnd + 1}] Sending model to all clients...")
            data = pickle.dumps(model.state_dict())

            for conn in clients:
                try:
                    conn.sendall(data)
                except Exception as e:
                    print(f"[Error] Failed to send model to client: {e}")

            weights_list = []
            for idx, conn in enumerate(clients):
                try:
                    client_data = conn.recv(10**6)
                    if not client_data:
                        print(f"[Warning] Client {idx+1} sent no data.")
                        continue
                    weights = pickle.loads(client_data)
                    weights_list.append(weights)
                    print(f"[Server] Received update from Client {idx + 1}")
                except Exception as e:
                    print(f"[Error] Failed to receive from Client {idx + 1}: {e}")

            if weights_list:
                model.load_state_dict(aggregate_weights(weights_list))
                print(f"[Round {rnd + 1}] Aggregation complete with {len(weights_list)} clients.\n")
            else:
                print(f"[Round {rnd + 1}] No updates received.\n")

        for conn in clients:
            conn.close()
        print("[Finished] Training completed.")

        # After training rounds
        for conn in clients:
            try:
                conn.sendall(pickle.dumps("FIN"))  # Send finish signal
                conn.close()
            except:
                pass
        print("[Finished] Training completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients', type=int, default=5, help='Number of clients to wait for')
    parser.add_argument('--rounds', type=int, default=10, help='Number of training rounds')
    args = parser.parse_args()

    main(args.clients, args.rounds)