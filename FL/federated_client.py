# federated_client.py

import socket, pickle, copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import SimpleNN
import argparse  # ← 追加

HOST = 'localhost'
PORT = 8000

def local_train(model, X, y, epochs=5, lr=0.01):
    model = copy.deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
    return model.state_dict()

def run_client(client_id):
    model = SimpleNN()
    X = torch.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).long()

    test_X = torch.randn(50, 2)
    test_y = (test_X[:, 0] + test_X[:, 1] > 0).long()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print(f"[Client {client_id}] Connected to server.")

        for round in range(10):
            data = s.recv(10 ** 6)
            message = pickle.loads(data)
            if message == "FIN":
                break  # gracefully exit
            model.load_state_dict(message)

            updated_weights = local_train(model, X, y)
            s.sendall(pickle.dumps(updated_weights))
            print(f"[Client {client_id}] Round {round + 1} completed.")

            # テスト精度の出力
            test_X = torch.randn(50, 2)
            test_y = (test_X[:, 0] + test_X[:, 1] > 0).long()
            acc = local_test(model, test_X, test_y)
            print(f"[Client {client_id}] Local test accuracy: {acc:.2%}")
        print(f"[Client {client_id}] Final local test accuracy: {acc:.2%}")

def local_test(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y).float().mean().item()
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0, help="Client ID")
    args = parser.parse_args()
    run_client(args.id)