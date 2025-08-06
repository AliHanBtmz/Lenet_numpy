import numpy as np
import matplotlib.pyplot as plt
import csv
from model.lnet5 import LNet5
from utils_.data_loader import load_mnist_images, load_mnist_label, one_hot_encode
from utils_.loss import CrossEntropyLoss
from utils_.optimizer import Adam
import os
def accuracy(preds, labels):
    return np.mean(np.argmax(preds, axis=1) == labels)

def train_one_epoch(model, loss_fn, x_train, y_train_onehot, batch_size, learning_rate, epoch, optimizer):
    permutation = np.random.permutation(x_train.shape[0])
    x_train_shuffled = x_train[permutation]
    y_train_onehot_shuffled = y_train_onehot[permutation]

    losses = []
    total_batches = x_train.shape[0] // batch_size

    for i in range(0, x_train.shape[0], batch_size):
        batch_idx = i // batch_size + 1
        x_batch = x_train_shuffled[i: i + batch_size]
        y_batch = y_train_onehot_shuffled[i: i + batch_size]
        if x_batch.ndim == 3:
            x_batch = np.expand_dims(x_batch, 1)

        preds = model.forward(x_batch)
        loss = loss_fn.forward(preds, y_batch)
        losses.append(loss)

        grad_loss = loss_fn.backward()
        model.backward(grad_loss)

        optimizer.parameters = model.get_parameters_and_grad()
        optimizer.step()

        # 💡 Her 100 batchte bir yazdır
        if batch_idx % 100 == 0 or batch_idx == total_batches:
            percent_done = (batch_idx / total_batches) * 100
            print(f"Epoch {epoch+1}, Batch {batch_idx}/{total_batches} ({percent_done:.1f}%): Loss = {loss:.4f}")

    return np.mean(losses)


def evaluation(model, x, y, batch_size=64):
    if x.ndim == 3:
        x = np.expand_dims(x, 1)
    preds = []
    for i in range(0, x.shape[0], batch_size):
        x_batch = x[i: i + batch_size]
        preds.append(model.forward(x_batch))
    preds = np.concatenate(preds, axis=0)
    return accuracy(preds, y)

def save_model(model, path="lenet5_weights_full_son.npz"):
    params = {}
    layer_idx = 0
    for layer in model.layers:
        if hasattr(layer, "weights"):
            params[f"layer{layer_idx}_weights"] = layer.weights
        if hasattr(layer, "biases"):
            params[f"layer{layer_idx}_biases"] = layer.biases
        layer_idx += 1
    np.savez(path, **params)



def plot_and_save_curves(train_acc_list, test_acc_list, loss_list, output_dir="results", acc_filename="accuracy_curve.png", loss_filename="loss_curve.png"):
    os.makedirs(output_dir, exist_ok=True)

    # Accuracy Plot
    plt.figure()
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, acc_filename))

    # Loss Plot
    plt.figure()
    plt.plot(loss_list, label='Training Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, loss_filename))


def save_csv_log(train_acc_list, test_acc_list, loss_list):
    with open("training_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Accuracy", "Test Accuracy", "Loss"])
        for epoch in range(len(train_acc_list)):
            writer.writerow([
                epoch + 1,
                f"{train_acc_list[epoch]:.4f}",
                f"{test_acc_list[epoch]:.4f}",
                f"{loss_list[epoch]:.4f}"
            ])

def main():
    x_train = load_mnist_images("data/train-images.idx3-ubyte")
    y_train = load_mnist_label("data/train-labels.idx1-ubyte")
    y_train_onehot = one_hot_encode(y_train)
    x_test = load_mnist_images("data/t10k-images.idx3-ubyte")
    y_test = load_mnist_label("data/t10k-labels.idx1-ubyte")

    EPOCHS = 3
    BATCH_SIZE = 128
    LR = 0.001

    model = LNet5()
    optimizer = Adam(model.get_parameters_and_grad(), learning_rate=LR)
    loss_func = CrossEntropyLoss()

    train_acc_list = []
    test_acc_list = []
    loss_list = []

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, loss_func, x_train, y_train_onehot, BATCH_SIZE, LR, epoch, optimizer)
        train_acc = evaluation(model, x_train, y_train, BATCH_SIZE)
        test_acc = evaluation(model, x_test, y_test, BATCH_SIZE)

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}, Loss = {loss:.4f}")

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        loss_list.append(loss)

    save_model(model)
    plot_and_save_curves(train_acc_list, test_acc_list, loss_list)
    save_csv_log(train_acc_list, test_acc_list, loss_list)

if __name__ == "__main__":
    main()
