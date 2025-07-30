from model.lnet5 import LNet5
from utils_.data_loader import load_mnist_images, load_mnist_label, one_hot_encode
from utils_.loss import CrossEntropyLoss
from utils_.optimizer import Adam
import numpy as np


def accuracy(preds, labels):
    return np.mean(np.argmax(preds, axis=1) == labels)


def train_one_epoch(
    model, loss_fn, x_train, y_train_onehot, batch_size, learning_rate, epoch, optimizer
):
    permutation = np.random.permutation(x_train.shape[0])
    x_train_shuffled = x_train[permutation]
    y_train_onehot_shuffled = y_train_onehot[permutation]

    losses = []
    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train_shuffled[i : i + batch_size]
        y_batch = y_train_onehot_shuffled[i : i + batch_size]
        if x_batch.ndim == 3:
            x_batch = np.expand_dims(x_batch, 1)

        preds = model.forward(x_batch)
        loss = loss_fn.forward(preds, y_batch)
        losses.append(loss)

        grad_loss = loss_fn.backward()
        model.backward(grad_loss)

        optimizer.parameters = model.get_parameters_and_grad()
        optimizer.step()

        batch_no = i // batch_size
        if batch_no < 3:
            print(
                f"[Epoch {epoch+1}][Batch {batch_no}] Loss={loss:.4f} preds[0]={preds[0]}"
            )
        elif batch_no % 100 == 0:
            print(f"[Epoch {epoch+1}][Batch {batch_no}] Loss={loss:.4f}")
    return np.mean(losses)


def evaluation(model, x, y, batch_size=64):
    if x.ndim == 3:
        x = np.expand_dims(x, 1)
    preds = []
    for i in range(0, x.shape[0], batch_size):
        x_batch = x[i : i + batch_size]
        preds.append(model.forward(x_batch))
    preds = np.concatenate(preds, axis=0)
    acc = accuracy(preds, y)
    return acc


def save_model(model, path="lenet5_weights_full.npz"):
    params = {}
    layer_idx = 0
    for layer in model.layers:
        if hasattr(layer, "weights"):
            params[f"layer{layer_idx}_weights"] = layer.weights
        if hasattr(layer, "biases"):
            params[f"layer{layer_idx}_biases"] = layer.biases
        layer_idx += 1
    np.savez(path, **params)
    print(f"Model saved to {path}")


def main():
    x_train = load_mnist_images("data/train-images.idx3-ubyte")
    y_train = load_mnist_label("data/train-labels.idx1-ubyte")
    y_train_onehot = one_hot_encode(y_train)
    # ////////////////////////////////////////////////////////////
    # x_train = x_train[:10000]
    # y_train = y_train[:10000]
    # y_train_onehot = y_train_onehot[:10000]
    # ////////////////////////////////////////////////////////////
    x_test = load_mnist_images("data/t10k-images.idx3-ubyte")
    y_test = load_mnist_label("data/t10k-labels.idx1-ubyte")

    EPOCHS = 2
    BATCH_SIZE = 256
    LR = 0.001

    model = LNet5()
    optimizer = Adam(model.get_parameters_and_grad(), learning_rate=LR)
    loss_func = CrossEntropyLoss()

    for epoch in range(EPOCHS):
        vg_loss = train_one_epoch(
            model,
            loss_func,
            x_train,
            y_train_onehot,
            BATCH_SIZE,
            LR,
            epoch,
            optimizer,
        )
        train_acc = evaluation(model, x_train[:1000], y_train[:1000], BATCH_SIZE)
        test_acc = evaluation(model, x_test[:1000], y_test[:1000], BATCH_SIZE)
        print(
            f"Epoch {epoch+1}: Train acc={train_acc:.4f}, Test acc={test_acc:.4f}, Loss={vg_loss:.4f}"
        )

    # Eğitimden sonra modeli kaydet
    save_model(model)


if __name__ == "__main__":
    main()
