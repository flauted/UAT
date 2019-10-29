import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, activation_function=nn.ReLU()):
        super().__init__()
        self._input_size = input_size
        self.W1 = nn.Linear(input_size, hidden_size, bias=True)
        self.W2 = nn.Linear(hidden_size, 1, bias=False)

        self.phi = activation_function

    def forward(self, x):
        assert x.shape[1] == self._input_size
        z1 = self.W1(x)
        a1 = self.phi(z1)
        z2 = self.W2(a1)
        assert z2.shape[1] == 1
        return z2


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # initialize the weight tensor, here we use a normal distribution
            m.weight.data.normal_(0, 1)
            if m.bias is not None:
                m.bias.data.zero_()


def main():
    model = MLP(2, 4)
    weights_init(model)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = model.to(dev)

    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02)

    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [[0], [1], [1], [0]]

    epochs = 201
    steps = len(X)
    assert len(X) == len(Y)

    all_youts = []

    SNAPSHOTS = 5

    for i in range(epochs):
        samples = list(range(steps))
        random.shuffle(samples)
        batches = [[sample] for sample in samples]
        for batch in batches:
            x_var = torch.tensor([X[samp] for samp in batch], dtype=torch.float32, device=dev)
            y_var = torch.tensor([Y[samp] for samp in batch], dtype=torch.float32, device=dev)

            optimizer.zero_grad()
            y_hat = model(x_var)
            loss = loss_func(y_hat, y_var)
            loss.backward()
            optimizer.step()

        if i % SNAPSHOTS == 0:
            print("Epoch: {0}, Loss: {1}, ".format(i, loss.item()))

            x0 = np.arange(0, 1.1, 0.1)
            x1 = np.arange(0, 1.1, 0.1)
            X0, X1 = np.meshgrid(x0, x1)
            x0_vec = X0.flatten()
            x1_vec = X1.flatten()
            x = np.stack([x0_vec, x1_vec], axis=-1)
            with torch.no_grad():
                x = torch.tensor(x, dtype=torch.float32, device=dev)
                y_out = model(x)
                y_out = y_out.squeeze(-1).to(torch.device("cpu"))

            all_youts.append(y_out)

    fps = 10

    all_youts = np.stack(all_youts, axis=-1)

    def update_plot(frame_number, zarray, plot):
        plot[0].remove()
        plot[0] = ax.scatter(x0_vec, x1_vec, zarray[:, frame_number], color='b')
        plot[1].remove()
        plot[1] = ax.scatter(x0_truth, x1_truth, xor_truth, color='r', marker="x")
        ax.set_title(f"Step = {frame_number * SNAPSHOTS}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(r"$Net(x, y) \approx XOR(x, y)$")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x0_truth = np.array([0., 0., 1., 1.], dtype=np.float)
    x1_truth = np.array([0., 1., 0., 1.], dtype=np.float)
    xor_truth = np.array([0., 1., 1., 0.], dtype=np.float)

    plot = [ax.scatter(x0_vec, x1_vec, all_youts[:, 0], color="0.75"),
            ax.scatter(x0_truth, x1_truth, xor_truth, color='r', marker="*")]
    ax.set_zlim(0, 1.1)
    ani = animation.FuncAnimation(fig, update_plot, all_youts.shape[1], fargs=(all_youts, plot), interval=1000 / fps)

    plt.show()

    fn = 'plot_surface_animation_funcanimation'
    ani.save(fn + '.mp4', writer='ffmpeg', fps=fps)


if __name__ == "__main__":
    main()
