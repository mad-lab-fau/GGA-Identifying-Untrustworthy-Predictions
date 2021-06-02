from robustbench.data import load_cifar10
from robustbench.utils import load_model
from src.gga.CSM import cosine_similarity_maps
from gga.CSM import cosine_similarity_maps
import matplotlib.pyplot as plt
import torch

def run_experiment(rows=3, cols=3):
    x, y = load_cifar10(n_examples=50, data_dir="../data")
    x = x.cuda()
    x_noisy = x + torch.rand_like(x)

    model = load_model(model_name="Standard", model_dir="../Models", dataset="cifar10")
    model = model.cuda()

    plot_csm(model, x, y, rows, cols, "CSMs of Clean Data")
    plot_csm(model, x_noisy, y, rows, cols, "CSMs of Noisy Data")

def plot_csm(model, x, y, rows, cols, title):
    csm = cosine_similarity_maps(model, x, True, False)
    csm = csm.cpu().numpy()
    labels = y.numpy()

    fig, ax = plt.subplots(rows, cols)
    total = rows * cols
    for i in range(total):
        r = i // cols
        c = i % cols
        ax[r, c].imshow(csm[i], cmap="coolwarm")
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set_title("Label: " + str(labels[i]))
    plt.tight_layout()
    fig.suptitle(title)

    plt.savefig("./Images/CIFAR10_" + title, dpi=800)
    plt.clf()

if __name__ == "__main__":
    run_experiment(rows=2, cols=2)