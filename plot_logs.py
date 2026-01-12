import glob
import pickle
from matplotlib import pyplot as plt

logs = "vae.py-2026-01-09_094034-bs=100,d=dataset,dl=[500, 500],el=[500, 500],zd=10.pkl"

with open(f"logs/{logs}", "rb") as log_file:
    history = pickle.load(log_file)


def plot_history(history, plot_name, axis_labels=None, save_name=None):
    metrics = list(history.keys())
    
    for label in metrics:
        plt.plot(history[label])
    
    plt.title(plot_name)
    if axis_labels and len(axis_labels) == 2:
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])

    
    plt.legend(metrics)
    if save_name:
        plt.savefig(f"{save_name}.svg")
    else:
        plt.show()

    plt.cla()


def join_logs(regex) -> dict:
    files = glob.glob(regex)
    files.sort()
    
    history = {}

    for file in files:
        with open(file, "rb") as log_file:
            logs = pickle.load(log_file) 

        for key in logs.keys():
            history[key] = history.get(key, []) + logs[key]

    return history


def exp_smoothing(values, alpha=1):
    result = [values[0]]
    for v in values[1:]:
        result.append(result[-1] + alpha * (v - result[-1]))
    return result
    

z_dim_logs = {}
for n in [10, 50, 150]:
    h = join_logs(f"logs/vae*={n}.pkl")
    z_dim_logs[f"z_dim={n}"] = h["loss"]
z_dim_logs = {k: exp_smoothing(v) for k,v in z_dim_logs.items()}
plot_history(z_dim_logs, "Latent Space Size vs Accuracy Of Training", ("epochs", "loss"), "src/zdim_loss")
