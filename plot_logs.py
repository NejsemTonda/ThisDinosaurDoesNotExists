import pickle
from matplotlib import pyplot as plt

logs = "dcgan.py-2026-01-05_182541-bs=50,d=mnist_small,e=10,gf=None,ng=1,r=False,rf=None,std=None,s=42,t=1,ts=None,zd=100.pkl"

with open(f"logs/{logs}", "rb") as log_file:
    history = pickle.load(log_file)


metrics = list(history.keys())

for label in metrics:
    plt.plot(history[label])

plt.title('Model training')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(metrics)

plt.show()






