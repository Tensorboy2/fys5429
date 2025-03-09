import torch 
import matplotlib.pyplot as plt
import numpy as np
k = torch.load('project_1/data/k.pt').numpy()

plt.plot(np.sort(k))
plt.show()