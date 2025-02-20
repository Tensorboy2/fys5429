'''Module just to plot activation functions'''
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
path = os.path.dirname(__file__)

x = np.linspace(-10,10,100)
plt.rcParams.update({'font.size': 14})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig , ax = plt.subplots(2,2,figsize=(8,6))

relu = np.maximum(0,x)
ax[0,0].plot(x,relu)
ax[0,0].legend(['ReLU'])
ax[0,0].grid()

gamma=0.01
leakyrelu = np.where(x > 0, x, x * gamma)  
ax[0,1].plot(x,leakyrelu)
ax[0,1].legend(['LeakyReLU'])
ax[0,1].text(-10, 6, r'$\gamma=$'+f'{gamma}', style='normal', bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 5})
ax[0,1].grid()

tanh = np.tanh(x)
ax[1,0].plot(x,tanh)
ax[1,0].legend(['Tanh'])
ax[1,0].grid()

sigmoid = 1/(1+np.exp(-x))
ax[1,1].plot(x,sigmoid)
ax[1,1].legend(['Sigmoid'])
ax[1,1].grid()



plt.savefig(os.path.join(path,'activation_functions.pdf'))
