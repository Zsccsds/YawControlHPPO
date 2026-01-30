# encoding: utf-8
"""
Description: Draw rose diagrams of the trainsets
@author: Dong Shuai
@contact: dongshuai@zsc.edu.cn
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib import rc

rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{amsmath}
\usepackage{newtxtext}    % Times 风格正文
\usepackage{mathptmx}    % Times 风格数学字体（关键！）
'''

plt.rc('text',usetex=True)

config = {
    "font.family":'Times New Roman',
    "axes.unicode_minus": False,
    "font.size": 18.0,
    # "font.style":'italic'
}
rcParams.update(config)

windfarm = ['1\#','2\#','3\#','4\#','5\#']

#load the statistics data of the first year
statisticsdatas = np.load('../wind_statistics.npy')
print(statisticsdatas.shape)
dir_para = statisticsdatas[:,0,0,:]
speed_para = statisticsdatas[:,0,1,:]

#draw the rose diagram of wind direction
ffig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=300, subplot_kw={'projection': 'polar'})
n = 16
theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
theta_closed = np.append(theta, theta[0])
for i in range(5):
    data = dir_para[i, :]
    data_closed = np.append(data, data[0])
    ax.plot(theta_closed, data_closed, label=windfarm[i], linewidth=1.5)

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_ylim([0.0, 0.18])
ax.set_rgrids(
    radii=[0.07, 0.14],
    angle=120,
    fontsize=18,
    horizontalalignment='left',
    verticalalignment='center',
)
ax.tick_params(axis='x', labelsize=18)
ax.legend(loc='lower left', bbox_to_anchor=(1.1, 0.5), fontsize=18, frameon=True)
plt.subplots_adjust(right=0.8)

plt.tight_layout()
plt.savefig('../Figures/wind_direction_distribution_polar.png')
plt.close()
# plt.show()

#draw the ros diagram of wind speed
fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=300, subplot_kw={'projection': 'polar'})
for i in range(5):
    data = speed_para[i, :]
    data_closed = np.append(data, data[0])
    ax.plot(theta_closed, data_closed, label=windfarm[i], linewidth=1.5)

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_ylim([0, 12])

ax.set_rgrids(
    radii=[6, 10],
    angle=250,
    fontsize=18,
    horizontalalignment = 'center',
    verticalalignment = 'top',
)

ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18,    pad = 0)
ax.legend(loc='lower left', bbox_to_anchor=(1.1, 0.5), fontsize=18, frameon=True)
plt.subplots_adjust(right=0.8)

plt.tight_layout()
plt.savefig('../Figures/wind_speed_distribution_polar.png')
plt.close()

