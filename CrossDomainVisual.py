import numpy as np
import matplotlib.pyplot as plt
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
def load_data(domain_idx):
    data = np.load(f'./data/trainset_domain{domain_idx+1}.npz')
    d = data['d']
    v = data['v']
    return d, v

domains = ['Sin', 'Rect', 'Tria', 'Serr']
t = np.arange(1200)


for i in range(4):
    d, v = load_data(i) #load data from data files
    plt.figure(figsize=(6.4,3.4),dpi=600)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    for j in range(0,6):
        plt.plot(t, d[j,:1200], linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Wind Direction (°)', fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=16)  # 主刻度
    plt.tick_params(axis='both', which='minor', labelsize=12)  # 次刻度（可选）
    plt.savefig(f'./Figures/wind_profiles_comparison_{domains[i]}.png')
    plt.close()


bins_d = np.linspace(-180, 180, 31)  # 30 个 bin
bin_centers_d = (bins_d[:-1] + bins_d[1:]) / 2

bins_v = np.linspace(0, 30, 31)  # 30 个 bin
bin_centers_v = (bins_v[:-1] + bins_v[1:]) / 2

freqs_d = []
freqs_v = []

for i in range(4):
    d, v = load_data(i)
    counts_d, _ = np.histogram(d.flatten(), bins=bins_d)
    counts_v, _ = np.histogram(v.flatten(), bins=bins_v)
    freqs_d.append(counts_d/ len(d.flatten()) )
    freqs_v.append(counts_v/ len(d.flatten()) )

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = domains
plt.figure(figsize=(6.4,3.4),dpi=600)
plt.subplots_adjust(left = 0.2, right = 0.9, top = 0.9, bottom = 0.2)
for i, (freq, color, label) in enumerate(zip(freqs_d, colors, labels)):
    plt.plot(bin_centers_d, freq, linewidth=1.5, color=color, label=label, marker='o', markersize=3)

plt.xlabel('Wind Direction (°)', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.savefig('./Figures/wind_direction_distribution_line.png')
plt.close()

plt.figure(figsize=(6.4,3.4),dpi=600)
plt.subplots_adjust(left = 0.2, right = 0.9, top = 0.9, bottom = 0.2)
for i, (freq, color, label) in enumerate(zip(freqs_v, colors, labels)):
    plt.plot(bin_centers_v, freq, linewidth=1.5, color=color, label=label, marker='o', markersize=3)

plt.xlabel('Wind Speed (m/s)', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.savefig('./Figures/wind_speed_distribution_line.png')
plt.close()