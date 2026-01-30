import numpy as np
import matplotlib.pyplot as plt
import os

datasetpath = r'E:\WorkspaceY9000P2025\YawControl\WTYCcodes\HPPO2026\data'
datasets = os.listdir(datasetpath)
datasets = [dataset for dataset in datasets if dataset.endswith('.npz')]


for dataset in datasets:
    data = np.load(os.path.join(datasetpath,dataset))
    d = data['d']
    v = data['v']
    print(dataset,d.shape,v.shape)
    plt.figure(figsize=(24,15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # plt.title(dataset)
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.plot(d[i*40,:])
    plt.savefig(os.path.join(datasetpath,dataset.split('.')[0]+'_d.png'),dpi=600)

    plt.figure(figsize=(24,15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # plt.title(dataset)
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.plot(v[i*40,:])
    plt.savefig(os.path.join(datasetpath,dataset.split('.')[0]+'_v.png'),dpi=600)
    plt.close()
# plt.show()