import matplotlib.pyplot as plt
import os

def plot4(env,savepath,i_epsoide, i_group_test, num= 4):
    for i in range(num):
        imgpath_trackingcurve = os.path.join(savepath, 'record_{}_{}_{}_TrackingCurve.png'.format(i_epsoide,i_group_test,i))
        imgpath_actioncurve = os.path.join(savepath, 'record_{}_{}_{}_ActionCure.png'.format(i_epsoide, i_group_test, i))
        plt.figure(figsize=(10, 7))
        plt.plot(env.PSI[i, 600:1200])
        plt.plot(env.Beta[i, 600:1200])
        plt.xlabel('Time (s)')
        plt.ylabel('Direction (deg)')
        plt.legend(['Wind', 'Yaw'])
        plt.savefig(imgpath_trackingcurve,dpi=600)
        plt.close()

        plt.figure(figsize=(10, 7))
        plt.plot(env.Action[i, 600:1200])
        plt.xlabel('Time (s)')
        plt.ylabel('Action')
        plt.yticks([-1,0,1],['-1','0','1'])
        plt.savefig(imgpath_actioncurve,dpi=600)
        plt.close()
