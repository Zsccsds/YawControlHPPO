import numpy as np
from numpy import bitwise_and as btand
from numpy import bitwise_or as btor
import yaml
# Redefining bitwise functions as logical operations for boolean arrays
def btand(a, b): return np.logical_and(a, b)
def btor(a, b):  return np.logical_or(a, b)
#state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5]
# van_1sec, van_30sec, van_5min, v_1sec, v_30sec, v_5min
class BangBangController:
    """
    Bang-Bang controller implementation for wind turbine yaw control.
    Makes decisions based on wind direction error with a deadband.
    """
    def __init__(self, num_traj, update_every=10, deadband=0.1):
        """
        Initialize the BangBang controller
        :param num_traj: Number of trajectories
        :param update_every: Update frequency (every N steps)
        :param deadband: Deadband threshold below which no action is taken
        """
        self.dim = num_traj
        self.update_every = update_every
        self.deadband = deadband
        self.counter = 0
        self.action = np.ones(num_traj, dtype=np.int8)  # 初始动作：不动 (1)

    def __call__(self, state):
        """
        Execute control decision
        :param state: State matrix, state[:, 0] is 1-second average wind direction error
        :return: Control actions for each trajectory
        """
        van_1sec = state[:, 0]

        if self.counter % self.update_every == 0:
            command = np.zeros(self.dim, dtype=np.int8)
            command[van_1sec > self.deadband] = 1   # CCW
            command[van_1sec < -self.deadband] = -1 # CW
            self.action = command + 1  # mapping {0: CW, 1: idle, 2: CCW}

        self.counter += 1
        return self.action

class HPPOController():
    """
    Hierarchical Proximal Policy Optimization (HPPO) controller for wind turbine yaw control.
    Uses learned policies to determine optimal yaw actions and periods.
    """
    def __init__(self,num_traj,hppo_model):
        """
        Initialize the HPPO controller
        :param num_traj: Number of trajectories
        :param hppo_model: Pre-trained HPPO model
        """
        self.dim = num_traj
        self.hppo = hppo_model

        self.escape = np.zeros(self.dim)
        self.action = np.ones(self.dim)
        self.period = np.zeros(self.dim)
        self.action_c = np.zeros(self.dim, dtype=np.int8)
        self.period_c = np.zeros(self.dim, dtype=np.int8)
    def __call__(self,state):
        """
        Execute control decision using HPPO policy
        :param state: Current state vector
        :return: Control actions for each trajectory
        """
        self.escape += 1
        # Get action and period from old policy
        self.action_c, self.period_c = self.hppo.policy_old.eval_act(state)
        for ind in range(len(self.action_c)):
            if self.escape[ind] >= self.period[ind]:
                self.period[ind] = self.period_c[ind] + 1
                self.escape[ind] = 0
                self.action[ind] = self.action_c[ind]
        return self.action


#Open-Loop Correction Controller
class OLCController():
    """
    Open-Loop Correction (OLC) controller for wind turbine yaw control.
    Uses 30-second average wind direction error to calculate appropriate actions.
    """
    def __init__(self,num_traj):
        """
        Initialize the OLC controller
        :param num_traj: Number of trajectories
        """
        self.dim = num_traj

        self.escape = np.zeros(self.dim)
        self.action = np.ones(self.dim)
        self.period = np.zeros(self.dim)

    def __call__(self,state):
        """
        Execute control decision using open-loop correction logic
        :param state: State vector containing various wind measurements
        :return: Control actions for each trajectory
        """
        # Extract state variables
        van_1sec, van_30sec, van_5min, v_1sec, v_30sec, v_5min = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5]

        # Calculate action based on sign of 30-second wind direction error
        action_c = np.sign(van_30sec)

        # Calculate period based on magnitude of wind direction error
        # 0.7 deg/s is assumed yaw rate
        period_c = np.clip(np.abs(van_30sec)/ np.deg2rad(0.7),0,15)
        self.escape += 1
        for ind in range(len(action_c)):
            # Update action when current period has elapsed
            if self.escape[ind] >= self.period[ind]:
                self.period[ind] = period_c[ind] + 1
                self.escape[ind] = 0
                self.action[ind] = action_c[ind]
        return self.action

class ClassicController():
    """
    Classic condition-based controller for wind turbine yaw control.
    Implements traditional wind turbine yaw control logic based on wind conditions.
    """
    def __init__(self,num_traj):
        """
        Initialize the classic controller
        :param num_traj: Number of trajectories
        """
        self.dim = num_traj

        # Boolean flags for different operational states
        self.bactive = np.zeros(num_traj,dtype=np.bool_)
        self.bfastCCW = np.zeros(num_traj,dtype=np.bool_)
        self.bfastCW = np.zeros(num_traj, dtype=np.bool_)
        self.yawtime = np.zeros(num_traj,dtype=np.int32)
        self.yawelapsed = np.zeros(num_traj,dtype=np.int32)
        self.bhalfyawtime = np.zeros(num_traj, dtype=np.bool_)
        self.d = np.zeros(self.dim, dtype=np.int8)
        self.action = np.zeros(self.dim,dtype=np.int8)


    def __call__(self, state):
        """
        Execute control decision using classic control logic
        :param state: State vector containing various wind measurements
        :return: Control actions for each trajectory
        """
        # Extract state variables
        van_1sec, van_30sec, van_5min, v_1sec, v_30sec, v_5min = state[:, 0], state[:, 1], state[:, 2], state[:,3], state[:,4], state[:,5]
        # print(van_1sec,van_30sec,van_5min,v_1sec,v_30sec,v_5min)

        # Determine if turbine should be active (minimum wind speed requirement)
        self.bactive = v_30sec>2.5
        command = np.zeros(self.dim,dtype=np.int8)

        # Reset all parameters for inactive turbines
        tmp0 = ~self.bactive
        self.yawtime[tmp0] = 0
        self.yawelapsed[tmp0] = 0
        self.bfastCCW[tmp0] = False
        self.bfastCW[tmp0] = False
        self.bhalfyawtime[tmp0] = False
        command[tmp0] = 0

        # Calculate dynamic threshold based on wind speed
        # Threshold ranges from 8° to 15° depending on wind speed
        self.v_threshold = np.deg2rad(5 + (v_30sec-2.5)/(7-2.5)*7)
        self.v_threshold = np.clip(self.v_threshold,np.deg2rad(8),np.deg2rad(15))

        # Check if fast rotations are needed
        self.bfastCCW = van_30sec>self.v_threshold
        self.bfastCW = van_30sec<-self.v_threshold
        tmp = btand(self.yawtime == 0, self.bfastCCW + self.bfastCW)

        # Calculate required yaw time based on error magnitude and yaw rate (0.7 deg/s)
        self.yawtime[tmp] = np.floor(np.abs(van_30sec[tmp] / np.deg2rad(0.7)))
        self.yawtime[tmp] = np.clip(self.yawtime[tmp], 0, int(180/np.deg2rad(0.7))).astype(np.int32)
        self.yawelapsed[tmp] = 0
        self.bhalfyawtime[tmp] = False
        self.d[btand (tmp, self.bfastCCW)] =1
        self.d[btand (tmp,~ self.bfastCCW)] = -1

        # Process turbines with active yaw maneuvers
        tmp2 = self.yawtime>0
        self.yawelapsed[tmp2] +=1
        command[tmp2] = self.d[tmp2]

        # Check if yaw maneuver is complete
        tmp6 = btand(tmp2, self.yawelapsed>=self.yawtime)
        self.yawtime[tmp6] = 0
        self.yawelapsed[tmp6] = 0
        self.bfastCCW[tmp6] = False
        self.bfastCW[tmp6]  = False

        self.action =command+1
        return self.action

class WindTurbine():
    """
    Wind turbine environment for reinforcement learning.
    Simulates wind turbine behavior and provides rewards based on power generation and tracking performance.
    """
    def __init__(self, configfile, command, reward_weight):
        """
        Initialize the wind turbine environment
        :param configfile: Path to configuration file
        :param command: Control mode ('VEL' for velocity or 'ACC' for acceleration)
        :param reward_weight: Weight for action penalty in reward calculation
        """
        self.Command = command
        self.configfile = configfile
        with open(self.configfile, 'r', encoding='utf8') as f:
            self.config = yaml.safe_load(f)
            self.pho = self.config['AIRDENSITY']
            self.A = np.pi * self.config['R'] ** 2 #叶轮面积
            self.a = 0.5*self.pho*self.A*0.42 #参数
            self.windpath =  self.config['WINDDATA']
            self.WTID = self.config['ID']
            self.WINDFARM = self.config['WINDFARM']
            self.bocclusion = self.config['OCCLUSION']
            self.dirocc =  self.config['OCCLUSIONATTR']['DIR']
            self.rangeocc = self.config['OCCLUSIONATTR']['RANAGE']
            self.ampocc = self.config['OCCLUSIONATTR']['AMPLITUDE']
            self.bmeasure_err = self.config['WINDERROR']

            self.reward_weight = reward_weight

    def step(self, a, wa=0):
        """
        Execute one simulation step
        :param a: Actions from the controller
        :param wa: Weight for action penalty in reward calculation
        :return: Next state, reward, done flag, additional info, and placeholder value
        """
        self.ind += 1
        # Check if episode is done
        if self.ind == self.traj_len - 200:
            self.done = True

        # Convert actions from {0, 1, 2} to {-1, 0, 1}
        action = a - 1

        # Apply control command based on selected mode
        if self.Command=='VEL':
            self.Yaw_Vel = action # Direct velocity control
        if self.Command =='ACC':
            self.Yaw_Vel = self.Yaw_Vel + action  # Acceleration control
            self.Yaw_Vel = np.clip(self.Yaw_Vel,-1,1)

        # Update action and yaw angle
        self.Action[:, self.ind] = self.Yaw_Vel
        self.Beta[:, self.ind] = self.Beta[:, self.ind-1] + self.Yaw_Vel * 0.7

        i = self.ind

        # Calculate wind angle relative to turbine orientation over past 600 steps
        Van = self.PSI[:, i - 600:i + 1] - self.Beta[:, i - 600:i + 1]
        Van = np.arctan2(np.sin(Van / 180 * np.pi), np.cos(Van / 180 * np.pi)) * 180 / np.pi
        Van = np.clip(Van, -179.999, 179.999)

        # Calculate various time-averaged wind direction errors
        van_1sec = Van[:, -1]  # 1-second average
        van_30sec = np.mean(Van[:, -30:], axis=1)  # 30-second average
        van_5min = np.mean(Van[:, -300:], axis=1)  # 5-minute average

        # Calculate various time-averaged wind speeds
        v_1sec = self.V[:, i]  # Instantaneous wind speed
        v_30sec = np.mean(self.V[:, i - 29:i + 1], axis=1)  # 30-second average
        v_5min = np.mean(self.V[:, i - 299:i + 1], axis=1)  # 5-minute average

        # Construct state vector with normalized values
        s = np.array([van_1sec / 90, van_30sec / 90, van_5min / 90, np.clip((v_1sec - 3) / 7, 0, 1),
                          np.clip((v_30sec - 3) / 7, 0, 1), np.clip((v_5min - 3) / 7, 0, 1), self.Action[:, -1],self.Beta[:, i + 1]/90]).T

        # Add measurement errors if enabled
        if self.bmeasure_err:
            psi = self.PSI[:, i] + self.config['WINDERRORATTR']['DIR_ERROR']
            v = self.V[:, i] + self.config['WINDERRORATTR']['SPEED_ERROR']
        else:
            psi = self.PSI[:, i]
            v = self.V[:, i]

        # Calculate power output using wind turbine power equation
        P = self.a * v ** 3 * np.cos((self.Beta[:, i] - psi) / 57.3) ** 3

        # Apply wake effect if enabled
        if self.bocclusion:
            err_van = np.abs(psi - self.dirocc) % 360
            b_occ = err_van < self.rangeocc
            rate_occ = self.ampocc
            rate_occ = rate_occ * b_occ
            P = P * (1 - rate_occ)

        # Apply power limits and action penalties
        P = np.clip(P, 0, 2000000) - np.abs(action) * 40000
        P = P / 3600 / 1000
        self.POWER[:, self.ind] = P

        # Calculate reward components
        reward_power = P # Power generation reward
        reward_tracking_error = -np.abs(van_1sec)# Negative tracking error
        reward_action = - np.abs(action - self.a_previous) * 2000000 # action
        reward = reward_power + reward_tracking_error + wa * reward_action
        self.a_previous = action
        return s, reward, self.done, [P, np.abs(self.Beta[:, i] - self.PSI[:, i])], 0

    def reset(self, winddir,v):
        """
        Reset the environment to initial state
        :param winddir: Wind direction array
        :param v: Wind speed array
        :return: Initial state and zero reward
        """
        self.PSI = np.array(winddir)  # Wind direction
        self.V = np.array(v)  # Wind speed
        self.POWER = np.zeros_like(self.PSI)  # Power output
        self.traj_num, self.traj_len = self.PSI.shape[0], self.PSI.shape[1]  # Trajectory dimensions
        self.Beta = np.zeros_like(self.PSI)  # Yaw angle

        self.Action = np.zeros_like(self.PSI)  # Action history

        self.ind = 600  # Current index
        self.done = False  # Done flag

        i = self.ind

        # Calculate initial wind angle relative to turbine orientation
        Van = self.PSI[:, i - 600:i + 1] - self.Beta[:, i - 600:i + 1]
        Van = np.arctan2(np.sin(Van / 180 * np.pi), np.cos(Van / 180 * np.pi)) * 180 / np.pi
        Van = np.clip(Van, -179.999, 179.999)

        # Calculate initial time-averaged wind direction errors
        van_1sec = Van[:, -1]
        van_30sec = np.mean(Van[:, -30:], axis=1)
        van_5min = np.mean(Van[:, -300:], axis=1)

        # Calculate initial time-averaged wind speeds
        v_1sec = self.V[:, i]
        v_30sec = np.mean(self.V[:, i - 29:i + 1], axis=1)
        v_5min = np.mean(self.V[:, i - 299:i + 1], axis=1)

        # Construct initial state vector
        s = np.array([van_1sec / 90, van_30sec / 90, van_5min / 90, np.clip((v_1sec - 3) / 7, 0, 1),
                          np.clip((v_30sec - 3) / 7, 0, 1), np.clip((v_5min - 3) / 7, 0, 1), self.Action[:, -1],self.Beta[:, i + 1]/90]).T


        self.a_previous = 0  # Previous action
        self.Yaw_Vel = 0  # Current yaw velocity
        return s, 0

