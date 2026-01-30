import numpy as np
import control as ctl
def noisegenerator(v_mean,turb_int,roughness,length):
    """
    Generate wind velocity with turbulence noise using frequency domain filters

    Args:
        v_mean: Mean wind velocity (m/s)
        turb_int: Turbulence intensity (%)
        roughness: Surface roughness length (m)
        length: Length of generated sequence

    Returns:
        Generated wind velocity array with turbulence components
    """
    omg_r = 13*np.pi/30
    D_rotor = 110
    t_sample = 1

    r = D_rotor/2
    sigma = v_mean*turb_int/100
    c = roughness/(2*np.pi*v_mean)
    d = r/v_mean
    trans = sigma*np.sqrt(roughness/v_mean)

    #Kamal model for wind statistic process
    T = np.arange(0,length)*t_sample
    WhiteNoise = np.random.randn(length,3)*np.sqrt(1)/np.sqrt(t_sample)
    KamalFilter = trans/np.sqrt(2)*ctl.tf([0.0182*c**2,1.3653*c,0.9846],[1.3453*c**2,3.7583*c,1])
    HarmonicFilter0 = ctl.series(KamalFilter,ctl.tf([4.7869*d,0.9904],[7.6823*d**2,7.3518*d,1]))
    HarmonicFilter3_1 = ctl.series(KamalFilter,ctl.tf([0.2766 * d,0.0307],[4.3691 * d ** 2, 1.7722 * d,1]))
    HarmonicFilter3_2 = ctl.series(KamalFilter,ctl.tf([0.2766 * d, 0.0307],[0.3691 * d **2, 1.7722 * d, 1]))
    y0 = ctl.forced_response(HarmonicFilter0,T,WhiteNoise[:,0]).outputs
    y1 = ctl.forced_response(HarmonicFilter3_1, T, WhiteNoise[:, 0]).outputs
    y2 = ctl.forced_response(HarmonicFilter3_2, T, WhiteNoise[:, 0]).outputs
    return v_mean + y0+ y1*(np.sqrt(2)*np.cos(3*omg_r*T))+y2*(np.sqrt(2)*np.sin(3*omg_r*T))

def genWindDir(Tau, A, s=1, l=2000):
    """
        Generate various types of wind direction patterns

        Args:
            Tau: Period of wind variation (time units)
            A: Amplitude of wind variation (degrees)
            s: Sampling interval (default: 1)
            l: Length of sequence (default: 2000)

        Returns:
            List containing different wind direction patterns:
            - Sinusoidal wave pattern
            - Triangular wave pattern
            - Rectangular wave pattern
            - Sawtooth wave pattern
            - Composite patterns with random variations
    """
    #sine profile
    T = np.linspace(0, l - s, int(l / s))
    w_sin = A * np.sin(T*2*np.pi/Tau)
    w_sin = np.sign(np.random.uniform(-1, 1)) * w_sin

    #rectangle profile
    w_rect = A * np.sin(T*2*np.pi/Tau)
    w_rect[w_rect > 0] = A
    w_rect[w_rect < 0] = -A
    w_rect = np.sign(np.random.uniform(-1, 1)) * w_rect

    #triangle profile
    dtmp = np.diff(A * np.sin(T*2*np.pi/Tau), append=0)
    d_max = 4*A / Tau
    dtmp[dtmp > 0] = d_max
    dtmp[dtmp < 0] = -d_max
    w_tria = np.cumsum(dtmp)
    w_tria = np.sign(np.random.uniform(-1, 1)) * w_tria

    #sawtooth profile
    w_serr = np.mod(T, Tau) / Tau * 2 * A - A
    w_serr = np.sign(np.random.uniform(-1,1))*w_serr

    out = [w_sin[:l],w_tria[:l], w_rect[:l], w_serr[:l]]

    #two additional combined profile
    T = np.linspace(0, l + 600 - s, int(l + 600 / s))
    for i in range(2):
        Tau_new = Tau
        w_sin = A * np.sin(T * 2 * np.pi / Tau_new)
        w_sin = np.sign(np.random.uniform(-1, 1)) * w_sin

        w_rect = np.random.randint(0,A) * np.sin(T * 2 * np.pi / Tau_new)
        w_rect[w_rect > 0] = np.random.randint(0,A)
        w_rect[w_rect < 0] = -np.random.randint(0,A)
        w_rect = np.sign(np.random.uniform(-1, 1)) * w_rect

        dtmp = np.diff(A * np.sin(T * 2 * np.pi / Tau_new), append=0)
        d_max = 4 * np.random.randint(0,A) / Tau
        dtmp[dtmp > 0] = d_max
        dtmp[dtmp < 0] = -d_max
        w_tria = np.cumsum(dtmp)

        w_tria = np.sign(np.random.uniform(-1,1))*w_tria
        w_serr = np.mod(T, Tau_new) / Tau_new * 2 * A - A
        w_serr = np.sign(np.random.uniform(-1, 1)) * w_serr
        offset = np.random.randint(0,600,4)
        w = np.random.uniform(0,1,4)
        w = w/sum(w)
        w_conf = (w[0]*w_sin[offset[0]:offset[0]+l]
                  +w[1]*w_tria[offset[1]:offset[1]+l]
                  +w[2]*w_rect[offset[2]:offset[2]+l]
                  +w[3]*w_serr[offset[3]:offset[3]+l])
        out.append(w_conf)

    return out

def genwind(n, Tau=800, A=20, v_mean=7,turbulence_intensity=18,roughness=400,sequence_length=1400):
    """
        Generate paired wind direction and velocity data

        Args:
            n: Number of wind sequences to generate
            Tau: Period of wind variation (default: 800)
            A: Amplitude of wind variation (default: 20)
            v_mean: Mean wind velocity (default: 7 m/s)
            turbulence_intensity: Turbulence intensity percentage (default: 18%)
            roughness: Surface roughness length (default: 400m)
            sequence_length: Length of each sequence (default: 1400)

        Returns:
            Tuple of (wind_directions, wind_velocities)
    """
    wind_dir, wind_v = [], []
    for i in range(n):
        d = genWindDir(Tau, A, 1, 1400)[:4] #without combined profiles
        wind_dir += d
        wind_v += [noisegenerator(v_mean,turbulence_intensity,roughness,sequence_length) for i in range(4)]
    return wind_dir, wind_v

def genwind_noise(n, Tau=800, A=20,  v_mean=7,turbulence_intensity=18,roughness=400,sequence_length=1400):
    """
       Generate wind data with additional noise in direction component

       Args:
           n: Number of wind sequences to generate
           Tau: Period of wind variation (default: 800)
           A: Amplitude of wind variation (default: 20)
           v_mean: Mean wind velocity (default: 7 m/s)
           turbulence_intensity: Turbulence intensity percentage (default: 18%)
           roughness: Surface roughness length (default: 400m)
           sequence_length: Length of each sequence (default: 1400)

       Returns:
           Tuple of (noisy_wind_directions, wind_velocities)
       """
    wind_dir, wind_v = [], []
    for i in range(n):
        d = genWindDir(Tau, A, 1, 1400)[:4]  #without combined profiles
        dnoise = [d[i]+noisegenerator(v_mean*10,turbulence_intensity,roughness,sequence_length) for i in range(4)]
        wind_dir += dnoise
        wind_v += [noisegenerator(v_mean,turbulence_intensity,roughness,sequence_length) for i in range(4)]
    return wind_dir, wind_v

def savewinddata(data_path,d,v):
    """
    Save wind direction and velocity data to compressed file

    Args:
        data_path: Path to save the data file
        d: Wind direction data array
        v: Wind velocity data array
    """
    np.savez(data_path,d=d,v=v)

def loadwinddata(data_path):
    """
    Load wind direction and velocity data from compressed file

    Args:
        data_path: Path to the data file

    Returns:
        Tuple of (wind_directions, wind_velocities)
    """
    data = np.load(data_path)
    d = data['d']
    v = data['v']
    return d,v





