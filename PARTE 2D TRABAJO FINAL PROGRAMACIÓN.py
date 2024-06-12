import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2

# Body masses (in Kg)
m1 = 5.972e24  # Earth's mass
m2 = 1.898e27  # Jupiter's mass
m3 = 1.989e30  # Sun's mass
m4 = 2.2e14    # Comet Halley

# Initial positions (in meters)
r1 = np.array([1.496e11, 0.0])  # Initial position of Earth
r2 = np.array([7.779e11, 0.0])  # Initial position of Jupiter
r3 = np.array([0.0, 0.0])       # Initial position of Sun
r4 = np.array([8.976e10, 0.0])  # Initial position of Comet Halley

# Initial velocities (in m/s)
v1 = np.array([0.0, 29.78e3])    # Initial velocity of Earth
v2 = np.array([0.0, 13.056e3])   # Initial velocity of Jupiter
v3 = np.array([0.0, 0.0])        # Initial velocity of the Sun
v4 = np.array([0.0, 54.166e3])   # Initial velocity of Comet Halley


def acceleration(r1, r2, r3, r4, m1, m2, m3, m4):
    r12 = np.linalg.norm(r2 - r1)
    r13 = np.linalg.norm(r3 - r1)
    r14 = np.linalg.norm(r4 - r1)
    r23 = np.linalg.norm(r3 - r2)
    r24 = np.linalg.norm(r4 - r2)
    r34 = np.linalg.norm(r4 - r3)
    
    a1 = G * m2 * (r2 - r1) / r12**3 + G * m3 * (r3 - r1) / r13**3 + G * m4 * (r4 - r1) / r14**3
    a2 = G * m1 * (r1 - r2) / r12**3 + G * m3 * (r3 - r2) / r23**3 + G * m4 * (r4 - r2) / r24**3
    a3 = G * m1 * (r1 - r3) / r13**3 + G * m2 * (r2 - r3) / r23**3 + G * m4 * (r4 - r3) / r34**3
    a4 = G * m1 * (r1 - r4) / r14**3 + G * m2 * (r2 - r4) / r24**3 + G * m3 * (r3 - r4) / r34**3

    
    return a1, a2, a3, a4

def rk4_step(r1, r2, r3, r4, v1, v2, v3, v4, m1, m2, m3, m4, dt):
    k1_v1, k1_v2, k1_v3, k1_v4= acceleration(r1, r2, r3, r4, m1, m2, m3, m4)
    k1_r1, k1_r2, k1_r3, k1_r4 = v1, v2, v3, v4

    k2_v1, k2_v2, k2_v3, k2_v4 = acceleration(r1 + k1_r1*dt/2, r2 + k1_r2*dt/2, r3 + k1_r3*dt/2, r4 + k1_r4*dt/2, m1, m2, m3, m4)
    k2_r1, k2_r2, k2_r3, k2_r4 = v1 + k1_v1*dt/2, v2 + k1_v2*dt/2, v3 + k1_v3*dt/2, v4 + k1_v4*dt/2

    k3_v1, k3_v2, k3_v3, k3_v4 = acceleration(r1 + k2_r1*dt/2, r2 + k2_r2*dt/2, r3 + k2_r3*dt/2, r4 + k2_r4*dt/2, m1, m2, m3, m4)
    k3_r1, k3_r2, k3_r3, k3_r4 = v1 + k2_v1*dt/2, v2 + k2_v2*dt/2, v3 + k2_v3*dt/2, v4 + k2_v4*dt/2

    k4_v1, k4_v2, k4_v3, k4_v4 = acceleration(r1 + k3_r1*dt, r2 + k3_r2*dt, r3 + k3_r3*dt, r4 + k3_r4*dt, m1, m2, m3, m4)
    k4_r1, k4_r2, k4_r3, k4_r4 = v1 + k3_v1*dt, v2 + k3_v2*dt, v3 + k3_v3*dt, v4 + k3_v4*dt

    r1_new = r1 + dt/6 * (k1_r1 + 2*k2_r1 + 2*k3_r1 + k4_r1)
    r2_new = r2 + dt/6 * (k1_r2 + 2*k2_r2 + 2*k3_r2 + k4_r2)
    r3_new = r3 + dt/6 * (k1_r3 + 2*k2_r3 + 2*k3_r3 + k4_r3)
    r4_new = r4 + dt/6 * (k1_r4 + 2*k2_r4 + 2*k3_r4 + k4_r4)

    v1_new = v1 + dt/6 * (k1_v1 + 2*k2_v1 + 2*k3_v1 + k4_v1)
    v2_new = v2 + dt/6 * (k1_v2 + 2*k2_v2 + 2*k3_v2 + k4_v2)
    v3_new = v3 + dt/6 * (k1_v3 + 2*k2_v3 + 2*k3_v3 + k4_v3)
    v4_new = v4 + dt/6 * (k1_v4 + 2*k2_v4 + 2*k3_v4 + k4_v4)

    return r1_new, r2_new, r3_new, r4_new, v1_new, v2_new, v3_new, v4_new


t_final = 3.154e10 
dt = 1e6

pos1, pos2, pos3, pos4 = [], [], [], []

t = 0

while t < t_final:
    pos1.append(r1)
    pos2.append(r2)
    pos3.append(r3)
    pos4.append(r4)
    
    r1, r2, r3, r4, v1, v2, v3, v4 = rk4_step(r1, r2, r3, r4, v1, v2, v3, v4, m1, m2, m3, m4, dt)
    t += dt


pos1 = np.array(pos1)
pos2 = np.array(pos2)
pos3 = np.array(pos3)
pos4 = np.array(pos4)

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-2e12, 2e12)
ax.set_ylim(-2e12, 2e12)

line1, = ax.plot([], [], label = 'Earth')
line2, = ax.plot([], [], label = 'Jupiter')
line3, = ax.plot([], [], label = 'Sun')
line4, = ax.plot([], [], label = 'Halley Comet')

def init():

    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line4.set_data([], [])

    return line1, line2, line3, line4

def update(frame):

    line1.set_data(pos1[:frame, 0], pos1[:frame, 1])
    line2.set_data(pos2[:frame, 0], pos2[:frame, 1])
    line3.set_data(pos3[:frame, 0], pos3[:frame, 1])
    line4.set_data(pos4[:frame, 0], pos4[:frame, 1])

    return line1, line2, line3, line4

ani = FuncAnimation(fig, update, frames=len(pos1), init_func=init, blit=True, interval=5)


plt.legend()
plt.title('Animation for the N body problem')
plt.show()
