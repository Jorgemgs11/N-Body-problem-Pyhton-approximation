''' 
N BODY PROBLEM, SIMULATION USING PYTHON

This code will be a simulation/animation for the N body problem, recreating the orbits that all the bodies in the system will follow.

The code's goals and requirements are the following:
    - The orbits need to chance according to the initial parameters given.
    - The code have to use the Runge-Kutta4 numerical method for the analitical solutions.
    - A main star will always be in the center, the other bodies will have to orbit around it.
    - Simulation needs to be at least for 1 year.
    - At least two bodies need to orbit the main star.
    - The orbits have to make sence (F.E: In a Sun, Earth and Moon system, the moon will need to orbit the earth while orbiting the sun).
    - 3D modeling for detail orbits and 2D modeling for simple orbits (Example simple orbit: Sun, Earth, Mars. Example detail orbit: Sun, Earth, Moon).
    - On 3D modeling, the star will need to move through space.
'''
'''
The Libraries we're going to use for this project, will be the following:

Numpy: Numpy will allow us to handle the arrays(The arrays let us keep multiple values of the same data type. It is like a list) and some functions.
Matplotlib: Matplotlib will allow us to graph all the data we store.
Matplotlib.animation: It is a module of Matplotlib that allow us to create animation with the data we have.
mpl_toolkits.mplot3d: It is the module of Matplotlib that allow us to graph in 3D

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

'''
Now that we define all the libraries we're going to use, it is time to start the code.

The first thing to do, is defining all the constants and the variables we're going to use for the calculations. In this case, I need to set the gravitational constant,
the masses for the bodies, their initial positions and initial velocities. For the first try, I will set the data for the Sun, earth and moon.

'''

G = 6.67430e-11  # Gravitational constant, on m^3 kg^-1 s^-2

# Bodies masses (in kg)

m1 = 5.972e24  # Mass of the Earth
m2 = 7.348e22  # Mass of the Moon
m3 = 1.989e30  # Mass of the Sun

# Initial positions (in meters)

r1 = np.array([1.496e11, 0.0, 0.0])               # Earth
r2 = np.array([1.496e11 + 3.844e8, 0.0, 0.0])     # Moon
r3 = np.array([0.0, 0.0, 0.0])                    # Sun

# Initial velocities(in m/s)
v1 = np.array([0.0, 23e4 + 29.78e3, 0.0])               # Earth
v2 = np.array([0.0, 23e4 + 29.78e3 + 1.022e3, 0.0 ])    # Moon
v3 = np.array([0.0, 23e4, 0.0])                         # Sun

'''

Now, we procede to calculate the motion ecuations:

    -We start with the acceleration of the bodies, in this case, the Numpy library will be esencial, because we will use function numpy.linalg.norm, what it does
    consist on calculate the minumum distance beetween two bodies, if we have two vectors, numpy.linalg.norm will calculate this: sqrt((v2_i - v1_i)^2 + (v2_j - v1_j)^2).
    -Once we define the acceleration, the numerical method of Runge-Kutta4 start. This method consist on an aproximation to the solution of an diferential ecuation with
    initial parameters. In order to find the solutions, we need to:

        1. Formulate the ecuation of motion: 
        
        The N body problem, consist of N masses, m_1, m_2,...,m_n, all of them with an initial position r_n(t) and velocity v_n(t) at a time t
        The gravitational force acting on each mass is determined by Newton's law of gravitation.

        2. Setting the system for the ordinaries diferential ecuations: 

        We need to solve a system of first-order for each mass: Y_i = [r,v]
        it can be written in the form of: dY_i/dt = f_i(t, Y_1, Y_2,..., Y_n)

        The RK4 method approximates the solution of the ODEs by considering the value of the function at several points within each time step. For a system of ODEs dy/dt = f(t, Y), the equivalent for Runge Kutta 4 is:

        Y_n+1 = Y_n + (h/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)

        where:
        
        k_1 = f(t_n, Y_n)
        k_2 = f(t_n + h/2, Y_n + (k_1)/2)
        k_3 = f(t_n + h/2, Y_n + (k_2)/2)
        k_4 = f(t_n + h, Y_n + k_3)

        For the three-body problem, these steps are applied to each of the N bodies. At each time step t_n we compute the values for k_1, k_2, k_3 and k_4 for both position
        and velocity of each body, and then, update the positions and velocities accordingly.

'''

def acceleration(r1, r2, r3, m1, m2, m3):

    r12 = np.linalg.norm(r2 - r1)
    r13 = np.linalg.norm(r3 - r1)
    r23 = np.linalg.norm(r3 - r2)
    
    a1 = G * m2 * (r2 - r1) / r12**3 + G * m3 * (r3 - r1) / r13**3
    a2 = G * m1 * (r1 - r2) / r12**3 + G * m3 * (r3 - r2) / r23**3
    a3 = G * m1 * (r1 - r3) / r13**3 + G * m2 * (r2 - r3) / r23**3
    
    return a1, a2, a3

def rk4_step(r1, r2, r3, v1, v2, v3, m1, m2, m3, dt):

    k1_v1, k1_v2, k1_v3 = acceleration(r1, r2, r3, m1, m2, m3)
    k1_r1, k1_r2, k1_r3 = v1, v2, v3

    k2_v1, k2_v2, k2_v3 = acceleration(r1 + k1_r1*dt/2, r2 + k1_r2*dt/2, r3 + k1_r3*dt/2, m1, m2, m3)
    k2_r1, k2_r2, k2_r3 = v1 + k1_v1*dt/2, v2 + k1_v2*dt/2, v3 + k1_v3*dt/2

    k3_v1, k3_v2, k3_v3 = acceleration(r1 + k2_r1*dt/2, r2 + k2_r2*dt/2, r3 + k2_r3*dt/2, m1, m2, m3)
    k3_r1, k3_r2, k3_r3 = v1 + k2_v1*dt/2, v2 + k2_v2*dt/2, v3 + k2_v3*dt/2

    k4_v1, k4_v2, k4_v3 = acceleration(r1 + k3_r1*dt, r2 + k3_r2*dt, r3 + k3_r3*dt, m1, m2, m3)
    k4_r1, k4_r2, k4_r3 = v1 + k3_v1*dt, v2 + k3_v2*dt, v3 + k3_v3*dt

    r1_new = r1 + dt/6 * (k1_r1 + 2*k2_r1 + 2*k3_r1 + k4_r1)
    r2_new = r2 + dt/6 * (k1_r2 + 2*k2_r2 + 2*k3_r2 + k4_r2)
    r3_new = r3 + dt/6 * (k1_r3 + 2*k2_r3 + 2*k3_r3 + k4_r3)

    v1_new = v1 + dt/6 * (k1_v1 + 2*k2_v1 + 2*k3_v1 + k4_v1)
    v2_new = v2 + dt/6 * (k1_v2 + 2*k2_v2 + 2*k3_v2 + k4_v2)
    v3_new = v3 + dt/6 * (k1_v3 + 2*k2_v3 + 2*k3_v3 + k4_v3)

    return r1_new, r2_new, r3_new, v1_new, v2_new, v3_new


t_final = 3.154e7   #A year in seconds
dt = 1e5            #Intervals of time (h)

#Lists to save the position of each object
pos1 = []
pos2 = []
pos3 = []       

'''
To update the current position for the bodies, we use a while cicle. In this case, we use append because we're working with a list, and we need to add a new data every time
the position change.

'''

t = 0
while t < t_final:
    pos1.append(r1)
    pos2.append(r2)
    pos3.append(r3)
    
    r1, r2, r3, v1, v2, v3 = rk4_step(r1, r2, r3, v1, v2, v3, m1, m2, m3, dt)
    t += dt

'''
In order to graph the solutions, we need to turn the lists into arrays

'''

pos1 = np.array(pos1)
pos2 = np.array(pos2)
pos3 = np.array(pos3)

'''
The nexts steps will define and create the graphs and animations.

'''

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2e11, 2e12)
ax.set_ylim(-2e11, 2e12)
ax.set_zlim(-2e11, 2e12)

'''
This lines will be the path the bodies will follow

'''

line1, = ax.plot([], [], [], 'o', label='Earth')          
line2, = ax.plot([], [], [], label='Moon')
line3, = ax.plot([], [], [], label='Sun')

'''
In the next step, what init does is to set up the initial state of the animation, before it starts.
It clears all the previous data and prepare them to start with the initial conditions.

What update does is taking the new data and save it, to update the position on the graph

'''

def init():

    line1.set_data([], [])
    line1.set_3d_properties([])
    line2.set_data([], [])
    line2.set_3d_properties([])
    line3.set_data([], [])
    line3.set_3d_properties([])
    return line1, line2, line3

def update(frame):
    line1.set_data(pos1[:frame, 0], pos1[:frame, 1])
    line1.set_3d_properties(pos1[:frame, 2])
    line2.set_data(pos2[:frame, 0], pos2[:frame, 1])
    line2.set_3d_properties(pos2[:frame, 2])
    line3.set_data(pos3[:frame, 0], pos3[:frame, 1])
    line3.set_3d_properties(pos3[:frame, 2])
    return line1, line2, line3


'''
Now, to plot the animation, we stablish the animation function, with the following parameters:
    -fig = The figure that will show the animation
    -update = Actualization function that is called up in every frame, it stablish the new position of each body
    -frames=len(pos1) = It stablish the total number of frames the animation is going to have, we can also use pos2 or pos3
    -init_func=init = This one is the inicialization function, it is called up at the begining of the animation 
    -interval = The time interval between frames in miliseconds.
    
'''

ani = FuncAnimation(fig, update, frames=len(pos1), init_func=init, interval=50)


plt.legend()
plt.title('Animation for the N body system on 3D')
plt.show()
