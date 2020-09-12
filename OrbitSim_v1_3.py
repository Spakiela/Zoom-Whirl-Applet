import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import odeint
from matplotlib.widgets import Slider, TextBox
import matplotlib.gridspec as gridspec

# Start Time
t_i = 0
# End Time
t_f = 10000
# Step Interval = h
Step = .1
# speed of Light
c = 1
# Mass of massive body
M = 1
# Mass of smaller body
m = 1
# Reduced Mass of the system
RMass = M*m/(M+m)
# Gravitational Constant
G = 1
# Total Energy
E = 1
# Angular Momentum
L = 4
# Specific Angular Momentum
h = L/RMass
# Schwartzchild Radius
r_s = 2*G*M/c**2
# Array Size
size = (t_f - t_i)/Step
size = int(size)
# Time Array
Time = np.arange(t_i, t_f, Step)
# Initial radius
r_i = 5
# Initial radial velocity
rdot_i = 0
# Initial Phi
phi_i = 0

a = h/c
b = h*RMass*c/E


# Creating the effective potential function
r = sp.Symbol('r')
# U_Eff = (-G*M*m/r + L**2/(2*RMass*r**2) - G*(M+m)*(L**2)/((c**2)*RMass*(r**3)))
U_Eff = (-G*M/r + L**2/(2*r**2) - G*M*(L**2)/r**3)/M
U_Eff_Func = sp.lambdify(r, U_Eff)
Eff_Force = -sp.diff(U_Eff, r)
Eff_Force_Func = sp.lambdify(r, Eff_Force)
# Phi = 1/((r**2)*sp.sqrt((1/b**2)-((1-(r_s/r))*((1/a**2)+(1/r**2)))))
Phi_dot = h/r**2
Phi_dot_Func = sp.lambdify(r, Phi_dot)

# Plotting Effective Potential
x = np.linspace(1, 10, 1000)
y = np.zeros_like(x)
for i in range(y.size):
    y[i] = U_Eff_Func(x[i])


# Creating a line to represent particles energy
def energy_line(Radius):
    y1 = np.zeros_like(x)
    for i in range(x.size):
        y1[i] = U_Eff_Func(Radius)
    return y1


# Solving Differential
def deriv(x, t):
    r = x[0]
    r_dot = x[1]
    phi = x[2]
    dXdt = [r_dot, Eff_Force_Func(r), Phi_dot_Func(r)]
    return dXdt


x0 = [r_i, rdot_i, phi_i]
sol = odeint(deriv, x0, Time)


# Creates Plotting window
fig1 = plt.figure(num='Orbit Applet', figsize=(10, 5.625))
plt.subplots_adjust(bottom=.25)
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig1)
# Plotting orbit
ax1 = fig1.add_subplot(spec[0, 0], projection='polar')
ax1.set_title('Orbit')
orbit, = plt.polar(sol[:, [2]], sol[:, [0]])
plt.ylim(0, 25)
# Plotting Effective Potential
ax2 = fig1.add_subplot(spec[0, 1])
ax2.set_title('Effective Potential')
a1, = plt.plot(x, y)
plt.ylim(-.2, .2)
# Adds Initial Radius on Effective Potential
a2, = plt.plot(r_i, U_Eff_Func(r_i), "or")
# Adds Energy Line on Effective Potential
a3, = plt.plot(x, energy_line(r_i), "--b", alpha=.3)
# Displays current Energy
ax2.text(r_i+.02, U_Eff_Func(r_i)+.02, 'U_eff=' + str(int(U_Eff_Func(r_i)*(10**5))/(10**5)))
# Add slider for initial radius
ax_r = plt.axes([.15, .15, .5, .04], facecolor='lightgoldenrodyellow')
s_r = Slider(ax_r, 'Initial Radius', 1, 10, valinit=r_i, valstep=.0001)
# Add slider for initial angle
ax_phi = plt.axes([.15, .075, .5, .04], facecolor='lightgoldenrodyellow')
s_phi = Slider(ax_phi, 'Initial Phi', 0, 2*np.pi, valinit=phi_i, valstep=.0001)
# Add text box to manually input initial radius
ax_text = plt.axes([.725, .15, .05, .04])
text_bot = TextBox(ax_text, ' ', initial=r_i)


# Function to zoom in and out of orbit plot w/ scroll wheel
def zoom(ax, scale):
    def zoom_event(event):
        cur_ylim = ax.get_ylim()
        if event.button == 'up':
            scale_fac = 1/scale
        elif event.button == 'down':
            scale_fac = scale
        else:
            scale_fac = 1
        ax.set_ylim([0, cur_ylim[1]*scale_fac])
        plt.draw()

    fig = ax.get_figure()
    fig.canvas.mpl_connect('scroll_event', zoom_event)

    return zoom_event


f = zoom(ax1, 1.5)


# Function to update plots when initial radius is changed
def update_r(val):
    temp_r = s_r.val
    temp_x0 = [temp_r, rdot_i, s_phi.val]
    sol = odeint(deriv, temp_x0, Time)
    orbit.set_data(sol[:, [2]], sol[:, [0]])
    a2.set_data(temp_r, U_Eff_Func(temp_r))
    a3.set_data(x, energy_line(temp_r))
    del ax2.texts[-1]
    ax2.text(temp_r+.02, U_Eff_Func(temp_r)+.02, 'U_eff=' + str(int(U_Eff_Func(temp_r)*(10**5))/(10**5)))
    text_bot.set_val(int(temp_r*10**3)/10**3)
    fig1.canvas.draw_idle()


# Function to update orbit plot when initial angle is changed
def update_phi(val):
    temp_phi = s_phi.val
    temp_x0 = [s_r.val, rdot_i, temp_phi]
    sol = odeint(deriv, temp_x0, Time)
    orbit.set_data(sol[:, [2]], sol[:, [0]])
    fig1.canvas.draw_idle()


# Function to update initial radius from text box
def submit(text):
    s_r.set_val(eval(text))
    update_r(text)


# calls updating functions for sliders and text box
s_r.on_changed(update_r)
s_phi.on_changed(update_phi)
text_bot.on_submit(submit)


plt.show()
