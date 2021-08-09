import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import odeint, solve_ivp
from matplotlib.widgets import Slider, TextBox
import matplotlib.gridspec as gridspec

# Start Time
t_i = 0
# End Time
t_f = 50000
# Step Interval = h
Step = 1
# Mass of massive body
M = 1
# Gravitational Constant
G = 1
# Angular Momentum
L = 4
# Array Size
size = (t_f - t_i)/Step
size = int(size)
# Time Array
Time = np.arange(t_i, t_f, Step)
# Initial radius
r_i = 5.244
# Initial radial velocity
rdot_i = 0
# Initial Phi
phi_i = 0


# Creating the effective potential function
r = sp.Symbol('r')
U_Eff = (-G*M/r + L**2/(2*r**2) - G*M*(L**2)/r**3)
U_Eff_Func = sp.lambdify(r, U_Eff)
Eff_Force = -sp.diff(U_Eff, r)
Eff_Force_Func = sp.lambdify(r, Eff_Force)
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


# Integration method for solve_ivp
def deriv(t, y):
    return [y[1], Eff_Force_Func(y[0]), Phi_dot_Func(y[0])]


# Array of initial conditions
y0 = [r_i, rdot_i, phi_i]
# Array of solutions
sol = solve_ivp(deriv, y0=y0, t_span=[t_i, t_f], t_eval=Time, rtol=1e-8, atol=1e-8)
# Autocorrelation of radius
r_cor = np.correlate(sol.y[0], sol.y[0], 'full')
# Autocorrelation of phi
phi_cor = np.correlate(sol.y[2] % np.pi, sol.y[2] % np.pi, 'full')
# Crosscorrelation of radius and phi
c_cor = np.correlate(r_cor, phi_cor, 'full')


# Creates Plotting window
fig1 = plt.figure(num='Orbit Applet', figsize=(10, 5.625))
plt.subplots_adjust(bottom=.25)
spec1 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig1)
# Plotting orbit
ax1 = fig1.add_subplot(spec1[0, 0], projection='polar')
ax1.set_title('Orbit')
orbit, = plt.polar(sol.y[2], sol.y[0])
plt.ylim(0, 25)
# Plotting Effective Potential
ax2 = fig1.add_subplot(spec1[0, 1])
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
text_bot = TextBox(ax_text, ' ', initial=str(r_i))


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
    sol = solve_ivp(deriv, y0= temp_x0, t_eval=Time, t_span=[t_i, t_f])
    orbit.set_data(sol.y[2], sol.y[0])
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
    sol = solve_ivp(deriv, y0= temp_x0, t_eval=Time, t_span=[t_i, t_f])
    orbit.set_data(sol.y[2], sol.y[0])
    fig1.canvas.draw_idle()


# Function to update initial radius from text box
def submit(text):
    s_r.set_val(eval(text))
    update_r(text)


# calls updating functions for sliders and text box
s_r.on_changed(update_r)
s_phi.on_changed(update_phi)
text_bot.on_submit(submit)


fig2 = plt.figure(num='Plots', figsize=(10, 5.625))
plt.subplots_adjust(bottom=.25)
spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)
bx1 = fig2.add_subplot(spec2[0, 0])
b1, = plt.plot(sol.t, sol.y[0])
bx1.set_title('Radius vs Time')
bx2 = fig2.add_subplot(spec2[0, 1])
b2, = plt.plot(sol.t, sol.y[2] % (2*np.pi))
bx2.set_title('Phi/(2*Pi) vs Time')
bx3 = fig2.add_subplot(spec2[1, 0])
b3, = plt.plot(r_cor)
bx4 = fig2.add_subplot(spec2[1, 1])
b4, = plt.plot(phi_cor)

fig3 = plt.figure()
plt.plot(phi_cor)


plt.show()
