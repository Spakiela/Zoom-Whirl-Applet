import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve, newton, bisect
from matplotlib.widgets import Slider, TextBox, Button
import matplotlib.gridspec as gridspec
from fractions import Fraction as frac
from matplotlib.animation import FuncAnimation

# Fix sliders
# Animation time, (set number of frames? increase in run time decrease in step size?)
# App closes when animation is finished
# U_Eff scaling to see periapsis? Is this useful or maybe a control
# Change time to coordinate time
# Add meaning to red bars on sliders


# Start Time
t_i = 0
# End Time
t_f = 4000
# Step Interval = h
Step = 2
# Time array
Time = np.arange(t_i, t_f, Step)
# Mass of massive body
M = 1
# Mass of particle (For radiation and waveform)
m = M/100000
# Gravitational Constant
G = 1
# Angular Momentum
L = 4
# Initial radius
r_i = 4
r_p = r_i
# Initial radial velocity
rdot_i = 0
# Initial Phi
phi_i = 0
# Radius of innermost stable circular orbit (Moore 10.11)
ISCO = (6 * G * M) / (1 - np.sqrt(1 - 12 * (G * M / L) ** 2))
# Radius of innermost unstable circular orbit (Moore 10.11)
IUCO = (6 * G * M) / (1 + np.sqrt(1 - 12 * (G * M / L) ** 2))


# Creating the effective potential function
r = sp.Symbol('r')
U_Eff = (-G*M/r + L**2/(2*r**2) - G*M*(L**2)/r**3)
U_Eff_Func = sp.lambdify(r, U_Eff)
Eff_Force = -sp.diff(U_Eff, r)
Eff_Force_Func = sp.lambdify(r, Eff_Force)
Phi_dot = L/r**2
Phi_dot_Func = sp.lambdify(r, Phi_dot)
U_n_Eff = -G*M/r + L**2/(2*r**2)
U_n_Eff_Func = sp.lambdify(r, U_n_Eff)


# Instantiating E
E = U_Eff_Func(r_i)


# Plotting Effective Potential
Ueff_Array = np.linspace(1, 50, 1000)
Zeros_Array = np.zeros_like(Ueff_Array)
for i in range(Zeros_Array.size):
    Zeros_Array[i] = U_Eff_Func(Ueff_Array[i])


# Creating a line to represent particles energy
def energy_line(Radius):
    y1 = np.zeros_like(Ueff_Array)
    for i in range(Ueff_Array.size):
        y1[i] = U_Eff_Func(Radius)
        # y1[i] = U_n_Eff_Func(Radius)
    return y1


root1 = sp.lambdify(r,  E - G*M/r + L**2/(2*r**2) - G*M*(L**2)/r**3)


def get_e(p, a):
    return (a - p) / (p + a)


# Integration method for solve_ivp
def deriv(t, y):
    return [y[1], Eff_Force_Func(y[0]), Phi_dot_Func(y[0])]


# event tracking method to track apoapsis terminal
def apoapsis(t, y):
    return y[1]


def apoapsis_nt(t, y):
    return y[1]


apoapsis.terminal = True
apoapsis.direction = -1
apoapsis_nt.terminal = False
apoapsis_nt.direction = -1


# Array of initial conditions
y0 = [r_i, rdot_i, phi_i]
# Array of solutions
sol = solve_ivp(deriv, y0=y0, t_span=[t_i, t_f], t_eval=Time, rtol=1e-8, atol=1e-8, events=apoapsis_nt)

Ecc = get_e(r_i, sol.y_events[0][0][0])


# Creates I_ddot tensor
def get_H():
    XY = np.zeros(shape=(2, sol.y[0].size))
    I = np.zeros(shape=(2, 2, sol.y[0].size))
    I_dot = np.zeros(shape=(2, 2, sol.y[0].size - 2))
    I_ddot = np.zeros(shape=(2, 2, sol.y[0].size - 4))
    for i in range(sol.y[0].size):
        XY[0][i] = sol.y[0][i]*np.cos(sol.y[2][i])
        XY[1][i] = sol.y[0][i]*np.sin(sol.y[2][i])
        I[0][0][i] = m*((XY[0][i]**2) - (1/3)*(sol.y[0][i]**2))
        I[1][1][i] = m*((XY[1][i]**2) - (1/3)*(sol.y[0][i]**2))
        I[0][1][i] = 2*m*(XY[0][i]*XY[1][i])
        I[1][0][i] = 2*m*(XY[0][i]*XY[1][i])
    for i in range(sol.y[0].size - 2):
        I_dot[0][0][i] = (I[0][0][i + 2] - I[0][0][i])/(sol.t[i + 2] - sol.t[i])
        I_dot[1][1][i] = (I[1][1][i + 2] - I[1][1][i]) / (sol.t[i + 2] - sol.t[i])
        I_dot[1][0][i] = (I[1][0][i + 2] - I[1][0][i]) / (sol.t[i + 2] - sol.t[i])
        I_dot[0][1][i] = (I[0][1][i + 2] - I[0][1][i]) / (sol.t[i + 2] - sol.t[i])
    for i in range(sol.y[0].size - 4):
        I_ddot[0][0][i] = (I_dot[0][0][i + 2] - I_dot[0][0][i])/(sol.t[i + 2] - sol.t[i])
        I_ddot[1][1][i] = (I_dot[1][1][i + 2] - I_dot[1][1][i]) / (sol.t[i + 2] - sol.t[i])
        I_ddot[1][0][i] = (I_dot[1][0][i + 2] - I_dot[1][0][i]) / (sol.t[i + 2] - sol.t[i])
        I_ddot[0][1][i] = (I_dot[0][1][i + 2] - I_dot[0][1][i]) / (sol.t[i + 2] - sol.t[i])
    return I_ddot


# Creates Plotting window
fig1 = plt.figure(num='Orbit Applet', figsize=(16, 9))
plt.subplots_adjust(bottom=.25, left=.2)
spec1 = gridspec.GridSpec(nrows=2, ncols=2, figure=fig1)

# Plotting orbit
ax1 = fig1.add_subplot(spec1[0, 0], projection='polar')
# ax1.set_title('Orbit')
orbit, = plt.polar(sol.y[2], sol.y[0])
# orbit_dot, = plt.polar(sol.y[2][0], sol.y[0][0], "or")
plt.ylim(0, 20)

# Plotting GW
ax3 = fig1.add_subplot(spec1[1, 0])
ax3.set_title('H+')
h_xx, = plt.plot(sol.t[10: - 4], get_H()[0][0][10:])
# h_xx_dot, = plt.plot(sol.t[10], get_H()[0][0][10], "or")
plt.xlabel('t')

ax4 = fig1.add_subplot(spec1[1, 1])
ax4.set_title('Hx')
h_xy, = plt.plot(sol.t[10: - 4], get_H()[1][0][10:])
# h_xy_dot, = plt.plot(sol.t[10], get_H()[1][0][10], "or")
plt.xlabel('t')


# Plotting Effective Potential
ax2 = fig1.add_subplot(spec1[0, 1])
ax2.set_title('Effective Potential')
a1, = plt.plot(Ueff_Array, Zeros_Array)
plt.xlabel('r/M')
plt.ylabel('E')

# plt.plot(x, z, color='black')
plt.ylim(-.05, .05)

# Adds periapsis on Effective Potential
# a2, = plt.plot(r_i, U_Eff_Func(r_i), "or")

# Adds Energy Line on Effective Potential
a3, = plt.plot(Ueff_Array, energy_line(r_i), "--r", alpha=.3)

# Add slider for energy
ax_E = plt.axes([.45, .15, .4, .02], facecolor='lightgoldenrodyellow')
s_E = Slider(ax_E, 'Energy', -.05, .05, valinit=U_Eff_Func(r_i), valstep=.00001)

# Add slider for angular momentum
ax_L = plt.axes([.45, .1, .4, .02], facecolor='lightgoldenrodyellow')
s_L = Slider(ax_L, 'Ang Momentum', 0, 10, valinit=L, valstep=.01)

# Add Slider for eccentricity
ax_e = plt.axes([.05, .25, .012, .5], facecolor='lightgoldenrodyellow')
s_e = Slider(ax_e, 'Eccentricity', 0, 1, valinit=Ecc, valstep=.001, orientation='vertical')

# Add Slider for periapsis
ax_rp = plt.axes([.1, .25, .012, .5], facecolor='lightgoldenrodyellow')
s_rp = Slider(ax_rp, 'Periapsis', 0, 10, valinit=r_p, valstep=.01, orientation='vertical')


# Add text box to manually input energy
ax_text_L = plt.axes([.25, .15, .1, .025])
text_bot_E = TextBox(ax_text_L, ' ', initial=str(int(U_Eff_Func(r_i) * (10 ** 5)) / (10 ** 5)))


# Add text box to manually input angular momentum
ax_text_E = plt.axes([.25, .1, .1, .025])
text_bot_L = TextBox(ax_text_E, ' ', initial=str(L))


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


# Function to update plots when energy is changed
def update_e(val):
    update_figures(s_E.val, L)


# Function to change angular momentum
def update_l(val):
    update_figures(E, s_L.val)


def update_rp_ecc(val):
    global Ecc
    rp = s_rp.val
    Ecc = s_e.val
    temp_E = (M * (1 - Ecc) * (4 * M - (1 + Ecc) * rp)) / (2 * rp * ((1 + Ecc) * rp - (3 + Ecc ** 2) * M))
    temp_L = ((1 + Ecc) * rp) / (np.sqrt((1 + Ecc) * (rp / M) - (3 + Ecc ** 2)))
    update_figures(temp_E, temp_L)


def update_figures(Eval, Lval):
    global E, L, ISCO, IUCO, root1, r_i, U_Eff, U_Eff_Func, Eff_Force, Eff_Force_Func, Phi_dot, Phi_dot_Func, y0, sol
    print("---------")
    print("E:" + str(E))
    print("L:" + str(L))
    if (Eval != E) and (Lval != L):
        print("Rp or Ecc was updated")
        E = Eval
        s_L.set_val(Lval)
    elif Lval != L:
        print("L slider was updated")
        L = Lval
        ISCO = (6 * G * M) / (1 - np.sqrt(1 - 12 * (G * M / L) ** 2))
        IUCO = (6 * G * M) / (1 + np.sqrt(1 - 12 * (G * M / L) ** 2))
        print("ISCO:" + str(ISCO))
        print("IUCO:" + str(IUCO))
        U_Eff = (-(G * M / r) + ((L ** 2) / (2 * (r ** 2))) - ((G * M * (L ** 2)) / (r ** 3)))
        U_Eff_Func = sp.lambdify(r, U_Eff)
        print("E_IUCO:" + str(U_Eff_Func(IUCO)))
        Eff_Force = -sp.diff(U_Eff, r)
        Eff_Force_Func = sp.lambdify(r, Eff_Force)
        Phi_dot = L / r ** 2
        Phi_dot_Func = sp.lambdify(r, Phi_dot)
        # Ecc is not being updated and is causing sliders to stop working
        if s_rp.val == int(IUCO * (10 ** 3)) / (10 ** 3):
            r_i = s_rp.val
            E = U_Eff_Func(IUCO)
            s_E.set_val(E)
        elif s_rp.val == int(ISCO * (10 ** 3)) / (10 ** 3):
            r_i = ISCO
            E = U_Eff_Func(ISCO)
            s_E.set_val(E)
        else:
            root1 = sp.lambdify(r, E + (G * M / r) - ((L ** 2) / (2 * (r ** 2))) + ((G * M * (L ** 2)) / (r ** 3)))
            r_i = bisect(root1, a=IUCO, b=ISCO, disp=True)
            s_E.set_val(E)
    elif Eval != E:
        print("E slider was updated")
        E = Eval
        root1 = sp.lambdify(r, E + (G * M / r) - (L ** 2 / (2 * (r ** 2))) + ((G * M * (L ** 2)) / (r ** 3)))
        r_i = bisect(root1, a=IUCO, b=ISCO, disp=True)
        s_E.set_val(E)
    else:
        print("Sliders are now updated")
        y0 = [r_i, rdot_i, phi_i]
        sol = solve_ivp(deriv, y0=y0, t_eval=Time, t_span=[t_i, t_f], rtol=1e-8, atol=1e-8, events=apoapsis_nt)
        orbit.set_data(sol.y[2], sol.y[0])
        y = np.zeros_like(Ueff_Array)
        for i in range(y.size):
            y[i] = U_Eff_Func(Ueff_Array[i])
        a1.set_data(Ueff_Array, y)
        # a2.set_data(r_i, U_Eff_Func(r_i))
        a3.set_data(Ueff_Array, energy_line(r_i))
        h_xx.set_data(sol.t[0: - 4], get_H()[0][0])
        h_xy.set_data(sol.t[0: - 4], get_H()[1][0])
        h_xx_dot.set_data(sol.t[0], get_H()[0][0][0])
        h_xy_dot.set_data(sol.t[0], get_H()[1][0][0])
        # h_yx.set_data(sol.t[10: - 4], get_H()[1][0][10:])
        # h_yy.set_data(sol.t[10: - 4], get_H()[1][1][10:])
        ax3.relim()
        ax3.autoscale_view()
        ax4.relim()
        ax4.autoscale_view()
        fig1.canvas.draw_idle()


# Function to update Energy from text box
def submit_E(text):
    s_E.set_val(eval(text))


def submit_L(text):
    s_L.set_val(eval(text))


# calls updating functions for sliders and text box
s_E.on_changed(update_e)
s_L.on_changed(update_l)
s_rp.on_changed(update_rp_ecc)
s_e.on_changed(update_rp_ecc)
text_bot_E.on_submit(submit_E)
text_bot_L.on_submit(submit_L)


def update(frame_number):
    orbit_dot.set_data(sol.y[2][frame_number], sol.y[0][frame_number])
    a2.set_data(sol.y[0][frame_number], U_Eff_Func(r_i))
    h_xx_dot.set_data(sol.t[frame_number], get_H()[0][0][frame_number])
    h_xy_dot.set_data(sol.t[frame_number], get_H()[1][0][frame_number])
    # fig1.canvas.draw_idle()


# animation = FuncAnimation(fig1, update, interval=.1)

plt.show()
