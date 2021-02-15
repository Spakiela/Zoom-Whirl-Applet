import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve, newton
from matplotlib.widgets import Slider, TextBox
import matplotlib.gridspec as gridspec
import seaborn as sb
from numba import vectorize
from timeit import default_timer as timer
from fractions import Fraction as frac

# GUI updates: Solve recursion problem to add slider for periapsis, autoscale effective potential plot, dont allow
# objects to move outside of the plot, add user optimization to only allow excepted values for sliders, add text boxes
# for angular momentum and periapsis values, add some way to change simulation run time.

# GUI items needing to be added: Gravitational wave plots, gravitational radiation

# Add in quadrapole gravitational waveform, decrease energy and angular momentum from one orbital period at apopapsis

# Add Levins classification system to GUI, possibly add constant Q lines in E vs L space (may not be trivial)


# Start Time
t_i = 0
# End Time
t_f = 10000
# Step Interval = h
Step = 1
# Time array
Time = np.arange(t_i, t_f, Step)
# Mass of massive body
M = 1
# Mass of particle (For radiation and waveform)
m = M/100000
# Gravitational Constant
G = 1
# Angular Momentum
L = 3.9
# Initial radius
r_i = 4.5
# Initial radial velocity
rdot_i = 0
# Initial Phi
phi_i = 0
# Radius of innermost stable circular orbit
ISCO = (6 * G * M) / (1 + np.sqrt(1 - 12 * (G * M / L) ** 2))
# Radius of innermost unstable circular orbit
IUCO = (6 * G * M) / (1 - np.sqrt(1 - 12 * (G * M / L) ** 2))


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


def root1(r):
    return E + G*M/r - L**2/(2*r**2) + G*M*(L**2)/r**3 - (1/2) * rdot_i**2


def root2(r):
    return G*M/r**2 - L**2/r**3 + G*M*(L**2)/(3*r**4)


# Integration method for solve_ivp
def deriv(t, y):
    return [y[1], Eff_Force_Func(y[0]), Phi_dot_Func(y[0])]


# event tracking method to track apoapsis
def apoapsis(t, y):
    return y[1]


apoapsis.terminal = True
apoapsis.direction = -1


# Array of initial conditions
y0 = [r_i, rdot_i, phi_i]
# Array of solutions
sol = solve_ivp(deriv, y0=y0, t_span=[t_i, t_f], t_eval=Time, rtol=1e-8, atol=1e-8)


# Creates I_ddot tensor
def get_H():
    XYZ = np.zeros(shape=(2, sol.y[0].size))
    I = np.zeros(shape=(2, 2, sol.y[0].size))
    I_dot = np.zeros(shape=(2, 2, sol.y[0].size - 2))
    I_ddot = np.zeros(shape=(2, 2, sol.y[0].size - 4))
    for i in range(sol.y[0].size):
        XYZ[0][i] = sol.y[0][i]*np.cos(sol.y[2][i])
        XYZ[1][i] = sol.y[0][i]*np.sin(sol.y[2][i])
        I[0][0][i] = m*((XYZ[0][i])**2 - (1/3)*(sol.y[0][i])**2)
        I[1][1][i] = m*(XYZ[1][i]**2 - (1/3)*sol.y[0][i]**2)
        I[0][1][i] = m*(XYZ[0][i]**2 + XYZ[1][i]**2)
        I[1][0][i] = m*(XYZ[0][i]**2 + XYZ[1][i]**2)
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
    return 2 * M * I_ddot


# Creates Plotting window
fig1 = plt.figure(num='Orbit Applet', figsize=(16, 9))
plt.subplots_adjust(bottom=.25)
spec1 = gridspec.GridSpec(nrows=7, ncols=8, figure=fig1)

# Plotting orbit
ax1 = fig1.add_subplot(spec1[:, :3], projection='polar')
ax1.set_title('Orbit')
orbit, = plt.polar(sol.y[2], sol.y[0])
plt.ylim(0, 20)

# Plotting GW
ax3 = fig1.add_subplot(spec1[:3, 4:])
ax3.set_title('Gravitational Waves')
h_xx, = plt.plot(sol.t[10: - 4], get_H()[0][0][10:])
h_xy, = plt.plot(sol.t[10: - 4], get_H()[0][1][10:])
h_yx, = plt.plot(sol.t[10: - 4], get_H()[1][0][10:])
h_yy, = plt.plot(sol.t[10: - 4], get_H()[1][1][10:])

# Plotting Effective Potential
ax2 = fig1.add_subplot(spec1[4:, 4:])
ax2.set_title('Effective Potential')
a1, = plt.plot(Ueff_Array, Zeros_Array)

# plt.plot(x, z, color='black')
plt.ylim(-.05, .05)

# Adds periapsis on Effective Potential
a2, = plt.plot(r_i, U_Eff_Func(r_i), "or")
# x3, = plt.plot(8, U_n_Eff_Func(8), "og")
# x4, = plt.plot(16, U_n_Eff_Func(16), marker='o', color='orange')

# Adds Energy Line on Effective Potential
a3, = plt.plot(Ueff_Array, energy_line(r_i), "--r", alpha=.3)
# x1, = plt.plot(x, energy_line(8), "--g", alpha=.3)
# x2, = plt.plot(x, energy_line(16), '--', color='orange', alpha=.3)

# Displays current Energy
ax2.text(r_i+.01, U_Eff_Func(r_i) + .002, 'E=' + str(int(U_Eff_Func(r_i)*(10**5))/(10**5)))
# ax2.text(4+.01, U_Eff_Func(4) + .002, 'E=' + str(int(U_Eff_Func(4)*(10**5))/(10**5)))
# ax2.text(12+.02, U_Eff_Func(12)-.003, 'E=' + str(int(U_Eff_Func(12)*(10**5))/(10**5)))

# Add slider for energy
ax_E = plt.axes([.15, .15, .5, .02], facecolor='lightgoldenrodyellow')
s_E = Slider(ax_E, 'Energy', -.05, .05, valinit=U_Eff_Func(r_i), valstep=.00001)

# Add slider for angular momentum
ax_L = plt.axes([.15, .1, .5, .02], facecolor='lightgoldenrodyellow')
s_L = Slider(ax_L, 'Ang Momentum', 0, 10, valinit=L, valstep=.01)

# Add slider for periapse
ax_R = plt.axes([.15, .05, .5, .02], facecolor='lightgoldenrodyellow')
s_R = Slider(ax_R, 'Periapsis', 0, 10, valinit=r_i, valstep=.001)

# Add text box to manually input initial radius
ax_text = plt.axes([.75, .15, .1, .03])
text_bot = TextBox(ax_text, ' ', initial=str(int(U_Eff_Func(r_i) * (10 ** 5)) / (10 ** 5)))


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
    global E
    E = s_E.val
    global r_i
    r_i = newton(root1, x0=IUCO, x1=ISCO)
    s_R.set_val(r_i)
    global y0
    y0 = [r_i, rdot_i, phi_i]
    global sol
    sol = solve_ivp(deriv, y0=y0, t_span=[t_i, t_f], rtol=1e-8, atol=1e-8)
    orbit.set_data(sol.y[2], sol.y[0])
    a2.set_data(r_i, U_Eff_Func(r_i))
    a3.set_data(Ueff_Array, energy_line(r_i))
    del ax2.texts[-1]
    ax2.text(r_i + .02, U_Eff_Func(r_i) + .02, 'Energy=' + str(int(U_Eff_Func(r_i) * (10 ** 5)) / (10 ** 5)))
    text_bot.set_val(int(E * 10 ** 5) / 10 ** 5)
    h_xx.set_data(sol.t[10: - 4], get_H()[0][0][10:])
    h_xy.set_data(sol.t[10: - 4], get_H()[0][1][10:])
    h_yx.set_data(sol.t[10: - 4], get_H()[1][0][10:])
    h_yy.set_data(sol.t[10: - 4], get_H()[1][1][10:])
    fig1.canvas.draw_idle()


# Function to change angular momentum
def update_l(val):
    global L
    L = s_L.val
    global E
    E = s_E.val
    global r_i
    r_i = newton(root1, x0=4, maxiter=200, x1=11.57)
    global U_Eff
    U_Eff = (-G * M / r + L ** 2 / (2 * r ** 2) - G * M * (L ** 2) / r ** 3)
    global U_Eff_Func
    U_Eff_Func = sp.lambdify(r, U_Eff)
    global Eff_Force
    Eff_Force = -sp.diff(U_Eff, r)
    global Eff_Force_Func
    Eff_Force_Func = sp.lambdify(r, Eff_Force)
    global Phi_dot
    Phi_dot = L / r ** 2
    global Phi_dot_Func
    Phi_dot_Func = sp.lambdify(r, Phi_dot)
    global y0
    y0 = [r_i, rdot_i, phi_i]
    global sol
    sol = solve_ivp(deriv, y0=y0, t_span=[t_i, t_f], rtol=1e-8, atol=1e-8)
    orbit.set_data(sol.y[2], sol.y[0])
    y = np.zeros_like(Ueff_Array)
    for i in range(y.size):
        y[i] = U_Eff_Func(Ueff_Array[i])
    a1.set_data(Ueff_Array, y)
    a2.set_data(r_i, U_Eff_Func(r_i))
    a3.set_data(Ueff_Array, energy_line(r_i))
    del ax2.texts[-1]
    ax2.text(r_i + .02, U_Eff_Func(r_i) + .02, 'Energy=' + str(int(E * (10 ** 5)) / (10 ** 5)))
    h_xx.set_data(sol.t[10: - 4], get_H()[0][0][10:])
    h_xy.set_data(sol.t[10: - 4], get_H()[0][1][10:])
    h_yx.set_data(sol.t[10: - 4], get_H()[1][0][10:])
    h_yy.set_data(sol.t[10: - 4], get_H()[1][1][10:])
    fig1.canvas.draw_idle()


# Function to update periapsis
def update_r(val):
    global r_i
    r_i = s_R.val
    global y0
    y0 = [r_i, rdot_i, phi_i]
    global E
    E = U_Eff_Func(r_i)
    s_E.set_val(E)
    text_bot.set_val(E)
    temp_sol = solve_ivp(deriv, y0=y0, t_span=[t_i, t_f], rtol=1e-8, atol=1e-8)
    orbit.set_data(temp_sol.y[2], temp_sol.y[0])
    a2.set_data(r_i, U_Eff_Func(r_i))
    a3.set_data(Ueff_Array, energy_line(r_i))
    fig1.canvas.draw_idle()


# Function to update Energy from text box
def submit(text):
    s_E.set_val(eval(text))
    update_e(text)


# calls updating functions for sliders and text box
s_E.on_changed(update_e)
s_L.on_changed(update_l)
# s_R.on_changed(update_r)
text_bot.on_submit(submit)


def get_q():
    temp_y0_1 = [r_i, rdot_i, phi_i]
    temp_sol_1 = solve_ivp(deriv, y0=temp_y0_1, t_span=[0, 100000000], rtol=1e-8, atol=1e-8, events=apoapsis)
    temp_y0_2 = [temp_sol_1.y_events[0][0][0], temp_sol_1.y_events[0][0][1], temp_sol_1.y_events[0][0][2]]
    temp_sol_2 = solve_ivp(deriv, y0=temp_y0_2, t_span=[0, 1], rtol=1e-8, atol=1e-8)
    temp_y0_3 = [temp_sol_2.y[0][1], temp_sol_2.y[1][1], temp_sol_2.y[2][1]]
    temp_sol_3 = solve_ivp(deriv, y0=temp_y0_3, t_span=[0, 100000000], rtol=1e-8, atol=1e-8, events=apoapsis)
    dPhi = np.abs((temp_sol_3.y_events[0][0][2] % (2 * np.pi)) - (temp_sol_1.y_events[0][0][2] % (2 * np.pi)))
    w = int(np.abs((temp_sol_3.y_events[0][0][2] - temp_sol_1.y_events[0][0][2])) / (2*np.pi))
    Q = w + (dPhi/(2*np.pi))
    print("dPhi: " + str(dPhi))
    print("v/z: " + str(frac(dPhi/(2*np.pi)).limit_denominator()))
    print("Q: " + str(Q))
    print("Whirls: " + str(w))


def resonance_test_r():
    ISCO = (6 * G * M) / (1 + np.sqrt(1 - 12 * (G * M / L) ** 2))
    r_max = (6 * G * M) / (1 - np.sqrt(1 - 12 * (G * M / L) ** 2))
    r_array = np.arange(ISCO + .1, r_max, step)
    energy_array = np.zeros_like(r_array)
    n_array = np.zeros_like(r_array)
    l_value = np.full_like(r_array, l_array[i])
    r_iterations = int((abs(r_max) - abs(ISCO + .1)) / step)
    for j in range(r_array.size):
        temp_r = r_array[j]
        temp_y0_1 = [temp_r, rdot_i, phi_i]
        temp_sol_1 = solve_ivp(deriv, y0=temp_y0_1, t_span=[0, 100000000], rtol=1e-8, atol=1e-8, events=apoapsis)
        temp_y0_2 = [temp_sol_1.y_events[0][0][0], temp_sol_1.y_events[0][0][1], temp_sol_1.y_events[0][0][2]]
        temp_sol_2 = solve_ivp(deriv, y0=temp_y0_2, t_span=[0, 1], rtol=1e-8, atol=1e-8)
        temp_y0_3 = [temp_sol_2.y[0][1], temp_sol_2.y[1][1], temp_sol_2.y[2][1]]
        temp_sol_3 = solve_ivp(deriv, y0=temp_y0_3, t_span=[0, 100000000], rtol=1e-8, atol=1e-8, events=apoapsis)
        n_array[j] = \
            ((2 * np.pi) / ((temp_sol_3.y_events[0][0][2] - temp_sol_1.y_events[0][0][2]) % (2 * np.pi))) % 1
        energy_array[j] = U_Eff_Func(temp_r)


def resonance():
    l_min = np.sqrt(12) + .01
    l_max = 4
    step = .01
    l_array = np.arange(start=l_min, stop=l_max, step=step)
    l_iterations = int((abs(l_max) - abs(l_min)) / step)
    resonance_fig = plt.figure('Parameter space')
    ax1 = resonance_fig.add_subplot()
    ax1.set_ylabel('Angular Momentum (L)')
    ax1.set_xlabel('Energy (E)')
    for i in range(l_array.size):
        global L
        L = l_array[i]
        global U_Eff
        U_Eff = (-G * M / r + L ** 2 / (2 * r ** 2) - G * M * (L ** 2) / r ** 3)
        global U_Eff_Func
        U_Eff_Func = sp.lambdify(r, U_Eff)
        global Eff_Force
        Eff_Force = -sp.diff(U_Eff, r)
        global Eff_Force_Func
        Eff_Force_Func = sp.lambdify(r, Eff_Force)
        global Phi_dot
        Phi_dot = L / r ** 2
        global Phi_dot_Func
        Phi_dot_Func = sp.lambdify(r, Phi_dot)
        ISCO = (6 * G * M) / (1 + np.sqrt(1 - 12 * (G * M / L) ** 2))
        r_max = (6 * G * M) / (1 - np.sqrt(1 - 12 * (G * M / L) ** 2))
        r_array = np.arange(ISCO + .1, r_max, step)
        energy_array = np.zeros_like(r_array)
        n_array = np.zeros_like(r_array)
        l_value = np.full_like(r_array, l_array[i])
        r_iterations = int((abs(r_max) - abs(ISCO + .1)) / step)
        for j in range(r_array.size):
            temp_r = r_array[j]
            temp_y0_1 = [temp_r, rdot_i, phi_i]
            temp_sol_1 = solve_ivp(deriv, y0=temp_y0_1, t_span=[0, 100000000], rtol=1e-8, atol=1e-8, events=apoapsis)
            temp_y0_2 = [temp_sol_1.y_events[0][0][0], temp_sol_1.y_events[0][0][1], temp_sol_1.y_events[0][0][2]]
            temp_sol_2 = solve_ivp(deriv, y0=temp_y0_2, t_span=[0, 1], rtol=1e-8, atol=1e-8)
            temp_y0_3 = [temp_sol_2.y[0][1], temp_sol_2.y[1][1], temp_sol_2.y[2][1]]
            temp_sol_3 = solve_ivp(deriv, y0=temp_y0_3, t_span=[0, 100000000], rtol=1e-8, atol=1e-8, events=apoapsis)
            n_array[j] = \
                ((2 * np.pi) / np.abs((temp_sol_3.y_events[0][0][2] % (2 * np.pi)) - (temp_sol_1.y_events[0][0][2] % (2 * np.pi)))) % 1
            energy_array[j] = U_Eff_Func(temp_r)
            print("L: " + str(i) + "/" + str(l_iterations) + "  R: " + str(j) + "/" + str(r_iterations))
        plt.scatter(energy_array[:], l_value[:], c=n_array[:], cmap='inferno')
        plt.plot(U_Eff_Func(ISCO), L, color='black', marker='o', markersize='4')
        plt.plot(U_Eff_Func(r_max), L, color='black', marker='o', markersize='4')


def radiation(a):
    print('PROCEED WITH CAUTION!')
    print('Radiation in Progress')
    R_array = np.zeros(a)
    # Fitting Numbers
    A_E = -0.141421
    B_E = 0.752091
    C_E = -4.634643
    A_L = -1.13137
    B_L = 1.31899
    C_L = -4.149103
    # temp r for switching between orbits
    temp_r = r_i
    # Figure for testing Plots
    fig1 = plt.figure(num='Test')
    # Integration sequence to reach first apoapsis given that initial conditions start at periapsis
    start_y0 = [temp_r, rdot_i, phi_i]
    start_sol = solve_ivp(deriv, y0=start_y0, t_span=[0, 100000000], rtol=1e-8, atol=1e-8, events=apoapsis)
    start_y0 = [start_sol.y_events[0][0][0], start_sol.y_events[0][0][1], start_sol.y_events[0][0][2]]
    # Radiation integration sequence
    for i in range(a):
        temp_sol_1 = solve_ivp(deriv, y0=start_y0, t_span=[0, 1], rtol=1e-8, atol=1e-8)
        temp_y0_2 = [temp_sol_1.y[0][1], temp_sol_1.y[1][1], temp_sol_1.y[2][1]]
        temp_sol_2 = solve_ivp(deriv, y0=temp_y0_2, t_span=[0, 100000000], rtol=1e-8, atol=1e-8, events=apoapsis)
        # Plot of apoapsis to apoapsis
        plt.polar(temp_sol_2.y[2], temp_sol_2.y[0], c='red', alpha=0.7)
        # Radiation calculations
        E_Rad = \
            (m / M) * (A_E * np.arccosh(1 + B_E * (4 * M / temp_r)**6 * (M / (temp_r - 4 * M))) + C_E * (temp_r / M - 4) * (M / temp_r)**(9/2))
        L_Rad = m*(A_L*np.arccosh(1 + B_L * (4 * M / temp_r)**3 * (M / (temp_r - 4 * M))) + C_L * (temp_r / M - 4) * (M / temp_r)**3)
        # Updates to global variables affected by radiation
        global E
        E = E + E_Rad
        global L
        L = L + L_Rad
        global U_Eff
        U_Eff = (-G * M / r + L ** 2 / (2 * r ** 2) - G * M * (L ** 2) / r ** 3)
        global U_Eff_Func
        U_Eff_Func = sp.lambdify(r, U_Eff)
        global Eff_Force
        Eff_Force = -sp.diff(U_Eff, r)
        global Eff_Force_Func
        Eff_Force_Func = sp.lambdify(r, Eff_Force)
        global Phi_dot
        Phi_dot = L / r ** 2
        global Phi_dot_Func
        Phi_dot_Func = sp.lambdify(r, Phi_dot)
        global ISCO
        ISCO = (6 * G * M) / (1 + np.sqrt(1 - 12 * (G * M / L) ** 2))
        global IUCO
        IUCO = (6 * G * M) / (1 - np.sqrt(1 - 12 * (G * M / L) ** 2))
        temp_r = newton(root1, x0=IUCO, x1=ISCO)
        # Update to initial conditions
        start_y0 = [temp_r, rdot_i, phi_i]
        start_sol = solve_ivp(deriv, y0=start_y0, t_span=[0, 100000000], rtol=1e-8, atol=1e-8, events=apoapsis)
        start_y0 = [start_sol.y_events[0][0][0], start_sol.y_events[0][0][1], temp_sol_2.y_events[0][0][2]]


try:
    radiation(10)
except RuntimeError:
    print('Oh no, we\'ve fallen into the black hole')

get_q()
plt.show()

