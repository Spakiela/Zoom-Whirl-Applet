import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve, newton, bisect
from matplotlib.widgets import Slider, TextBox, Button
import matplotlib.gridspec as gridspec
from fractions import Fraction as frac

# GUI updates: Solve recursion problem to add slider for periapsis, autoscale effective potential plot, dont allow
# objects to move outside of the plot, add user optimization to only allow excepted values for sliders, add text boxes
# for angular momentum and periapsis values, add some way to change simulation run time.

# GUI items needing to be added: Gravitational wave plots, gravitational radiation

# Add in quadrapole gravitational waveform, decrease energy and angular momentum from one orbital period at apopapsis

# Add Levins classification system to GUI, possibly add constant Q lines in E vs L space (may not be trivial)

# Find units on time and compare using keplers 3rd

# Start Time
t_i = 0
# End Time
t_f = 100000
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
L = 3.8
# Initial radius
r_i = 4.5
r_p = r_i
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


# Creates I_ddot tensor
def get_H():
    XY = np.zeros(shape=(2, sol.y[0].size))
    I = np.zeros(shape=(2, 2, sol.y[0].size))
    I_dot = np.zeros(shape=(2, 2, sol.y[0].size - 2))
    I_ddot = np.zeros(shape=(2, 2, sol.y[0].size - 4))
    for i in range(sol.y[0].size):
        XY[0][i] = sol.y[0][i]*np.cos(sol.y[2][i])
        XY[1][i] = sol.y[0][i]*np.sin(sol.y[2][i])
        I[0][0][i] = m*(XY[0][i]**2 - (1/3)*(sol.y[0][i]**2))
        I[1][1][i] = m*(XY[1][i]**2 - (1/3)*(sol.y[0][i]**2))
        I[0][1][i] = m*(XY[0][i]**2 + XY[1][i]**2)
        I[1][0][i] = m*(XY[0][i]**2 + XY[1][i]**2)
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
ax1.set_title('Orbit')
orbit, = plt.polar(sol.y[2], sol.y[0])
plt.ylim(0, 20)

# Plotting GW
ax3 = fig1.add_subplot(spec1[1, 0])
ax3.set_title('H+')
h_xx, = plt.plot(sol.t[10: - 4], get_H()[0][0][10:])
plt.xlabel('t')

ax4 = fig1.add_subplot(spec1[1, 1])
ax4.set_title('Hx')
h_xy, = plt.plot(sol.t[10: - 4], get_H()[0][1][10:])
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
a2, = plt.plot(r_i, U_Eff_Func(r_i), "or")
# x3, = plt.plot(8, U_n_Eff_Func(8), "og")
# x4, = plt.plot(16, U_n_Eff_Func(16), marker='o', color='orange')

# Adds Energy Line on Effective Potential
a3, = plt.plot(Ueff_Array, energy_line(r_i), "--r", alpha=.3)
# x1, = plt.plot(x, energy_line(8), "--g", alpha=.3)
# x2, = plt.plot(x, energy_line(16), '--', color='orange', alpha=.3)

# ax2.text(4+.01, U_Eff_Func(4) + .002, 'E=' + str(int(U_Eff_Func(4)*(10**5))/(10**5)))
# ax2.text(12+.02, U_Eff_Func(12)-.003, 'E=' + str(int(U_Eff_Func(12)*(10**5))/(10**5)))

# Add slider for energy
ax_E = plt.axes([.45, .15, .4, .02], facecolor='lightgoldenrodyellow')
s_E = Slider(ax_E, 'Energy', -.05, .05, valinit=U_Eff_Func(r_i), valstep=.00001)

# Add slider for angular momentum
ax_L = plt.axes([.45, .1, .4, .02], facecolor='lightgoldenrodyellow')
s_L = Slider(ax_L, 'Ang Momentum', 0, 10, valinit=L, valstep=.01)

# Add Slider for eccentricity
ax_e = plt.axes([.05, .25, .012, .5], facecolor='lightgoldenrodyellow')
s_e = Slider(ax_e, 'Eccentricity', 0, 1, valinit=get_e(r_i, sol.y_events[0][0][0]), valstep=.001, orientation='vertical')

# Add Slider for periapsis
ax_rp = plt.axes([.1, .25, .012, .5], facecolor='lightgoldenrodyellow')
s_rp = Slider(ax_rp, 'Periapsis', 0, 10, valinit=r_p, valstep=.1, orientation='vertical')


# Add text box to manually input energy
ax_text_L = plt.axes([.25, .15, .1, .025])
text_bot_E = TextBox(ax_text_L, ' ', initial=str(int(U_Eff_Func(r_i) * (10 ** 5)) / (10 ** 5)))


# Add text box to manually input angular momentum
ax_text_E = plt.axes([.25, .1, .1, .025])
text_bot_L = TextBox(ax_text_E, ' ', initial=str(L))


# Button to go to radiation
# axrad = plt.axes([.9, .1, .05, .05], fc='grey')
# rad_button = Button(axrad, 'Radiation')


print("ISCO = " + str(ISCO))
print("IUCO = " + str(IUCO))
print("E1 = " + str(root1(ISCO)))
print("E2 = " + str(root1(IUCO)))


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
    global root1
    root1 = sp.lambdify(r, E - G * M / r + L ** 2 / (2 * r ** 2) - G * M * (L ** 2) / r ** 3)
    print("ISCO = " + str(ISCO))
    print("IUCO = " + str(IUCO))
    print("E1 = " + str(root1(ISCO)))
    print("E2 = " + str(root1(IUCO)))
    global r_i
    r_i = bisect(root1, a=IUCO, b=ISCO, disp=True)
    print("rp = " + str(r_i))
    print("**********************")
    global y0
    y0 = [r_i, rdot_i, phi_i]
    global sol
    sol = solve_ivp(deriv, y0=y0, t_span=[t_i, t_f], rtol=1e-8, atol=1e-8, events=apoapsis_nt)
    orbit.set_data(sol.y[2], sol.y[0])
    a2.set_data(r_i, U_Eff_Func(r_i))
    a3.set_data(Ueff_Array, energy_line(r_i))
    text_bot_E.set_val(int(E * 10 ** 5) / 10 ** 5)
    h_xx.set_data(sol.t[10: - 4], get_H()[0][0][10:])
    h_xy.set_data(sol.t[10: - 4], get_H()[0][1][10:])
    # h_yx.set_data(sol.t[10: - 4], get_H()[1][0][10:])
    # h_yy.set_data(sol.t[10: - 4], get_H()[1][1][10:])
    ax3.relim()
    ax3.autoscale_view()
    ax4.relim()
    ax4.autoscale_view()
    fig1.canvas.draw_idle()


# Function to change angular momentum
def update_l(val):
    global L
    L = s_L.val
    global E
    E = s_E.val
    print("ISCO = " + str(ISCO))
    print("IUCO = " + str(IUCO))
    global root1
    root1 = sp.lambdify(r, E - G * M / r + L ** 2 / (2 * r ** 2) - G * M * (L ** 2) / r ** 3)
    global r_i
    r_i = bisect(root1, a=IUCO, b=ISCO, disp=True)
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
    sol = solve_ivp(deriv, y0=y0, t_span=[t_i, t_f], rtol=1e-8, atol=1e-8, events=apoapsis_nt)
    orbit.set_data(sol.y[2], sol.y[0])
    y = np.zeros_like(Ueff_Array)
    for i in range(y.size):
        y[i] = U_Eff_Func(Ueff_Array[i])
    a1.set_data(Ueff_Array, y)
    a2.set_data(r_i, U_Eff_Func(r_i))
    a3.set_data(Ueff_Array, energy_line(r_i))
    text_bot_L.set_val(int(L * 10 ** 5) / 10 ** 5)
    h_xx.set_data(sol.t[10: - 4], get_H()[0][0][10:])
    h_xy.set_data(sol.t[10: - 4], get_H()[0][1][10:])
    # h_yx.set_data(sol.t[10: - 4], get_H()[1][0][10:])
    # h_yy.set_data(sol.t[10: - 4], get_H()[1][1][10:])
    ax3.relim()
    ax3.autoscale_view()
    ax4.relim()
    ax4.autoscale_view()
    fig1.canvas.draw_idle()


# change E to be radial kinetic energy rather than relativistic energy using sqrt(E^2 - 1)
def update_rp_ecc(val):
    rp = s_rp.val
    e = s_e.val
    global E
    E = (M * (1 - e) * (4 * M - (1 + e) * rp)) / (rp * ((1 + e) * rp - (3 + e**2) * M))
    global L
    L = ((1 + e) * rp) / (np.sqrt((1 + e) * (rp / M) - (3 + e**2)))
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
    y0 = [rp, rdot_i, phi_i]
    global sol
    sol = solve_ivp(deriv, y0=y0, t_span=[t_i, t_f], t_eval=Time, rtol=1e-8, atol=1e-8, events=apoapsis_nt)
    orbit.set_data(sol.y[2], sol.y[0])
    y = np.zeros_like(Ueff_Array)
    for i in range(y.size):
        y[i] = U_Eff_Func(Ueff_Array[i])
    a1.set_data(Ueff_Array, y)
    a2.set_data(rp, U_Eff_Func(rp))
    a3.set_data(Ueff_Array, energy_line(rp))
    h_xx.set_data(sol.t[10: - 4], get_H()[0][0][10:])
    h_xy.set_data(sol.t[10: - 4], get_H()[0][1][10:])
    # h_yx.set_data(sol.t[10: - 4], get_H()[1][0][10:])
    # h_yy.set_data(sol.t[10: - 4], get_H()[1][1][10:])
    ax3.relim()
    ax3.autoscale_view()
    ax4.relim()
    ax4.autoscale_view()
    fig1.canvas.draw_idle()


# Button to switch to radiation orbits
# def radiation_figure(event):
    # radiation(10000)
    # plt.show()
    # plt.close('Orbit Applet')


# Function to update Energy from text box
def submit_E(text):
    s_E.set_val(eval(text))
    update_e(text)


def submit_L(text):
    s_L.set_val(eval(text))
    update_l(text)


# calls updating functions for sliders and text box
s_E.on_changed(update_e)
s_L.on_changed(update_l)
s_rp.on_changed(update_rp_ecc)
s_e.on_changed(update_rp_ecc)
text_bot_E.on_submit(submit_E)
text_bot_L.on_submit(submit_L)
# rad_button.on_clicked(radiation_figure)


def radiation(a):
    global E
    global L
    print('PROCEED WITH CAUTION!')
    print('Radiation in Progress')
    Ra_array = np.zeros(1)
    Rp_array = np.zeros(1)
    t_gw_array = np.zeros(1)
    t_array = np.zeros(1)
    E_array = np.zeros(1)
    E_array[0] = E
    L_array = np.zeros(1)
    L_array[0] = L
    e_array = np.zeros(1)
    a_array = np.zeros(1)
    Q_array = np.zeros(1)
    Sol_Array = np.zeros(shape=(3, 1))
    XY = np.zeros(shape=(2, 1))
    I = np.zeros(shape=(2, 1))
    count = 0
    # Fitting Numbers
    A_E = -0.141421
    B_E = 0.752091
    C_E = -4.634643
    A_L = -1.13137
    B_L = 1.31899
    C_L = -4.149103
    # temp r for switching between orbits
    temp_r = r_i
    # Integration sequence to reach first apoapsis given that initial conditions start at periapsis
    start_y0 = [temp_r, rdot_i, phi_i]
    start_sol = solve_ivp(deriv, y0=start_y0, t_span=[t_array[0], 1000000000], rtol=1e-8, atol=1e-8, events=apoapsis)
    a_array[0] = (start_sol.y_events[0][0][0] - temp_r) / 2
    e_array[0] = get_e(temp_r, start_sol.y_events[0][0][0])
    start_y0 = [start_sol.y_events[0][0][0], start_sol.y_events[0][0][1], start_sol.y_events[0][0][2]]
    t_gw_array[0] = start_sol.t_events[0][0]
    XY[0] = start_y0[0] * np.cos(start_y0[2])
    XY[1] = start_y0[0] * np.sin(start_y0[2])
    I[0] = m * (XY[0][-1] ** 2 - (1 / 3) * (start_y0[0] ** 2))
    I[1] = m * (XY[0][-1] ** 2 + XY[1][-1] ** 2)
    t_array[0] = start_sol.t_events[0][0]
    Ra_array[0] = start_sol.y_events[0][0][0]
    Rp_array[0] = temp_r
    # Radiation integration sequence
    try:
        for i in range(a):
            temp_sol_1 = solve_ivp(deriv, y0=start_y0, t_span=[t_array[-1], t_array[-1] + .001], rtol=1e-8, atol=1e-8)
            temp_y0_2 = [temp_sol_1.y[0][1], temp_sol_1.y[1][1], temp_sol_1.y[2][1]]
            temp_sol_2 = solve_ivp(deriv, y0=temp_y0_2, t_span=[temp_sol_1.t[-1], 1000000000], rtol=1e-8, atol=1e-8, events=apoapsis)
            s = temp_sol_2.t.size
            # Plot of apoapsis to apoapsis
            Sol_Array = np.append(Sol_Array, [temp_sol_2.y[0], temp_sol_2.y[1], temp_sol_2.y[2]], axis=1)
            # Levin's Q
            dPhi = np.abs((temp_sol_2.y_events[0][0][2] % (2 * np.pi)) - (start_y0[2] % (2 * np.pi)))
            w = int(np.abs((temp_sol_2.y_events[0][0][2] - start_y0[2])) / (2 * np.pi))
            Q_array = np.append(Q_array, w + (dPhi / (2 * np.pi)))
            # Gravitational Wave calculation
            for j in range(s):
                XY = np.append(XY, [[temp_sol_2.y[0][j] * np.cos(temp_sol_2.y[2][j])], [temp_sol_2.y[0][j] * np.sin(temp_sol_2.y[2][j])]], axis=1)
                I = np.append(I, [[m * (XY[0][-1] ** 2 - (1 / 3) * (temp_sol_2.y[0][j]) ** 2)], [m * (XY[0][-1] ** 2 + XY[1][-1] ** 2)]], axis=1)
                t_gw_array = np.append(t_gw_array, temp_sol_2.t[j])
            t_array = np.append(t_array, temp_sol_2.t_events[0][0])
            # Radiation calculations
            E_Rad = \
                (m / M) * (A_E * np.arccosh(1 + B_E * (4 * M / temp_r)**6 * (M / (temp_r - 4 * M))) + C_E * (temp_r / M - 4) * (M / temp_r)**(9/2))
            L_Rad = m*(A_L*np.arccosh(1 + B_L * (4 * M / temp_r)**3 * (M / (temp_r - 4 * M))) + C_L * (temp_r / M - 4) * (M / temp_r)**3)
            # Updates to global variables affected by radiation
            E = E + E_Rad
            E_array = np.append(E_array, E)
            L = L + L_Rad
            L_array = np.append(L_array, L)
            global root1
            root1 = sp.lambdify(r, E + G * M / r - L ** 2 / (2 * r ** 2) + G * M * (L ** 2) / r ** 3 - (1 / 2) * rdot_i ** 2)
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
            global r_p
            r_p = bisect(root1, a=IUCO, b=ISCO, disp=True)
            # Update to initial conditions
            start_y0 = [r_p, rdot_i, phi_i]
            start_sol = solve_ivp(deriv, y0=start_y0, t_span=[0, 1000000000], rtol=1e-8, atol=1e-8, events=apoapsis)
            a_array = np.append(a_array, (start_sol.y_events[0][0][0] - r_p) / 2)
            e_array = np.append(e_array, get_e(r_p, start_sol.y_events[0][0][0]))
            start_y0 = [start_sol.y_events[0][0][0], start_sol.y_events[0][0][1], temp_sol_2.y_events[0][0][2]]
            Ra_array = np.append(Ra_array, start_sol.y_events[0][0][0])
            Rp_array = np.append(Rp_array, r_p)
            count = count + 1
            print(str(count) + '/' + str(a))
        I_dot = np.zeros(shape=(2, I[0].size - 2))
        I_ddot = np.zeros(shape=(2, I_dot[0].size - 2))
        for i in range(I_dot[0].size):
            I_dot[0][i] = (I[0][i + 2] - I[0][i]) / (t_gw_array[i + 2] - t_gw_array[i])
            I_dot[1][i] = (I[1][i + 2] - I[1][i]) / (t_gw_array[i + 2] - t_gw_array[i])
        t_gw_array = np.delete(t_gw_array, 0)
        t_gw_array = np.delete(t_gw_array, -1)
        for i in range(I_ddot[0].size - 2):
            I_ddot[0][i] = (I_dot[0][i + 2] - I_dot[0][i]) / (t_gw_array[i + 2] - t_gw_array[i])
            I_ddot[1][i] = (I_dot[1][i + 2] - I_dot[1][i]) / (t_gw_array[i + 2] - t_gw_array[i])
        t_gw_array = np.delete(t_gw_array, 0)
        t_gw_array = np.delete(t_gw_array, -1)
        dEdt_array = np.zeros(E_array.size - 2)
        dLdt_array = np.zeros(L_array.size - 2)
        for n in range(t_array.size - 2):
            dEdt_array[n] = (E_array[n + 2] - E_array[n]) / (t_array[n + 2] - t_array[n])
            dLdt_array[n] = (L_array[n + 2] - L_array[n]) / (t_array[n + 2] - t_array[n])
        fig1 = plt.figure(figsize=(16, 9))
        plt.subplots_adjust(bottom=.25)
        spec = gridspec.GridSpec(nrows=2, ncols=2, figure=fig1)
        ax1 = fig1.add_subplot(spec[0, 0])
        # plt.scatter(abs(e_array[:]), a_array[:])
        plt.scatter(Rp_array[1:a], np.abs(dEdt_array[:]), s=4)
        plt.xlabel('Radius of Periapsis')
        plt.ylabel('|dE/dt|')
        ax2 = fig1.add_subplot(spec[1, 0])
        # plt.plot(t_gw_array[:], 2 * M * I_ddot[0][:])
        # plt.scatter(t_array[:], E_array[:])
        # plt.scatter(Rp_array[1:a], np.abs(dEdt_array[:]))
        plt.scatter(Rp_array[1:a], np.abs(dLdt_array[:]), s=4)
        plt.xlabel('Radius of Periapsis')
        plt.ylabel('|dL/dt|')
        ax3 = fig1.add_subplot(spec[1, 1])
        # plt.scatter(XY[0], XY[1])
        # plt.scatter(Sol_Array[2][1:], Sol_Array[0][1:])
        # plt.scatter(E_array[:-1], Q_array[1:])
        plt.scatter(Rp_array[:], e_array[:], s=4)
        plt.xlabel('r_p')
        plt.ylabel('e')
        ax4 = fig1.add_subplot(spec[0, 1])
        # plt.polar(Sol_Array[2][1:], Sol_Array[0][1:])
        plt.plot(E_array[:], L_array[:])
        # plt.xlabel('Energy')
        # plt.ylabel('Angular Momentum')
        # plt.plot(t_gw_array[:], 2 * M * I_ddot[0][:], alpha=.7)
        # plt.plot(sol.t[10: - 4], 2 * get_H()[0][0][10:], alpha=.7)
        # plt.xlim(1000, 2000)
        # plt.ylim(-.000025, .000025)
    except (RuntimeError, ValueError):
        I_dot = np.zeros(shape=(2, I[0].size - 2))
        I_ddot = np.zeros(shape=(2, I_dot[0].size - 2))
        for i in range(I_dot[0].size):
            I_dot[0][i] = (I[0][i + 2] - I[0][i]) / (t_gw_array[i + 2] - t_gw_array[i])
            I_dot[1][i] = (I[1][i + 2] - I[1][i]) / (t_gw_array[i + 2] - t_gw_array[i])
        t_gw_array = np.delete(t_gw_array, 0)
        t_gw_array = np.delete(t_gw_array, -1)
        for i in range(I_ddot[0].size - 2):
            I_ddot[0][i] = (I_dot[0][i + 2] - I_dot[0][i]) / (t_gw_array[i + 2] - t_gw_array[i])
            I_ddot[1][i] = (I_dot[1][i + 2] - I_dot[1][i]) / (t_gw_array[i + 2] - t_gw_array[i])
        t_gw_array = np.delete(t_gw_array, 0)
        t_gw_array = np.delete(t_gw_array, -1)
        dEdt_array = np.zeros(E_array.size - 2)
        dLdt_array = np.zeros(L_array.size - 2)
        for n in range(t_array.size - 2):
            dEdt_array[n] = (E_array[n + 2] - E_array[n]) / (t_array[n + 2] - t_array[n])
            dLdt_array[n] = (L_array[n + 2] - L_array[n]) / (t_array[n + 2] - t_array[n])
        fig1 = plt.figure(figsize=(16, 9))
        plt.subplots_adjust(bottom=.25)
        spec = gridspec.GridSpec(nrows=2, ncols=2, figure=fig1)
        ax1 = fig1.add_subplot(spec[0, 0])
        # plt.scatter(abs(e_array[:]), a_array[:])
        plt.scatter(Rp_array[1:a], np.abs(dEdt_array[:]), s=4)
        plt.xlabel('Radius of Periapsis')
        plt.ylabel('|dE/dt|')
        ax2 = fig1.add_subplot(spec[1, 0])
        # plt.plot(t_gw_array[:], 2 * M * I_ddot[0][:])
        # plt.scatter(t_array[:], E_array[:])
        # plt.scatter(Rp_array[1:a], np.abs(dEdt_array[:]))
        plt.scatter(Rp_array[1:a], np.abs(dLdt_array[:]), s=4)
        plt.xlabel('Radius of Periapsis')
        plt.ylabel('|dL/dt|')
        ax3 = fig1.add_subplot(spec[1, 1])
        # plt.scatter(XY[0], XY[1])
        # plt.scatter(Sol_Array[2][1:], Sol_Array[0][1:])
        # plt.scatter(E_array[:-1], Q_array[1:])
        plt.scatter(Rp_array[:], e_array[:], s=4)
        plt.xlabel('r_p')
        plt.ylabel('e')
        ax4 = fig1.add_subplot(spec[0, 1])
        # plt.polar(Sol_Array[2][1:], Sol_Array[0][1:])
        plt.scatter(E_array[:], L_array[:], s=4)
        plt.xlabel('Energy')
        plt.ylabel('Angular Momentum')
        # plt.plot(t_gw_array[:], 2 * M * I_ddot[0][:], alpha=.7)
        # plt.plot(sol.t[10: - 4], 2 * get_H()[0][0][10:], alpha=.7)
        # plt.xlim(1000, 2000)
        # plt.ylim(-.000025, .000025)

# radiation(2)
plt.show()
