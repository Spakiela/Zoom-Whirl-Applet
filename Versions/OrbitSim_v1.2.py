import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import odeint
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

# Start Time
t_i = 0
# End Time
t_f = 20000
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
r_i = 4.5
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
print(a/r_s)

# Solving Differential
def deriv(X, t):
    r = X[0]
    r_dot = X[1]
    phi = X[2]
    dXdt = [r_dot, Eff_Force_Func(r), Phi_dot_Func(r)]
    return dXdt


X0 = [r_i, rdot_i, phi_i]
sol = odeint(deriv, X0, Time)

# Plotting Effective Potential and Orbit
fig = plt.figure(figsize=(10, 5.625))
plt.subplots_adjust(bottom=.25)
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
ax1 = fig.add_subplot(spec[0, 0], projection='polar')
ax1.set_title('Orbit')
orbit, = plt.polar(sol[:, [2]], sol[:, [0]])
plt.ylim(0, 25)
ax2 = fig.add_subplot(spec[0, 1])
ax2.set_title('Effective Potential')
a1, = plt.plot(x, y)
plt.ylim(-.5, .5)
a2, = plt.plot(r_i, U_Eff_Func(r_i), "or")
# Add slider for radius
ax_r = plt.axes([.25, .1, .65, .03], facecolor='lightgoldenrodyellow')
s_r = Slider(ax_r, 'Radius', 1, 10, valinit=r_i, valstep=.0001)


def update(val):
    temp_r = s_r.val
    temp_X0 = [temp_r, rdot_i, phi_i]
    sol = odeint(deriv, temp_X0, Time)
    orbit.set_data(sol[:, [2]], sol[:, [0]])
    a2.set_data(temp_r, U_Eff_Func(temp_r))
    fig.canvas.draw_idle()


s_r.on_changed(update)


plt.show()
