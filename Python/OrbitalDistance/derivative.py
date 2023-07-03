import os
import re
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy.utilities import lambdify


def derivative(a1, e1, i1, w1, O1, T1, a2, e2, i2, w2, O2, T2):
    u, v = sp.symbols('u v')

    M1 = 2 * sp.pi / T1 * u
    E1 = M1 + e1 * sp.sin(M1) + 1/2 * e1 ** 2 * sp.sin(2 * M1)

    X1 = a1 * (sp.cos(E1) - e1)
    Y1 = a1 * sp.sqrt(1 - e1 ** 2) * sp.sin(E1)

    P11_1 = (sp.cos(O1) * sp.cos(w1) - sp.sin(O1) * sp.cos(i1) * sp.sin(w1))
    P12_1 = (- sp.cos(O1) * sp.sin(w1) - sp.sin(O1) * sp.cos(i1) * sp.cos(w1))
    P21_1 = (sp.sin(O1) * sp.cos(w1) + sp.cos(O1) * sp.cos(i1) * sp.sin(w1))
    P22_1 = (- sp.sin(O1) * sp.sin(w1) + sp.cos(O1) * sp.cos(i1) * sp.cos(w1))
    P31_1 = sp.sin(i1) * sp.sin(w1)
    P32_1 = sp.sin(i1) * sp.cos(w1)

    x1 = X1 * P11_1 + Y1 * P12_1
    y1 = X1 * P21_1 + Y1 * P22_1
    z1 = X1 * P31_1 + Y1 * P32_1

    M2 = 2 * sp.pi / T2 * v
    E2 = M2 + e2 * sp.sin(M2) + 1/2 * e2 ** 2 * sp.sin(2 * M2)

    X2 = a2 * (sp.cos(E2) - e2)
    Y2 = a2 * sp.sqrt(1 - e2 ** 2) * sp.sin(E2)

    P11_2 = (sp.cos(O2) * sp.cos(w2) - sp.sin(O2) * sp.cos(i2) * sp.sin(w2))
    P12_2 = (- sp.cos(O2) * sp.sin(w2) - sp.sin(O2) * sp.cos(i2) * sp.cos(w2))
    P21_2 = (sp.sin(O2) * sp.cos(w2) + sp.cos(O2) * sp.cos(i2) * sp.sin(w2))
    P22_2 = (- sp.sin(O2) * sp.sin(w2) + sp.cos(O2) * sp.cos(i2) * sp.cos(w2))
    P31_2 = sp.sin(i2) * sp.sin(w2)
    P32_2 = sp.sin(i2) * sp.cos(w2)

    x2 = X2 * P11_2 + Y2 * P12_2
    y2 = X2 * P21_2 + Y2 * P22_2
    z2 = X2 * P31_2 + Y2 * P32_2

    f = sp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    df_du = sp.diff(f, u)
    df_dv = sp.diff(f, v)

    # sol = sp.solve((df_du, df_dv), (u, v))

    df_du_simp = sp.simplify(df_du)
    df_dv_simp = sp.simplify(df_dv)

    print(df_du_simp)
    print("\n\n")
    print(df_dv_simp)


def calculate_constants(a1, e1, i1, w1, O1, a2, e2, i2, w2, O2):
    P11_1 = (np.cos(O1) * np.cos(w1) - np.sin(O1) * np.cos(i1) * np.sin(w1))
    P12_1 = (- np.cos(O1) * np.sin(w1) - np.sin(O1) * np.cos(i1) * np.cos(w1))
    P21_1 = (np.sin(O1) * np.cos(w1) + np.cos(O1) * np.cos(i1) * np.sin(w1))
    P22_1 = (- np.sin(O1) * np.sin(w1) + np.cos(O1) * np.cos(i1) * np.cos(w1))
    P31_1 = np.sin(i1) * np.sin(w1)
    P32_1 = np.sin(i1) * np.cos(w1)

    P11_2 = (np.cos(O2) * np.cos(w2) - np.sin(O2) * np.cos(i2) * np.sin(w2))
    P12_2 = (- np.cos(O2) * np.sin(w2) - np.sin(O2) * np.cos(i2) * np.cos(w2))
    P21_2 = (np.sin(O2) * np.cos(w2) + np.cos(O2) * np.cos(i2) * np.sin(w2))
    P22_2 = (- np.sin(O2) * np.sin(w2) + np.cos(O2) * np.cos(i2) * np.cos(w2))
    P31_2 = np.sin(i2) * np.sin(w2)
    P32_2 = np.sin(i2) * np.cos(w2)

    K0 = a1**2*(P11_1**2 + P21_1**2 + P31_1**2)
    K1 = a2**2*(P11_2**2 + P21_2**2 + P31_2**2)
    K2 = a1**2*(1 - e1**2)*(P12_1**2 + P22_1**2 + P32_1**2)
    K3 = a2**2*(1 - e2**2)*(P12_2**2 + P22_2**2 + P32_2**2)
    K4 = 2*a1**2*np.sqrt(1 - e1**2)*(P11_1*P12_1 + P21_1*P22_1 + P31_1*P32_1)
    K5 = 2*a2**2*np.sqrt(1 - e2**2)*(P11_2*P12_2 + P21_2*P22_2 + P31_2*P32_2)
    K6 = -2*a1*a2*(P11_1*P11_2 * P21_1*P21_2 + P31_1*P31_2)
    K7 = -2*a1*a2*np.sqrt(1 - e2**2)*(P11_1*P12_2 + P21_1*P22_2 + P31_1*P32_2)
    K8 = -2*a1*a2*np.sqrt(1 - e1**2)*(P12_1*P11_2 + P22_1*P21_2 + P32_1*P31_2)
    K9 = -2*a1*a2*np.sqrt(1 - e1**2)*np.sqrt(1 - e2**2)*(P12_1*P12_2 +
                                                         P22_1*P22_2 +
                                                         P32_1*P32_2)


    constants = [K0, K1, K2, K3, K4, K5, K6, K7, K8, K9]

    return constants


# Define the constants and symbols
a1 = 2000
a2 = 1000
e1 = 0.1
e2 = 0.2
i1 = 0.1
i2 = 0.8
w1 = 0.1
w2 = 0.1
O1 = 0.1
O2 = 0.1

const = calculate_constants(a1, e1, i1, w1, O1, a2, e2, i2, w2, O2)
print('Constants:')
for ii in range(len(const)):
    print(f'K{ii}: ', const[ii])
print('')

Ei, Ej, K0, K1, K2, K3, K4, K5, K6, K7, K8, K9, ei, ej = sp.symbols('Ei Ej K0 K1 K2 K3 K4 K5 K6 K7 K8 K9 ei ej')

# Define the expression
f = sp.sqrt(K0*(sp.cos(Ei) - ei)**2 + K1*(sp.cos(Ej) - ej)**2 + K2*(sp.sin(Ei))**2 + K3*(sp.sin(Ej))**2 + K4*(sp.cos(Ei) - ei)*sp.sin(Ei) + K5*(sp.cos(Ej) - ej)*sp.sin(Ej) + K6*(sp.cos(Ei) - ei)*(sp.cos(Ej) - ej) + K7*(sp.cos(Ei) - ei)*sp.sin(Ej) + K8*sp.sin(Ei)*(sp.cos(Ej) - ej) + K9*sp.sin(Ei)*sp.sin(Ej))

# Substitute constants and variables
f = f.subs({K0: const[0], K1: const[1], K2: const[2], K3: const[3],
            K4: const[4], K5: const[5], K6: const[6], K7: const[7],
            K8: const[8], K9: const[9], ei: e1, ej: e2})

# Define a NumPy-compatible function
f_np = lambdify((Ei, Ej), f, modules=['numpy'])

# Evaluate the function on a grid of values
E_i = np.linspace(0, 2*np.pi, 1000)
E_j = np.linspace(0, 2*np.pi, 1000)
Ei, Ej = np.meshgrid(E_i, E_j)

Z = f_np(Ei, Ej)

minDistance = round(np.min(Z), 2)
print('Minimum distance: ', minDistance)

# Plot the surface
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(Ei, Ej, Z, cmap='coolwarm')

# Add a legend
ax.text(0.8, 0.8, minDistance, "Minimum distance in m: {:.2f}".format(minDistance), transform=ax.transAxes)

path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(path, 'MinimumDistance.png')
plt.savefig(filepath, dpi=1200)
plt.show()
