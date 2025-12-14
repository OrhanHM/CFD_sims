import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from helper_functions import load_state


x = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

thick10 = [np.float64(4.3364095115252914e-05), np.float64(6.0037982755383245e-05), np.float64(7.769237987827833e-05), np.float64(7.686554681196469e-05), np.float64(9.557702215641677e-05), np.float64(9.334719546675496e-05)]
thick15 = [np.float64(0.00010753618070559248), np.float64(0.00012169882813711644), np.float64(0.00017855998087879446), np.float64(0.00020167439709608862), np.float64(0.000239976987760222), np.float64(0.00025267385856484007)]
thick125 = [np.float64(6.978113882642753e-05), np.float64(8.825657556808846e-05), np.float64(0.00012442877461853581), np.float64(0.00012992525971169372), np.float64(0.00015918402660251897), np.float64(0.00015610898261011834)]
thick175 = [np.float64(0.0001645522992871047), np.float64(0.00016550587142053145), np.float64(0.00023088001541189362), np.float64(0.00029251259676408024), np.float64(0.0003472004740941227), np.float64(0.0003904374871133594)]


fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, thick175, marker='o', linewidth=2, label='0.175 thickness')
ax.plot(x, thick15, marker='o', linewidth=2, label='0.15 thickness')
ax.plot(x, thick125, marker='o', linewidth=2, label='0.125 thickness')
ax.plot(x, thick10, marker='o', linewidth=2, label='0.1 thickness')


ax.set_xlabel('Airspeed (m/s)', fontsize=12)
ax.set_ylabel('Drag (N)', fontsize=12)
ax.set_title('Increasing drag with airspeed and thickness', fontsize=16)

plt.legend(loc='upper left')
plt.show()
# plt.savefig('variable_drag.png', dpi=300)
