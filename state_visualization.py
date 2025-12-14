import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd

from helper_functions import load_state


# u, v, p, X, Y, wing_points = load_state('./Paper_resources/designed.npz')

df = pd.read_csv('./NACA_testing/naca_results_df.csv', index_col='codes')
code = 5315
u, v, p, X, Y, wing_points = load_state(f'./NACA_testing/States/Naca_{code}.npz')


fig, ax = plt.subplots(figsize=(8, 8))

sx, sy = 16, 16
Xp, Yp = X[200:-200][::sx, ::sy], Y[200:-200][::sx, ::sy]
up, vp = u[200:-200][::sx, ::sy], v[200:-200][::sx, ::sy]
p -= np.mean(p[200:-200])

cmap = mpl.colormaps['bwr']
range = np.max(np.abs(p[200:-200]))
low = -range
high=range

cf = ax.contourf(X[200:-200], Y[200:-200], p[200:-200], levels=80, alpha=0.8, cmap=cmap, vmin=low, vmax=high)

path = mpl.path.Path(np.array(wing_points))
patch = mpl.patches.PathPatch(path, facecolor='grey', lw=1, alpha=0.7)
ax.add_patch(patch)

cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
cbar.set_label('Relative pressure (Pa)', fontsize=14, labelpad=10)

ax.set_aspect('equal')
ax.quiver(Xp, Yp, up, vp)

ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)

# ax.set_title('Custom airfoil: L/D=1.225', fontsize=15)
ax.set_title(f'NACA airfoil {code}: L/D={df.loc[code, 'ldr']:.3f}', fontsize=15)

plt.show()
# plt.savefig(f'./Paper_resources/designed.png', dpi=300, bbox_inches='tight')
# plt.savefig(f'./Paper_resources/Naca_{code}.png', dpi=300, bbox_inches='tight')

