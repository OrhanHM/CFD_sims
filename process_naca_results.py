import numpy as np
import pandas as pd

with open('./NACA_testing/stable_nacas.txt', 'r') as f:
    code_ls = f.readlines()

codes = [code[:-1] for code in code_ls]

drags = np.zeros(len(codes))
lifts = np.zeros_like(drags)
ldrs = np.zeros_like(drags)

for i, code in enumerate(codes):
    with open(f"./NACA_testing/Stats/Naca_{code}.txt", 'r') as f:
        info = f.readlines()
        lift, drag, ldr = info[0], info[1], info[2]
        
        for ind in range(len(lift)):
            try:
                float(lift[ind:-1])
            except ValueError:
                pass
            else:
                lifts[i] = float(lift[ind:-1])
                break

        for ind in range(len(drag)):
            try:
                float(drag[ind:-1])
            except ValueError:
                pass
            else:
                drags[i] = float(drag[ind:-1])
                break

        for ind in range(len(ldr)):
            try:
                float(ldr[ind:-1])
            except ValueError:
                pass
            else:
                ldrs[i] = float(ldr[ind:-1])
                break


df = pd.DataFrame()
df['codes'] = codes
df['drag'] = drags
df['lift'] = lifts
df['ldr'] = ldrs
df.set_index('codes', inplace=True)

df.to_csv('./NACA_testing/naca_results_df.csv', index=True)

