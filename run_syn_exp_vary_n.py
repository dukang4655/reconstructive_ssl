import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from syn_utils import run_exp

import argparse


# Create the parser
parser = argparse.ArgumentParser(description="parameter 'num_runs'.")


# Add the 'num_runs' argument as an integer without explicit positivity check
parser.add_argument('--num_runs', type=int, required=True,
                    help="Number of experiments to run (must be a positive integer).")

# Parse the arguments
args = parser.parse_args()




s_values = [100, 200, 400, 800, 1600]

num_runs = args.num_runs

accuracy_results = {s: [] for s in s_values}

accuracy_results_sl= {s: [] for s in s_values}

accuracy_results_sl2= {s: [] for s in s_values}

for s in s_values:
    print(f"Running experiments for s={s}")
    for _ in range(num_runs):
        accuracy, accuracy_direct, accuracy_direct2 = run_exp(n_samples_downstream=s)
        accuracy_results[s].append(accuracy)
        accuracy_results_sl[s].append(accuracy_direct)
        accuracy_results_sl2[s].append(accuracy_direct2)


accuracy_means = [np.mean(accuracy_results[s]) for s in s_values]
accuracy_stds = [np.std(accuracy_results[s]) for s in s_values]


accuracy_means_sl = [np.mean(accuracy_results_sl[s]) for s in s_values]
accuracy_stds_sl = [np.std(accuracy_results_sl[s]) for s in s_values]

accuracy_means_sl2 = [np.mean(accuracy_results_sl2[s]) for s in s_values]
accuracy_stds_sl2 = [np.std(accuracy_results_sl2[s]) for s in s_values]


results = pd.DataFrame({
    's_values': s_values,
    'accuracy_means': accuracy_means,
    'accuracy_stds': accuracy_stds,
    'accuracy_means_sl': accuracy_means_sl,
    'accuracy_stds_sl': accuracy_stds_sl,
    'accuracy_means_sl2': accuracy_means_sl2,
    'accuracy_stds_sl2': accuracy_stds_sl2
})

# Save the DataFrame as a CSV file
results.to_csv('syn_results/exp_syn_vary_n.csv', index=False)


# Create plots
plt.figure(figsize=(6, 5))



fontsize1=18

# Accuracy plot
plt.xticks(s_values,s_values,fontsize=fontsize1,rotation = 45)
plt.yticks(fontsize=fontsize1)
#plt.yticks(yticks2,yticks2,fontsize=fontsize1)
plt.fill_between(s_values, np.subtract(accuracy_means, accuracy_stds), np.add(accuracy_means, accuracy_stds), color='mistyrose')

plt.fill_between(s_values, np.subtract(accuracy_means_sl, accuracy_stds_sl), np.add(accuracy_means_sl, accuracy_stds_sl), color='lightblue')
plt.plot(s_values, accuracy_means, 'o-', color='red')
plt.plot(s_values, accuracy_means_sl, 's-', color='blue')
plt.plot(s_values, accuracy_means_sl2, 'd-', color='green')
plt.fill_between(s_values, np.subtract(accuracy_means_sl2, accuracy_stds_sl2), np.add(accuracy_means_sl2, accuracy_stds_sl2), color='honeydew')


plt.title('Accuracy vs n')
plt.ylabel('Accuracy',fontsize=fontsize1)
plt.xlabel('n',fontsize=fontsize1)
plt.grid(False)

plt.tight_layout()

plt.savefig('syn_results/final_SSL_vary_n.png', dpi=300, bbox_inches='tight')

plt.show()
