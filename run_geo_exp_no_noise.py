import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from geo_utils import *

import argparse

import warnings

warnings.filterwarnings(action='once')


# Create the parser
parser = argparse.ArgumentParser(description="parameters 'num_runs'.")

# Add the 'num_runs' argument as an integer without explicit positivity check
parser.add_argument('--num_runs', type=int, required=True,
                    help="Number of experiments to run (must be a positive integer).")

parser.add_argument('--shape', type=int, required=True,
                    help="Choose shape circle (0) vs triangle or pentagon (1) vs triangle")



# Parse the arguments
args = parser.parse_args()


s_values = [10,20,40,60,80,100,150,200]
num_runs = args.num_runs

accuracy_results = {s: [] for s in s_values}

accuracy_results_sl= {s: [] for s in s_values}



for s in s_values:
    print(f"Running experiments for s={s}")
    for _ in range(num_runs):
        accuracy, accuracy_direct = run_exp(num_training_samples = 20000, num_labled_samples=s, num_test_samples =1000,noise_level=0, pair = args.shape)
        accuracy_results[s].append(accuracy)
        accuracy_results_sl[s].append(accuracy_direct)


# Calculate mean and standard deviation for accuracy

accuracy_means = [np.mean(accuracy_results[s]) for s in s_values]
accuracy_stds = [np.std(accuracy_results[s]) for s in s_values]


accuracy_means_sl = [np.mean(accuracy_results_sl[s]) for s in s_values]
accuracy_stds_sl = [np.std(accuracy_results_sl[s]) for s in s_values]


results = pd.DataFrame({
    's_values': s_values,
    'accuracy_means': accuracy_means,
    'accuracy_stds': accuracy_stds,
    'accuracy_means_sl': accuracy_means_sl,
    'accuracy_stds_sl': accuracy_stds_sl
})

shape_str = "circle" if args.shape==0 else "pentagon"


results.to_csv("geo_results/" +shape_str +"_no_noise.csv", index=False)


fontsize1= 18

# Create plots
plt.figure(figsize=(6, 5))

plt.xticks(s_values,s_values,fontsize= fontsize1)
plt.yticks(fontsize = fontsize1)
plt.fill_between(s_values, np.subtract(accuracy_means, accuracy_stds), np.add(accuracy_means, accuracy_stds), color='mistyrose')

plt.fill_between(s_values, np.subtract(accuracy_means_sl, accuracy_stds_sl), np.add(accuracy_means_sl, accuracy_stds_sl), color='lightblue')
plt.plot(s_values, accuracy_means, 'o-', color='red')
plt.plot(s_values, accuracy_means_sl, 's-', color='blue')


plt.title('Accuracy vs s')
plt.ylabel('Accuracy',fontsize = fontsize1)
plt.xlabel('n',fontsize = fontsize1)
plt.grid(False)

plt.tight_layout()

plt.savefig("geo_results/" +shape_str +"_no_noise.png", dpi=300, bbox_inches='tight')

plt.show()

