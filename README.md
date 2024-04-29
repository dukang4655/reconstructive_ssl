
Implementation of the experiments from the work 

"Low-Rank Approximation of Structural Redundancy for Self-Supervised Learning" by Kang Du and Yu Xiang.


> Synthetic Data Experiment

`data/syn_data.py`: data generation

`utils/syn_utils.py` utility functions defined for model training and testing 

Run the experiment with fixed parameter `s` varying labeled sample size `n` (e.g., 200 datasets): 

`python3 run_syn_exp_vary_n.py --num_runs 200`

Run the experiment with fixed labeled sample size `n` and varying `s`: 

`python3 run_syn_exp_vary_s.py --num_runs 200`


> Geometric Shapes Experiment

`data/geo_data.py`: data generation

`utils/geo_utils.py` utility functions defined for model training and testing 

Run experiment triangle v.s. circles with no background pattern:

`python3 run_geo_exp_no_noise.py --num_runs 200 --shape 0`

Run experiment triangle v.s. circles with background pattern (noise_type: dot (0) or dash (1)):

`python3 run_geo_exp_no_noise.py --num_runs 200 --shape 0 --noise_type 0 --max_noise_space 32`

Run experiment triangle v.s. pentagon with no background pattern:

`python3 run_geo_exp_no_noise.py --num_runs 200 --shape 1`




> Stylized MNIST Experiment

`data/minist_data.py`: data generation

`utils/mnist_utils.py` utility functions defined for model training and testing 

Run experiment mnist with background pattern (noise_type: dot (0) or dash (1)):

`python3 run_mnist_with_noise.py --num_runs 200 --noise_type 0 --max_noise_space 32`

>References:
