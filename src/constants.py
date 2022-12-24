from src.parser import *
from src.folderconstants import *

# Threshold parameters
lm_d = {
		'SMD': [(0.99995, 1.04), (0.99995, 1.06), (0.99999, 1.06)],
		'synthetic': [(0.999, 1), (0.999, 1), (0.999, 1)],
		'SWaT': [(0.993, 1), (0.993, 1), (0.993, 1)],
		'UCR': [(0.993, 1), (0.99935, 1), (0.99935, 1)],
		'NAB': [(0.991, 1), (0.99, 1), (0.992, 1.3)],
		'SMAP': [(0.98, 1), (0.98, 1), (0.99, 1.06)],
		'WADI': [(0.99, 1), (0.999, 1), (0.999, 1.1)],
		'MSDS': [(0.91, 1), (0.9, 1.04), (0.9, 1.04)],
		'MBA': [(0.87, 1), (0.93, 1.04), (0.93, 1.04)],
		'PSM': [(0.84, 1.04), (0.84, 1.04), (0.84, 1.04)],
	}
lm = lm_d[args.dataset][1 if args.model in ['TranAD'] else 2 if args.model in ['ATF_UAD'] else 0]

# Hyperparameters
lr_d = {
		'SMD': [0.0001, 0.0001, 0.05],
		'synthetic': [0.0001, 0.0001, 0.0001],
		'SWaT': [0.05, 0.05, 0.1],
		'SMAP': [0.001, 0.001, 0.05],
		'WADI': [0.0001, 0.0001, 0.05],
		'MSDS': [0.001, 0.001, 0.001],
		'UCR': [0.3, 0.3, 0.05],
		'NAB': [0.009, 0.009, 0.05],
		'MBA': [0.03, 0.03, 0.05],
		'PSM': [0.001, 0.001, 0.05],
	}
lr = lr_d[args.dataset][1 if args.model in ['TranAD'] else 2 if args.model in ['ATF_UAD'] else 0]

# Debugging
percentiles = {
		'SMD': [(98, 2000), (98, 2000), (98, 2000)],
		'synthetic': [(95, 10), (95, 10), (95, 10)],
		'SWaT': [(95, 10), (95, 10), (95, 10)],
		'SMAP': [(97, 5000), (97, 5000), (97, 5000)],
		'WADI': [(99, 1200), (99, 1200), (99, 1200)],
		'MSDS': [(96, 30), (96, 30), (96, 30)],
		'UCR': [(98, 2), (98, 2), (99, 2)],
		'NAB': [(98, 2), (98, 2), (99, 2)],
		'MBA': [(99, 2), (99, 2), (99, 2)],
		'PSM': [(98, 100), (98, 100), (98, 100)],
	}

beta_list = {
		'SMD': 0.5,
		'SWaT': 0.8,
		'SMAP': 0.5,
		'WADI': 0.2,
		'MSDS': 0.5,
		'UCR': 0.5,
		'NAB': 0.5,
		'MBA': 0.4,
		'PSM': 0.5,
}

percentile_merlin = percentiles[args.dataset][2 if args.model in ['ATF_UAD'] else 1 if args.model in ['TranAD'] else 0][0]
cvp = percentiles[args.dataset][1]
preds = []
debug = 9
beta = beta_list[args.dataset]