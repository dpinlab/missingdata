import os
import sys
import numpy as np
import benchpots
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pypots.optim import Adam
from pypots.imputation import SAITS, BRITS, USGAN, GPVAE, MRNN
from Components.Models import Models as md
from pypots.utils.random import set_random_seed

set_random_seed()
physionet2012_dataset = benchpots.datasets.preprocess_physionet2012(subset="all", rate=0.1)
print(physionet2012_dataset.keys())

dataset_for_IMPU_training = {
    "X": physionet2012_dataset['train_X'],
}

dataset_for_IMPU_validating = {
    "X": physionet2012_dataset['val_X'],
    "X_ori": physionet2012_dataset['val_X_ori'],
}

dataset_for_IMPU_testing = {
    "X": physionet2012_dataset['test_X'],
}

test_X_indicating_mask = np.isnan(physionet2012_dataset['test_X_ori']) ^ np.isnan(physionet2012_dataset['test_X'])
test_X_ori = np.nan_to_num(physionet2012_dataset['test_X_ori'])  


saits = md.model("saits", dataset=physionet2012_dataset, train=False)

brits = md.model("brits", dataset=physionet2012_dataset, train=False)

us_gan = md.model("usgan", dataset=physionet2012_dataset, train=False)

gp_vae = md.model("gpvae", dataset=physionet2012_dataset, train=False)

mrnn = md.model("mrnn", dataset=physionet2012_dataset, train=False)

def get_total_params (model_x):
    total_params = sum(p.numel() for p in model_x.model.parameters() if p.requires_grad)
    return total_params


total_params_saits = get_total_params(saits)
print(f"Total de parâmetros treináveis SAITS: {total_params_saits}")

total_params_brits = get_total_params(brits)
print(f"Total de parâmetros treináveis BRITS: {total_params_brits}")

total_params_us_gan = get_total_params(us_gan)
print(f"Total de parâmetros treináveis USGAN: {total_params_us_gan}")

total_params_gp_vae = get_total_params(gp_vae)
print(f"Total de parâmetros treináveis GPVAE: {total_params_gp_vae}")

total_params_mrnn = get_total_params(mrnn)
print(f"Total de parâmetros treináveis MRNN: {total_params_mrnn}")