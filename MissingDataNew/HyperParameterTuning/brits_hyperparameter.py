import pypots
from pypots.imputation import SAITS, BRITS, USGAN, GPVAE, MRNN
from pypots.optim import Adam
import numpy as np
import benchpots
from pypots.utils.random import set_random_seed
import optuna
from pypots.nn.functional import calc_mae
import torch


import numpy as np
import benchpots
from pypots.utils.random import set_random_seed

set_random_seed()

# Load the PhysioNet-2012 dataset
physionet2012_dataset = benchpots.datasets.preprocess_physionet2012(
    subset="set-a",
    rate=0.1,  # the rate of missing values artificially created to evaluate algorithms
)

# Take a look at the generated PhysioNet-2012 dataset, you'll find that everything has been prepared for you,
# data splitting, normalization, additional artificially-missing values for evaluation, etc.
print(physionet2012_dataset.keys())


# assemble the datasets for training
dataset_for_IMPU_training = {
    "X": physionet2012_dataset['train_X'],
}
# assemble the datasets for validation
dataset_for_IMPU_validating = {
    "X": physionet2012_dataset['val_X'],
    "X_ori": physionet2012_dataset['val_X_ori'],
}
# assemble the datasets for test
dataset_for_IMPU_testing = {
    "X": physionet2012_dataset['test_X'],
}
## calculate the mask to indicate the ground truth positions in test_X_ori, will be used by metric funcs to evaluate models
test_X_indicating_mask = np.isnan(physionet2012_dataset['test_X_ori']) ^ np.isnan(physionet2012_dataset['test_X'])
test_X_ori = np.nan_to_num(physionet2012_dataset['test_X_ori'])  # metric functions do not accpet input with NaNs, hence fill NaNs with 0

def objective(trial):

    rnn_hidden_size = trial.suggest_categorical("rnn_hidden_size", [64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    
        
    brits = BRITS(
        n_steps=physionet2012_dataset['n_steps'],
        n_features=physionet2012_dataset['n_features'],
        rnn_hidden_size=rnn_hidden_size,
        batch_size=batch_size,
        epochs=10,
        patience=3,
        optimizer=Adam(lr=1e-3),
        num_workers=0,
        device=None,
        model_saving_strategy="best",
    )

    brits.fit(train_set=dataset_for_IMPU_training, val_set=dataset_for_IMPU_validating)

    brits_results = brits.predict(dataset_for_IMPU_testing)
    brits_imputation = brits_results["imputation"]

    testing_mae = calc_mae(
        brits_imputation,
        test_X_ori,
        test_X_indicating_mask,
    )

    return testing_mae



study = optuna.create_study(direction="minimize")  # minimizar MAE
study.optimize(objective, n_trials=100)

print("Melhores hiperparâmetros:", study.best_params)
print("Melhor score:", study.best_value)