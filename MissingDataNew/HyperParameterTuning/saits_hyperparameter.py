import pypots
from pypots.imputation import SAITS, BRITS, USGAN, GPVAE, MRNN
from pypots.optim import Adam
import numpy as np
import benchpots
from pypots.utils.random import set_random_seed
import optuna
from pypots.nn.functional import calc_mae


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

    n_layers = trial.suggest_int("n_layers", 1, 4)
    d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])

    # Calcula dinamicamente
    d_k = d_model // n_heads
    d_v = d_model // n_heads

    d_ffn = trial.suggest_categorical("d_ffn", [128, 256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    ORT_weight = trial.suggest_categorical("ORT_weight", [0.01, 0.1, 1.0])
    MIT_weight = trial.suggest_categorical("MIT_weight", [0.1, 0.5, 1.0])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])



    saits = SAITS(
        n_steps=physionet2012_dataset['n_steps'],
        n_features=physionet2012_dataset['n_features'],
        n_layers=n_layers,
        d_model=d_model,
        d_ffn=d_ffn,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        dropout=dropout,
        ORT_weight=ORT_weight,  # you can adjust the weight values of arguments ORT_weight
        # and MIT_weight to make the SAITS model focus more on one task. Usually you can just leave them to the default values, i.e. 1.
        MIT_weight=MIT_weight,
        batch_size=batch_size,
        # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
        epochs=10,
        # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
        # You can leave it to defualt as None to disable early stopping.
        patience=3,
        # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
        # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
        optimizer=Adam(lr=1e-3),
        # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
        # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
        # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
        num_workers=0,
        # just leave it to default as None, PyPOTS will automatically assign the best device for you.
        # Set it as 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices, even parallelly on ['cuda:0', 'cuda:1']
        device=None,
        # set the path for saving tensorboard and trained model files
        saving_path="tutorial_results/imputation/saits",
        # only save the best model after training finished.
        # You can also set it as "better" to save models performing better ever during training.
        model_saving_strategy="best",
    )

    saits.fit(
        train_set = dataset_for_IMPU_training,
        val_set=dataset_for_IMPU_validating
    )

    # the testing stage, impute the originally-missing values and artificially-missing values in the test set
    saits_results = saits.predict(dataset_for_IMPU_testing)
    saits_imputation = saits_results["imputation"]

    # calculate mean absolute error on the ground truth (artificially-missing values)
    testing_mae = calc_mae(
        saits_imputation,
        test_X_ori,
        test_X_indicating_mask,
    )

    return testing_mae


study = optuna.create_study(direction="minimize")  # minimizar MAE
study.optimize(objective, n_trials=100)

print("Melhores hiperparâmetros:", study.best_params)
print("Melhor score:", study.best_value)