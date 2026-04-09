import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from Components.loadDataset import loadDataset as ld
from Components.Models import Models as md 

#Carrega o dataset physionet
physionet2012_dataset = ld.load_dataset_pypots("physionet_2012", "all", 0.1)

#Separa o dataset physionet em treino, validação e teste 
dataset_for_training, dataset_for_validating, dataset_for_testing = ld.separating_dataset(physionet2012_dataset)

#Cria o indicating mask para o test
test_X_indicating_mask = ld.create_indicating_mask(physionet2012_dataset["test_X_ori"], physionet2012_dataset["test_X"])

#Tranforma os nan do dataset em zero
test_X_ori = ld.transform_nan_to_zero(physionet2012_dataset["test_X_ori"])

#Cria a instância do modelo com seus parâmetros
model = md.model("saits", physionet2012_dataset, False)

#Carrega treinamento do modelo existente
path = "tutorial_results/imputation/saits/20250819_T224532/SAITS.pypots"
md.train_load_model(model, dataset_for_training, dataset_for_validating, False, path)

#Realiza a imputação e salva o dataset imputado
path_save_imputation = "SaitsImputedDataset"
model_imputation = md.imputation(model, dataset_for_testing, path_save_imputation)



#calcula MAE/AE -> Divide

