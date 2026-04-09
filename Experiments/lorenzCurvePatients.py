import os
import sys
import numpy as np
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from Components.loadDataset import loadDataset as ld
from Components.Models import Models as md 
from Components.Results import Results as rs
from Components.Views import Views as vs
from MAEModify.error import calc_mae

#Carrega o dataset physionet
physionet2012_dataset = ld.load_dataset_pypots_modify("physionet_2012", "all", 0.1)

#Separa o dataset physionet em treino, validação e teste 
dataset_for_training, dataset_for_validating, dataset_for_testing = ld.separating_dataset(physionet2012_dataset)

#Cria o indicating mask para o test
test_X_indicating_mask = ld.create_indicating_mask(physionet2012_dataset["test_X_ori"], physionet2012_dataset["test_X"])

#Tranforma os nan do dataset em zero
test_X_ori = ld.transform_nan_to_zero(physionet2012_dataset["test_X_ori"])

#Cria a instância do modelo com seus parâmetros
model_saits = md.model("saits", physionet2012_dataset, False)
model_brits = md.model("brits", physionet2012_dataset, False)
model_usgan = md.model("usgan", physionet2012_dataset, False)
model_gpvae = md.model("gpvae", physionet2012_dataset, False)
model_mrnn = md.model("mrnn", physionet2012_dataset, False)

#Treina ou carrega modelo existente
md.train_load_model(model_saits, dataset_for_training, dataset_for_validating, False, "tutorial_results/imputation/saits/20251109_T170425/SAITS.pypots")
md.train_load_model(model_brits, dataset_for_training, dataset_for_validating, False, "tutorial_results/imputation/brits/20251109_T170432/BRITS.pypots")
md.train_load_model(model_usgan, dataset_for_training, dataset_for_validating, False, "tutorial_results/imputation/usgan/20251109_T174804/USGAN.pypots")
md.train_load_model(model_gpvae, dataset_for_training, dataset_for_validating, False, "tutorial_results/imputation/gpvae/20251109_T174824/GPVAE.pypots")
md.train_load_model(model_mrnn, dataset_for_training, dataset_for_validating, False, "tutorial_results/imputation/mrnn/20251109_T174856/MRNN.pypots")


#Realiza a imputação e salva o dataset imputado
imputation_saits = md.imputation(model_saits, dataset_for_testing)
imputation_brits = md.imputation(model_brits, dataset_for_testing)
imputation_gpvae = md.imputation(model_gpvae, dataset_for_testing)
imputation_usgan = md.imputation(model_usgan, dataset_for_testing)
imputation_mrnn = md.imputation(model_mrnn, dataset_for_testing)

#Calcula MAE e AE para imputação do modelo
saits_mae, saits_ae = calc_mae(imputation_saits, test_X_ori, test_X_indicating_mask)
brits_mae, brits_ae = calc_mae(imputation_brits, test_X_ori, test_X_indicating_mask)
usgan_mae, usgan_ae = calc_mae(imputation_usgan, test_X_ori, test_X_indicating_mask)
imputation_gpvae = np.squeeze(imputation_gpvae, axis=1)
gpvae_mae, gpvae_ae = calc_mae(imputation_gpvae, test_X_ori, test_X_indicating_mask)
mrnn_mae, mrnn_ae = calc_mae(imputation_mrnn, test_X_ori, test_X_indicating_mask)

#reshape para pacientes
test_X_indicating_mask = rs.reshape_for_patients(test_X_indicating_mask)
saits_ae_reshape = rs.reshape_for_patients(saits_ae)
brits_ae_reshape = rs.reshape_for_patients(brits_ae)
usgan_ae_reshape = rs.reshape_for_patients(usgan_ae)
gpvae_ae_reshape = rs.reshape_for_patients(gpvae_ae)
usgan_ae_reshape = rs.reshape_for_patients(usgan_ae)
mrnn_ae_reshape = rs.reshape_for_patients(mrnn_ae)

saits_ae_reshape = rs.ae_mask(saits_ae_reshape, test_X_indicating_mask)
brits_ae_reshape = rs.ae_mask(brits_ae_reshape, test_X_indicating_mask)
usgan_ae_reshape = rs.ae_mask(usgan_ae_reshape, test_X_indicating_mask)
gpvae_ae_reshape = rs.ae_mask(gpvae_ae_reshape, test_X_indicating_mask)
mrnn_ae_reshape = rs.ae_mask(mrnn_ae_reshape, test_X_indicating_mask)

#Soma os valores dos AEs
saits_ae_sum = rs.sum_aes(saits_ae_reshape)
brits_ae_sum = rs.sum_aes(brits_ae_reshape)
gpvae_ae_sum = rs.sum_aes(gpvae_ae_reshape)
usgan_ae_sum = rs.sum_aes(usgan_ae_reshape)
mrnn_ae_sum = rs.sum_aes(mrnn_ae_reshape)

vs.lorenz_curve_5(saits_ae_sum, brits_ae_sum, gpvae_ae_sum, usgan_ae_sum, mrnn_ae_sum, "Lorenz Curves by Model for All Models")


