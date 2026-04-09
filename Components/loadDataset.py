import os
import sys
import benchpots
import numpy as np
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pypots.utils.random import set_random_seed
from pypotsModify.benchpotsMAE.datasets import preprocess_physionet2012 as preprocess_physionet2012

class loadDataset:

    #Carrega dataset escolhido
    def load_dataset_pypots(dataset, subset, rate):
        set_random_seed()

        if dataset == "physionet_2012":
            dataset_load = benchpots.datasets.preprocess_physionet2012(subset, rate)
        
        print(dataset_load.keys())

        return dataset_load
    
    #Carrega dataset escolhido
    def load_dataset_pypots_modify(dataset, subset, rate, normalization = 1):
        set_random_seed()

        if dataset == "physionet_2012":
            dataset_load = preprocess_physionet2012(subset, rate, normalization)
        
        print(dataset_load.keys())

        return dataset_load
    
    #Separa o dataset em treino, validação e test
    def separating_dataset(dataset):
        dataset_for_train = {
            "X": dataset["train_X"]
        }

        dataset_for_validation = {
            "X": dataset["val_X"],
            "X_ori": dataset["val_X_ori"]
        }

        dataset_for_testing = {
            "X": dataset["test_X"]
        }

        return dataset_for_train, dataset_for_validation, dataset_for_testing
    
    def separating_dataset_by_subgroup(dataset):

        dataset_for_training = {
            "X": dataset['train_X'],
        }

        dataset_for_validating = {
            "X": dataset['val_X'],
            "X_ori": dataset['val_X_ori']
        }

        dataset_for_testing_ori = {
            "X_ori": dataset['test_X_ori'],
            "female_gender_test_X_ori": dataset['female_gender_test_X_ori'],
            "male_gender_test_X_ori": dataset['male_gender_test_X_ori'],
            "undefined_gender_test_X_ori": dataset['undefined_gender_test_X_ori'],
            "more_than_or_equal_to_65_test_X_ori":  dataset['more_than_or_equal_to_65_test_X_ori'],
            "less_than_65_test_X_ori": dataset['less_than_65_test_X_ori'],
            "classificacao_undefined_test_X_ori": dataset['classificacao_undefined_test_X_ori'],
            "classificacao_baixo_peso_test_X_ori": dataset['classificacao_baixo_peso_test_X_ori'],
            "classificacao_normal_peso_test_X_ori": dataset['classificacao_normal_peso_test_X_ori'],
            "classificacao_sobrepeso_test_X_ori": dataset['classificacao_sobrepeso_test_X_ori'],
            "classificacao_obesidade_test_X_ori": dataset['classificacao_obesidade_test_X_ori'],
        }

        dataset_for_testing = {
            "X": dataset['test_X'],
            "female_gender_test_X": dataset['female_gender_test_X'],
            "male_gender_test_X": dataset['male_gender_test_X'],
            "undefined_gender_test_X": dataset['undefined_gender_test_X'],
            "more_than_or_equal_to_65_test_X":  dataset['more_than_or_equal_to_65_test_X'],
            "less_than_65_test_X": dataset['less_than_65_test_X'],
            "classificacao_undefined_test_X": dataset['classificacao_undefined_test_X'],
            "classificacao_baixo_peso_test_X": dataset['classificacao_baixo_peso_test_X'],
            "classificacao_normal_peso_test_X": dataset['classificacao_normal_peso_test_X'],
            "classificacao_sobrepeso_test_X": dataset['classificacao_sobrepeso_test_X'],
            "classificacao_obesidade_test_X": dataset['classificacao_obesidade_test_X'],
        }

        return dataset_for_training, dataset_for_validating, dataset_for_testing_ori, dataset_for_testing    

    #Cria a indicating mask para o test
    def create_indicating_mask(dataset_testing_ori, dataset_testing):
        indicating_mask = np.isnan(dataset_testing_ori) ^ np.isnan(dataset_testing)

        return indicating_mask
    
    #Tranforma nan em zero no dataset
    def transform_nan_to_zero(dataset):
        
        nan_to_zero = np.nan_to_num(dataset)

        return nan_to_zero
    

    def dict_to_list(dataset):
        dataset_list = []
        for i in dataset.values():
            dataset_list.append(i) 


        return dataset_list
    


    def components_mae(dataset_for_testing_ori, dataset_for_testing):
        test_X_indicating_mask = []
        test_X_ori = []
        for i in range(len(dataset_for_testing)):
            test_X_indicating_mask.append(np.isnan(dataset_for_testing_ori[i]) ^ np.isnan(dataset_for_testing[i]))
            test_X_ori.append(np.nan_to_num(dataset_for_testing_ori[i]))

        return test_X_indicating_mask, test_X_ori
    