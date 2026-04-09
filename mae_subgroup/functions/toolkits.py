import numpy as np 

class toolkits:
    def separating_dataset(dataset):

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
    
    def dict_to_list(dataset):
        dataset_list = []
        for i in dataset.values():
            dataset_list.append(i) 
        
        return dataset_list
    
    def components_mae(dataset_for_testing_ori_standard, dataset_for_testing_standard):
        test_X_indicating_mask = []
        test_X_ori = []
        for i in range(len(dataset_for_testing_standard)):
            test_X_indicating_mask.append(np.isnan(dataset_for_testing_ori_standard[i]) ^ np.isnan(dataset_for_testing_standard[i]))
            test_X_ori.append(np.nan_to_num(dataset_for_testing_ori_standard[i]))

        return test_X_indicating_mask, test_X_ori
    
    def bootstrap_v2(ae, mask,subgrupo, n_resamples):
        distribution_bootstrap = []

        for i in range(n_resamples):
            indices = np.random.randint(0, len(ae[subgrupo]) - 1, size = len(ae[subgrupo]))
            resampling_ae = ae[subgrupo][indices]
            resampling_mask = mask[subgrupo][indices]
            mae = sum(resampling_ae*resampling_mask)/(sum(resampling_mask)+1e-12)
            distribution_bootstrap.append(mae)
        
        return distribution_bootstrap