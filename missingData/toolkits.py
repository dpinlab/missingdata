import numpy as np
import pypots
import random
import pandas as pd
import math
from MAEModify.error import calc_mae
import scipy.stats as st

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
    
    

    def components_mae(dataset_for_testing_ori_standard, dataset_for_testing_standard):
        test_X_indicating_mask = []
        test_X_ori = []
        for i in range(len(dataset_for_testing_standard)):
            test_X_indicating_mask.append(np.isnan(dataset_for_testing_ori_standard[i]) ^ np.isnan(dataset_for_testing_standard[i]))
            test_X_ori.append(np.nan_to_num(dataset_for_testing_ori_standard[i]))# metric functions do not accpet input with NaNs, hence fill NaNs with 0
            
        return test_X_indicating_mask, test_X_ori
    
    def pre_reshape(dataset):
        for i in range(len(dataset)):
            dataset[i] = dataset[i].reshape(len(dataset[i])*48, 37)
        return dataset
    
    def reshape_variable(dataset):
        listaMed = []
        listaAux = []
        dataset_variable = []

        for i in range(len(dataset)):
            for j in range(37):
                for k in range(len(dataset[i])):
                    listaAux.append(dataset[i][k][j])
                listaMed.append(listaAux) 
                listaAux = []
            listaMed = np.array(listaMed)
            dataset_variable.append(listaMed)
            listaMed = []

        return dataset_variable
    
    def model_imputation(dataset_for_testing, model):
        model_imputation = []
        for value in  dataset_for_testing.values():
            _dict = {'X':value}
            model_results = model.predict(_dict)
            model_imputation.append(model_results["imputation"])
        return model_imputation
    
    def calculate_mae(model_imputation, test_X_ori, indicating_mask):
        testing_mae_model_append_subgroups = []
        testing_mae_model_append_variables = []
        testing_ae_model_append_subgroups = []
        testing_ae_model_append_variables = []
        for i in range(len(model_imputation)):
            for j in range(len(model_imputation[i])):
                aux_mae, aux_ae = calc_mae(model_imputation[i][j], test_X_ori[i][j], indicating_mask[i][j])
                testing_mae_model_append_variables.append(aux_mae)
                testing_ae_model_append_variables.append(aux_ae)
            testing_mae_model_append_subgroups.append(testing_mae_model_append_variables)
            testing_ae_model_append_subgroups.append(testing_ae_model_append_variables)
            testing_mae_model_append_variables = []
            testing_ae_model_append_variables = []
        
        return testing_mae_model_append_subgroups, testing_ae_model_append_subgroups
    
    #Mae per model
    def show_mae(testing_mae_model, subgroups, variables):

        for i in range(len(subgroups)): 
                print(subgroups[i]) 
                print("-------------")
                for j in range(len(variables)):
                    print(variables[j], ":" ,testing_mae_model[i][j])


    #Create table per model
    def create_table(testing_mae_model, subgroups, variables):

        df_model_mae = pd.DataFrame(variables)

        for i in range(len(subgroups)):
                df_model_mae[subgroups[i]] = testing_mae_model[i]


        return df_model_mae
    
    def min_value_in_subgroup(model, subgroups, variables):
        for i in range(len(subgroups)):
            value = model[subgroups[i]].min()
            print(subgroups[i])
            for j in range(len(variables)):
                if(model[subgroups[i]][j] == value):
                    var = variables[j]
            print(var)
            print(value)        
            print("--------------------")

    def max_value_in_subgroup(model, subgroups, variables):
        for i in range(len(subgroups)):
            value = model[subgroups[i]].max()
            print(subgroups[i])
            for j in range(len(variables)):
                if(model[subgroups[i]][j] == value):
                    var = variables[j]
            print(var)
            print(value)        
            print("--------------------")    

    def desnormalization(dataset, scaler):
        dataset_desnormalized = []
        for i in range(len(dataset)):
            dataset_desnormalized.append(scaler.inverse_transform(dataset[i]))

        return dataset_desnormalized

    def dict_to_list(dataset):
        dataset_list = []
        for i in dataset.values():
            dataset_list.append(i) 
        
        return dataset_list
    

    def table_latex(dataframe):

        latex_code = dataframe.to_latex(index=False)

        with open("tabela.tex", "w") as f:
            f.write(latex_code)

        print(latex_code)

    def split_subgroup(dataframe, subgroup):
        
        if subgroup == "male" or subgroup == "female" or subgroup == "undefined gender":

            if subgroup == "female":
                param = 0
            elif subgroup == "male": 
                param = 1
            else:
                param = -1
            
            dataframe_ids_test = dataframe[dataframe["Gender"] == param]
            dataframe_ids_test  = list(set(dataframe_ids_test["RecordID"].to_list()))
            dataframe_ids_train = dataframe[~dataframe["RecordID"].isin(dataframe_ids_test)]
            dataframe_ids_train = list(set(dataframe_ids_train["RecordID"].to_list()))
        
        return dataframe_ids_train, dataframe_ids_test
    
    def bootstrap(ae, mask, subgrupo, variavel, n_resamples):
        resampling_ae = np.zeros(len(ae[subgrupo][variavel]))
        resampling_mask = np.zeros(len(ae[subgrupo][variavel]))
        distribution_bootstrap = []

        for i in range(n_resamples):
            for j in range(len(resampling_ae)):
                index = random.randint(0, len(resampling_ae) - 1)
                resampling_ae[j] = ae[subgrupo][variavel][index]
                resampling_mask[j] = mask[subgrupo][variavel][index]

            mae = sum(resampling_ae * resampling_mask)/(sum(resampling_mask)+1e-12)
            distribution_bootstrap.append(mae)
            resampling_ae = np.zeros(len(ae[subgrupo][variavel]))
            resampling_mask = np.zeros(len(ae[subgrupo][variavel]))
        
        return distribution_bootstrap
    
    def bootstrap_v2(ae, mask, subgrupo, variavel, n_resamples):
        distribution_bootstrap = []

        for i in range(n_resamples):
            indices = np.random.randint(0, len(ae[subgrupo][variavel]) - 1, size = len(ae[subgrupo][variavel]))
            resampling_ae = ae[subgrupo][variavel][indices]
            resampling_mask = mask[subgrupo][variavel][indices]
            mae = sum(resampling_ae * resampling_mask)/(sum(resampling_mask)+1e-12)
            distribution_bootstrap.append(mae)
        
        return distribution_bootstrap
    
    def bootstrap_v3(ae, mask, subgrupo, variavel, n_resamples):
        distribution_bootstrap = []

        for i in range(n_resamples):
            indices = np.random.randint(0, len(ae[subgrupo][variavel]) - 1, size = len(ae[subgrupo][variavel]))
            resampling_ae = ae[subgrupo][variavel][indices]
            resampling_mask = mask[subgrupo][variavel][indices]
            resampling_std = resampling_ae[resampling_mask == True]
            std = np.std(resampling_std)
            distribution_bootstrap.append(std)
        
        return distribution_bootstrap
    
        

    def diff_mae_top_5(mae_model, subgrupo1, subgrupo2, variables):
        
        diff = []

        for i in range(len(mae_model[0])):
            diff.append((variables[i], math.fabs(mae_model[subgrupo1][i] - mae_model[subgrupo2][i])))
            
        
        diff = sorted(diff, key=lambda x: x[1], reverse=True)
        diff = diff[0:5]
        

        return diff
    
    def calc_mean_and_standard_deviation(bootstrap_results_for_the_model):
        means_bootstraps = [] 
        standards_deviations = []
        for i in range(len(bootstrap_results_for_the_model)):
            means_bootstraps.append(np.mean(bootstrap_results_for_the_model[i]))
            standards_deviations.append(np.std(bootstrap_results_for_the_model[i]))
        return means_bootstraps, standards_deviations
    
    def calc_lower_and_upper_bound(bootstrap_results_for_the_model,means_bootstraps, standards_deviations):
        lower_bounds = []
        upper_bounds = []
        for i in range(len(means_bootstraps)):
            lower_bounds.append(means_bootstraps[i] - st.norm.ppf(1-0.05/2) * standards_deviations[i])
            upper_bounds.append(means_bootstraps[i] + st.norm.ppf(1-0.05/2) * standards_deviations[i])
        return lower_bounds, upper_bounds
    
    def calc_lower_and_upper_bound_percentile(bootstrap_results_for_the_model):

        lower_bounds = []
        upper_bounds = []

        for i in range(len(bootstrap_results_for_the_model)):
            lower_bounds.append(np.percentile(bootstrap_results_for_the_model[i], 2.5))
            upper_bounds.append(np.percentile(bootstrap_results_for_the_model[i], 97.5))

        return lower_bounds, upper_bounds

    
    def calc_mean_values_ci(lower_bounds, upper_bounds):
        mean_values_ci = []
        for i in range(len(lower_bounds)):
            mean_values_ci.append((lower_bounds[i] + upper_bounds[i]) / 2)
        return mean_values_ci  


        

            
        