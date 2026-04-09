import numpy as np
from MAEModify.error import calc_mae

class Results:

    def reshape_for_patients(model_aes):
        dataset_imputed_reshape = model_aes.reshape(len(model_aes), 48 * 37)
        return dataset_imputed_reshape
    
    def reshape_for_patients_subgroups(model_aes):
        model_aes_reshape = []
        for i in range(len(model_aes)):
            model_aes_reshape.append(model_aes[i].reshape(len(model_aes[i]), 48 * 37))
        
        return model_aes_reshape
            
    
    def sum_aes(model_aes):
        model_ae_sum  = []

        for model_ae in model_aes:
            model_ae_sum.append(np.sum(model_ae))
        
        return model_ae_sum

    def sum_aes_subgroup(model_aes):
        model_ae_sum  = []
        model_ae_sum_subgroup  = []
        for subgroup in model_aes:
            for patient in subgroup:
                model_ae_sum.append(np.sum(patient))
            model_ae_sum_subgroup.append(model_ae_sum)
            model_ae_sum = []
        
        return model_ae_sum_subgroup
    
    def ae_mask(model_aes, indicating_mask):
        model_ae_mask = []
        model_ae_mask_aux = []

        for i in range(len(indicating_mask)):
            for j in range(len(indicating_mask[i])):
                if indicating_mask[i][j] == True:
                    model_ae_mask_aux.append(model_aes[i][j])

            model_ae_mask.append(model_ae_mask_aux)
            model_ae_mask_aux = []


        return model_ae_mask
    
    
    def ae_mask_subgroup(model_aes, indicating_mask):
        model_ae_mask_patients = []
        model_ae_mask_aux = []
        model_ae_subgroup = [] 

        for i in range(len(indicating_mask)):

            for j in range(len(indicating_mask[i])):
                for k in range(len(indicating_mask[i][j])):
                    if indicating_mask[i][j][k] == True:
                        model_ae_mask_aux.append(model_aes[i][j][k])

                model_ae_mask_patients.append(model_ae_mask_aux)
                model_ae_mask_aux = []

            model_ae_subgroup.append(model_ae_mask_patients)
            model_ae_mask_patients = []

        return model_ae_subgroup
    
    

    def calc_mae_subgroup(model_imputation, test_x_ori_subgroup, indicating_mask_subgroup):

        model_mae = []
        model_ae = []

        for i in range(len(model_imputation)):
            aux_mae, aux_ae = calc_mae(model_imputation[i], test_x_ori_subgroup[i], indicating_mask_subgroup[i])
            model_mae.append(aux_mae)
            model_ae.append(aux_ae)
        

        return model_mae, model_ae

