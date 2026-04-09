import numpy as np
import matplotlib.pyplot as plt

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
    
    def gini(model_ae):
        sorted_ae = model_ae.copy()
        sorted_ae.sort()
        n = model_ae.size
        coef_ = 2./n
        const_ = (n+1.)/n
        weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_ae)])
        return coef_*weighted_sum/(sorted_ae.sum()) - const_
    
    def lorenz_curve(X):
        X = np.sort(X)  # ordenar os dados
        X_lorenz = X.cumsum() / X.sum()
        X_lorenz = np.insert(X_lorenz, 0, 0)  # inserir 0 no início
        x_vals = np.linspace(0, 1, len(X_lorenz))  # eixo x normalizado

        fig, ax = plt.subplots(figsize=[6,6])
        
        # Curva de Lorenz com linha contínua
        ax.plot(x_vals, X_lorenz, color='darkgreen', linewidth=2)
        
        # Linha de igualdade
        ax.plot([0, 1], [0, 1], color='black', linestyle='--')
        
        ax.set_title("Curva de Lorenz")
        ax.set_xlabel("População acumulada (fração)")
        ax.set_ylabel("Renda acumulada (fração)")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def lorenz_curve_5(X1, X2, X3, X4, X5, labels=None, colors=None):
        arrays = [X1, X2, X3, X4, X5]
        
        if labels is None:
            labels = ["SAITS", "BRITS", "US-GAN", "GP-VAE", "MRNN"]
        if colors is None:
            colors = ['blue', 'green', 'orange', 'purple', 'red']
        
        fig, ax = plt.subplots(figsize=[6,6])

        for i, X in enumerate(arrays):
            X = np.sort(X)
            X_lorenz = X.cumsum() / X.sum()
            X_lorenz = np.insert(X_lorenz, 0, 0)
            x_vals = np.linspace(0, 1, len(X_lorenz))

            ax.plot(x_vals, X_lorenz, linewidth=2, label=labels[i], color=colors[i])

        # Linha de igualdade
        ax.plot([0, 1], [0, 1], color='black', linestyle='--', label='perfect equality')

        ax.set_title("Lorenz Curve - 5 Models")
        ax.set_xlabel("Patients")
        ax.set_ylabel("Absolute error (AE)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


    def bootstrap_v2(ae, subgrupo, n_resamples):
        distribution_bootstrap = []

        for i in range(n_resamples):
            indices = np.random.randint(0, len(ae[subgrupo]) - 1, size = len(ae[subgrupo]))
            resampling_ae = ae[subgrupo][indices]
            gini = toolkits.gini(resampling_ae)
            distribution_bootstrap.append(gini)
        
        return distribution_bootstrap