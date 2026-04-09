import numpy as np

class toolkits:
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
   
   def reshape_variables_v2(dataset):
       
        listaAux = []
        dataset_variable = []

        for i in range(37):

            for j in range(len(dataset)):
               
               listaAux.append(dataset[j][i])
            
            dataset_variable.append(listaAux)
            listaAux = []
        
        return dataset_variable
   

   def gini(model_ae):
        sorted_ae = model_ae.copy()
        sorted_ae.sort()
        n = model_ae.size
        coef_ = 2./n
        const_ = (n+1.)/n
        weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_ae)])
        return coef_*weighted_sum/(sorted_ae.sum()) - const_
   
   def bootstrap_v2(ae, subgrupo, n_resamples):
        distribution_bootstrap = []

        for i in range(n_resamples):
            indices = np.random.randint(0, len(ae[subgrupo]) - 1, size = len(ae[subgrupo]))
            resampling_ae = ae[subgrupo][indices]
            gini = toolkits.gini(resampling_ae)
            distribution_bootstrap.append(gini)
        
        return distribution_bootstrap
               
