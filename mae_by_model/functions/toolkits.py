import numpy as np

class toolkits:

    def bootstrap_v3(ae, mask, n_resamples):
        distribution_bootstrap = []

        for i in range(n_resamples):
            indices = np.random.randint(0, len(ae) - 1, size = len(ae))
            resampling_ae = ae[indices]
            resampling_mask = mask[indices]
            mae = sum(resampling_ae * resampling_mask)/(sum(resampling_mask)+1e-12)
            distribution_bootstrap.append(mae)
        
        return distribution_bootstrap  
    

    def calc_lower_and_upper_bound_percentile(bootstrap_results_for_the_model):

        lower_bounds = np.percentile(bootstrap_results_for_the_model, 2.5)
        upper_bounds = np.percentile(bootstrap_results_for_the_model, 97.5)

        return lower_bounds, upper_bounds

    
    def calc_mean_values_ci(lower_bounds, upper_bounds):

        mean_values_ci = (lower_bounds + upper_bounds) / 2

        return mean_values_ci  