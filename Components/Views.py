import matplotlib.pyplot as plt
import numpy as np

class Views:

     def lorenz_curve_5(X1, X2, X3, X4, X5, title, labels=None, colors=None):
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

        ax.set_title(title)
        ax.set_xlabel("Cummulative Share of Imputation per Patients")
        ax.set_ylabel("Cummulative Share of Imputation Errors")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig("lorenz_curves_by_model.pdf", format='pdf', bbox_inches='tight')
        plt.show()