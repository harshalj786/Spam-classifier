from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

class MY_PCA:

    def __init__(self, n_components):
        self.n_components = n_components

    def convert_to_pca(self, x):
        # Store the input data
        self.x = x
        
        # Scale the data
        scalar = StandardScaler()
        self.x_scaled = scalar.fit_transform(self.x)
        
        # Apply PCA
        pca = PCA(n_components=self.n_components)
        x_pca = pca.fit_transform(self.x_scaled)
        
        # Create column names
        pc_cols = [f"PC{i+1}" for i in range(x_pca.shape[1])]
        
        # Create and return DataFrame
        X_pca_df = pd.DataFrame(x_pca, columns=pc_cols, index=self.x.index)
        
        # Store PCA object for variance info
        self.pca = pca
        
        return X_pca_df
    
    def get_explained_variance(self):
        if hasattr(self, 'pca'):
            return self.pca.explained_variance_ratio_
        else:
            return "Call convert_to_pca() first"