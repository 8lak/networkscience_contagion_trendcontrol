import networkx as nx
import numpy as np
from scipy import stats
import powerlaw

# Load graph
G = nx.read_edgelist("facebook_combined.txt", nodetype=int)
nodes = list(G.nodes())

# Compute centralities
degree = np.array([G.degree(n) for n in nodes])
bc = nx.betweenness_centrality(G)
betweenness = np.array([bc[n] for n in nodes])


# Bin by degree
n_bins = 15
degree_percentiles = np.percentile(degree, np.linspace(0, 100, n_bins + 1))

results = []
for i in range(n_bins):
    mask = (degree >= degree_percentiles[i]) & (degree < degree_percentiles[i + 1])
    B_given_d = betweenness[mask]
    B_given_d = B_given_d[B_given_d > 0]  # Lognormal needs positive values
    d_mean = degree[mask].mean()
    
    if len(B_given_d) < 30:
        continue
    
    # Sample moments
    SM_1 = np.mean(B_given_d)
    SM_2 = np.mean(B_given_d**2)
    
    # Fit lognormal: log(X) ~ Normal(mu, sigma^2)
    log_B = np.log(B_given_d)
    mu_fit = np.mean(log_B)
    sigma_fit = np.std(log_B)
    
    # Theoretical moments for lognormal
    TM_1 = np.exp(mu_fit + sigma_fit**2 / 2)
    TM_2 = np.exp(2 * mu_fit + 2 * sigma_fit**2)
    
    ratio_1 = TM_1 / SM_1
    ratio_2 = TM_2 / SM_2
    
    results.append({
        'd_mean': d_mean,
        'SM_1': SM_1,
        'TM_1': TM_1,
        'ratio_1': ratio_1,
        'ratio_2': ratio_2,
        'mu': mu_fit,
        'sigma': sigma_fit,
        'n_samples': mask.sum()
    })

print("d_mean\t\tSM_1\t\tTM_1\t\tTM/SM_1\t\tTM/SM_2\t\tn")
for r in results:
    print(f"{r['d_mean']:.1f}\t\t{r['SM_1']:.6f}\t{r['TM_1']:.6f}\t{r['ratio_1']:.3f}\t\t{r['ratio_2']:.3f}\t\t{r['n_samples']}")