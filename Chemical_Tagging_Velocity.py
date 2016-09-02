# coding: utf-8

# In[1]:

__author__ = 'Erwin Chen'
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
from astropy.table import Table
import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.metrics.cluster import v_measure_score
from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope

# In[2]:

elements = np.array(['Al', 'Ca', 'C', 'Fe', 'K', 'Mg', 'Mn', 'Na', 'Ni', 'N', 'O', 'Si', 'S', 'Ti', 'V'])


def get_corr(chem, elements):
    '''
    get correlations among 15 elements.
    
    :param chem:
        The matrix that contains chemical abundances with 
        the shape (num_stars, num_elements)
        
    :param elements:
        The array that contains the names of element abundances
        
    :returns:
    The correlation matrix
    '''
    n_elements = len(elements)
    # get the correlation matrix for all stars
    correlation = np.corrcoef(chem.T)
    # plot correlation matrix
    fig_corr = plt.figure()
    plt.pcolor(correlation)
    plt.colorbar()
    plt.yticks(np.arange(0.5, n_elements + .5), elements)
    plt.xticks(np.arange(0.5, n_elements + .5), elements)
    plt.xlim(0, n_elements)
    plt.ylim(0, n_elements)
    plt.title('Correlation Matrix for 15 Elements')
    plt.show()
    return correlation


# In[3]:

def get_PCA(chem):
    '''
    get the transformed data after PCA.
    
    :param chem:
        The matrix that contains chemical abundances with 
        the shape (num_stars, num_elements)
        
    :returns:
    transform data
    '''
    # perform PCA
    n_components = chem[0].size
    pca = PCA(n_components=n_components)
    chem_pca = pca.fit_transform(chem)
    # plot results from PCA
    ratio = np.around(pca.explained_variance_ratio_, decimals=3)
    # plot explained variance ratio
    ratio_increment = [sum(ratio[:n + 1]) for n in range(n_components)]
    ratio_increment = np.around(ratio_increment, decimals=3)
    print ratio_increment
    f, ax = plt.subplots()
    x = range(1, n_components + 1)
    ax.plot(x, ratio_increment)
    ax.set_title('PCA Explained Variance Ratio for All Stars')
    ax.set_xlabel('principal component')
    ax.set_ylabel('ratio')
    ax.set_xticks(x)
    ax.set_xlim(0.5, x[-1] + .5)
    ax.set_ylim(ratio_increment[0] * .95, 1)
    plt.show()
    return chem_pca


# In[4]:

# # load data from APOGEE
# load data from fits file
ap_file = fits.open('allStar-v603.fits')
ap_file.info()
ap_data = ap_file[1].data
feature_names = ['APOGEE_ID', 'GLON', 'GLAT', 'RA', 'DEC', 'VHELIO_AVG', 'LOGG', 'TEFF', 'PMRA', 'PMDEC', 'AL_H',
                 'CA_H', 'C_H',
                 'FE_H', 'K_H', 'MG_H', 'MN_H', 'NA_H', 'NI_H', 'N_H', 'O_H', 'SI_H', 'S_H', 'TI_H', 'V_H', 'SNR']
element_names = ['AL_H', 'CA_H', 'C_H', 'FE_H', 'K_H', 'MG_H', 'MN_H', 'NA_H', 'NI_H', 'N_H', 'O_H', 'SI_H', 'S_H',
                 'TI_H', 'V_H']
ap_cols = []
for name in feature_names:
    ap_cols.append(ap_data.field(name))
ap_cols = np.array(ap_cols)
ap_cols = ap_cols.T
dtype = ['float' for n in range(len(feature_names))]
dtype[0] = 'string'
ap_table = Table(data=ap_cols, names=feature_names, dtype=dtype, meta={'name': 'apogee table'})

# In[5]:

# get stars with 15 elements
ap_stars_15 = np.where((ap_table['AL_H'] > -9999.0)
                       * (ap_table['CA_H'] > -9999.0)
                       * (ap_table['C_H'] > -9999.0)
                       * (ap_table['FE_H'] > -9999.0)
                       * (ap_table['K_H'] > -9999.0)
                       * (ap_table['MG_H'] > -9999.0)
                       * (ap_table['MN_H'] > -9999.0)
                       * (ap_table['NA_H'] > -9999.0)
                       * (ap_table['NI_H'] > -9999.0)
                       * (ap_table['N_H'] > -9999.0)
                       * (ap_table['O_H'] > -9999.0)
                       * (ap_table['SI_H'] > -9999.0)
                       * (ap_table['S_H'] > -9999.0)
                       * (ap_table['TI_H'] > -9999.0)
                       * (ap_table['V_H'] > -9999.0)
                       )[0]
ap_table_15 = ap_table[ap_stars_15]

# In[6]:

# correlation and PCA for 15 elements
ap_chem_15 = np.array([np.array(ap_table_15[element_names][element], dtype=float) for element in element_names])
ap_chem_15 = ap_chem_15.T
ap_15_corr = get_corr(ap_chem_15, elements)
ap_15_PCA = get_PCA(ap_chem_15)

# In[7]:

# correlation and PCA for 10 elements
element_names_10 = ['AL_H', 'CA_H', 'FE_H', 'K_H', 'MG_H', 'MN_H', 'NI_H', 'O_H', 'SI_H', 'S_H']
elements_10 = ['Al', 'Ca', 'Fe', 'K', 'Mg', 'Mn', 'Ni', 'O', 'Si', 'S']
ap_chem_10 = np.array([np.array(ap_table_15[element_names_10][element], dtype=float) for element in element_names_10])
ap_chem_10 = ap_chem_10.T
ap_10_corr = get_corr(ap_chem_10, elements_10)
ap_10_PCA = get_PCA(ap_chem_10)

# In[8]:

# load known members
known_clusters = np.loadtxt('table4.dat', usecols=(0, 1), dtype=('S', 'S'), unpack=True)
member_IDs = np.intersect1d(ap_table_15['APOGEE_ID'], known_clusters[0])
members_kc = np.array([np.where(ID == known_clusters[0])[0][0] for ID in member_IDs])
members_table = np.array([np.where(ID == ap_table_15['APOGEE_ID'])[0][0] for ID in member_IDs])
ap_tb_mem = ap_table_15[members_table]
name_col = Table.Column(name='cluster_name', data=known_clusters[1][members_kc])
ap_tb_mem.add_column(name_col)

# In[9]:

# get clusters with more than 5 stars
ap_tb_mem = ap_tb_mem.group_by('cluster_name')
# print ap_tb_mem.groups.keys
# print ap_tb_mem.groups.indices
k = 0
index = []
for group in ap_tb_mem.groups:
    if group['AL_H'].size > 5:
        index.append(k)
    k += 1
ap_tb_check = ap_tb_mem.groups[index]

# In[43]:

# plot clusters in chemical space with respect to H
from scipy.ndimage.filters import gaussian_filter

cluster_labels = np.zeros(len(ap_tb_check))
for n in ap_tb_check.groups.indices[1:]:
    cluster_labels[n:] += 1
cluster_labels = cluster_labels.astype('int')
for n in range(len(element_names)):
    fig = plt.figure()
    #     H, xedges, yedges = np.histogram2d(ap_table_15[element_names[n-1]], ap_table_15[element_names[n]], bins=200)
    #     x = (xedges[:-1] + xedges[1:]) / 2
    #     y = (yedges[:-1] + yedges[1:]) / 2
    #     levels = np.array([0.01, 0.08, 0.1, 0.5]) * H.max()
    #     print levels
    #     plt.contour(x, y, H, cmap='Greys', levels=levels)
    plt.plot(ap_table_15[element_names[n - 1]], ap_table_15[element_names[n]], '.')
    plt.plot(ap_tb_check[element_names[n - 1]], ap_tb_check[element_names[n]], 'o')
    plt.xlabel('[%s/H]' % elements[n - 1])
    plt.ylabel('[%s/H]' % elements[n])
    plt.show()

# In[ ]:

# plot clusters in chemical space with respect to Fe
cluster_labels = np.zeros(len(ap_tb_check))
for n in ap_tb_check.groups.indices[1:]:
    cluster_labels[n:] += 1
cluster_labels = cluster_labels.astype('int')
for n in range(len(element_names)):
    if (element_names[n - 1] != 'FE_H') * (element_names[n] != 'FE_H'):
        fig = plt.figure()
        plt.scatter(ap_tb_check[element_names[n - 1]] - ap_tb_check['FE_H'],
                    ap_tb_check[element_names[n]] - ap_tb_check['FE_H'],
                    c=cluster_labels, s=8, linewidth=0, alpha=0.8)
        plt.xlabel('[%s/Fe]' % elements[n - 1])
        plt.ylabel('[%s/Fe]' % elements[n])
        plt.show()
    elif element_names[n - 1] != 'FE_H':
        fig = plt.figure()
        plt.scatter(ap_tb_check[element_names[n - 1]] - ap_tb_check['FE_H'], ap_tb_check[element_names[n]],
                    c=cluster_labels, s=8, linewidth=0, alpha=0.8)
        plt.xlabel('[%s/Fe]' % elements[n - 1])
        plt.ylabel('[%s/H]' % elements[n])
        plt.show()
    elif element_names[n] != 'FE_H':
        fig = plt.figure()
        plt.scatter(ap_tb_check[element_names[n - 1]], ap_tb_check[element_names[n]] - ap_tb_check['FE_H'],
                    c=cluster_labels, s=8, linewidth=0, alpha=0.8)
        plt.xlabel('[%s/H]' % elements[n - 1])
        plt.ylabel('[%s/Fe]' % elements[n])
        plt.show()

# In[ ]:

# plot other features
fig = plt.figure()
plt.scatter(ap_tb_check['VHELIO_AVG'], ap_tb_check['LOGG'], c=cluster_labels, s=8, linewidth=0, alpha=0.8)
plt.xlabel('VHELIO_AVG')
plt.ylabel('LOGG')
plt.show()


# In[ ]:
