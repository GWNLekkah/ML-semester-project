from prince import PCA, MCA
import numpy as np

def perform_PCA(data, comp, n=3):
    pca = PCA(n_components=comp, n_iter=n, random_state=101)
    pca.fit(data)
    trans_pca = pca.transform(data)
    #print(trans_pca.head())
    #print(sum(pca.explained_inertia_))
    return sum(pca.explained_inertia_)

# Does not work yet
def perform_MCA(data, comp, n=3):
    mca = MCA(n_components=comp, n_iter=n, random_state=101)
    # Change all 1.0 and 0.0 to string booleans to solve a known bug in mca ("True" and "False" did not work)
    data[data == 1.0] = 1.000001
    data[data == 0.0] = 0.000001

    mca.fit(data)
    trans_mca = mca.transform(data)
    trans_mca.head()
    #print(trans_mca.head())
    #print("Coverage: ", sum(mca.explained_inertia_))
    return sum(mca.explained_inertia_)


def apply_dimreduction(data, method, components = 2):
    if method == "PCA":
        perform_PCA(data, components)
    elif method == "MCA":
        perform_MCA(data, components)
    else:
        raise 'Specify a valid dimension reduction method'
