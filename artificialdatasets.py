import numpy as np
import pandas as pd
from sklearn.preprocessing import Binarizer


def bernoulli_generation(instances, dimensions, n_classes=2,
                         density=0.05, density_sd=0.02, target=None):
    """Creates a binary DataFrame using independent Bernoulli distribution.
    Each class uses its own distribution for each attribute.
    Uses a normal distribution to find the distribution for each value `p`.

    :param dimensions: int - number of features
    :param instances: int - number of instances
    :param n_classes: int - number of classes
    :param density: float - approximate ratio of 1s in the data
    :param density_sd: float - standard deviation of the density

    :returns: a pandas DataFrame with `dimensions` numbered columns,
              and a target column
    """

    if target is None:
        target = np.random.choice(n_classes, size=instances, replace=True)
    random_matrix = np.random.rand(instances, dimensions)

    df = pd.DataFrame(random_matrix).assign(target=target)

    for cls in range(n_classes):
        for dim in range(dimensions):
            density = np.random.rand() * density_sd + density
            threshold = 1.0 - density
            binner = Binarizer(threshold)
            df.loc[df.target == cls, dim] = binner.transform(
                    df.loc[df.target == cls, dim].values.reshape(-1, 1)
                    ).reshape(-1, 1)[0]

    return df


def binomial_generation(instances, dimensions, n_classes=2,
                        p_mean=0.2, p_sd=0.05,
                        n_min=5, n_max=20, target=None):
    """Creates an integer DataFrame using a binomial distribution.
    Each class has its own distribution for each feature.
    Uses a normal distribution to find the values of `p`,
    and a uniform distribution for the values of `n`.
    """

    if target is None:
        target = np.random.choice(n_classes, size=instances, replace=True)
    random_matrix = np.zeros((instances, dimensions))

    df = pd.DataFrame(random_matrix).assign(target=target)

    for cls in range(n_classes):
        for dim in range(dimensions):
            p = np.random.randn() * p_sd + p_mean
            n = np.random.randint(n_min, n_max + 1)
            size = df.loc[df.target == cls, dim].shape
            df.loc[df.target == cls, dim] = np.random.binomial(n, p, size)

    return df


def sparse_binomial_generation(instances, dimensions, n_classes=2,
                               density=0.05, density_sd=0.02,
                               p_mean=0.2, p_sd=0.05,
                               n_min=5, n_max=20):

    base = bernoulli_generation(instances, dimensions, n_classes,
                                density, density_sd)
    weights = binomial_generation(instances, dimensions, n_classes,
                                  p_mean, p_sd, n_min, n_max,
                                  target=base.target)
    base_matrix = base.drop('target', axis=1).as_matrix()
    weights_matrix = weights.drop('target', axis=1).as_matrix()
    final_matrix = base_matrix * weights_matrix

    return pd.DataFrame(final_matrix).assign(target=base.target)
