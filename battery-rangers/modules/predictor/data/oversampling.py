"""Oversampling functions."""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

# pylint: disable=invalid-name


def dist(x1, x2):
    """Calculates the Euclidean distance between two data points."""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def get_rare_cases_relavance(y, te):
    """
    Identifies rare cases based on the relevance of the target variable values.
    :param y: dataset target variable
    :param te: a threshold for the relevance of the target variable
    :return:
    """
    relevance_func = create_inverse_density_relevance_function(y)

    # Determine the median of the target variable
    tilde_y = np.median(y)

    # Identify rare low and high cases
    relevances = relevance_func(y)
    rareL_indices = np.where((relevances > te) & (y < tilde_y))[0]
    rareH_indices = np.where((relevances > te) & (y > tilde_y))[0]

    rare_ids = np.concatenate((rareL_indices, rareH_indices), axis=0) if len(rareL_indices) > 0 and len(rareH_indices) > 0 else rareL_indices if len(rareL_indices) > 0 else rareH_indices

    return rare_ids


def get_rare_cases_bins(y, te):
    """
    Identifies rare cases based number of bins.
    :param y: a dataset target variable
    :param te: a number of bins
    :return:
    """
    rare_ids = []
    rng = np.random.default_rng(seed=42)
    bins = pd.cut(y, bins=te, labels=False)
    unique_bins, counts = np.unique(bins, return_counts=True)
    largest_bin = np.argmax(counts)
    lragest_bin_size = counts[largest_bin]
    lragest_bin_size *= 0.67
    lragest_bin_size = int(lragest_bin_size)
    for b in unique_bins:
        if b != largest_bin:
            small_ids = np.argwhere(bins == b).flatten()
            oversampling_count = lragest_bin_size - len(small_ids)
            while oversampling_count >= len(small_ids):
                rare_ids.extend(small_ids)
                oversampling_count -= len(small_ids)
            if oversampling_count > 0:
                sampled_rare = rng.choice(small_ids, oversampling_count, replace=False)
                rare_ids.extend(sampled_rare)
    return rare_ids


def smoter(X, y, te, o, k, numeric_cols, oversampling_type="SMOTER"):
    """
    Implements the SMOTER algorithm for regression with rare extreme values.

    Args:
        X (np.ndarray): The training data features.
        y (np.ndarray): The training data target variable.
        te (float): The threshold for relevance of the target variable values.
        o (float): The percentage of over-sampling (e.g., 200 for 200%).
        k (int): The number of nearest neighbors used in case generation.
        numeric_cols (list): A list of column indices representing numeric attributes.

    Returns:
        tuple: A tuple containing the new training data features (X_resampled) and
               the new training data target variable (y_resampled).
    """

    if oversampling_type == "SMOTER":
        rare_ids = get_rare_cases_relavance(y, te)
    else:
        rare_ids = get_rare_cases_bins(y, te)

    # Generate synthetic cases for rare low and high values
    newCases_X, newCases_y = gen_synth_cases(rare_ids, o, k, numeric_cols, X, y)

    X_resampled = np.concatenate((X, newCases_X), axis=0) if len(newCases_X) > 0 else X
    y_resampled = np.concatenate((y, newCases_y), axis=0) if len(newCases_y) > 0 else y

    return X_resampled, y_resampled


def gen_synth_cases(rare_ids, o, k, numeric_cols, X, y):
    """
    Generates synthetic cases using the SMOTER approach.

    Args:
        X_rare (np.ndarray): The subset of the features for over-sampling (rare values).
        y_rare (np.ndarray): The subset of the target variable for over-sampling (rare values).
        o (float): The percentage of over-sampling (e.g., 200 for 200%).
        k (int): The number of nearest neighbors to consider.
        numeric_cols (list): A list of column indices representing numeric attributes.
        X (np.ndarray): The original training data features.
        y (np.ndarray): The original training data target variable..

    Returns:
        tuple: A tuple containing the synthetic features (X_synthetic) and the
               synthetic target variable (y_synthetic).
    """
    X_synthetic = []
    y_synthetic = []
    rng = np.random.default_rng(seed=42)
    ng = o / 100.0  # Number of new cases to generate for each existing case

    if len(rare_ids) == 0:
        return np.array([]), np.array([])

    # Calculate nearest neighbors using the original dataset for wider coverage
    categorical_cols = [i for i in range(X.shape[1]) if i not in numeric_cols]
    if len(categorical_cols) > 0:
        enc = OneHotEncoder(drop="first")
        enc_features = enc.fit_transform(X[:, categorical_cols])
        X_encoded = np.concatenate((X[:, numeric_cols], enc_features.toarray()), axis=1)
    else:
        X_encoded = X
    knn = NearestNeighbors(n_neighbors=2 * k)  # +1 to exclude the case itself
    knn.fit(X_encoded)

    for i in rare_ids:
        case_X = X[i]
        case_y = y[i]
        case_X_encoded = X_encoded[i]

        # Find k-Nearest Neighbors, but exclude the case itself for a valid neighborhood
        neighbors = knn.kneighbors(case_X_encoded.reshape(1, -1), return_distance=False)[0]  # [1:]  # Exclude self
        neighbors = pd.DataFrame(neighbors).drop_duplicates(keep="first").values.flatten()
        neighbors = np.array([n for n in neighbors if not np.all(case_X == X[n])])[:k]
        for _ in range(int(ng)):
            # Randomly choose neighbor
            x_index = rng.choice(neighbors)
            neighbor_X = X[x_index]
            neighbor_X_encoded = X_encoded[x_index]
            neighbor_y = y[x_index]

            new_X = np.copy(case_X)  # Initialize the new features with a copy of 'case_X'

            # Generate Attribute Values
            for a in range(X.shape[1]):  # Iterate through the columns
                if a in numeric_cols:
                    diff = case_X[a] - neighbor_X[a]
                    new_X[a] = case_X[a] + rng.uniform(low=0, high=1) * diff
                    if isinstance(case_X[a], int) and isinstance(neighbor_X[a], int):
                        new_X[a] = int(new_X[a])
                else:
                    new_X[a] = rng.choice([case_X[a], neighbor_X[a]])  # Randomly select

            # Decide the target value
            if len(categorical_cols) > 0:
                enc_features = enc.transform(new_X[categorical_cols].reshape(1, -1)).toarray().flatten()
                new_X_encoded = np.concatenate((new_X[numeric_cols], enc_features))
            else:
                new_X_encoded = new_X
            d1 = dist(new_X_encoded, case_X_encoded)
            d2 = dist(new_X_encoded, neighbor_X_encoded)
            new_y = (d2 * case_y + d1 * neighbor_y) / (d1 + d2)

            X_synthetic.append(new_X)
            y_synthetic.append(new_y)

    return np.array(X_synthetic), np.array(y_synthetic)


def create_inverse_density_relevance_function(Y):
    """
    Creates a relevance function that assigns higher relevance to values
    in regions of lower density.  This implements the idea that rare values
    are more relevant.

    Args:
        Y (np.ndarray): The target variable values.

    Returns:
        callable: A function that takes a target value and returns its relevance.
    """
    kde = gaussian_kde(Y)

    def relevance(y):
        """
        Calculates the relevance as the inverse of the probability density,
        normalized to the range [0, 1].
        """
        density = kde(y)
        if np.isscalar(density):
            if density == 0:
                return 1.0  # Handle zero density (rare case)
            return 1.0 / density
        density[density == 0] = np.finfo(float).eps  # Avoid division by zero
        return 1.0 / density

    # Normalize relevance values to [0, 1]
    relevance_values = relevance(Y)
    min_relevance = np.min(relevance_values)
    max_relevance = np.max(relevance_values)

    def normalized_relevance(y):
        raw_relevance = relevance(y)
        normalized = (raw_relevance - min_relevance) / (max_relevance - min_relevance)
        return normalized

    return normalized_relevance


if __name__ == "__main__":
    df = pd.read_csv("../../../old/old/fingerprints_maccs/data_experts1.csv")
    df_features = df.drop(columns=["smiles", "capacity_max"])
    X = df.drop(columns=["smiles", "capacity_max"]).values
    y = df["capacity_max"].values
    print(df.duplicated(subset=[c for c in df.columns if c not in ["smiles", "capacity_max"]]))
    print(df.iloc[[9, 21]][["smiles", "capacity_max"]])
    print(df.columns)
    te = 0.5  # Relevance threshold
    o = 300  # Over-sampling percentage
    k = 5  # Number of nearest neighbors
    numeric_cols = [i for i in range(len(df_features.columns)) if df_features.columns[i] not in ["symmetry"]]  # Indices of numeric attributes
    numeric_cols = []
    # Create the relevance function
    relevance_func = create_inverse_density_relevance_function(y)

    # You can then pass this relevance_func to the smoter function:
    X_resampled, y_resampled = smoter(X, y, te, o, k, numeric_cols, oversampling_type="SMOTER")

    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)
    print("Resampled X shape:", X_resampled.shape)
    print("Resampled y shape:", y_resampled.shape)
    print("Resampled X:\n", X_resampled)
    print("Resampled y:\n", y_resampled)

    print(f"Duplicated rows: {pd.DataFrame(X_resampled).duplicated().sum()}")
