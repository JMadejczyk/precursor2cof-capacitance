"""Paper (batch) corrections of the target J.

Every correction is fitted on training papers only. A test paper's offset is unknown at prediction time,
so predictions are mapped back to the J scale of an average paper.
"""

import numpy as np
import pandas as pd


class NoCorrection:
    """Raw J."""

    def fit_transform(self, y: np.ndarray, papers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        :param y: J of the training rows.
        :param papers: paper of each training row.
        :return: corrected target and a mask of training rows to keep.
        """
        return y.copy(), np.ones(len(y), dtype=bool)

    def to_j(self, pred: np.ndarray) -> np.ndarray:
        """Map predictions back to the J scale of an average paper."""
        return pred


class PaperCenter(NoCorrection):
    """J minus its paper's mean. Papers with one material carry no information and are dropped."""

    def fit_transform(self, y, papers):
        s = pd.Series(y)
        keep = s.groupby(papers).transform("size").to_numpy() >= 2
        self.grand_mean = y[keep].mean()
        return y - s.groupby(papers).transform("mean").to_numpy(), keep

    def to_j(self, pred):
        return pred + self.grand_mean


class PaperZScore(NoCorrection):
    """Per-paper z-score (location and scale). Papers with one material are dropped."""

    def fit_transform(self, y, papers):
        s = pd.Series(y)
        keep = s.groupby(papers).transform("size").to_numpy() >= 2
        mean = s.groupby(papers).transform("mean").to_numpy()
        std = s.groupby(papers).transform(lambda v: v.std(ddof=0)).to_numpy()
        std = np.where(std > 0, std, 1.0)
        self.grand_mean = y[keep].mean()
        self.pooled_std = np.sqrt(((y - mean)[keep] ** 2).mean())
        return (y - mean) / std, keep

    def to_j(self, pred):
        return pred * self.pooled_std + self.grand_mean


class ComBat1D(NoCorrection):
    """
    ComBat (Johnson et al. 2007) for a single variable: empirical-Bayes shrinkage of each paper's location and
    scale toward priors estimated across papers. Single-material papers are kept (their scale comes from the prior).
    """

    def fit_transform(self, y, papers, n_iter: int = 100):
        df = pd.DataFrame({"y": y, "p": papers})
        alpha = y.mean()
        batch_mean = df.groupby("p")["y"].transform("mean").to_numpy()
        sigma = np.sqrt(((y - batch_mean) ** 2).mean())
        z = (y - alpha) / sigma
        df["z"] = z
        g = df.groupby("p")["z"]
        n, gamma_hat = g.size(), g.mean()
        delta_hat = g.var(ddof=1)
        gamma_bar, tau2 = gamma_hat.mean(), gamma_hat.var(ddof=1)
        d = delta_hat.dropna()
        m, s2 = d.mean(), d.var(ddof=1)
        a, b = (2 * s2 + m**2) / s2, (m * s2 + m**3) / s2
        delta_star = delta_hat.fillna(m)
        for _ in range(n_iter):
            gamma_star = (n * tau2 * gamma_hat + delta_star * gamma_bar) / (n * tau2 + delta_star)
            ss = df.assign(r=(df["z"] - df["p"].map(gamma_star)) ** 2).groupby("p")["r"].sum()
            new_delta = (b + 0.5 * ss) / (n / 2 + a - 1)
            if np.allclose(new_delta, delta_star, rtol=1e-6):
                break
            delta_star = new_delta
        z_adj = (z - df["p"].map(gamma_star).to_numpy()) / np.sqrt(df["p"].map(delta_star).to_numpy())
        return sigma * z_adj + alpha, np.ones(len(y), dtype=bool)


CORRECTIONS = {"none": NoCorrection, "center": PaperCenter, "zscore": PaperZScore, "combat": ComBat1D}
METHODS = ["none", "center", "zscore", "combat", "mixed"]


def fit_mixed(model: object, X: np.ndarray, y: np.ndarray, papers: np.ndarray, n_iter: int = 10) -> object:
    """
    Random intercept per paper fitted jointly with the model (MERF-style EM): alternate between fitting the model
    on y - b[paper] and re-estimating shrunken paper intercepts b from the residuals. Random forests use
    out-of-bag predictions for the residuals so that they do not absorb the paper offsets.
    :return: the fitted model (fixed part); an unseen paper gets intercept 0. Intercepts in model.paper_intercepts_.
    """
    codes, uniq = pd.factorize(papers)
    b = np.zeros(len(uniq))
    sigma2_e, sigma2_b = y.var() / 2, y.var() / 2
    if hasattr(model, "oob_score"):
        model.set_params(oob_score=True)
    for _ in range(n_iter):
        model.fit(X, y - b[codes])
        fitted = model.oob_prediction_ if hasattr(model, "oob_prediction_") else model.predict(X)
        r = y - fitted
        n_i = np.bincount(codes)
        sum_r = np.bincount(codes, weights=r)
        post_var = 1.0 / (n_i / sigma2_e + 1.0 / sigma2_b)
        b = post_var * sum_r / sigma2_e
        sigma2_b = max(np.mean(b**2 + post_var), 1e-6)
        sigma2_e = max(np.mean((r - b[codes]) ** 2) + np.mean(n_i * post_var) / len(y), 1e-6)
    model.paper_intercepts_ = pd.Series(b, index=uniq)
    return model


def fit_predict(method: str, model: object, X_train: np.ndarray, y_train: np.ndarray, papers_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Fit a correction and a model on training rows, predict test rows on the J scale of an average paper.
    :return: (predictions, fitted model, fitted correction or None for 'mixed', mask of training rows used).
    """
    if method == "mixed":
        model = fit_mixed(model, X_train, y_train, papers_train)
        return model.predict(X_test).flatten(), model, None, np.ones(len(y_train), dtype=bool)
    corr = CORRECTIONS[method]()
    y_adj, keep = corr.fit_transform(y_train, papers_train)
    model.fit(X_train[keep], y_adj[keep])
    return corr.to_j(model.predict(X_test).flatten()), model, corr, keep
