"""Flatness feature calculation."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from sklearn.linear_model import LinearRegression

from modules.features.utils import compute_conformer


def get_flatness_mol(molecule: Mol | str, plot_visualization: bool = False, **kwargs) -> float | None:
    """
    Calculate the flatness of a molecule.

    Fit a plane to the molecule's 3D coordinates and calculate the RMSD of the
    molecule's atoms from the plane.

    Args:
        molecule: An RDKit molecule or a SMILES string.
        plot_visualization: Whether to plot the molecule and the fitted plane.
        **kwargs: Additional arguments for the compute_conformer function.
    Returns:
        The flatness of the molecule.
    """

    if isinstance(molecule, str):
        rdkit_mol = compute_conformer(molecule, **kwargs)
    else:
        rdkit_mol = molecule
        # check if rdkit_mol has a conformation
        if rdkit_mol.GetNumConformers() == 0:
            rdkit_mol = compute_conformer(rdkit_mol)

    # check if conformer is present
    if rdkit_mol is None:
        return None

    conformers = rdkit_mol.GetNumConformers()
    rmsds = np.zeros(conformers)
    for i in range(conformers):
        conf = rdkit_mol.GetConformer(i)
        coords = conf.GetPositions()

        # Fit a plane to the coordinates
        model = LinearRegression()
        model.fit(coords[:, :2], coords[:, 2])

        # Calculate the RMSD of the atoms from the plane
        z_pred = model.predict(coords[:, :2])
        rmsds[i] = np.sqrt(np.mean((coords[:, 2] - z_pred) ** 2))

        if plot_visualization:
            __visualize_the_plane(coords, model)

    return np.mean(rmsds)


def __visualize_the_plane(coords: np.ndarray, model: LinearRegression) -> None:
    """
    Visualize the plane fitted to the molecule's 3D coordinates.

    Args:
        coords: The molecule's 3D coordinates.
        model: The fitted plane.
    Returns:
        None
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])

    x = np.linspace(min(coords[:, 0]), max(coords[:, 0]), 10)
    y = np.linspace(min(coords[:, 1]), max(coords[:, 1]), 10)

    x, y = np.meshgrid(x, y)

    z = model.intercept_ + model.coef_[0] * x + model.coef_[1] * y

    ax.plot_surface(x, y, z, alpha=0.2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set ticks to integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))

    # bbox_inches='tight' removes the white space around the image
    plt.tight_layout()
    plt.savefig("flatness.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    SMILES = "Nc1ccc(-c2nc(-c3ccc(N)cc3)nc(-c3ccc(N4C(=O)c5ccc6c7c(ccc(c57)C4=O)C(=O)OC6=O)cc3)n2)cc1"

    f = get_flatness_mol(SMILES, True, max_attempts=1, num_conformers=20)
    logging.info(f"Flatness: {f}")

    mol = Chem.MolFromSmiles(SMILES)
