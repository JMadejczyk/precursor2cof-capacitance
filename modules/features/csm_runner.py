"""
Provides a high-level interface to run Continuous Symmetry Measure (CSM) calculations.

See: https://github.com/continuous-symmetry-measure/csm?tab=readme-ov-file for details.

This module defines two main components:
1.  `MoleculeCSMAnalysis`: A dataclass to store the aggregated results of CSM
    calculations for a single molecule across multiple symmetry point groups.
2.  `CSMRunner`: A class that manages all interactions with a Docker container
    running the 'teamcsm/csm' tool. It handles container lifecycle,
    file I/O, conformer generation via an external utility, and parsing of
    results into the `MoleculeCSMAnalysis` structure.
"""

import os
import shutil
import uuid
from dataclasses import dataclass, field

import docker
from docker.client import DockerClient
from docker.errors import ImageNotFound, NotFound
from docker.models.containers import Container
from rdkit import Chem
from rdkit.Chem import Mol

from modules.features.utils import compute_conformer


@dataclass
class MoleculeCSMResult:
    """
    Holds the complete CSM analysis for a molecule across various point groups.

    This dataclass acts as a structured container for all data generated during
    the analysis of a single SMILES string.

    Attributes:
        smiles: The input SMILES string representing the molecule.
        csm_results: A dictionary mapping each point group (e.g., 'c2') to its
            calculated CSM value (float).
        input_sdf_path: The absolute path to the generated 3D SDF input file.
        output_dirs: A dictionary mapping each point group to the path of its
            corresponding output directory created by the CSM tool.
        error_messages: A dictionary capturing any errors that occurred during
            the calculation for a specific point group.
    """

    smiles: str
    csm_results: dict[str, float] = field(default_factory=dict)
    input_sdf_path: str | None = None
    output_dirs: dict[str, str] = field(default_factory=dict)
    error_messages: dict[str, str] = field(default_factory=dict)

    @property
    def lowest_csm(self) -> tuple[str, float] | None:
        """
        Finds the point group with the minimum CSM value.

        This property provides a convenient way to identify the symmetry group
        that the molecule most closely matches, based on the principle that a
        lower CSM value indicates a better fit.

        Returns:
            A tuple containing the point group name (str) and its corresponding
            CSM value (float) for the best fit, or None if no successful

            results are available.
        """
        if not self.csm_results:
            return None
        return min(self.csm_results.items(), key=lambda item: item[1])


class CSMRunner:
    """
    Manages and executes CSM computations using a background Docker container.

    This class provides a simplified interface to the `teamcsm/csm`
    CLI tool. It handles starting/stopping the container, creating input files
    via a provided utility function, executing the CSM calculations, and parsing
    the output into a structured `MoleculeCSMAnalysis` object.
    """

    client: DockerClient
    image_name: str
    container_name: str
    host_data_dir: str
    container_data_dir: str
    current_user_id: int = os.getuid()
    current_group_id: int = os.getgid()

    def __init__(
        self,
        image_name: str = "teamcsm/csm:latest",
        container_name: str = "csm_runner_container",
        data_dir: str = "/tmp/csm_data",
    ) -> None:
        """
        Initializes the CSMRunner and sets up configuration.

        Args:
            image_name: The name of the Docker image to use for calculations.
            container_name: The name to assign to the running Docker container.
            data_dir: The local directory for storing input and output files.
                      This directory will be mounted as a volume in the container.
        """
        self.client = docker.from_env()
        self.image_name = image_name
        self.container_name = container_name
        self.host_data_dir = os.path.abspath(data_dir)
        self.container_data_dir = "/data"  # Mount point inside the container

        if not os.path.exists(self.host_data_dir):
            os.makedirs(self.host_data_dir)

    def _get_or_start_container(self, pull_image: bool) -> Container:
        """
        Ensures the Docker container is running and returns it.

        This method checks if the container exists. If it does, it starts it if
        it's stopped. If it doesn't exist, it creates and starts a new one.
        It can optionally pull the image first.

        Args:
            pull_image: If True, ensures the latest Docker image is pulled
                        before starting the container.

        Returns:
            The running `docker.models.containers.Container` object.

        Raises:
            DockerException: If there's an issue communicating with the Docker daemon.
        """
        if pull_image:
            try:
                self.client.images.get(self.image_name)
            except ImageNotFound:
                print(f"Pulling latest image '{self.image_name}'...")
                self.client.images.pull(self.image_name)
                print("Image pulled.")

        try:
            container = self.client.containers.get(self.container_name)
            if container.status != "running":
                container.start()
            return container
        except NotFound:
            print(f"Container '{self.container_name}' not found. Creating a new one.")
            return self.client.containers.run(
                self.image_name,
                name=self.container_name,
                detach=True,
                user=f"{self.current_user_id}:{self.current_group_id}",
                tty=True,  # Keeps the container running in the background
                volumes={self.host_data_dir: {"bind": self.container_data_dir, "mode": "rw"}},
            )

    def analyze_molecule(
        self,
        molecule: Mol | str,
        point_groups: list[str],
        pull_image: bool = False,
        exact: bool = True,
    ) -> MoleculeCSMResult | None:
        """
        Performs a full CSM analysis for a molecule against multiple point groups.

        This is the main public method of the class. It orchestrates the entire
        workflow: starting the container, creating the input SDF file, executing
        the CSM calculation for each specified point group, and parsing the results.

        Args:
            molecule: The SMILES string of the molecule to analyze or an RDKit Mol object.
            point_groups: A list of point group strings (e.g., ['c2', 'd6'])
                          to calculate the CSM against.
            pull_image: If True, ensures the Docker image is up-to-date before running.
            exact: If True, exact calculations are performed, otherwise approximate.
        Returns:
            A `MoleculeCSMAnalysis` object containing the aggregated results and
            any errors encountered.
        """
        container = self._get_or_start_container(pull_image)

        input_filename = f"{uuid.uuid4().hex}.sdf"
        host_input_path = os.path.join(self.host_data_dir, input_filename)

        molecule = compute_conformer(molecule=molecule, save_file=True, max_attempts=50, num_conformers=100, filename=host_input_path)

        if molecule is None:
            return None

        analysis = MoleculeCSMResult(smiles=Chem.MolToSmiles(molecule))

        analysis.input_sdf_path = host_input_path
        container_input_path = os.path.join(self.container_data_dir, os.path.basename(host_input_path))

        for pg in point_groups:
            output_dirname = f"output_{pg}_{uuid.uuid4().hex}"
            host_output_path = os.path.join(self.host_data_dir, output_dirname)
            container_output_path = os.path.join(self.container_data_dir, output_dirname)

            command = [
                "csm",
                "exact" if exact else "approx",
                pg,
                "--input",
                container_input_path,
                "--output",
                container_output_path,
                "--keep-structure",
            ]
            exit_code, (_, stderr) = container.exec_run(command, demux=True)

            if exit_code != 0:
                analysis.error_messages[pg] = stderr.decode("utf-8") if stderr else "Execution failed with no stderr."
                continue

            try:
                csm_txt_path = os.path.join(host_output_path, "csm.txt")
                with open(csm_txt_path, "r") as f:
                    data_line = next(line for line in f if not line.startswith("#"))
                    csm_value = float(data_line.strip().split()[-1])
                analysis.csm_results[pg] = csm_value
                analysis.output_dirs[pg] = host_output_path
            except (IOError, StopIteration, IndexError, ValueError) as e:
                analysis.error_messages[pg] = f"Failed to parse output file: {e}"

        return analysis

    def cleanup(self) -> None:
        """
        Removes all contents of the data directory.

        This method provides a simple way to reset the data directory by deleting
        every file and sub-directory within it, ensuring a clean state for
        subsequent runs.
        """
        print(f"Cleaning all contents of the data directory: {self.host_data_dir}")
        for item_name in os.listdir(self.host_data_dir):
            item_path = os.path.join(self.host_data_dir, item_name)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}. Reason: {e}")
        print("Cleanup complete.")
