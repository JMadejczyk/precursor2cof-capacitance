"""Cross-validation pipeline for Chemprop models."""
import copy
import os
import random

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from modules.predictor.data.utils import custom_data_split
from modules.predictor.training_and_evaluation.training_pipeline import ModelTrainingPipeline

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ChempropTrainingPipeline(ModelTrainingPipeline):
    """
    Cross-validation pipeline class for Chemprop models.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        feature_types: dict,
        folds: list,
        metrics: list,
        save_dir: str,
        data_name: str,
        model_path: str,
        hyperparam_opt: bool = True,
        num_bins: int = 5,
        verbose: bool = False,
    ):
        """
        Initialize the cross-validation pipeline.
        :param X: dataframe with features.
        :param y: dataframe with target variable.
        :param folds: list with cross-validation folds.
        :param metrics: list with metrics to evaluate.
        :param save_dir: path to save scores.
        :param data_name: name of the dataset.
        :param model_path: path to saved Chameleon model.
        :param hyperparam_opt: whether to perform hyperparameter optimization.
        :param num_bins: number of bins for stratified splitting for hyperparam optimization.
        :param verbose: whether to print model scores.
        """
        super().__init__(X, y, feature_types, folds, metrics, save_dir, data_name, hyperparam_opt, num_bins, verbose)
        self.model_path = model_path
        self.num_workers = 8

    def tune_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model: object, param_grid: dict | None) -> object:
        return None

    def train_pipeline(self, model_name: str, model_path: str | None = None) -> tuple:
        """
        Train the model.
        :param model_name: name of the model.
        :param model_path: path to saved model.
        :return: tuple with results, scores, explanations, model parameters.
        """
        proper_model_name = "ChemProp model"
        self.init_scores()

        if self.verbose:
            print(f"Training model {proper_model_name}")

        for i, fold in enumerate(self.folds):
            train_idx, test_idx = fold

            X_train_all = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)
            y_train_all = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)
            X_test = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
            y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)

            folds = custom_data_split(X_train_all, y_train_all, num_bins=self.num_bins, train_size=0.7, random_state=42)
            X_train, y_train = X_train_all.iloc[folds[0][0], :], y_train_all.iloc[folds[0][0], :]
            X_val, y_val = X_train_all.iloc[folds[0][1], :], y_train_all.iloc[folds[0][1], :]

            train_smiles = X_train.loc[:, "smiles"].values.flatten()
            additional_descriptors_train = X_train.drop(columns=["smiles"]).values
            ys = y_train[y_train.columns[0]].values.flatten().reshape(-1, 1)
            val_smiles = X_val.loc[:, "smiles"].values.flatten()
            additional_descriptors_val = X_val.drop(columns=["smiles"]).values
            y_vals = y_val[y_val.columns[0]].values.flatten().reshape(-1, 1)

            if additional_descriptors_train.shape[1] == 0:
                train_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(train_smiles, ys)]
                val_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(val_smiles, y_vals)]
            else:
                train_data = [data.MoleculeDatapoint.from_smi(smi, y, x_d=X_d) for smi, y, X_d in zip(train_smiles, ys, additional_descriptors_train)]
                val_data = [data.MoleculeDatapoint.from_smi(smi, y, x_d=X_d) for smi, y, X_d in zip(val_smiles, y_vals, additional_descriptors_val)]

            featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
            agg = nn.MeanAggregation()
            chemeleon_mp = torch.load(self.model_path, weights_only=True)
            mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
            mp.load_state_dict(chemeleon_mp["state_dict"])

            train_dset = data.MoleculeDataset(train_data, featurizer)
            scaler = train_dset.normalize_targets()
            if additional_descriptors_train.shape[1] != 0:
                extra_datapoint_descriptors_scaler = train_dset.normalize_inputs("X_d")
            train_loader = data.build_dataloader(train_dset, num_workers=self.num_workers)

            val_dset = data.MoleculeDataset(val_data, featurizer)
            val_dset.normalize_targets(scaler)
            if additional_descriptors_train.shape[1] != 0:
                val_dset.normalize_inputs("X_d", extra_datapoint_descriptors_scaler)
            val_loader = data.build_dataloader(val_dset, num_workers=self.num_workers, shuffle=False)

            output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

            metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]

            if additional_descriptors_train.shape[1] != 0:
                X_d_transform = nn.ScaleTransform.from_standard_scaler(extra_datapoint_descriptors_scaler)
                ffn_input_dim = mp.output_dim + additional_descriptors_train.shape[1]
                ffn = nn.RegressionFFN(output_transform=output_transform, input_dim=ffn_input_dim)
                mpnn = models.MPNN(mp, agg, ffn, X_d_transform=X_d_transform, batch_norm=False, metrics=metric_list)
            else:
                ffn = nn.RegressionFFN(output_transform=output_transform, input_dim=mp.output_dim)
                mpnn = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=metric_list)

            mpnn.message_passing.apply(lambda module: module.requires_grad_(False))
            mpnn.message_passing.eval()
            mpnn.bn.apply(lambda module: module.requires_grad_(False))
            mpnn.bn.eval()

            checkpoint_path = os.path.join(self.save_dir, "models", f"checkpoints_{i}")
            os.makedirs(checkpoint_path, exist_ok=True)

            checkpointing = ModelCheckpoint(
                checkpoint_path,  # Directory where model checkpoints will be saved
                "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
                "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
                mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
                save_last=True,  # Always save the most recent checkpoint, even if it's not the best
            )
            trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=True,
                # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
                enable_progress_bar=True,
                accelerator="auto",
                devices=1,
                max_epochs=20,  # number of epochs to train for
                callbacks=[checkpointing],  # Use the configured checkpoint callback
            )
            trainer.fit(mpnn, train_loader, val_loader)

            # evaluate on test set
            test_smiles = X_test["smiles"].tolist()
            additional_descriptors_test = X_test.drop(columns=["smiles"]).values
            if additional_descriptors_test.shape[1] == 0:
                test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in test_smiles]
            else:
                test_data = [data.MoleculeDatapoint.from_smi(smi, x_d=X_d) for smi, X_d in zip(test_smiles, additional_descriptors_test)]

            test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
            test_loader = data.build_dataloader(test_dset, shuffle=False, num_workers=self.num_workers)

            test_results = trainer.predict(mpnn, test_loader)
            y_pred = np.concatenate(test_results, axis=0).flatten()
            f_imp = ()

            y_test_numpy = y_test.to_numpy().flatten()
            y_pred_eval = self.eval_model(y_pred, y_test_numpy)

            baseline = np.median(y_train_all[y_train_all.columns[0]].to_numpy()) * np.ones_like(y_test_numpy)
            baselines = self.eval_model(baseline, y_test_numpy)

            self.update_scores(y_pred_eval, baselines, f_imp)

        results = self.aggregate_scores()
        self.save_results(results, proper_model_name, None)
        return results, self.scores, self.feature_importance, None

    def calculate_f_importance(self, model: object, method: str, X_test: pd.DataFrame, X_train: pd.DataFrame) -> tuple:
        pass
