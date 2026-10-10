"""Sklearn training pipeline module."""
import ast
import copy
import os
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder, StandardScaler
from tabpfn_extensions.interpretability import shapiq

from modules.predictor.data.utils import custom_data_split
from modules.predictor.training_and_evaluation.evaluation_metrics import EvalMetrics
from modules.predictor.training_and_evaluation.model_factory import Models
from modules.predictor.training_and_evaluation.training_pipeline import ModelTrainingPipeline

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")


class SklearnTrainingPipeline(ModelTrainingPipeline):
    """
    Cross-validation pipeline class for sklearn models.
    """

    def tune_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model: object, param_grid: dict | None) -> object:
        """
        Perform model tuning.
        :param X_train: training data.
        :param y_train: target variable data.
        :param model: prediction model.
        :param param_grid: dictionary with parameter grid.
        :return: model with optimized parameters.
        """
        if self.hyperparam_opt and len(param_grid) > 0:
            folds = custom_data_split(X_train, y_train, num_bins=self.num_bins, train_size=0.7)
            x_train_opt_non, y_train_opt = X_train.iloc[folds[0][0], :], y_train.iloc[folds[0][0], :]
            x_test_opt_non, y_test_opt = X_train.iloc[folds[0][1], :], y_train.iloc[folds[0][1], :]

            x_train_opt = self._data_preparation(x_train_opt_non, x_train_opt_non)
            x_test_opt = self._data_preparation(x_train_opt_non, x_test_opt_non)

            def objective_rmse(trial):
                params = {}
                for key, value in param_grid.items():
                    if isinstance(value, list):
                        suggest = trial.suggest_categorical(key, value)
                        if key == "hidden_layer_sizes":
                            suggest = ast.literal_eval(suggest)
                        params[key] = suggest
                    elif isinstance(value, tuple):
                        if isinstance(value[0], int):
                            params[key] = trial.suggest_int(key, value[0], value[1])
                        else:
                            params[key] = trial.suggest_float(key, value[0], value[1])
                    else:
                        raise ValueError(f"Unsupported parameter type: {type(value)}")

                model.set_params(**params)
                try:
                    model.fit(x_train_opt.to_numpy(), y_train_opt[y_train.columns[0]].to_numpy())
                    y_pred = model.predict(x_test_opt.to_numpy()).flatten()
                    score = root_mean_squared_error(y_test_opt.to_numpy(), y_pred)
                except ValueError:
                    score = float("inf")
                return score

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.QMCSampler(seed=42))

            study.optimize(objective_rmse, n_trials=100, show_progress_bar=False)
            best_params = study.best_params
            if "hidden_layer_sizes" in best_params:
                best_params["hidden_layer_sizes"] = ast.literal_eval(best_params["hidden_layer_sizes"])
            best_score = study.best_value
            model.set_params(**best_params)
            if self.verbose:
                print(f"Best score: {best_score}\n Best params: {best_params}")
        return model

    def tune_model_folds(self, X_train: pd.DataFrame, y_train: pd.DataFrame, model: object, param_grid: dict | None) -> object:
        if self.hyperparam_opt and len(param_grid) > 0:
            folds = copy.deepcopy(self.folds)

            def objective_rmse(trial):
                params = {}
                for key, value in param_grid.items():
                    if isinstance(value, list):
                        suggest = trial.suggest_categorical(key, value)
                        if key == "hidden_layer_sizes":
                            suggest = ast.literal_eval(suggest)
                        params[key] = suggest
                    elif isinstance(value, tuple):
                        if isinstance(value[0], int):
                            params[key] = trial.suggest_int(key, value[0], value[1])
                        else:
                            params[key] = trial.suggest_float(key, value[0], value[1])
                    else:
                        raise ValueError(f"Unsupported parameter type: {type(value)}")

                model.set_params(**params)

                scores = []
                for i, fold in enumerate(folds):
                    try:
                        train_idx, test_idx = fold

                        x_train_opt_non = copy.deepcopy(X_train.loc[train_idx, :]).reset_index(drop=True)
                        y_train_opt = copy.deepcopy(y_train.loc[train_idx, :]).reset_index(drop=True)
                        x_test_opt_non = copy.deepcopy(X_train.loc[test_idx, :]).reset_index(drop=True)
                        y_test_opt = copy.deepcopy(y_train.loc[test_idx, :]).reset_index(drop=True)

                        x_train_opt = self._data_preparation(x_train_opt_non, x_train_opt_non)
                        x_test_opt = self._data_preparation(x_train_opt_non, x_test_opt_non)

                        model.fit(x_train_opt.to_numpy(), y_train_opt[y_train.columns[0]].to_numpy())
                        y_pred = model.predict(x_test_opt.to_numpy()).flatten()
                        score = root_mean_squared_error(y_test_opt.to_numpy(), y_pred)
                    except ValueError:
                        score = float("inf")
                    scores.append(score)
                return np.mean(scores)

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.QMCSampler(seed=42))

            study.optimize(objective_rmse, n_trials=100, show_progress_bar=False)
            best_params = study.best_params
            if "hidden_layer_sizes" in best_params:
                best_params["hidden_layer_sizes"] = ast.literal_eval(best_params["hidden_layer_sizes"])
            best_score = study.best_value
            model.set_params(**best_params)
            if self.verbose:
                print(f"Best score: {best_score}\n Best params: {best_params}")

        return model

    def _data_preparation(self, X_fit: pd.DataFrame, X_transform: pd.DataFrame) -> pd.DataFrame:
        """
        Data preparation: encoding categorical features and scaling numerical features.
        :param X_fit: dataframe to fit transformers.
        :param X_transform: dataframe to transform.
        :return: transformed dataframe.
        """
        numerical_features = self.feature_types["numerical"]
        categorical_features = self.feature_types["categorical"]
        df_to_transform = copy.deepcopy(X_transform)
        numerical_features = [f for f in numerical_features if f in df_to_transform.columns]
        categorical_features = [f for f in categorical_features if f in df_to_transform.columns]
        if len(categorical_features) > 0:
            enc = OneHotEncoder(drop="first", handle_unknown="ignore")
            enc.fit(X_fit[categorical_features])
            enc_features_to_transform = enc.transform(df_to_transform[categorical_features]).toarray()
            enc_features_names = enc.get_feature_names_out(categorical_features)
            df_to_transform[enc_features_names] = enc_features_to_transform
            df_to_transform.drop(categorical_features, axis=1, inplace=True)
        if len(numerical_features) > 0:
            scaler = StandardScaler().fit(X_fit[numerical_features].values)
            st_features_to_transform = scaler.transform(df_to_transform[numerical_features].values)
            df_to_transform[numerical_features] = st_features_to_transform
        df_to_transform.fillna(0, inplace=True)
        return df_to_transform

    def train_pipeline(self, model_name: str, model_path: str | None = None) -> tuple:
        """
        Train the model.
        :param model_name: name of the model.
        :param model_path: path to saved model.
        :return: tuple with results, scores, explanations, model parameters.
        """
        proper_model_name, model, param_grid, f_importance_method = Models().get_model(model_name, model_path=model_path)
        self.init_scores()

        if self.verbose:
            print(f"Training model {proper_model_name}")

        model_params = []

        for i, fold in enumerate(self.folds):
            train_idx, test_idx = fold

            # train-test split
            X_train_non = copy.deepcopy(self.X.loc[train_idx, :]).reset_index(drop=True)
            y_train = copy.deepcopy(self.y.loc[train_idx, :]).reset_index(drop=True)
            X_test_non = copy.deepcopy(self.X.loc[test_idx, :]).reset_index(drop=True)
            y_test = copy.deepcopy(self.y.loc[test_idx, :]).reset_index(drop=True)

            model = self.tune_model(X_train_non, y_train, model, param_grid)

            X_train = self._data_preparation(X_train_non, X_train_non)
            X_test = self._data_preparation(X_train_non, X_test_non)

            # model training
            model.fit(X_train.to_numpy(), y_train[y_train.columns[0]].to_numpy())

            y_pred = model.predict(X_test.to_numpy()).flatten()

            f_imp = self.calculate_f_importance(model, f_importance_method, X_test, X_train)

            # model eval
            y_test_numpy = y_test.to_numpy().flatten()
            y_pred_eval = self.eval_model(y_pred, y_test_numpy)

            baseline = np.median(y_train[y_train.columns[0]].to_numpy()) * np.ones_like(y_test_numpy)
            baselines = self.eval_model(baseline, y_test_numpy)

            self.update_scores(y_pred_eval, baselines, f_imp)
            if len(self.save_dir) > 0:
                self.save_model(model, fold_num=i)

            model_params.append(model.get_params())

        results = self.aggregate_scores()
        self.save_results(results, proper_model_name, model_params)
        return results, self.scores, self.feature_importance, model_params

    def calculate_f_importance(self, model: object, method: str, X_test: pd.DataFrame, X_train: pd.DataFrame) -> tuple:
        """
        Calculate feature importance using SHAP or model coefficients (for linear models).
        :param model: model to be explained.
        :param method: method of feature importance calculation ('kernel', 'tree', 'coefficients').
        :param X_test: test data.
        :param X_train: train data.
        :return: tuple with method and SHAP values or coefficients.
        """
        if method == "kernel":
            explainer = shap.KernelExplainer(model.predict, X_train.to_numpy())
            shap_values = explainer.shap_values(X_test.to_numpy(), gc_collect=True)
            return method, shap_values
        elif method == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.to_numpy())
            return method, shap_values
        elif method == "coefficients":
            coefficients = model.coef_
            intercept = model.intercept_
            coefficients = list(coefficients) + [intercept]
            features = X_train.columns
            features = list(features) + ["intercept"]
            coef_df = {"feature": features, "coefficient": coefficients}
            return method, coef_df
        elif method == "linear":
            explainer = shap.LinearExplainer(model, X_train.to_numpy())
            shap_values = explainer.shap_values(X_test.to_numpy())
            return method, shap_values
        return None, None

    def train_and_save_model(self, model_name: str, model_path: str | None = None) -> object:
        """
        Train the model on the entire dataset and save it.
        :param model_name: name of the model.
        :param model_path: path to saved model.
        :return: trained model.
        """
        proper_model_name, model, param_grid, f_importance_method = Models().get_model(model_name, model_path=model_path)

        if self.verbose:
            print(f"Training final model {proper_model_name} on the entire dataset")

        X_prepared = self._data_preparation(self.X, self.X)

        standard_scaler = StandardScaler()
        numerical_features = [f for f in self.feature_types["numerical"] if f in self.X.columns]
        standard_scaler.fit(self.X[numerical_features].values)
        model = self.tune_model(self.X, self.y, model, param_grid)

        model.fit(X_prepared.to_numpy(), self.y[self.y.columns[0]].to_numpy())

        f_imp = self.calculate_f_importance(model, f_importance_method, X_prepared, X_prepared)

        if len(self.save_dir) > 0:
            self.save_model(standard_scaler, fold_num="scaler_final")
            self.save_model(model, fold_num="final")
            f_importance_path = os.path.join(self.save_dir, f"feature_importance.pickle")
            with open(f_importance_path, "wb") as f:
                joblib.dump(f_imp, f)
            X_prepared.to_csv(os.path.join(self.save_dir, "data.csv"), index=False)

        return model

    def predict_model(self, model: object, X_predict: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        :param model: trained model.
        :param X_predict: data for prediction.
        :return: predicted values.
        """
        X_prepared = self._prepare_features_for_model(model, self.X, X_predict)
        y_pred = model.predict(X_prepared.to_numpy()).flatten()
        return y_pred
