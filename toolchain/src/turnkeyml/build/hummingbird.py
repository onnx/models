import os
import onnx
import onnx.tools.update_model_dims
import numpy as np

import turnkeyml.common.build as build
import turnkeyml.build.stage as stage
import turnkeyml.build.export as export
import turnkeyml.common.exceptions as exp
import turnkeyml.common.filesystem as fs

try:
    # An initial selection of Hummingbird-supported models.
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import IsolationForest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import SGDClassifier
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    sklearn_available = True
except ImportError as e:
    sklearn_available = False

try:
    from xgboost import XGBClassifier
    from xgboost import XGBRegressor

    xgboost_available = True
except ImportError as e:
    xgboost_available = False

try:
    from lightgbm import LGBMClassifier
    from lightgbm import LGBMRegressor

    lightgbm_available = True
except ImportError as e:
    lightgbm_available = False


def is_supported_sklearn_model(model) -> bool:
    return (
        isinstance(model, ExtraTreesClassifier)
        or isinstance(model, GradientBoostingClassifier)
        or isinstance(model, IsolationForest)
        or isinstance(model, RandomForestClassifier)
        or isinstance(model, RandomForestRegressor)
        or isinstance(model, SGDClassifier)
        or isinstance(model, BernoulliNB)
        or isinstance(model, GaussianNB)
        or isinstance(model, MultinomialNB)
        or isinstance(model, KNeighborsClassifier)
        or isinstance(model, MLPClassifier)
        or isinstance(model, Pipeline)
        or isinstance(model, StandardScaler)
        or isinstance(model, LinearSVC)
        or isinstance(model, DecisionTreeClassifier)
    )


def is_supported_xgboost_model(model) -> bool:
    return isinstance(model, XGBClassifier) or isinstance(model, XGBRegressor)


def is_supported_lightgbm_model(model) -> bool:
    return isinstance(model, LGBMClassifier) or isinstance(model, LGBMRegressor)


def is_supported_model(model) -> bool:
    return (
        (sklearn_available and is_supported_sklearn_model(model))
        or (xgboost_available and is_supported_xgboost_model(model))
        or (lightgbm_available and is_supported_lightgbm_model(model))
    )


class ConvertHummingbirdModel(stage.Stage):
    """
    Stage that takes an SKLearn, XGBoost, or LightGBM model instance, in state.model, and
    converts it to an ONNX file via Hummingbird.

    Expected inputs:
     - state.model is an SKLearn, XGBoost, or LightGBM model object
     - state.inputs is a dict of the form {"input_0": <data>} where <data>
        is a numpy array as would be provided, e.g., to sklearn's predict method.

    Outputs:
     - A *-base.onnx file that implements state.model given state.inputs
    """

    def __init__(self):
        super().__init__(
            unique_name="hummingbird_conversion",
            monitor_message="Converting model to ONNX with Hummingbird",
        )

    def fire(self, state: build.State):
        # TODO: Temporarily inlined to avoid warning message in hummingbird-ml<=0.46.
        import hummingbird.ml  # pylint: disable=import-error
        from hummingbird.ml.exceptions import (  # pylint: disable=import-error
            ConstantError,
            MissingConverter,
            MissingBackend,
        )

        if not is_supported_model(state.model):
            msg = f"""
            The current stage (ConvertHummingbirdModel) is only compatible with
            certain scikit-learn, xgboost, and lightgbm models, however the stage
            received an unsupported model of type {type(state.model)}.

            Support scikit-learn models:
              - sklearn.ensemble.ExtraTreesClassifier
              - sklearn.ensemble.GradientBoostingClassifier
              - sklearn.ensemble.IsolationForest
              - sklearn.ensemble.RandomForestClassifier
              - sklearn.ensemble.RandomForestRegressor
              - sklearn.linear_model.SGDClassifier
              - sklearn.naive_bayes.BernoulliNB
              - sklearn.naive_bayes.GaussianNB
              - sklearn.naive_bayes.MultinomialNB
              - sklearn.neighbors.KNeighborsClassifier
              - sklearn.neural_network.MLPClassifier
              - sklearn.pipeline.Pipeline
              - sklearn.preprocessing.StandardScaler
              - sklearn.svm.LinearSVC
              - sklearn.tree.DecisionTreeClassifier

            Supported xgboost models:
              - xgboost.XGBClassifier
              - xgboost.XGBRegressor

            Supported lightgbm models:
              - lightgbm.LGBMClassifier
              - lightgbm.LGBMRegressor
            """
            raise exp.StageError(msg)

        # TODO: By default the strategy will be chosen wih Hummingbird's logic.
        # Ideally, this would also be a parameter.
        tree_implementation_strategy = "gemm"  # or "tree_trav" or "perf_tree_trav"

        inputs = state.inputs
        if inputs is None:
            raise exp.StageError(
                "Hummingbird conversion requires inputs to be provided,"
                " however `inputs` is None."
            )
        test_X = inputs["input_0"]
        batch_size = test_X.shape[0]
        if test_X.dtype == np.float64:
            raise exp.StageError(
                "Fitting a model with float64 inputs can cause issues"
                " with conversion and compilation. This can be corrected by changing"
                " code like model.fit(X, y) to model.fit(X.astype(numpy.float32), y)."
            )

        extra_config = {
            "onnx_target_opset": state.config.onnx_opset,
            "tree_implementation": tree_implementation_strategy,
            "batch_size": batch_size,
        }

        try:
            onnx_model = hummingbird.ml.convert(
                state.model, "onnx", test_X, extra_config=extra_config
            ).model
        except (
            RuntimeError,
            IndexError,
            ValueError,
            ConstantError,
            MissingConverter,
            MissingBackend,
        ) as e:
            raise exp.StageError(f"Hummingbird conversion failed with error: {e}")

        input_dims = {
            "input_0": [
                batch_size,
                onnx_model.graph.input[0].type.tensor_type.shape.dim[1].dim_value,
            ]
        }
        if len(onnx_model.graph.output) > 1:
            output_dims = {
                "variable": [batch_size],
                onnx_model.graph.output[1].name: [
                    batch_size,
                    onnx_model.graph.output[1].type.tensor_type.shape.dim[1].dim_value,
                ],
            }
        else:
            output_dims = {"variable": [batch_size]}

        # Concretize symbolic shape parameter
        onnx_model = onnx.tools.update_model_dims.update_inputs_outputs_dims(
            onnx_model, input_dims, output_dims
        )

        # Save output node names
        state.expected_output_names = export.get_output_names(onnx_model)

        output_path = export.base_onnx_file(state)
        os.makedirs(export.onnx_dir(state))
        onnx.save(onnx_model, output_path)

        np.save(state.original_inputs_file, state.inputs)

        state.intermediate_results = [output_path]
        stats = fs.Stats(state.cache_dir, state.config.build_name)
        stats.add_sub_stat(
            state.stats_id,
            fs.Keys.ONNX_FILE,
            output_path,
        )

        return state
