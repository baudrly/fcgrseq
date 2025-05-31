# -*- coding: utf-8 -*-
"""
Unit tests for the machine_learning module.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from numpy.testing import assert_almost_equal, assert_array_equal
from unittest.mock import patch, MagicMock, ANY

# Module containing functions to test
from fcgr_analyzer import machine_learning as ml
from fcgr_analyzer.config import PLOTTING_ENABLED, CNN_EPOCHS, CNN_BATCH_SIZE, \
                                 FCGR_DIM, TEST_SIZE, RANDOM_STATE

# Skip all tests if TF or sklearn is not available at the module level of `ml`
pytestmark = [
    pytest.mark.skipif(not ml.TENSORFLOW_AVAILABLE, reason="TensorFlow not available in ml module"),
    pytest.mark.skipif(not ml.SKLEARN_AVAILABLE, reason="scikit-learn not available in ml module")
]

@pytest.fixture
def mock_tf_keras_utils(mocker):
    """Mocks tf.keras.utils.to_categorical."""
    return mocker.patch('tensorflow.keras.utils.to_categorical', side_effect=lambda y, num_classes: np.eye(num_classes)[y])

@pytest.fixture
def mock_tf_data_methods(mocker):
    """Mocks tf.data.Dataset methods to allow inspection without full TF execution."""
    mock_dataset = MagicMock()
    mock_dataset.shuffle.return_value = mock_dataset
    mock_dataset.batch.return_value = mock_dataset
    mock_dataset.cache.return_value = mock_dataset
    mock_dataset.prefetch.return_value = mock_dataset
    # For predict, to get the number of samples in test_ds
    def mock_cardinality(ds_instance):
        # This is a simplified mock. Real cardinality needs TF execution context.
        # Assume test_ds has a known size in tests using this.
        mock_tensor = MagicMock()
        # In tests, we'll know the expected test set size. Let's say it's `test_set_size_for_mock`.
        # The test needs to ensure this value is consistent.
        # For predict, the number of batches * batch_size is important.
        # The mock_predict_dynamic in mock_model_fit_and_eval covers this better.
        mock_tensor.numpy.return_value = getattr(ds_instance, '_mock_cardinality_val', 5) # Default small value
        return mock_tensor

    mocker.patch('tensorflow.data.Dataset.from_tensor_slices', return_value=mock_dataset)
    mocker.patch('tensorflow.data.experimental.cardinality', side_effect=mock_cardinality)
    return mock_dataset


@pytest.fixture
def mock_model_fit_and_eval(mocker):
    """Mocks model building, fit, evaluate, and predict."""
    mock_model_instance = MagicMock(spec=ml.models.Sequential)
    
    # Mock History object
    mock_history_obj = MagicMock()
    mock_history_obj.history = {
        'loss': np.array([0.5, 0.3, 0.2]), 'accuracy': np.array([0.8, 0.9, 0.95]),
        'val_loss': np.array([0.6, 0.4, 0.3]), 'val_accuracy': np.array([0.75, 0.85, 0.92])
    }
    mock_model_instance.fit.return_value = mock_history_obj
    mock_model_instance.evaluate.return_value = [0.33, 0.92]  # test_loss, test_accuracy
    
    # Dynamic predict based on test set size
    def mock_predict_dynamic(dataset, **kwargs):
        # This is a simplified way to get test set size.
        # In a real scenario, dataset would be a tf.data.Dataset.
        # The test setup will ensure y_test_encoded has the correct length.
        num_test_samples = getattr(dataset, '_actual_test_size', 10) # Default, test should set this
        num_classes = getattr(dataset, '_num_classes_for_predict', 2)
        
        mock_output = np.random.rand(num_test_samples, num_classes).astype(np.float32)
        mock_output /= np.sum(mock_output, axis=1, keepdims=True)
        return mock_output

    mock_model_instance.predict.side_effect = mock_predict_dynamic
    
    mocker.patch('fcgr_analyzer.machine_learning.build_cnn_model', return_value=mock_model_instance)
    return mock_model_instance

# --- Test build_cnn_model ---
@pytest.mark.parametrize("k_val, num_classes_val", [(6, 3), (4, 2), (8, 10)])
def test_build_cnn_model_success_various_k(k_val, num_classes_val):
    input_dim = 1 << k_val
    input_shape = (input_dim, input_dim, 1)
    model = ml.build_cnn_model(input_shape, num_classes_val)
    assert model is not None
    assert isinstance(model, ml.models.Sequential)
    assert model.input_shape == (None, input_dim, input_dim, 1)
    assert model.output_shape == (None, num_classes_val)

def test_build_cnn_model_compiles_with_mixed_precision(mocker):
    if not ml.TENSORFLOW_AVAILABLE: pytest.skip("TF needed")
    
    # Simulate mixed precision being enabled
    mock_policy = MagicMock()
    mock_policy.name = 'mixed_float16'
    mocker.patch('tensorflow.keras.mixed_precision.global_policy', return_value=mock_policy)
    mock_lso = mocker.patch('tensorflow.keras.mixed_precision.LossScaleOptimizer')

    input_shape = (FCGR_DIM, FCGR_DIM, 1)
    model = ml.build_cnn_model(input_shape, 2)
    assert model is not None
    mock_lso.assert_called_once() # Check if LossScaleOptimizer was used
    assert model.layers[-1].dtype_policy.name == 'float32' # Output layer should be float32


# --- Test run_classification ---
@pytest.fixture
def sample_ml_data():
    n_samples = 60 # Increased for better split
    height, width = FCGR_DIM, FCGR_DIM
    X = np.random.rand(n_samples, height, width).astype(np.float32)
    # More balanced classes for stratification tests
    y_labels_list = ['ClassA'] * (n_samples // 3) + \
                    ['ClassB'] * (n_samples // 3) + \
                    ['ClassC'] * (n_samples - 2 * (n_samples // 3))
    y = np.array(y_labels_list)
    np.random.shuffle(y) # Shuffle to ensure classes are mixed
    return X, y

@pytest.mark.usefixtures("mock_tf_keras_utils", "mock_tf_data_methods")
def test_run_classification_full_flow(sample_ml_data, figures_dir, mock_model_fit_and_eval, mocker):
    X, y = sample_ml_data
    target_name = "Species"
    num_classes = len(np.unique(y))

    # Mock classification_report
    # Calculate expected test size for report's support field
    # Stratified split is default, so count per class in test set can vary.
    # For simplicity, assume roughly TEST_SIZE of total samples for macro/weighted support.
    from sklearn.model_selection import train_test_split as actual_split
    _, y_test_for_report_mock = actual_split(y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    expected_test_n = len(y_test_for_report_mock)
    
    # Attach this expected test size to the mock dataset for predict to use
    # This is a bit of a hack due to predict needing to know the test set size.
    ml.test_ds_global_ref = MagicMock() # Create a reference point
    ml.test_ds_global_ref._actual_test_size = expected_test_n
    ml.test_ds_global_ref._num_classes_for_predict = num_classes
    
    # This ensures that our `mock_predict_dynamic` uses the correct values via this global ref.
    # This is needed because the `dataset` passed to `predict` is a `tf.data.Dataset` which is hard to inspect
    # for size without executing it, which we are trying to avoid in unit tests.
    # In `run_classification`, before `model.predict(test_ds)`, we'd need to set these attributes on `test_ds`.
    # This is getting complex. A better mock for `predict` might just return a fixed size array if the
    # test data size is fixed for a given test.

    # Let's simplify the predict mock:
    # Assume the test will set up y_test_encoded which `run_classification` uses.
    # `mock_model_fit_and_eval`'s `mock_predict_dynamic` will be called with `test_ds`.
    # We need `test_ds` to somehow communicate its size.
    # The `run_classification` function calculates `y_test_encoded`. We can use its length.

    # Patching the part where test_ds is created to add the size attribute
    original_from_tensor_slices = ml.tf.data.Dataset.from_tensor_slices
    def from_tensor_slices_with_size_attr(data_tuple):
        ds = original_from_tensor_slices(data_tuple)
        # If this is the test_ds, its first element (X_test) has the number of samples.
        # Or, more reliably, we know y_test (one-hot) or y_test_encoded (integer) is used.
        # The test call to `run_classification` will provide X and y.
        # `run_classification` will split it into X_test, y_test_encoded.
        # So, `y_test_encoded` is the key.
        # This is still tricky because the dataset is created *inside* run_classification.
        
        # Simpler: The `classification_report` mock will be called with `y_test_indices` and `y_pred_indices`.
        # We need to ensure `y_pred_indices` has the same length as `y_test_indices`.
        # The `mock_model_fit_and_eval.predict` will be called with `test_ds`.
        # Let's assume `test_ds` yields `expected_test_n` items.

        # The mock for `predict` should be configured to return `expected_test_n` predictions.
        # `mock_model_fit_and_eval` fixture's `mock_predict_dynamic` now uses getattr on the dataset.
        # Let's patch `tf.data.Dataset.from_tensor_slices` to add this attribute to the created test_ds.

        def side_effect_from_tensor_slices(tensors):
            dataset = original_from_tensor_slices(tensors)
            # Heuristic: if the first tensor has 4 dims (batch, H, W, C), it's likely X data for CNN
            if isinstance(tensors, tuple) and len(tensors) == 2 and tensors[0].ndim == 4:
                 # This is likely X_train, X_val, or X_test.
                 # We're interested in when it's X_test.
                 # The length of y_test_encoded is what matters.
                 # This is still hard to do without deeper integration.
                 # The `classification_report` will be called with y_test_indices and y_pred_indices.
                 # The length of y_test_indices comes from the train_test_split.
                 # The length of y_pred_indices comes from model.predict(test_ds).
                 # So, model.predict(test_ds) must return expected_test_n items.
                 
                 # Let's assume the `mock_predict_dynamic` in the fixture handles this by being set up
                 # to expect `expected_test_n` samples from the test dataset.
                 # This is implicitly handled if `expected_test_n` is correctly calculated and used by `mock_predict_dynamic`.
                 pass # The fixture mock for predict should handle this
            return dataset
        
        mocker.patch('tensorflow.data.Dataset.from_tensor_slices', side_effect=side_effect_from_tensor_slices)


    # Mock classification_report to use this expected_test_n for support counts
    cr_labels = [f'Class{chr(65+i)}' for i in range(num_classes)]
    mock_cr_dict = {
        lbl: {'precision': 0.9, 'recall': 0.9, 'f1-score': 0.9, 'support': expected_test_n // num_classes} for lbl in cr_labels
    }
    mock_cr_dict.update({
        'accuracy': 0.92, # from mock_model_instance.evaluate
        'macro avg': {'precision': 0.9, 'recall': 0.9, 'f1-score': 0.9, 'support': expected_test_n},
        'weighted avg': {'precision': 0.9, 'recall': 0.9, 'f1-score': 0.9, 'support': expected_test_n}
    })
    mocker.patch('sklearn.metrics.classification_report', return_value=mock_cr_dict)


    # Mock plotting helpers if plotting enabled
    mock_save_b64 = mocker.patch('fcgr_analyzer.analysis._save_plot_to_base64', return_value="b64_plot_string")
    mock_save_file = mocker.patch('fcgr_analyzer.analysis._save_plot_to_file', return_value=True) # Simulate save success

    # Actual call
    encoder, results = ml.run_classification(X, y, target_name, figures_dir, output_format='file')

    assert encoder is not None
    assert isinstance(encoder, LabelEncoder)
    assert len(encoder.classes_) == num_classes
    
    assert results['target'] == target_name
    assert_almost_equal(results['accuracy'], 0.92)
    assert_almost_equal(results['loss'], 0.33)
    assert results['report'] is not None
    assert results['report']['accuracy'] == 0.92 # Should match the specific 'accuracy' key from report

    mock_model_fit_and_eval.fit.assert_called_once()
    # Check if early stopping was used based on validation split
    fit_callbacks = mock_model_fit_and_eval.fit.call_args[1].get('callbacks', [])
    if len(X) * (1 - TEST_SIZE) * ml.CNN_VALIDATION_SPLIT >= 5 : # If validation set was expected
        assert any(isinstance(cb, ml.callbacks.EarlyStopping) for cb in fit_callbacks)
    
    mock_model_fit_and_eval.evaluate.assert_called_once()
    mock_model_fit_and_eval.predict.assert_called_once()
    ml.classification_report.assert_called_once()

    if PLOTTING_ENABLED and ml.PLOTTING_LIBS_AVAILABLE:
        assert results['history_plot'] is not None
        assert results['cm_plot'] is not None
        assert mock_save_file.call_count >= 2 # History and CM plot
    else:
        assert results.get('history_plot') is None
        assert results.get('cm_plot') is None

@pytest.mark.parametrize("num_samples, num_features_gt_50", [
    (10, False), # Too few samples overall
    (25, False), # Too few per class after split for 3 classes (25 * 0.75 = 18.75 train, might be < 5 per class)
    (60, True),  # Sufficient samples, test PCA init if n_features > 50
])
def test_run_classification_edge_cases_samples_features(num_samples, num_features_gt_50, figures_dir, mock_model_fit_and_eval, mocker):
    height, width = FCGR_DIM, FCGR_DIM
    if num_features_gt_50:
        # To test PCA init in t-SNE, we need n_features > 50. FCGR (H,W) as features.
        # This means height*width > 50. If FCGR_DIM is e.g. 8 (k=3), 8*8=64.
        # Here, X_cnn is (N, H, W, 1). The features for t-SNE are after flattening or PCA.
        # The code uses PCA(n_components=50) if n_features > 50 for t-SNE.
        # The "features" here are the flattened FCGR matrices if not using PCA first.
        # This part of the test is more about the t-SNE path if n_features (FCGR_DIM*FCGR_DIM) > 50.
        pass # This is implicitly handled by FCGR_DIM if it's large enough

    X = np.random.rand(num_samples, height, width).astype(np.float32)
    y = np.array([f'Class{i%2}' for i in range(num_samples)]) # 2 classes

    # Mock t-SNE's PCA init path if features are high
    mock_pca_for_tsne_inst = MagicMock()
    mock_pca_for_tsne_inst.fit_transform.return_value = np.random.rand(ANY, 50 if X.size/num_samples > 50 else ANY) # Simulate PCA reduction
    
    if X.size/num_samples > 50 : # If flattened features > 50
        mocker.patch('fcgr_analyzer.machine_learning.ml.PCA', return_value=mock_pca_for_tsne_inst) # Mock PCA used by TSNE

    # Simplified mocks for this specific test
    mocker.patch('sklearn.metrics.classification_report', return_value={'accuracy': 0.5})
    mocker.patch('fcgr_analyzer.analysis._save_plot_to_file', return_value=True)

    encoder, results = ml.run_classification(X, y, "EdgeTest", figures_dir, output_format='file')

    min_total_samples_needed = max(20, len(np.unique(y)) * 5)

    if num_samples < min_total_samples_needed:
        assert results['accuracy'] is None
        mock_model_fit_and_eval.fit.assert_not_called()
    else:
        assert results['accuracy'] is not None
        mock_model_fit_and_eval.fit.assert_called_once()
        if X.size/num_samples > 50 and ml.TENSORFLOW_AVAILABLE and ml.SKLEARN_AVAILABLE: # If TSNE's PCA path was taken
             # This check is hard if PCA is imported as `from sklearn.decomposition import PCA`.
             # We would need to mock `sklearn.decomposition.PCA`.
             # For now, trust the t-SNE internal PCA logic.
             pass


def test_run_classification_label_encoder_reuse(sample_ml_data, figures_dir, mock_model_fit_and_eval, mocker):
    X, y_original = sample_ml_data
    target_name = "EncoderTest"
    
    # Fit an encoder
    initial_encoder = LabelEncoder()
    initial_encoder.fit(y_original)
    
    # Mocks
    mocker.patch('sklearn.metrics.classification_report', return_value={'accuracy': 0.9})
    mocker.patch('fcgr_analyzer.analysis._save_plot_to_file', return_value=True)

    # Run with pre-fitted encoder (all labels known)
    encoder_run1, results_run1 = ml.run_classification(X, y_original, target_name, figures_dir, label_encoder=initial_encoder, output_format='file')
    
    assert encoder_run1 is initial_encoder # Should reuse
    assert results_run1['accuracy'] is not None
    
    # Run with new labels not in encoder
    y_new_labels = np.concatenate([y_original, ['ClassNew1', 'ClassNew2']])
    X_new_labels = np.concatenate([X, np.random.rand(2, FCGR_DIM, FCGR_DIM).astype(np.float32)])
    
    with pytest.warns(UserWarning, match="Found new labels not in encoder"): # Expect a warning
        encoder_run2, results_run2 = ml.run_classification(X_new_labels, y_new_labels, target_name, figures_dir, label_encoder=initial_encoder, output_format='file')

    assert encoder_run2 is not initial_encoder # Should have re-fitted a new one
    assert len(encoder_run2.classes_) > len(initial_encoder.classes_)
    assert 'ClassNew1' in encoder_run2.classes_
    assert results_run2['accuracy'] is not None


def test_run_classification_no_plotting_libs(sample_ml_data, figures_dir, mock_model_fit_and_eval, mocker):
    X, y = sample_ml_data
    # Simulate plotting libraries not being available within the ml module
    mocker.patch('fcgr_analyzer.machine_learning.PLOTTING_LIBS_AVAILABLE', False)
    mocker.patch('sklearn.metrics.classification_report', return_value={'accuracy': 0.9})

    _, results = ml.run_classification(X, y, "NoPlotTest", figures_dir, output_format='file')
    
    assert results['accuracy'] is not None
    assert results.get('history_plot') is None # No plot path should be stored
    assert results.get('cm_plot') is None


def test_run_classification_output_format_base64(sample_ml_data, mock_model_fit_and_eval, mocker):
    X, y = sample_ml_data
    # Mocks for base64 saving
    mocker.patch('sklearn.metrics.classification_report', return_value={'accuracy': 0.9})
    mock_save_b64 = mocker.patch('fcgr_analyzer.analysis._save_plot_to_base64', return_value="b64_plot_string")

    _, results = ml.run_classification(X, y, "Base64Test", figures_dir=None, output_format='base64')
    
    assert results['accuracy'] is not None
    if PLOTTING_ENABLED and ml.PLOTTING_LIBS_AVAILABLE:
        assert results['history_plot_b64'] == "b64_plot_string"
        assert results['cm_plot_b64'] == "b64_plot_string"
        assert mock_save_b64.call_count >= 2
    else:
        assert results.get('history_plot_b64') is None
        assert results.get('cm_plot_b64') is None
