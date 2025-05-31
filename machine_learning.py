# -*- coding: utf-8 -*-
"""
Machine Learning Module for FCGR Classification using TensorFlow/Keras.
Fixed version with proper tuple returns and no data leakage.
"""
import os
import logging
import numpy as np
import pandas as pd
import time
import io
import base64

# Import framework components and check flags
from .config import RANDOM_STATE, CNN_EPOCHS, CNN_BATCH_SIZE, TEST_SIZE, \
                    CNN_VALIDATION_SPLIT, CNN_EARLY_STOPPING_PATIENCE, EPSILON, \
                    PLOTTING_ENABLED, PLOT_SAVEFIG_DPI
from .utils import IS_PYODIDE, TENSORFLOW_AVAILABLE, safe_filename
from .analysis import SKLEARN_AVAILABLE

# Import analysis helpers for plotting conditionally
_save_plot_to_base64 = None
_save_plot_to_file = None
PLOTTING_LIBS_AVAILABLE = False
plt = None
sns = None

if PLOTTING_ENABLED:
    try:
        from .analysis import _save_plot_to_base64, _save_plot_to_file, PLOTTING_LIBS_AVAILABLE as ANALYSIS_PLOTTING_AVAILABLE
        if ANALYSIS_PLOTTING_AVAILABLE:
            import matplotlib.pyplot as plt_
            plt = plt_
            import seaborn as sns_
            sns = sns_
            PLOTTING_LIBS_AVAILABLE = True
            logging.debug("ML Plotting dependencies loaded.")
        else:
            logging.debug("ML Plots: analysis module reported plotting unavailable.")
    except ImportError:
        logging.debug("ML Plots: Failed to import plotting helpers from analysis module.")
        PLOTTING_LIBS_AVAILABLE = False
else:
    logging.debug("ML Plots: Plotting disabled globally.")

# --- Conditional TF/Sklearn Imports ---
tf = None; layers = None; models = None; utils = None; callbacks = None; AUTOTUNE = None
LabelEncoder = None; train_test_split = None; classification_report = None; confusion_matrix = None

if TENSORFLOW_AVAILABLE:
    try:
        import tensorflow as tf_
        tf = tf_
        from tensorflow.keras import layers as layers_
        layers = layers_
        from tensorflow.keras import models as models_
        models = models_
        from tensorflow.keras import utils as utils_
        utils = utils_
        from tensorflow.keras import callbacks as callbacks_
        callbacks = callbacks_
        AUTOTUNE = tf.data.AUTOTUNE
    except ImportError as e:
        logging.error(f"TensorFlow import failed even though TENSORFLOW_AVAILABLE=True: {e}")
        TENSORFLOW_AVAILABLE = False

if SKLEARN_AVAILABLE:
    try:
        from sklearn.model_selection import train_test_split as tts
        train_test_split = tts
        from sklearn.preprocessing import LabelEncoder as le
        LabelEncoder = le
        from sklearn.metrics import classification_report as cr
        classification_report = cr
        from sklearn.metrics import confusion_matrix as cm
        confusion_matrix = cm
    except ImportError as e:
        logging.error(f"Sklearn import failed even though SKLEARN_AVAILABLE=True: {e}")
        SKLEARN_AVAILABLE = False


def build_cnn_model(input_shape: tuple, num_classes: int) -> models.Sequential | None:
    """Builds the CNN model, guarded by dependency checks."""
    if not TENSORFLOW_AVAILABLE or not models or not layers:
        logging.error("Cannot build CNN model: TensorFlow/Keras components unavailable.")
        return None
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        logging.error(f"Invalid input_shape for CNN: {input_shape}.")
        return None
    if not isinstance(num_classes, int) or num_classes < 2:
        logging.error(f"Invalid num_classes for CNN: {num_classes}.")
        return None

    logging.info(f"Building CNN Model: Input={input_shape}, Classes={num_classes}")
    try:
        policy = tf.keras.mixed_precision.global_policy()
        use_mixed_precision = policy.name == 'mixed_float16'
    except Exception:
        use_mixed_precision = False

    final_activation = 'softmax'
    output_dtype = 'float32'

    try:
        model = models.Sequential(name=f"FCGR_CNN_{input_shape[0]}x{input_shape[1]}")
        model.add(layers.Input(shape=input_shape, name='input_fcgr'))
        
        # Architecture
        model.add(layers.Conv2D(32, (3, 3), padding='same', name='conv1a'))
        model.add(layers.BatchNormalization(name='bn1a'))
        model.add(layers.Activation('relu', name='act1a'))
        model.add(layers.Conv2D(32, (3, 3), padding='same', name='conv1b'))
        model.add(layers.BatchNormalization(name='bn1b'))
        model.add(layers.Activation('relu', name='act1b'))
        model.add(layers.MaxPooling2D((2, 2), name='pool1'))

        model.add(layers.Conv2D(64, (3, 3), padding='same', name='conv2a'))
        model.add(layers.BatchNormalization(name='bn2a'))
        model.add(layers.Activation('relu', name='act2a'))
        model.add(layers.Conv2D(64, (3, 3), padding='same', name='conv2b'))
        model.add(layers.BatchNormalization(name='bn2b'))
        model.add(layers.Activation('relu', name='act2b'))
        model.add(layers.MaxPooling2D((2, 2), name='pool2'))

        min_dim_after_pool2 = input_shape[0] // 4
        if min_dim_after_pool2 >= 8:
            model.add(layers.Conv2D(128, (3, 3), padding='same', name='conv3a'))
            model.add(layers.BatchNormalization(name='bn3a'))
            model.add(layers.Activation('relu', name='act3a'))
            model.add(layers.MaxPooling2D((2, 2), name='pool3'))

        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(128, name='dense1'))
        model.add(layers.BatchNormalization(name='bn_dense1'))
        model.add(layers.Activation('relu', name='act_dense1'))
        model.add(layers.Dropout(0.5, name='dropout_final'))
        model.add(layers.Dense(num_classes, activation=final_activation, name='output', dtype=output_dtype))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        if use_mixed_precision:
            try:
                optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            except Exception as lso_e:
                logging.warning(f"Failed LossScaleOptimizer: {lso_e}")

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        logging.info("CNN Model Summary:\n" + "\n".join(summary_list))
        return model

    except Exception as build_e:
        logging.error(f"Error during CNN model definition: {build_e}", exc_info=True)
        return None


def run_classification(X_data: np.ndarray, y_labels: np.ndarray, target_name: str,
                       figures_dir: str | None = None,
                       label_encoder=None,
                       output_format: str = 'file') -> tuple:
    """Runs classification with proper validation split and no data leakage."""
    plot_key_suffix = "_b64" if output_format == 'base64' else ""
    # Initialize results
    results = {
        'target': target_name, 'accuracy': None, 'loss': None, 'report': None,
        f'history_plot{plot_key_suffix}': None, f'cm_plot{plot_key_suffix}': None
    }
    hist_plot_key = f'history_plot{plot_key_suffix}'
    cm_plot_key = f'cm_plot{plot_key_suffix}'

    # --- Guard Clauses - ALWAYS RETURN TUPLE ---
    if IS_PYODIDE: 
        logging.warning(f"ML '{target_name}' skipped: Pyodide env.")
        return None, results
    if not TENSORFLOW_AVAILABLE or not SKLEARN_AVAILABLE: 
        logging.error(f"ML '{target_name}' skipped: Dep unavailable.")
        return None, results
    if not all([train_test_split, LabelEncoder, classification_report, confusion_matrix, utils, callbacks, AUTOTUNE, tf]):
        logging.error(f"ML '{target_name}' skipped: Core function unavailable.")
        return None, results
    if X_data is None or y_labels is None or len(X_data) == 0: 
        logging.error(f"ML '{target_name}' skipped: No data.")
        return None, results
    if len(X_data) != len(y_labels): 
        logging.error(f"ML '{target_name}' skipped: X/y len mismatch.")
        return None, results
    if X_data.ndim != 3: 
        logging.error(f"ML '{target_name}' skipped: X dim != 3.")
        return None, results

    logging.info(f"\n--- Running Classifier for Target: {target_name} ---")

    # --- Label Encoding ---
    try:
        if label_encoder is None: 
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_labels)
        else:
            known_labels = set(label_encoder.classes_)
            new_labels = set(y_labels) - known_labels
            if new_labels:
                logging.warning(f"Found new labels not in encoder: {new_labels}. Re-fitting encoder.")
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y_labels)
            else:
                y_encoded = label_encoder.transform(y_labels)
                
        num_classes = len(label_encoder.classes_)
        logging.info(f"Target: {target_name} - Classes: {num_classes} -> {list(label_encoder.classes_)}")
        
        unique, counts = np.unique(y_encoded, return_counts=True)
        logging.info(f"Class distribution: {dict(zip(unique, counts))}")
        
        if num_classes < 2: 
            logging.error(f"ML '{target_name}' skipped: Need >= 2 classes.")
            return label_encoder, results
    except Exception as e: 
        logging.error(f"Label encoding failed for '{target_name}': {e}")
        return label_encoder, results

    # --- Check sample size ---
    min_samples_per_class = 5
    min_total_samples = max(20, num_classes * min_samples_per_class)
        
    if len(X_data) < min_total_samples: 
        logging.error(f"ML '{target_name}' skipped: Samples {len(X_data)} < required {min_total_samples}.")
        return label_encoder, results

    # --- Data Splitting & TF Datasets ---
    try:
        X_cnn = np.expand_dims(X_data, axis=-1).astype(np.float32)
        input_shape = X_cnn.shape[1:]
        y_one_hot = utils.to_categorical(y_encoded, num_classes=num_classes)
        indices = np.arange(len(X_cnn))
        
        # Try stratified split, fall back to random if it fails
        try: 
            train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded)
        except ValueError as e: 
            logging.warning(f"Stratified split failed for '{target_name}': {e}. Using random split.")
            train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            
        X_train, X_test = X_cnn[train_idx], X_cnn[test_idx]
        y_train, y_test = y_one_hot[train_idx], y_one_hot[test_idx]
        y_test_encoded = y_encoded[test_idx]  # Keep encoded labels for confusion matrix
        
        logging.info(f"Data split for '{target_name}': Train={len(X_train)}, Test={len(X_test)}")
        
        # Debug: Check train/test class distribution
        train_classes, train_counts = np.unique(y_encoded[train_idx], return_counts=True)
        test_classes, test_counts = np.unique(y_encoded[test_idx], return_counts=True)
        logging.info(f"Train class distribution: {dict(zip(train_classes, train_counts))}")
        logging.info(f"Test class distribution: {dict(zip(test_classes, test_counts))}")

        # FIXED: Proper validation split without data leakage
        val_size = int(len(X_train) * CNN_VALIDATION_SPLIT)
        use_validation = val_size >= 5 and val_size < len(X_train) * 0.5
        early_stopping = None
        val_ds = None
        
        if use_validation:
            # Create validation split from training data only
            train_indices = np.arange(len(X_train))
            np.random.seed(RANDOM_STATE)
            np.random.shuffle(train_indices)
            
            val_indices = train_indices[:val_size]
            train_indices = train_indices[val_size:]
            
            X_val = X_train[val_indices]
            y_val = y_train[val_indices]
            X_train_final = X_train[train_indices]
            y_train_final = y_train[train_indices]
            
            # Create datasets
            train_ds = tf.data.Dataset.from_tensor_slices((X_train_final, y_train_final))
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            
            # Configure datasets
            train_ds = train_ds.shuffle(len(X_train_final), seed=RANDOM_STATE).batch(CNN_BATCH_SIZE).cache().prefetch(AUTOTUNE)
            val_ds = val_ds.batch(CNN_BATCH_SIZE).cache().prefetch(AUTOTUNE)
            
            # Enable early stopping with validation data
            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=CNN_EARLY_STOPPING_PATIENCE, verbose=1, restore_best_weights=True)
            
            logging.info(f"Proper split: Train={len(X_train_final)}, Val={len(X_val)}, Test={len(X_test)}")
        else:
            # No validation set - train on all training data
            logging.warning(f"Validation set too small ({val_size}). Training without validation split.")
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            
            train_ds = train_ds.shuffle(len(X_train), seed=RANDOM_STATE).batch(CNN_BATCH_SIZE).cache().prefetch(AUTOTUNE)
            
        test_ds = test_ds.batch(CNN_BATCH_SIZE).prefetch(AUTOTUNE)
        logging.info(f"tf.data pipelines ready.")
        
    except Exception as data_prep_e: 
        logging.error(f"Data prep failed for '{target_name}': {data_prep_e}", exc_info=True)
        return label_encoder, results

    # --- Build and Train Model ---
    model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)
    if model is None: 
        logging.error(f"Model building failed for '{target_name}'.")
        return label_encoder, results
        
    history = None
    try:
        logging.info(f"Starting CNN training for '{target_name}'...")
        start_time = time.time()
        
        # Train with or without validation
        if use_validation and val_ds is not None:
            history = model.fit(
                train_ds, 
                epochs=CNN_EPOCHS, 
                validation_data=val_ds, 
                callbacks=[early_stopping], 
                verbose=2
            )
        else:
            history = model.fit(
                train_ds, 
                epochs=CNN_EPOCHS, 
                verbose=2
            )
        
        training_time = time.time() - start_time
        logging.info(f"CNN training finished for '{target_name}' in {training_time:.2f}s.")
    except Exception as train_e: 
        logging.error(f"CNN training failed for '{target_name}': {train_e}", exc_info=True)
        return label_encoder, results

    # --- Evaluate Model ---
    try:
        logging.info(f"Evaluating CNN model on test set for '{target_name}'...")
        loss, accuracy = model.evaluate(test_ds, verbose=0)
        results['accuracy'] = float(accuracy) if np.isfinite(accuracy) else None
        results['loss'] = float(loss) if np.isfinite(loss) else None
        print(f"\nCNN Eval Results ({target_name}): Loss={results['loss']:.4f}, Acc={results['accuracy']:.4f}")
    except Exception as eval_e: 
        logging.error(f"Model evaluation failed for '{target_name}': {eval_e}", exc_info=True)

    # --- Generate Predictions and Report ---
    y_pred_indices = None
    y_test_indices = None
    report_failed = False
    
    try:
        y_pred_proba = model.predict(test_ds)
        y_pred_indices = np.argmax(y_pred_proba, axis=1)
        
        # Use the stored test indices
        y_test_indices = y_test_encoded
        
        logging.info(f"Prediction shape: {y_pred_indices.shape}, Test labels shape: {y_test_indices.shape}")

        if len(y_pred_indices) != len(y_test_indices):
            logging.error(f"Length mismatch predicted ({len(y_pred_indices)}) vs true ({len(y_test_indices)}) labels for '{target_name}'. Skipping report/CM.")
            report_failed = True
        else:
            # Debug predictions
            unique_pred, pred_counts = np.unique(y_pred_indices, return_counts=True)
            logging.info(f"Predicted class distribution: {dict(zip(unique_pred, pred_counts))}")
            
            print(f"\nClassification Report ({target_name}):")
            results['report'] = classification_report(y_test_indices, y_pred_indices, labels=range(num_classes), target_names=label_encoder.classes_, zero_division=0, output_dict=True)
            print(classification_report(y_test_indices, y_pred_indices, labels=range(num_classes), target_names=label_encoder.classes_, zero_division=0))
    except Exception as report_e:
        logging.error(f"Prediction or report generation failed for '{target_name}': {report_e}", exc_info=True)
        report_failed = True
        results['report'] = None

    # --- Plotting (Conditional) ---
    plot_cm = not report_failed

    if PLOTTING_ENABLED and PLOTTING_LIBS_AVAILABLE and plt and sns and _save_plot_to_base64 and _save_plot_to_file:
        # Plot History
        fig_hist = None
        if history and history.history and 'loss' in history.history and len(history.history['loss']) > 1:
            try:
                fig_hist = plt.figure(figsize=(14, 6))
                
                # Plot Accuracy
                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
                if 'val_accuracy' in history.history:
                    plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Accuracy', fontsize=12)
                min_acc = min(min(history.history.get('accuracy',[0])), 
                             min(history.history.get('val_accuracy', history.history.get('accuracy',[0]))))
                plt.ylim([max(0, min_acc - 0.1), 1.05])
                plt.legend(loc='best', fontsize=11)
                plt.title('Model Accuracy', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.6)
                
                # Plot Loss
                plt.subplot(1, 2, 2)
                plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
                if 'val_loss' in history.history:
                    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.legend(loc='best', fontsize=11)
                plt.title('Model Loss', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.6)
                
                plt.suptitle(f'CNN Training History ({target_name})', fontsize=16, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                # Save
                if output_format == 'base64': 
                    results[hist_plot_key] = _save_plot_to_base64(fig_hist, dpi=100)
                elif output_format == 'file' and figures_dir: 
                    hist_fname = safe_filename(f"cnn_history_{target_name}") + ".png"
                    hist_fpath = os.path.join(figures_dir, hist_fname)
                    success = _save_plot_to_file(fig_hist, hist_fpath, dpi=PLOT_SAVEFIG_DPI)
                    results['history_plot'] = os.path.join(os.path.basename(figures_dir), hist_fname) if success else None
                else: 
                    plt.close(fig_hist)
                fig_hist = None
            except Exception as e: 
                logging.error(f"Failed ML history plot for '{target_name}': {e}", exc_info=False)
            finally:
                if fig_hist: 
                    plt.close(fig_hist)
        else: 
            logging.warning(f"Skipping ML history plot for '{target_name}': Insufficient data.")

        # Plot Confusion Matrix
        fig_cm = None
        if plot_cm:
            try:
                fig_cm = plt.figure(figsize=(max(10, num_classes * 0.8), max(8, num_classes * 0.7)))
                cm_data = confusion_matrix(y_test_indices, y_pred_indices, labels=range(num_classes))
                
                # Normalize confusion matrix by rows (true labels)
                cm_normalized = cm_data.astype('float') / cm_data.sum(axis=1)[:, np.newaxis]
                
                sns.heatmap(cm_normalized, annot=cm_data, fmt='d', cmap='Blues', 
                           xticklabels=label_encoder.classes_, 
                           yticklabels=label_encoder.classes_, 
                           annot_kws={"size": 10})
                plt.xlabel('Predicted Label', fontsize=14)
                plt.ylabel('True Label', fontsize=14)
                plt.title(f'Confusion Matrix ({target_name})', fontsize=16, fontweight='bold')
                plt.xticks(rotation=45, ha='right', fontsize=11)
                plt.yticks(rotation=0, fontsize=11)
                plt.tight_layout()
                
                # Save
                if output_format == 'base64': 
                    results[cm_plot_key] = _save_plot_to_base64(fig_cm, dpi=100)
                elif output_format == 'file' and figures_dir: 
                    cm_fname = safe_filename(f"cnn_cm_{target_name}") + ".png"
                    cm_fpath = os.path.join(figures_dir, cm_fname)
                    success = _save_plot_to_file(fig_cm, cm_fpath, dpi=PLOT_SAVEFIG_DPI)
                    results['cm_plot'] = os.path.join(os.path.basename(figures_dir), cm_fname) if success else None
                else: 
                    plt.close(fig_cm)
                fig_cm = None
            except Exception as e: 
                logging.error(f"Failed CM plot for '{target_name}': {e}", exc_info=False)
            finally:
                if fig_cm: 
                    plt.close(fig_cm)
        else: 
            logging.warning(f"Skipping CM plot for '{target_name}': Report failed.")

    # Return results with potentially adjusted keys
    final_results = { 
        f'history_plot{plot_key_suffix}': results.get(hist_plot_key), 
        f'cm_plot{plot_key_suffix}': results.get(cm_plot_key) 
    }
    final_results.update({k:v for k,v in results.items() if not k.startswith(('history_plot','cm_plot'))})

    return label_encoder, final_results