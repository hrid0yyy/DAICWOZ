
---

## **CNN.ipynb**

This notebook implements a CNN for binary classification of spectrograms. Below is a detailed breakdown of its components, design choices, and implementation details.

### **1. Imports and Setup**
- **Libraries**: Imports standard libraries for data handling (`os`, `numpy`, `pandas`), PyTorch (`torch`, `torch.nn`, `torch.optim`), and data loading (`Dataset`, `DataLoader` from `torch.utils.data`).
- **Purpose**: These libraries provide tools for file handling, tensor operations, neural network construction, optimization, and data loading.

### **2. Data Loading and Preprocessing**
- **Dataset**: Loads a CSV file (`dataset.csv`) containing labels and iterates over `.npy` files in the `Spectrograms` directory, each representing a spectrogram.
  - Spectrograms are loaded as NumPy arrays and stored in list `X`.
  - Labels are extracted from the `label` column of the CSV and stored in list `y`.
  - Both `X` and `y` are converted to NumPy arrays with `dtype=np.float32` for consistency with PyTorch’s tensor requirements.
- **Output**: Prints the number of loaded spectrograms and labels (1883 samples).
- **Splitting**: Uses `train_test_split` from `sklearn.model_selection` to split data into training (80%, 1506 samples) and test sets (20%, 377 samples) with a fixed `random_state=42` for reproducibility.
- **Tensor Conversion**:
  - Converts NumPy arrays to PyTorch tensors with `dtype=torch.float32`.
  - Moves tensors to the appropriate device (`cuda` if available, else `cpu`).
  - A warning is noted about using `torch.tensor` directly on tensors, recommending `detach().clone()` for better practice (though not critical here since inputs are NumPy arrays).
- **Dataset Class**:
  - Defines `SpectrogramDataset`, a custom PyTorch `Dataset` subclass.
  - Stores `X` (spectrograms) and `y` (labels) as tensors.
  - Implements `__len__` (returns dataset size) and `__getitem__` (returns a spectrogram-label pair for a given index).
- **DataLoader**:
  - Creates `train_loader` and `test_loader` with a batch size of 32.
  - `train_loader` shuffles data (`shuffle=True`) for better training, while `test_loader` does not (`shuffle=False`).

### **3. CNN Model Architecture**
- **Class**: `CNNClassifier` (inherits `nn.Module`).
- **Architecture**:
  - **Input**: Spectrograms with shape `(batch_size, 128, 251)`, interpreted as 2D images with 1 channel (height=128, width=251).
  - **Conv1**: `nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)` → Output: `(batch_size, 16, 128, 251)`.
    - Applies 16 filters of size 3x3, preserving spatial dimensions via padding.
  - **ReLU1**: Applies ReLU activation for non-linearity.
  - **Pool1**: `nn.MaxPool2d(kernel_size=2, stride=2)` → Output: `(batch_size, 16, 64, 125)`.
    - Reduces spatial dimensions by half.
  - **Conv2**: `nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)` → Output: `(batch_size, 32, 64, 125)`.
  - **ReLU2** and **Pool2**: Further reduces to `(batch_size, 32, 32, 62)`.
  - **Conv3**: `nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)` → Output: `(batch_size, 64, 32, 62)`.
  - **ReLU3** and **Pool3**: Reduces to `(batch_size, 64, 16, 31)`.
  - **Flatten**: Reshapes to `(batch_size, 64 * 16 * 31 = 31744)`.
  - **FC1**: `nn.Linear(31744, 128)` → Output: `(batch_size, 128)`.
  - **ReLU4** and **Dropout**: Applies ReLU and 50% dropout for regularization.
  - **FC2**: `nn.Linear(128, 1)` → Output: `(batch_size, 1)` for binary classification (logits).
- **Forward Pass**:
  - Adds a channel dimension to input: `(batch_size, 128, 251)` → `(batch_size, 1, 128, 251)`.
  - Applies convolutional, ReLU, and pooling layers sequentially.
  - Flattens the output and passes it through fully connected layers.
  - No sigmoid activation in `forward` (handled by loss function).

### **4. Training**
- **Setup**:
  - Uses `nn.BCEWithLogitsLoss` (combines sigmoid and binary cross-entropy for numerical stability).
  - Uses Adam optimizer with learning rate `lr=0.001`.
- **Training Function** (`train_model`):
  - Iterates over `num_epochs=10`.
  - For each batch:
    - Moves inputs and labels to the device.
    - Zeroes gradients, computes logits, calculates loss, backpropagates, and updates weights.
    - Computes predictions using `torch.sigmoid(outputs) > 0.5` for binary classification.
    - Tracks running loss and accuracy.
  - Prints epoch loss and accuracy.
- **Output**:
  - Loss decreases from 0.6885 (Epoch 1) to 0.5973 (Epoch 10).
  - Accuracy increases from 57.44% to 68.53%.
  - Test loss: 0.6348, Test accuracy: 67.64%.

### **5. Evaluation**
- **Evaluation Function** (`evaluate_model`):
  - Sets model to evaluation mode (`model.eval()`).
  - Computes test loss and accuracy using the same logic as training but without gradients.
- **Metrics Function** (`evaluate_metrics`):
  - Computes precision, recall, and F1 score using `sklearn.metrics`.
  - Predictions: `torch.sigmoid(outputs) > 0.5`.
  - Results:
    - Precision: 0.8219 (high, indicating low false positives).
    - Recall: 0.3550 (low, indicating many false negatives).
    - F1 Score: 0.4959 (moderate, balancing precision and recall).

### **6. Key Observations**
- **Model Fit**: The CNN achieves moderate accuracy (67.64%) but struggles with recall (0.3550), suggesting it misses many positive cases (likely imbalanced classes).
- **Architecture**: Typical CNN for image-like data (spectrograms), leveraging spatial hierarchies via convolutions and pooling.
- **Loss Function**: `BCEWithLogitsLoss` is appropriate for binary classification, avoiding explicit sigmoid in the model.
- **Potential Issues**:
  - Low recall suggests class imbalance or insufficient model capacity for complex patterns.
  - Warning about tensor conversion could be addressed by using `np.array(X)` before `torch.tensor`.
  - No data augmentation or preprocessing (e.g., normalization) mentioned, which could improve performance.

---

## **LSTM.ipynb**

This notebook implements an LSTM for classifying spectrograms, treating them as sequential data. Below is a detailed breakdown.

### **1. Imports and Setup**
- **Libraries**: Same as `CNN.ipynb` (`os`, `numpy`, `pandas`, PyTorch modules).
- **Purpose**: Identical to CNN for data handling and model building.

### **2. Data Loading and Preprocessing**
- **Dataset**: Similar to `CNN.ipynb`, loads spectrograms from `.npy` files and labels from `dataset.csv`.
  - Spectrograms stored in `X`, labels in `y` (no explicit `dtype` conversion to `float32` here, unlike CNN).
  - Prints 1883 spectrograms and labels.
- **Splitting**: Uses `train_test_split` with the same split (80% train: 1506 samples, 20% test: 377 samples, `random_state=42`).
- **Tensor Conversion**:
  - Converts `X_train`, `X_test` to `torch.float32` tensors and `y_train`, `y_test` to `torch.long` (for multi-class classification).
  - Warning about slow tensor creation from a list of NumPy arrays (same as CNN; could be optimized by converting `X` to a single NumPy array first).
  - Moves tensors to `cuda` or `cpu`.
- **Dataset Class**:
  - `SpectrogramDataset` is identical to CNN’s, but `y` is not explicitly converted to `torch.float32` (assumes `y` is already a tensor).
- **DataLoader**:
  - Same setup as CNN: batch size 32, shuffle for training, no shuffle for testing.

### **3. LSTM Model Architecture**
- **Class**: `LSTMClassifier` (inherits `nn.Module`).
- **Parameters**:
  - `input_size=251`: Number of frequency bins in each time step of the spectrogram.
  - `hidden_size=128`: Number of LSTM hidden units.
  - `num_layers=2`: Two stacked LSTM layers.
  - `num_classes=2`: For binary classification (though treated as multi-class here).
- **Architecture**:
  - **Input**: Spectrograms with shape `(batch_size, 128, 251)`, treated as sequences of length 128 (time steps) with 251 features per step.
  - **LSTM**: `nn.LSTM(input_size=251, hidden_size=128, num_layers=2, batch_first=True)`.
    - Processes the sequence, outputting `(batch_size, 128, 128)` (output for each time step).
  - **Dropout**: Applies 30% dropout for regularization.
  - **FC**: `nn.Linear(128, 2)` → Output: `(batch_size, 2)` for two classes.
- **Forward Pass**:
  - Initializes hidden and cell states (`h0`, `c0`) with zeros for each batch.
  - Passes input through LSTM, taking the output of the last time step: `(batch_size, 128)`.
  - Applies dropout and fully connected layer to produce logits for two classes.
- **Note**: Unlike CNN, no channel dimension is added; spectrograms are treated as time-series data.

### **4. Training**
- **Setup**:
  - Uses `nn.CrossEntropyLoss` (suitable for multi-class classification, expects `torch.long` labels).
  - Uses Adam optimizer with `lr=0.001`.
- **Training Loop**:
  - Iterates over `num_epochs=10`.
  - For each batch:
    - Zeroes gradients, computes logits, calculates loss, backpropagates, and updates weights.
    - Tracks running loss (no accuracy during training).
  - Prints epoch loss, decreasing from 0.6707 to 0.5226.
- **Evaluation**:
  - Computes test accuracy by taking the class with the highest logit (`torch.max(outputs, 1)`).
  - Test accuracy: 79.58%.

### **5. Evaluation and Metrics**
- **Evaluation**:
  - Similar to training loop but without gradients.
  - Computes accuracy (79.58%) and collects predictions and labels for metrics.
- **Metrics**:
  - Uses `sklearn.metrics` to compute precision, recall, and F1 score with `average='weighted'` (suitable for multi-class or imbalanced data).
  - Results:
    - Precision: 0.7929
    - Recall: 0.7958
    - F1 Score: 0.7942
- **Prediction Example**:
  - Loads a single spectrogram (`372_clip_0.npy`), converts to tensor, and predicts class (output: class 1).

### **6. Key Observations**
- **Model Fit**: The LSTM achieves higher test accuracy (79.58%) and better-balanced metrics (precision, recall, F1 ~0.79) than the CNN.
- **Architecture**: Treats spectrograms as sequences, leveraging LSTM’s ability to model temporal dependencies across the 128 time steps.
- **Loss Function**: `CrossEntropyLoss` assumes multi-class classification, which conflicts with the binary nature suggested by the dataset (labels are likely 0 or 1).
- **Potential Issues**:
  - Mismatch between `torch.long` labels and `num_classes=2` with `CrossEntropyLoss`. For binary classification, `BCEWithLogitsLoss` with a single output and sigmoid would be more appropriate.
  - No spectrogram preprocessing (e.g., normalization) mentioned.
  - Slow tensor creation warning (same as CNN).

---

## **Comparison of CNN and LSTM Notebooks**

### **1. Model Architecture**
- **CNN**:
  - Treats spectrograms as 2D images (1x128x251).
  - Uses convolutional layers to extract spatial features, followed by pooling to reduce dimensions.
  - Fully connected layers for classification.
  - Suited for capturing local patterns (e.g., frequency patterns in spectrograms).
- **LSTM**:
  - Treats spectrograms as sequences (128 time steps, 251 features each).
  - Uses LSTM to model temporal dependencies across time steps.
  - Takes the last time step’s output for classification.
  - Suited for sequential data where temporal relationships are critical.

### **2. Loss Function and Classification**
- **CNN**:
  - Uses `BCEWithLogitsLoss` for binary classification, outputting a single logit per sample.
  - Labels are `float32` (0.0 or 1.0), consistent with binary classification.
- **LSTM**:
  - Uses `CrossEntropyLoss` for multi-class classification, outputting two logits (for two classes).
  - Labels are `long` (0 or 1), which is inconsistent with binary classification; should use `BCEWithLogitsLoss` for consistency with the dataset.

### **3. Performance**
- **CNN**:
  - Test Accuracy: 67.64%
  - Precision: 0.8219, Recall: 0.3550, F1: 0.4959
  - Lower accuracy and poor recall suggest struggles with positive class detection (possibly due to class imbalance).
- **LSTM**:
  - Test Accuracy: 79.58%
  - Precision: 0.7929, Recall: 0.7958, F1: 0.7942
  - Better overall performance, with balanced metrics, indicating better generalization.

### **4. Data Handling**
- **CNN**:
  - Explicitly converts data to `float32` for both features and labels.
  - Adds channel dimension in the model for 2D convolution.
- **LSTM**:
  - Converts features to `float32` but labels to `long`, which is inconsistent with binary classification.
  - Treats spectrograms as sequences without channel dimension.

### **5. Potential Issues and Improvements**
- **Common Issues**:
  - **Data Preprocessing**: Neither notebook normalizes spectrograms (e.g., min-max or z-score normalization), which could improve model stability and convergence.
  - **Class Imbalance**: The CNN’s low recall suggests possible class imbalance, which neither notebook addresses (e.g., via class weights in the loss function or oversampling).
  - **Tensor Creation**: Both notebooks trigger a warning about slow tensor creation from lists of NumPy arrays. Convert `X` to a single NumPy array before tensor conversion:
    ```python
    X = np.stack(X)  # Instead of np.array(X, dtype=np.float32)
    X_train_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    ```
  - **Hyperparameter Tuning**: Both use default learning rates (`lr=0.001`) and batch sizes (32). Grid search or learning rate scheduling could improve performance.
- **CNN-Specific**:
  - Low recall suggests the model may need more capacity (e.g., deeper layers, more filters) or data augmentation (e.g., random crops, flips).
  - Consider adding batch normalization to stabilize training.
- **LSTM-Specific**:
  - Incorrect loss function (`CrossEntropyLoss` for binary classification). Replace with:
    ```python
    criterion = nn.BCEWithLogitsLoss()
    model.fc = nn.Linear(hidden_size, 1)  # Single output for binary
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)
    ```
  - LSTM may benefit from bidirectional processing (`bidirectional=True`) to capture dependencies in both directions.

### **6. Why LSTM Outperforms CNN**
- **Data Nature**: Spectrograms have both spatial and temporal components. The LSTM’s sequential modeling may better capture temporal patterns in the 128 time steps, while the CNN focuses on spatial patterns.
- **Feature Extraction**: LSTM processes the entire sequence, potentially retaining more temporal context, while CNN’s pooling layers may lose some information.
- **Metrics**: The LSTM’s balanced precision and recall suggest it handles the dataset’s classes better, possibly due to better generalization or less sensitivity to class imbalance.

---

## **Recommendations for Improvement**
1. **Data Preprocessing**:
   - Normalize spectrograms (e.g., `X = (X - X.mean()) / X.std()`).
   - Check for class imbalance in `dataset.csv` and apply class weights or oversampling if needed.
2. **Model Enhancements**:
   - **CNN**: Add batch normalization, increase model depth, or use residual connections (e.g., ResNet-like architecture).
   - **LSTM**: Use bidirectional LSTM, adjust `hidden_size` or `num_layers`, or try GRU for faster training.
   - Consider a hybrid CNN-LSTM model to capture both spatial and temporal features:
     - Use CNN to extract spatial features from spectrograms.
     - Feed CNN outputs to an LSTM for temporal modeling.
3. **Loss Function**:
   - For binary classification, both models should use `BCEWithLogitsLoss` with a single output and `float32` labels.
4. **Evaluation**:
   - Plot confusion matrices to analyze false positives/negatives.
   - Use cross-validation to ensure robust performance.
5. **Hyperparameter Tuning**:
   - Experiment with learning rates (e.g., 0.0001 to 0.01), batch sizes, and number of epochs.
   - Use learning rate schedulers (e.g., `torch.optim.lr_scheduler.ReduceLROnPlateau`).
6. **Data Augmentation**:
   - Apply time-frequency augmentations (e.g., time masking, frequency masking) using libraries like `torchaudio`.

---

## **Visualizing Performance**

To compare the performance of both models, here’s a bar chart showing their accuracy, precision, recall, and F1 score.

```chartjs
{
  "type": "bar",
  "data": {
    "labels": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "datasets": [
      {
        "label": "CNN",
        "data": [0.6764, 0.8219, 0.3550, 0.4959],
        "backgroundColor": "rgba(75, 192, 192, 0.6)",
        "borderColor": "rgba(75, 192, 192, 1)",
        "borderWidth": 1
      },
      {
        "label": "LSTM",
        "data": [0.7958, 0.7929, 0.7958, 0.7942],
        "backgroundColor": "rgba(255, 99, 132, 0.6)",
        "borderColor": "rgba(255, 99, 132, 1)",
        "borderWidth": 1
      }
    ]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": true,
        "max": 1.0,
        "title": {
          "display": true,
          "text": "Metric Value"
        }
      },
      "x": {
        "title": {
          "display": true,
          "text": "Metric"
        }
      }
    },
    "plugins": {
      "legend": {
        "display": true,
        "position": "top"
      },
      "title": {
        "display": true,
        "text": "CNN vs LSTM Performance Metrics"
      }
    }
  }
}
```

This chart clearly shows the LSTM outperforming the CNN across all metrics, particularly in recall and F1 score, highlighting its better handling of the classification task.

---

## **Conclusion**
- **CNN.ipynb**: Implements a CNN for binary classification, treating spectrograms as images. It achieves moderate accuracy but struggles with low recall, likely due to class imbalance or insufficient model capacity.
- **LSTM.ipynb**: Implements an LSTM, treating spectrograms as sequences. It achieves higher accuracy and balanced metrics but uses an incorrect loss function for binary classification.
- **Recommendation**: Fix the LSTM’s loss function to `BCEWithLogitsLoss`, preprocess data (normalization, class balancing), and consider a hybrid CNN-LSTM model for optimal performance. The LSTM’s superior performance suggests temporal modeling is more effective for this dataset, but both models could benefit from further tuning and preprocessing.