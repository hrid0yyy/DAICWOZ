Both documents (`CNN-DNN.ipynb` and `LSTM.ipynb`) implement deep learning models using PyTorch to classify spectrograms, likely for a binary classification task (e.g., detecting depression from audio spectrograms, as hinted in the code). The spectrograms are stored in `.npy` files in a directory called `Spectrograms`, and the labels are provided in a `dataset.csv` file with columns `id` and `label`. Below, I’ll provide an in-depth explanation of each file, focusing on their structure, implementation details, differences, and potential improvements, assuming you’re familiar with PyTorch, CNNs, LSTMs, and DNNs.

---

## **1. CNN-DNN.ipynb**

This notebook implements a hybrid **Convolutional Neural Network (CNN)** followed by a **Dense Neural Network (DNN)** for classifying spectrograms. The model processes spectrograms as 2D images (with a single channel) and uses convolutional layers to extract spatial features, followed by fully connected layers for classification.

### **Key Components**

#### **1.1. Imports and Setup**
- **Libraries**: The notebook uses `os`, `numpy`, `pandas`, `torch`, `torch.nn`, `torch.optim`, and `torch.utils.data` for handling file operations, data manipulation, and PyTorch-based model building.
- **Purpose**: These libraries support loading spectrograms, creating datasets, defining the model, and training it.

#### **1.2. SpectrogramDataset Class**
- **Definition**:
  ```python
  class SpectrogramDataset(Dataset):
      def __init__(self, spectrogram_folder, label_csv):
          self.spectrogram_folder = spectrogram_folder
          self.df = pd.read_csv(label_csv)
          self.ids = self.df['id'].astype(str).tolist()
          self.labels = self.df['label'].tolist()
  ```
- **Purpose**: A custom PyTorch `Dataset` class to load spectrograms (`.npy` files) and their corresponding labels from a CSV file.
- **Details**:
  - The `label_csv` file contains two columns: `id` (matching spectrogram filenames) and `label` (binary, e.g., 0 for "Not Depressed" and 1 for "Depressed").
  - `__getitem__(idx)` loads a spectrogram file (`{id}.npy`) from `spectrogram_folder`, adds a channel dimension to make it `(1, n_mels, time_steps)` (e.g., `(1, 128, 251)`), and converts it to a `torch.float32` tensor.
  - Labels are converted to `torch.long` for compatibility with `CrossEntropyLoss`.
- **Shape**: Assumes spectrograms are 2D arrays of shape `(n_mels, time_steps)`, e.g., `(128, 251)`, where `n_mels` is the number of mel frequency bins and `time_steps` is the temporal dimension.

#### **1.3. Dataset Splitting and DataLoader**
- **Splitting**:
  ```python
  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
  ```
  - The dataset is split into 80% training and 20% testing using `random_split`.
- **DataLoader**:
  ```python
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
  ```
  - Creates `DataLoader` objects for batching, with a batch size of 32. Shuffling is enabled for training to improve generalization but disabled for testing to maintain consistency.

#### **1.4. CNNDNN Model**
- **Definition**:
  ```python
  class CNNDNN(nn.Module):
      def __init__(self):
          super(CNNDNN, self).__init__()
          self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
          self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
          self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
          self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
          self.relu = nn.ReLU()
          self.batchnorm1 = nn.BatchNorm2d(32)
          self.batchnorm2 = nn.BatchNorm2d(64)
          self.batchnorm3 = nn.BatchNorm2d(128)
          self.flatten_size = 128 * (128 // 8) * (251 // 8)  # = 128 * 16 * 31 = 63,488
          self.fc1 = nn.Linear(self.flatten_size, 512)
          self.fc2 = nn.Linear(512, 128)
          self.fc3 = nn.Linear(128, 2)
          self.dropout = nn.Dropout(0.5)
  ```
- **Architecture**:
  - **Input**: Spectrograms of shape `(batch_size, 1, 128, 251)` (single channel, 128 mel bins, 251 time steps).
  - **CNN Layers**:
    - Three 2D convolutional layers (`Conv2d`) with increasing channels (1→32, 32→64, 64→128), each with a 3x3 kernel, stride 1, and padding 1 to preserve spatial dimensions.
    - Each convolution is followed by batch normalization (`BatchNorm2d`), ReLU activation, and 2x2 max-pooling (`MaxPool2d`) with stride 2, which reduces spatial dimensions by half.
    - After three max-pooling layers, the spatial dimensions are reduced by a factor of 8 (128/8 = 16, 251/8 ≈ 31), resulting in a feature map of shape `(batch_size, 128, 16, 31)`.
  - **Flattening**: The feature map is flattened to a vector of size `128 * 16 * 31 = 63,488`.
  - **DNN Layers**:
    - Three fully connected layers (`Linear`): 63,488→512, 512→128, 128→2.
    - ReLU activation and dropout (p=0.5) are applied after the first two `Linear` layers to prevent overfitting.
    - The final layer outputs logits for two classes (binary classification).
- **Forward Pass**:
  ```python
  def forward(self, x):
      x = self.pool(self.relu(self.batchnorm1(self.conv1(x))))
      x = self.pool(self.relu(self.batchnorm2(self.conv2(x))))
      x = self.pool(self.relu(self.batchnorm3(self.conv3(x))))
      x = x.view(x.size(0), -1)
      x = self.relu(self.fc1(x))
      x = self.dropout(x)
      x = self.relu(self.fc2(x))
      x = self.dropout(x)
      x = self.fc3(x)
      return x
  ```
  - Processes the input through the CNN layers, flattens the output, and passes it through the DNN layers to produce class logits.

#### **1.5. Training Setup**
- **Device**: Uses GPU (`cuda`) if available, otherwise CPU.
- **Model**: `CNNDNN` is instantiated and moved to the device.
- **Loss and Optimizer**:
  ```python
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  ```
  - `CrossEntropyLoss` combines log-softmax and negative log-likelihood loss, suitable for multi-class classification (here, binary).
  - Adam optimizer with a learning rate of 0.001.

#### **1.6. Training Loop**
- **Code**:
  ```python
  num_epochs = 20
  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for spectrograms, labels in train_loader:
          spectrograms, labels = spectrograms.to(device), labels.to(device)
          outputs = model(spectrograms)
          loss = criterion(outputs, labels)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
      print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
      model.eval()
      correct = 0
      total = 0
      with torch.no_grad():
          for spectrograms, labels in test_loader:
              spectrograms, labels = spectrograms.to(device), labels.to(device)
              outputs = model(spectrograms)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
      print(f"Validation Accuracy: {100 * correct / total:.2f}%")
  ```
- **Details**:
  - Trains for 20 epochs.
  - In each epoch, computes the loss for each batch, backpropagates gradients, and updates model parameters.
  - Prints average training loss per epoch.
  - Evaluates on the test set after each epoch, reporting validation accuracy.
  - Uses `model.train()` and `model.eval()` to toggle between training (enables dropout, batch norm) and evaluation modes.

#### **1.7. Final Evaluation**
- **Code**:
  ```python
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for spectrograms, labels in test_loader:
          spectrograms, labels = spectrograms.to(device), labels.to(device)
          outputs = model(spectrograms)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  print(f"Final Test Accuracy: {100 * correct / total:.2f}%")
  ```
- **Purpose**: Evaluates the model on the test set after training, reporting the final accuracy.
- **Model Saving** (commented out):
  ```python
  # torch.save(model.state_dict(), 'cnn_dnn_model.pth')
  ```
  - Optionally saves the trained model’s weights.

#### **1.8. Model Loading**
- **Code**:
  ```python
  model = CNNDNN().to(device)
  model.load_state_dict(torch.load('cnn_dnn_model.pth'))
  model.eval()
  ```
- **Purpose**: Loads a pre-trained model for inference or further training.

#### **1.9. Prediction on a Single Spectrogram**
- **Code**:
  ```python
  spec_path = 'Spectrograms/sample.npy'
  spectrogram = np.load(spec_path)
  spectrogram = spectrogram[np.newaxis, np.newaxis, :, :]  # Shape: (1, 1, n_mels, time_steps)
  spectrogram = torch.tensor(spectrogram, dtype=torch.float32).to(device)
  model.eval()
  with torch.no_grad():
      output = model(spectrogram)
      _, predicted = torch.max(output, 1)
      print(f"Predicted class: {'Depressed' if predicted.item() == 1 else 'Not Depressed'}")
  ```
- **Details**:
  - Loads a single spectrogram, adds batch and channel dimensions to match `(1, 1, 128, 251)`.
  - Runs inference and maps the predicted class index to a human-readable label ("Depressed" or "Not Depressed").

#### **1.10. Additional Considerations**
- **Data Augmentation**: Suggests random cropping, noise addition, or time/frequency masking using `torchaudio`.
- **Class Imbalance**: Proposes using a weighted `CrossEntropyLoss` to handle imbalanced datasets.
- **Hyperparameter Tuning**: Recommends experimenting with learning rate, batch size, and model architecture.
- **Early Stopping**: Suggests monitoring validation loss to prevent overfitting.
- **GPU Usage**: Advises ensuring CUDA is installed for faster training.

### **Strengths**
- **CNN Architecture**: Well-suited for spectrograms, as it captures spatial patterns (e.g., frequency-time relationships) effectively.
- **Batch Normalization**: Stabilizes and accelerates training by normalizing layer outputs.
- **Dropout**: Reduces overfitting with a high dropout rate (0.5).
- **Scalable**: The model can handle varying input sizes (though fixed here at 128x251).

### **Weaknesses**
- **Fixed Flatten Size**: The `flatten_size` (63,488) assumes a specific input shape `(1, 128, 251)`. Changes in spectrogram size require recomputing this value.
- **No Validation in Training Loop**: While validation accuracy is computed, there’s no early stopping or model checkpointing based on validation performance.
- **Limited Augmentation**: No data augmentation is implemented, which could limit generalization.
- **Class Imbalance**: The weighted loss is only suggested, not implemented, which could be problematic if the dataset is imbalanced.

---

## **2. LSTM.ipynb**

This notebook implements an **LSTM (Long Short-Term Memory)** model to classify spectrograms, treating them as sequential data (sequences of frequency bins over time). The LSTM processes the spectrogram as a sequence of 128 time steps, each with 251 features (frequency bins).

### **Key Components**

#### **2.1. Imports and Setup**
- **Libraries**: Same as `CNN-DNN.ipynb` (`os`, `numpy`, `pandas`, `torch`, etc.), with the addition of `sklearn.model_selection.train_test_split`.
- **Purpose**: Supports data loading, preprocessing, and model training.

#### **2.2. Data Loading**
- **Code**:
  ```python
  df = pd.read_csv('dataset.csv')
  spectrograms_dir = 'Spectrograms'
  X = []
  y = []
  for index, file in enumerate(os.listdir(spectrograms_dir)):
      if file.endswith('.npy'):
          spectrogram = np.load(os.path.join(spectrograms_dir, file))
          y.append(df.iloc[index]['label'])
          X.append(spectrogram)
  ```
- **Details**:
  - Loads all `.npy` files from the `Spectrograms` directory into a list `X`.
  - Extracts corresponding labels from `dataset.csv` using the index, assuming the CSV and directory are aligned.
  - Unlike `CNN-DNN.ipynb`, this approach loads all spectrograms into memory upfront, which could be memory-intensive for large datasets.

#### **2.3. SpectrogramDataset Class**
- **Definition**:
  ```python
  class SpectrogramDataset(Dataset):
      def __init__(self, X, y):
          self.X = X
          self.y = y
      def __len__(self):
          return len(self.X)
      def __getitem__(self, idx):
          return X[idx], y[idx]
  ```
- **Purpose**: A simpler `Dataset` class compared to `CNN-DNN.ipynb`, as spectrograms and labels are preloaded into lists `X` and `y`.
- **Details**:
  - Returns spectrograms as-is (shape `(128, 251)`) and labels as scalars.
  - No additional preprocessing (e.g., adding channel dimensions) is done here, as it’s handled later.

#### **2.4. Dataset Splitting**
- **Code**:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  print(len(X_train), len(X_test))  # Outputs: 371, 93
  ```
- **Details**:
  - Uses `sklearn`’s `train_test_split` to split data into 80% training (371 samples) and 20% testing (93 samples).
  - `random_state=42` ensures reproducibility.

#### **2.5. Tensor Conversion**
- **Code**:
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
  y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
  X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
  y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)
  ```
- **Details**:
  - Converts the lists `X_train`, `X_test`, `y_train`, and `y_test` to PyTorch tensors and moves them to the device (GPU/CPU).
  - A warning is noted about the inefficiency of converting a list of NumPy arrays to a tensor directly, suggesting `np.array(X_train)` first.

#### **2.6. DataLoader**
- **Code**:
  ```python
  train_dataset = SpectrogramDataset(X_train_tensor, y_train_tensor)
  test_dataset = SpectrogramDataset(X_test_tensor, y_test_tensor)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
  ```
- **Details**:
  - Creates `DataLoader` objects similar to `CNN-DNN.ipynb`, with batch size 32 and shuffling for training.

#### **2.7. LSTMClassifier Model**
- **Definition**:
  ```python
  class LSTMClassifier(nn.Module):
      def __init__(self, input_size, hidden_size, num_layers, num_classes):
          super(LSTMClassifier, self).__init__()
          self.hidden_size = hidden_size
          self.num_layers = num_layers
          self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
          self.fc = nn.Linear(hidden_size, num_classes)
          self.dropout = nn.Dropout(0.3)
  ```
- **Architecture**:
  - **Input**: Spectrograms of shape `(batch_size, 128, 251)`, interpreted as sequences of length 128 (time steps) with 251 features (frequency bins) per step.
  - **LSTM Layer**:
    - `input_size=251`: Number of frequency bins.
    - `hidden_size=128`: Number of hidden units in the LSTM.
    - `num_layers=2`: Two stacked LSTM layers.
    - `batch_first=True`: Expects input shape `(batch_size, seq_len, input_size)`.
  - **Output**: The LSTM outputs a tensor of shape `(batch_size, seq_len, hidden_size)`. The last time step’s output `(batch_size, hidden_size)` is used.
  - **Fully Connected Layer**: Maps the LSTM’s last output to `num_classes=2`.
  - **Dropout**: Applies dropout (p=0.3) before the final layer.
- **Forward Pass**:
  ```python
  def forward(self, x):
      h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
      c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
      out, _ = self.lstm(x, (h0, c0))
      out = out[:, -1, :]
      out = self.dropout(out)
      out = self.fc(out)
      return out
  ```
  - Initializes hidden and cell states (`h0`, `c0`) for the LSTM.
  - Processes the input sequence through the LSTM.
  - Takes the last time step’s output and passes it through dropout and the fully connected layer.

#### **2.8. Training Setup**
- **Hyperparameters**:
  ```python
  input_size = 251
  hidden_size = 128
  num_layers = 2
  num_classes = 2
  model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  ```
- **Details**:
  - Matches the spectrogram’s feature dimension (251 frequency bins).
  - Uses `CrossEntropyLoss` and Adam optimizer, similar to `CNN-DNN.ipynb`.

#### **2.9. Training Loop**
- **Code**:
  ```python
  num_epochs = 10
  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for i, (inputs, labels) in enumerate(train_loader):
          inputs, labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
  ```
- **Details**:
  - Trains for 10 epochs (fewer than `CNN-DNN.ipynb`’s 20).
  - Computes and prints average training loss per epoch.
  - No validation during training (unlike `CNN-DNN.ipynb`).

#### **2.10. Evaluation**
- **Code**:
  ```python
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for inputs, labels in test_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          outputs = model(inputs)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  accuracy = 100 * correct / total
  print(f'Test Accuracy: {accuracy:.2f}%')  # Outputs: 89.25%
  ```
- **Details**:
  - Evaluates the model on the test set, reporting an accuracy of 89.25%.
  - Similar to `CNN-DNN.ipynb`’s evaluation but only done after training.

#### **2.11. Prediction on a Single Spectrogram**
- **Code**:
  ```python
  new_spectrogram = np.load('Spectrograms/490_clip_0.npy')
  new_spectrogram_tensor = torch.tensor(new_spectrogram, dtype=torch.float32).unsqueeze(0).to(device)
  with torch.no_grad():
      output = model(new_spectrogram_tensor)
      _, predicted_class = torch.max(output, 1)
      print(f"Predicted Class for New Spectrogram: {predicted_class.item()}")
  ```
- **Details**:
  - Loads a spectrogram, adds a batch dimension to get `(1, 128, 251)`, and runs inference.
  - Outputs the predicted class index (e.g., 0 for "Not Depressed").

### **Strengths**
- **Sequential Modeling**: LSTMs are designed for sequential data, making them suitable for capturing temporal dependencies in spectrograms (e.g., how frequency patterns evolve over time).
- **Simpler Architecture**: Fewer layers than the CNN-DNN model, potentially faster to train for smaller datasets.
- **Achieved Accuracy**: Reports a test accuracy of 89.25%, indicating good performance for the task.

### **Weaknesses**
- **Memory Usage**: Loading all spectrograms into memory (`X` and `y` lists) can be inefficient for large datasets.
- **Inefficient Tensor Conversion**: The warning about `torch.tensor(X_train)` suggests a performance bottleneck; converting to a single NumPy array first would be better.
- **No Validation During Training**: Unlike `CNN-DNN.ipynb`, there’s no validation loop, so overfitting is harder to detect.
- **No Augmentation or Regularization**: Limited regularization (only dropout with p=0.3) and no data augmentation.
- **No Model Saving**: Unlike `CNN-DNN.ipynb`, the model weights are not saved, limiting reusability.

---

## **Comparison**

| **Aspect**                | **CNN-DNN.ipynb**                              | **LSTM.ipynb**                                |
|---------------------------|-----------------------------------------------|---------------------------------------------|
| **Model Type**            | CNN + DNN (3 conv layers + 3 dense layers)    | LSTM (2 layers) + 1 dense layer            |
| **Input Shape**           | `(batch_size, 1, 128, 251)` (image-like)      | `(batch_size, 128, 251)` (sequence-like)   |
| **Data Loading**          | On-demand via `SpectrogramDataset`            | Preloads all spectrograms into memory       |
| **Dataset Splitting**     | `random_split` (80/20)                        | `train_test_split` (80/20)                 |
| **Batch Size**            | 32                                            | 32                                         |
| **Epochs**                | 20                                            | 10                                         |
| **Loss Function**         | `CrossEntropyLoss`                            | `CrossEntropyLoss`                         |
| **Optimizer**             | Adam (lr=0.001)                               | Adam (lr=0.001)                            |
| **Regularization**        | Dropout (0.5), BatchNorm                      | Dropout (0.3)                              |
| **Validation**            | Per epoch                                     | Only at the end                             |
| **Reported Accuracy**     | Not reported                                  | 89.25%                                     |
| **Model Saving**          | Supported (commented out)                     | Not implemented                            |
| **Prediction**            | Maps to "Depressed"/"Not Depressed"           | Outputs class index                        |
| **Additional Features**   | Suggests augmentation, weighted loss, etc.     | None                                       |

### **Key Differences**
1. **Model Architecture**:
   - **CNN-DNN**: Treats spectrograms as 2D images, using convolutions to extract spatial features (e.g., patterns in frequency and time). Suitable for capturing local patterns.
   - **LSTM**: Treats spectrograms as sequences (128 time steps of 251 features), focusing on temporal dependencies. Better for sequential patterns but may miss spatial relationships.
2. **Data Handling**:
   - **CNN-DNN**: Loads spectrograms on-demand, reducing memory usage.
   - **LSTM**: Preloads all spectrograms, which is memory-intensive.
3. **Training**:
   - **CNN-DNN**: More epochs (20) and includes per-epoch validation, making it easier to monitor performance.
   - **LSTM**: Fewer epochs (10), no per-epoch validation, but achieves 89.25% test accuracy.
4. **Regularization**:
   - **CNN-DNN**: Uses batch normalization and higher dropout (0.5), likely more robust to overfitting.
   - **LSTM**: Only uses dropout (0.3), potentially less regularized.
5. **Flexibility**:
   - **CNN-DNN**: Provides suggestions for improvements (augmentation, weighted loss, early stopping).
   - **LSTM**: Minimal additional considerations, less extensible.

---

## **Potential Improvements for Both**

1. **Data Augmentation**:
   - Implement `torchaudio` transforms (e.g., `TimeMasking`, `FrequencyMasking`) to augment spectrograms, improving generalization.
   - Example:
     ```python
     import torchaudio.transforms as T
     transform = T.Compose([
         T.TimeMasking(time_mask_param=20),
         T.FrequencyMasking(freq_mask_param=20)
     ])
     ```

2. **Class Imbalance**:
   - Compute class distribution from `dataset.csv` and apply a weighted `CrossEntropyLoss`:
     ```python
     class_counts = pd.read_csv('dataset.csv')['label'].value_counts()
     weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
     criterion = nn.CrossEntropyLoss(weight=weights.to(device))
     ```

3. **Early Stopping**:
   - Add early stopping based on validation loss to prevent overfitting:
     ```python
     best_val_loss = float('inf')
     patience = 5
     counter = 0
     for epoch in range(num_epochs):
         # Training and validation
         val_loss = compute_validation_loss()
         if val_loss < best_val_loss:
             best_val_loss = val_loss
             torch.save(model.state_dict(), 'best_model.pth')
             counter = 0
         else:
             counter += 1
         if counter >= patience:
             print("Early stopping")
             break
     ```

4. **Hyperparameter Tuning**:
   - Use a library like `optuna` to tune learning rate, batch size, number of layers, etc.
   - Example:
     ```python
     import optuna
     def objective(trial):
         lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
         optimizer = optim.Adam(model.parameters(), lr=lr)
         # Train and evaluate
         return validation_accuracy
     study = optuna.create_study(direction="maximize")
     study.optimize(objective, n_trials=20)
     ```

5. **Efficient Data Loading**:
   - For `LSTM.ipynb`, convert the list of NumPy arrays to a single array before tensor conversion:
     ```python
     X_train = np.array(X_train)
     X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
     ```

6. **Model Ensembling**:
   - Combine CNN-DNN and LSTM predictions to leverage both spatial and temporal features:
     ```python
     cnn_model = CNNDNN().to(device)
     lstm_model = LSTMClassifier(...).to(device)
     cnn_model.eval()
     lstm_model.eval()
     with torch.no_grad():
         cnn_output = cnn_model(spectrogram)
         lstm_output = lstm_model(spectrogram)
         combined_output = (cnn_output + lstm_output) / 2
         _, predicted = torch.max(combined_output, 1)
     ```

7. **Cross-Validation**:
   - Implement k-fold cross-validation to better estimate model performance:
     ```python
     from sklearn.model_selection import KFold
     kf = KFold(n_splits=5, shuffle=True)
     for train_idx, val_idx in kf.split(X):
         train_data = SpectrogramDataset(X[train_idx], y[train_idx])
         val_data = SpectrogramDataset(X[val_idx], y[val_idx])
         # Train and evaluate
     ```

8. **Visualization**:
   - Plot training and validation loss/accuracy curves to monitor training dynamics:
     ```python
     import matplotlib.pyplot as plt
     train_losses = []
     val_accuracies = []
     for epoch in range(num_epochs):
         # Training and validation
         train_losses.append(running_loss / len(train_loader))
         val_accuracies.append(100 * correct / total)
     plt.plot(train_losses, label='Train Loss')
     plt.plot(val_accuracies, label='Val Accuracy')
     plt.legend()
     plt.show()
     ```

---

## **Conclusion**

- **CNN-DNN.ipynb** is better suited for tasks where spatial patterns in spectrograms are critical (e.g., identifying specific frequency-time structures). It’s more robust due to batch normalization, higher dropout, and per-epoch validation but requires careful handling of input size changes.
- **LSTM.ipynb** excels at modeling temporal dependencies, making it suitable for sequential patterns in spectrograms. However, its data loading is less efficient, and it lacks validation during training.
- **Recommendation**: For a robust solution, consider:
  1. Using the CNN-DNN model for its better regularization and validation strategy.
  2. Implementing the suggested improvements (augmentation, early stopping, etc.).
  3. Optionally, ensembling both models to capture both spatial and temporal features.
  4. For large datasets, optimize `LSTM.ipynb`’s data loading to match `CNN-DNN.ipynb`’s on-demand approach.

If you have specific questions about implementation details, performance issues, or want to dive deeper into any part, let me know!