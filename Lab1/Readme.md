# Image Classification with Hyperparameter Tuning and MLflow Tracking

This project focuses on image classification using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It includes **automated hyperparameter tuning with Optuna** and **training experiment tracking using MLflow**.
---
### 🧑‍💻 **Thông tin Nhóm**

| Họ và Tên              | MSSV      |
|------------------------|-----------|
| **Trần Ngọc Thiện**    | 21521465  |
| **Trịnh Thị Lan Anh**  | 22520083  |
| **Trần Quang Đạt**     | 22520236  |
| **Vương Dương Thái Hà** | 22520375  |
---

## 📂 DATASET AND FULL EXPERIMENT CODE 
[🔗 VIDEO DEMO →](https://drive.google.com/drive/u/0/folders/1wM8M4DMZ1YqE5N7a-1LAmO4F_cm4wwGI)

🔗 **You can view and download the full MLflow tracking logs and outputs from My Kaggle:**  
Due to the dataset and logs being **large in size**, they are not hosted directly on this repo.
[🔗 View on Kaggle →](https://www.kaggle.com/code/dattran0509/cs317-lab1#MLflow-Training-Pipeline-on-Kaggle-for-Animal-Classification-using-Custom-CNN-(Keras)-with-Optuna)

🔗 **ALL INFORMATION ABOUT TRACKING IN Output/mlruns**

[🔗 View dataset →](https://www.kaggle.com/datasets/dattran0509/animal)

## 📌 Project Features

- Image classification from custom dataset  
- Train/Validation split with preprocessing (resizing, normalization)  
- CNN model built with TensorFlow/Keras  
- Hyperparameter optimization using **Optuna**  
- Model tracking and versioning with **MLflow**  
- EarlyStopping to prevent overfitting  
- Visualizations: Accuracy/Loss plots & Confusion Matrix  
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

---

## 🛠 Technologies Used

| Tool/Library     | Purpose                               |
|------------------|----------------------------------------|
| Python           | Core programming language              |
| TensorFlow/Keras | Deep learning model building           |
| OpenCV           | Image reading and preprocessing        |
| Optuna           | Hyperparameter tuning                  |
| MLflow           | Experiment tracking & model registry   |
| Seaborn/Matplotlib | Plotting and visualization          |
| Scikit-learn     | Evaluation metrics & train/test split  |



## 🔄 Pipeline Overview


### 1. **Data Loading and Preprocessing**

- The dataset is loaded, and images are resized to a fixed size.
- Labels are encoded using `LabelEncoder` to convert class labels into numeric form.
- The data is split into training and testing sets using `train_test_split` from `scikit-learn` to ensure proper training and validation of the model.

### 2. **Model Building**

- A custom **Convolutional Neural Network (CNN)** is built using Keras. 
- The model architecture includes several convolutional and dense layers, followed by dropout for regularization and ReLU activations.

### 3. **Hyperparameter Tuning with Optuna**

- The pipeline integrates **Optuna**, a hyperparameter optimization framework, which is used to tune the following hyperparameters:
  - Learning rate
  - Batch size
  - Epochs
- Optuna performs an efficient search for the best hyperparameters, helping improve model performance by finding optimal settings.

### 4. **Model Training**

- The model is trained using the preprocessed dataset. 
- **Early stopping** is applied during training to monitor validation loss. If there's no improvement for 3 consecutive epochs, the training process is halted to avoid overfitting.

### 5. **Experiment Tracking with MLflow**

- All training parameters (such as learning rate, batch size, and epochs) and metrics (including accuracy, precision, recall, F1-score) are logged using **MLflow**.
- This allows easy comparison of multiple runs, hyperparameter combinations, and provides a detailed history of model performance.

### 6. **Model Evaluation**

- After training, the best model is evaluated on the test set.
- Key metrics such as accuracy, precision, recall, and F1-score are computed and printed.
- A **confusion matrix** is also generated and saved for better understanding of the model’s performance across different classes.

### 7. **Visualization**

- During training, both the **loss** and **accuracy** curves are saved for visualization. This helps in understanding the model's performance and learning curve.
- Additionally, confusion matrices are plotted for visual interpretation of how well the model is classifying different categories.
