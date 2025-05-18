# Brain Tumor Detection Using CNN
This project focuses on classifying brain MRI images as tumor or no tumor using Convolutional Neural Networks (CNN) built with TensorFlow/Keras. It includes data preprocessing, model training, evaluation, and visualization.

📁 Dataset
MRI images are categorized into two folders: yes (with tumor) and no (without tumor).

Images are loaded using Pathlib, labeled, and stored in a Pandas DataFrame for further processing.

A separate prediction set is used to test model generalization.

🧠 Model Architecture
Two CNN models were built and trained:

Model 1
Input size: 200x200 (grayscale)

Layers: 4 Conv2D layers with MaxPooling and Dropout

Dense layers: 512 (ReLU), output layer (Softmax)

Optimizer: RMSprop

Epochs: 30

Model 2
Similar to Model 1 but with more Conv2D layers and trained with data augmentation

Epochs: 50

📦 Libraries Used
TensorFlow / Keras

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

OpenCV, PIL

🔄 Data Augmentation
To enhance model performance and reduce overfitting, ImageDataGenerator is used with:

Rescaling

Brightness variation

Rotation

Horizontal flips

Width/Height shifts

🧪 Model Training & Evaluation
Training and validation data were split (90% train, 10% test).

Used categorical crossentropy loss.

Accuracy and loss metrics were tracked per epoch and plotted.

Evaluated model performance on unseen test data.

📊 Results
Accuracy and loss are visualized for both training and validation sets.

Final predictions are displayed using matplotlib with actual images.

📌 Highlights
Handled imbalanced data using proper labeling and augmentation.

Two models trained and compared for performance.

Predictions visualized with model-predicted labels.

📂 File Structure
yaml
Copy
Edit
project/
├── yes/                    # MRI images with tumor
├── no/                     # MRI images without tumor
├── predict/                # Test images
├── brain_tumor_cnn.py      # Main Python file (your code)
└── README.md               # Project description
🏁 How to Run
Install dependencies:

bash
Copy
Edit
pip install tensorflow keras pandas matplotlib seaborn opencv-python pillow scikit-learn
Set your dataset paths in the script.

Run the script:

bash
Copy
Edit
python brain_tumor_cnn.py
