# CIFAR-10 Image Classification with CNN

## Description
This project demonstrates the implementation of a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset**. The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes, including airplanes, cars, birds, and more. A CNN model is built using TensorFlow and Keras to achieve accurate predictions for image classification.

---

## Dataset
### CIFAR-10
The CIFAR-10 dataset is a well-known dataset for machine learning and computer vision tasks. It consists of:
- **50,000 training images**
- **10,000 testing images**
- **10 Classes**: `['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']`

Each image has dimensions **32x32x3**, representing 32x32 pixels with RGB color channels.

---

## Model Architecture
The CNN model used in this project includes the following layers:
1. **Convolutional Layer**:
   - Filters: 32, Kernel Size: (3x3), Activation: ReLU
   - Input Shape: (32, 32, 3)
2. **MaxPooling Layer**:
   - Pool Size: (2x2)
3. **Second Convolutional Layer**:
   - Filters: 64, Kernel Size: (3x3), Activation: ReLU
4. **Second MaxPooling Layer**:
   - Pool Size: (2x2)
5. **Flatten Layer**:
   - Converts the 2D feature maps into a 1D vector.
6. **Fully Connected Dense Layer**:
   - Units: 64, Activation: ReLU
7. **Output Layer**:
   - Units: 10 (for 10 classes), Activation: Softmax

---

## Steps to Run the Project

1. **Load Dataset**:
   - The CIFAR-10 dataset is loaded using TensorFlow's Keras API.
   - Images are normalized by dividing pixel values by 255 to scale them to the range [0, 1].

2. **Model Compilation**:
   - Optimizer: **Adam**
   - Loss Function: **Sparse Categorical Crossentropy** (for integer labels)
   - Metric: **Accuracy**

3. **Model Training**:
   - The model is trained on the training dataset for **50 epochs**.

4. **Model Evaluation**:
   - The model is evaluated on the test dataset to determine accuracy.

5. **Predictions**:
   - Predictions are made using the trained model, and the results are compared with the actual labels.
   - Sample images and their predictions are displayed using Matplotlib.

---

## How to Use
1. **Clone the repository.**
2. Ensure you have Python installed along with TensorFlow, NumPy, and Matplotlib.
3. Run the provided code in a Jupyter Notebook or any Python IDE.
4. Modify the `index` variable to test predictions on specific images from the test dataset.

---

## Example Prediction
The code includes functionality to display a test image, predict its class, and compare it with the actual class. Here's an example:

```python
# Select an index to test a single image
index = 0

# Display the image
plt.figure(figsize=(2, 2))
plt.imshow(X_test[index])
plt.axis('off')
plt.show()

# Predict the class of the selected image
prediction = cnn.predict(X_test[index].reshape(1, 32, 32, 3))
predicted_class = np.argmax(prediction)

# Display the predicted and actual class
print(f"Predicted Class: {classes[predicted_class]}")
print(f"Actual Class: {classes[y_test[index][0]]}")
```

---

## Results
The CNN achieves reasonable accuracy in classifying the CIFAR-10 images. Fine-tuning the model, using data augmentation, or increasing the number of epochs can further improve the accuracy.

---

## Requirements
- Python 3.7+
- TensorFlow
- NumPy
- Matplotlib

---

## Acknowledgments
The CIFAR-10 dataset is provided by the **Canadian Institute for Advanced Research**. TensorFlow and Keras are used to implement the CNN model.

