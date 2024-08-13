# Face Recognition Model Using K-Nearest Neighbors (KNN)

This notebook demonstrates the process of building a face recognition model using the K-Nearest Neighbors (KNN) algorithm. Below is a step-by-step explanation of the notebook.

## 1. Installation of Required Libraries
The necessary Python libraries are installed:
- `pip` is upgraded.
- `cmake`, `face_recognition`, and other essential libraries like `numpy`, `opencv` (cv2), and `scikit-learn` are installed.

## 2. Dataset Organization
The dataset is loaded from a CSV file (`Dataset.csv`), which contains image filenames and corresponding labels (celebrity names). The dataset is analyzed to see the distribution of images for each label (celebrity).

## 3. Face Encoding
A custom function `encode_images(df)` is created to:
- Read each image file from the dataset.
- Detect and encode the face(s) in the image using the `face_recognition` library.
- Store the face encodings and corresponding labels in separate arrays (`x` for encodings and `y` for labels).

## 4. Splitting the Data
The dataset is split into training and testing sets using `train_test_split`, with 15% of the data reserved for testing. Stratification ensures that the distribution of labels remains consistent between the training and testing sets.

## 5. Building the Classification Model
A K-Nearest Neighbors (KNN) model is created using `sklearn.neighbors.KNeighborsClassifier` with the following parameters:
- `n_neighbors=3`: Uses 3 nearest neighbors for classification.
- `algorithm='ball_tree'`: Efficiently handles the high-dimensional face encoding data.
- `weights='distance'`: Weights neighbors based on their distance from the query point.

The model is trained on the training data (`x_train`, `y_train`).

## 6. Evaluating the Model
The model's performance is evaluated on the training data and the test data:
- **Accuracy Score:** The model achieves an accuracy of 100% on the training set.
- **Important Scores:** Precision, Recall, and F1 Score are calculated using the test set. These metrics all yield a perfect score of 1.0, indicating excellent model performance.
- **Confusion Matrix:** A heatmap is generated to visualize the confusion matrix, showing how well the model distinguishes between different labels.

## 7. Prediction and Visualization
The model is tested on new images using a custom function `predict_and_visualize(image_path)`. The function:
- Loads the image and detects face locations and encodings.
- Uses the trained KNN model to predict the label for each detected face.
- Draws a bounding box around the face and labels it with the predicted name or "Unknown" if the face is not recognized.

**Example Outputs:**
- When tested on images of "The Rock" and "Billie Eilish & Camila Cabello", the model successfully identifies known faces.
- If a face (e.g., Michael Jackson) is not in the dataset, it is labeled as "Unknown".
- The model correctly identifies Tom Cruise, who is in the dataset, while Michael Jackson, who is not, is labeled as "Unknown".

## Summary
This notebook provides a complete workflow for creating a face recognition system, from data preparation and model training to evaluation and real-time face prediction. The use of the KNN algorithm allows for a simple yet effective way to recognize faces based on their encoded features.

## Additional Resources
- [Kaggle Notebook: Face Detection and Recognition](https://www.kaggle.com/code/abdelrahmanmohamed26/face-detection-and-recognition/notebook#7.-predicting-some-images)
- [Face Recognition Dataset](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset)
