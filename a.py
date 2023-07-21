import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Specify the path to the audio files and emotion labels
audio_path = '/path/to/audio/files'
labels = ['happy', 'sad', 'angry']  # Example emotion labels

# Initialize empty lists to store features and labels
X = []
y = []

# Iterate over the audio files
for filename in os.listdir(audio_path):
    # Load the audio file using librosa
    audio, sr = librosa.load(os.path.join(audio_path, filename))
    
    # Extract features from the audio using librosa
    # Here, we extract the Mel-Frequency Cepstral Coefficients (MFCCs)
    features = librosa.feature.mfcc(audio, sr=sr)
    
    # Append the features to X and corresponding label to y
    X.append(features)
    label = filename.split('_')[0]  # Assuming the label is part of the filename
    y.append(label)

# Convert the lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Step 2: Feature Preprocessing
# Flatten the feature matrix and reshape it to 2D
X = X.reshape(X.shape[0], -1)

# Step 3: Train-Test Split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
# Initialize a Support Vector Machine (SVM) classifier
svm = SVC()

# Train the SVM classifier on the training data
svm.fit(X_train, y_train)

# Step 5: Model Evaluation
# Evaluate the trained model on the testing data
accuracy = svm.score(X_test, y_test)
print("Accuracy:", accuracy)

# Step 6: Emotion Prediction
# Assuming you have a new audio sample for emotion prediction
new_audio, sr = librosa.load('/path/to/new/audio.wav')

# Extract features from the new audio sample
new_features = librosa.feature.mfcc(new_audio, sr=sr)
new_features = new_features.reshape(1, -1)

# Predict the emotion label using the trained model
predicted_label = svm.predict(new_features)
print("Predicted emotion:", predicted_label)
