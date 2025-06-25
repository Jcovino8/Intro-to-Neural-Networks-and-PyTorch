# Intro-to-Neural-Networks-and-PyTorch
The 'Labs' folder contains the labs for course 4-13 of my certificate

### 1. 🧠 League of Legends Match Outcome Predictor
Predicts the outcome (win/loss) of League of Legends matches using logistic regression models in PyTorch.

- **Tech:** PyTorch, scikit-learn, Matplotlib
- **Highlights:**
  - Logistic regression with and without L2 regularization
  - Confusion matrix, ROC curve, and classification report
  - Hyperparameter tuning for learning rate
  - Feature importance visualization
- **Results:** Achieved test accuracy over 80% with regularization

🔗 [`/lol_match_predictor`](./lol_match_predictor)

---

### 2. 🧪 Breast Cancer Classification (UCI Dataset)
Binary classification of breast tumors using a fully connected neural network in PyTorch.

- **Tech:** PyTorch, UCI ML Repo, scikit-learn
- **Highlights:**
  - Neural net classifier with experiments on optimizer (Adam, SGD) and architecture
  - Balanced dataset (200 Benign, 200 Malignant)
  - Loss curve tracking across experiments
- **Results:** Successfully classified malignant vs. benign tumors with high accuracy

🔗 [`/breast_cancer_classifier`](./breast_cancer_classifier)

---

### 3. ♻️ Waste Product Classifier Using Transfer Learning
Classifies waste as **organic (O)** or **recyclable (R)** using transfer learning with VGG16 in TensorFlow/Keras.

- **Tech:** TensorFlow/Keras, VGG16, ImageDataGenerator
- **Highlights:**
  - Transfer learning with feature extraction and fine-tuning
  - Real-time image augmentation
  - Accuracy/loss curves across epochs
  - Test predictions with annotated visual output
- **Results:** Fine-tuned model achieved higher validation accuracy than feature extraction
