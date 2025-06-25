# Final Project - League of Legends Match Predictor

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import itertools

# 1 - Data Loading and Preprocessing
# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# Step 2: Load the dataset
data = pd.read_csv('league_of_legends_data_large.csv')

# Step 3: Separate features and target
X = data.drop('win', axis=1)
y = data['win']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)








# 2 - Logistic Regression Model
# Step 1: Define model architecture
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # One output for binary classification

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Sigmoid activation for logistic regression

# Step 2: Initialize model, loss function, and optimizer
input_dim = X_train_tensor.shape[1]  # Number of input features

model = LogisticRegressionModel(input_dim)  # Instantiate the model
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent








# 3 - Train the model
# Binary Cross-Entropy Loss
criterion = nn.BCELoss()

# Initialize the model
input_dim = X_train_tensor.shape[1]
model_no_reg = LogisticRegressionModel(input_dim)

# Optimizer without L2 regularization
optimizer_no_reg = optim.SGD(model_no_reg.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model_no_reg.train()
    optimizer_no_reg.zero_grad()

    outputs = model_no_reg(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer_no_reg.step()

    if (epoch + 1) % 100 == 0:
        print(f"[No Reg] Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluation
model_no_reg.eval()
with torch.no_grad():
    train_preds = (model_no_reg(X_train_tensor) >= 0.5).float()
    test_preds = (model_no_reg(X_test_tensor) >= 0.5).float()

    train_acc_no_reg = (train_preds.eq(y_train_tensor).sum().item()) / y_train_tensor.size(0)
    test_acc_no_reg = (test_preds.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)

print(f"\n[No Reg] Training Accuracy: {train_acc_no_reg * 100:.2f}%")
print(f"[No Reg] Test Accuracy:     {test_acc_no_reg * 100:.2f}%")





# 4 - Model OPtimization and Evaluation
# Re-initialize a new model for a fair comparison
model_l2 = LogisticRegressionModel(input_dim)

# Optimizer with L2 regularization (weight decay)
optimizer_l2 = optim.SGD(model_l2.parameters(), lr=0.01, weight_decay=0.01)

# Training loop
for epoch in range(num_epochs):
    model_l2.train()
    optimizer_l2.zero_grad()

    outputs = model_l2(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer_l2.step()

    if (epoch + 1) % 100 == 0:
        print(f"[L2 Reg] Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluation
model_l2.eval()
with torch.no_grad():
    train_preds_l2 = (model_l2(X_train_tensor) >= 0.5).float()
    test_preds_l2 = (model_l2(X_test_tensor) >= 0.5).float()

    train_acc_l2 = (train_preds_l2.eq(y_train_tensor).sum().item()) / y_train_tensor.size(0)
    test_acc_l2 = (test_preds_l2.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)

print(f"\n[L2 Reg] Training Accuracy: {train_acc_l2 * 100:.2f}%")
print(f"[L2 Reg] Test Accuracy:     {test_acc_l2 * 100:.2f}%")




# 5 - Visualization and Interpretation
# Helper to plot confusion matrix and ROC for a given model
def evaluate_model(model, X_test_tensor, y_test_tensor, title_prefix=""):
    # Get predicted probabilities and labels
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
    y_pred_test_labels = (y_pred_test > 0.5).float()

    # Confusion matrix
    cm = confusion_matrix(y_test_tensor.cpu(), y_pred_test_labels.cpu())

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{title_prefix}Confusion Matrix')
    plt.colorbar()
    tick_marks = range(2)
    plt.xticks(tick_marks, ['Loss', 'Win'], rotation=45)
    plt.yticks(tick_marks, ['Loss', 'Win'])

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Classification report
    print(f"{title_prefix}Classification Report:\n",
          classification_report(y_test_tensor.cpu(), y_pred_test_labels.cpu(), target_names=['Loss', 'Win']))

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test_tensor.cpu(), y_pred_test.cpu())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title_prefix}Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


# Evaluate and visualize both models
evaluate_model(model_no_reg, X_test_tensor, y_test_tensor, title_prefix="Baseline – ")
evaluate_model(model_l2, X_test_tensor, y_test_tensor, title_prefix="L2-Regularized – ")




# 6 - Model Saving and Loading

# Save the model
MODEL_PATH = "lol_logreg_l2.pth"
torch.save(model_l2.state_dict(), MODEL_PATH)
print("Model parameters saved to", MODEL_PATH)


# Load the model
loaded_model = LogisticRegressionModel(input_dim)  # same architecture
loaded_model.load_state_dict(torch.load(MODEL_PATH))
print("Model parameters loaded into new model instance")


# Ensure the loaded model is in evaluation mode
loaded_model.eval()


# Evaluate the loaded model
with torch.no_grad():
    test_probs_loaded = loaded_model(X_test_tensor)
    test_preds_loaded = (test_probs_loaded >= 0.5).float()
    loaded_test_acc = (
        test_preds_loaded.eq(y_test_tensor).sum().item()
        / y_test_tensor.size(0)
    )

print("Loaded model test accuracy: {:.2f}%".format(loaded_test_acc))




# 7 - Hyperparameter Tuning
# Fixed number of epochs for each run
num_epochs = 100
learning_rates = [0.01, 0.05, 0.1]
test_accuracies = {}

# Loss function
criterion = nn.BCELoss()

# Evaluate function for test accuracy
def evaluate_accuracy(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        probs = model(X_test)
        preds = (probs >= 0.5).float()
        acc = preds.eq(y_test).sum().item() / y_test.size(0)
    return acc

# Loop through each learning rate
for lr in learning_rates:
    # Reinitialize model and optimizer
    model = LogisticRegressionModel(input_dim)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate on test set
    acc = evaluate_accuracy(model, X_test_tensor, y_test_tensor)
    test_accuracies[lr] = acc
    print("Learning Rate: {:.2f}  ->  Test Accuracy: {:.2f}%".format(lr, acc * 100))

# Identify best learning rate
best_lr = max(test_accuracies, key=test_accuracies.get)
best_acc = test_accuracies[best_lr]

print("\nBest Learning Rate: {:.2f}  with Test Accuracy: {:.2f}%".format(best_lr, best_acc * 100))



# 8 - Feature Importance
import pandas as pd
import matplotlib.pyplot as plt
# Extract the weights of the linear layer
## Write your code here
weights = model_l2.linear.weight.data.numpy().flatten()
feature_names = X_train.columns


# Create a DataFrame for feature importance
## Write your code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create DataFrame with feature names and corresponding weights
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Weight': weights,
    'Abs_Weight': np.abs(weights)
})

# Sort features by absolute weight (most important at top)
importance_df = importance_df.sort_values(by='Abs_Weight', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Weight'], color='steelblue')
plt.xlabel('Weight')
plt.title('Feature Importance from Logistic Regression')
plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Optional: print the sorted table
print(importance_df.drop(columns='Abs_Weight'))






