# Lab Task: KNN Classification and Map Visualization in Google Colab

This README outlines the tasks for implementing KNN classification and visualizing clusters using map data in Google Colab.

---

## Coding Exercise 1: Drawing the Map

1. **Upload Shape Files**
   - Load all 40 files from the "shape" folder into Google Colab by running the provided loader cell.

2. **Mark Diseased People on the Map**
   - Place random points representing diseased individuals on the map.
   - Display these points using the ‘x’ symbol.

---

## Coding Exercise 2: Classification and Evaluation

### Algorithm: Manual K-Nearest Neighbor (KNN)

#### **NearestNeighborClassifierManual**
```python
class NearestNeighborClassifierManual:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        # Store training data and labels
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        # Loop through each test point
        for x_test in X_test:
            # Compute distances between x_test and all training points
            distances = np.linalg.norm(self.X_train - x_test, axis=1)
            # Find the index of the nearest neighbor
            nearest_index = np.argmin(distances)
            # Append the label of the nearest neighbor to predictions
            predictions.append(self.y_train[nearest_index])
        return predictions
