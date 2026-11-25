import csv
import math
import os
import pickle
import random


class DecisionTree:

  def __init__(self, max_depth=10, min_samples_split=2, num_features=None):
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.num_features = num_features
    self.root = None

  def _gini(self, y):
    total = len(y)
    if total == 0:
      return 0.0
    counts = {}
    for label in y:
      counts[label] = counts.get(label, 0) + 1
    impurity = 1.0
    for count in counts.values():
      prob = count / total
      impurity -= prob * prob
    return impurity

  def _best_split(self, X, y, feat_indices):
    best_gain = -1.0
    best_idx = None
    best_thresh = None
    current_gini = self._gini(y)
    n_samples = len(y)

    for idx in feat_indices:
      values = sorted(set(row[idx] for row in X))
      if len(values) <= 1:
        continue
      # Candidate thresholds (midpoints)
      thresholds = [
          (values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)
      ]
      # Agar thresholds bahut zyada hon toh evenly sample kar lein taaki fast chale
      if len(thresholds) > 25:
        step = len(thresholds) / 25.0
        thresholds = [thresholds[int(i * step)] for i in range(25)]

      for thresh in thresholds:
        left_y = [y[i] for i in range(n_samples) if X[i][idx] <= thresh]
        right_y = [y[i] for i in range(n_samples) if X[i][idx] > thresh]
        if not left_y or not right_y:
          continue
        p_left = len(left_y) / n_samples
        p_right = len(right_y) / n_samples
        weighted_gini = p_left * self._gini(left_y) + p_right * self._gini(
            right_y
        )
        gain = current_gini - weighted_gini

        if gain > best_gain:
          best_gain = gain
          best_idx = idx
          best_thresh = thresh

    return best_idx, best_thresh, best_gain

  def _majority_label(self, y):
    counts = {}
    for label in y:
      counts[label] = counts.get(label, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]

  def _build_tree(self, X, y, depth=0):
    if (
        depth >= self.max_depth
        or len(y) < self.min_samples_split
        or len(set(y)) == 1
    ):
      return {"leaf": True, "label": self._majority_label(y)}

    n_features = len(X[0])
    k = (
        self.num_features
        if self.num_features
        else max(1, int(math.sqrt(n_features)))
    )
    feat_indices = random.sample(range(n_features), min(k, n_features))

    feat_idx, thresh, gain = self._best_split(X, y, feat_indices)
    if feat_idx is None or gain <= 0.0:
      return {"leaf": True, "label": self._majority_label(y)}

    left_X, left_y, right_X, right_y = [], [], [], []
    for i in range(len(X)):
      if X[i][feat_idx] <= thresh:
        left_X.append(X[i])
        left_y.append(y[i])
      else:
        right_X.append(X[i])
        right_y.append(y[i])

    return {
        "leaf": False,
        "feature_index": feat_idx,
        "threshold": thresh,
        "left": self._build_tree(left_X, left_y, depth + 1),
        "right": self._build_tree(right_X, right_y, depth + 1),
    }

  def fit(self, X, y):
    self.root = self._build_tree(X, y, depth=0)
    return self

  def _predict_one(self, row, node):
    if node["leaf"]:
      return node["label"]
    if row[node["feature_index"]] <= node["threshold"]:
      return self._predict_one(row, node["left"])
    return self._predict_one(row, node["right"])

  def predict(self, X):
    return [self._predict_one(row, self.root) for row in X]


class PureRandomForest:

  def __init__(
      self, n_trees=25, max_depth=12, min_samples_split=2, seed=42
  ):
    self.n_trees = n_trees
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.seed = seed
    self.trees = []

  def fit(self, X, y):
    random.seed(self.seed)
    self.trees = []
    n_samples = len(X)
    n_features = len(X[0]) if n_samples > 0 else 7
    num_feat = max(1, int(math.sqrt(n_features)))

    for _ in range(self.n_trees):
      indices = [random.randrange(n_samples) for _ in range(n_samples)]
      boot_X = [X[i] for i in indices]
      boot_y = [y[i] for i in indices]
      tree = DecisionTree(
          max_depth=self.max_depth,
          min_samples_split=self.min_samples_split,
          num_features=num_feat,
      )
      tree.fit(boot_X, boot_y)
      self.trees.append(tree)
    return self

  def predict(self, X):
    if not self.trees:
      return []
    all_preds = [tree.predict(X) for tree in self.trees]
    results = []
    for i in range(len(X)):
      votes = {}
      for tree_preds in all_preds:
        lbl = tree_preds[i]
        votes[lbl] = votes.get(lbl, 0) + 1
      best_lbl = max(votes.items(), key=lambda item: item[1])[0]
      results.append(best_lbl)
    return results


def load_dataset(csv_path):
  X = []
  y = []
  with open(csv_path, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
      X.append([
          float(r["N"]),
          float(r["P"]),
          float(r["K"]),
          float(r["temperature"]),
          float(r["humidity"]),
          float(r["ph"]),
          float(r["rainfall"]),
      ])
      y.append(r["label"].strip())

  # Wheat (Gehun), Mustard (Sarson), aur Sugarcane (Ganna) ke sample rows
  extra_samples = [
      # --- WHEAT (Gehun) ---
      ([50.0, 40.0, 45.0, 20.0, 55.0, 6.5, 70.0], "wheat"),
      ([52.0, 42.0, 44.0, 19.5, 56.0, 6.6, 68.0], "wheat"),
      ([48.0, 38.0, 42.0, 20.5, 54.0, 6.4, 72.0], "wheat"),
      ([55.0, 45.0, 46.0, 18.5, 58.0, 6.7, 65.0], "wheat"),
      ([46.0, 36.0, 40.0, 21.0, 52.0, 6.5, 75.0], "wheat"),
      ([54.0, 40.0, 45.0, 19.0, 57.0, 6.8, 62.0], "wheat"),
      ([49.0, 39.0, 43.0, 20.2, 55.5, 6.6, 70.0], "wheat"),
      ([51.0, 41.0, 44.0, 19.8, 54.5, 6.5, 69.0], "wheat"),
      ([53.0, 43.0, 47.0, 18.8, 59.0, 6.7, 64.0], "wheat"),
      ([47.0, 37.0, 41.0, 20.8, 53.0, 6.4, 74.0], "wheat"),
      ([56.0, 44.0, 46.0, 18.2, 60.0, 6.9, 60.0], "wheat"),
      ([50.0, 40.0, 43.0, 20.0, 55.0, 6.6, 71.0], "wheat"),
      ([52.0, 39.0, 45.0, 19.2, 56.5, 6.5, 67.0], "wheat"),
      ([48.0, 41.0, 42.0, 20.4, 54.0, 6.7, 73.0], "wheat"),
      ([54.0, 42.0, 45.0, 19.0, 57.5, 6.6, 66.0], "wheat"),
      # --- MUSTARD (Sarson) ---
      ([60.0, 40.0, 35.0, 18.0, 50.0, 6.8, 45.0], "mustard"),
      ([62.0, 42.0, 36.0, 17.5, 52.0, 7.0, 40.0], "mustard"),
      ([58.0, 38.0, 34.0, 19.0, 48.0, 6.6, 50.0], "mustard"),
      ([65.0, 45.0, 38.0, 17.0, 54.0, 7.1, 38.0], "mustard"),
      ([55.0, 36.0, 32.0, 20.0, 46.0, 6.5, 52.0], "mustard"),
      ([61.0, 40.0, 35.0, 18.5, 51.0, 6.9, 44.0], "mustard"),
      ([59.0, 39.0, 34.0, 19.2, 49.0, 6.7, 47.0], "mustard"),
      ([63.0, 43.0, 37.0, 17.8, 53.0, 7.0, 42.0], "mustard"),
      ([57.0, 37.0, 33.0, 19.5, 47.0, 6.6, 49.0], "mustard"),
      ([60.0, 41.0, 36.0, 18.2, 50.5, 6.8, 45.0], "mustard"),
      ([64.0, 44.0, 38.0, 17.2, 55.0, 7.2, 39.0], "mustard"),
      ([56.0, 38.0, 34.0, 19.8, 48.5, 6.7, 51.0], "mustard"),
      ([61.0, 41.0, 35.0, 18.6, 51.5, 6.9, 43.0], "mustard"),
      ([58.0, 40.0, 35.0, 19.1, 49.5, 6.8, 46.0], "mustard"),
      ([62.0, 42.0, 37.0, 18.0, 52.5, 7.0, 41.0], "mustard"),
      # --- SUGARCANE (Ganna) ---
      ([120.0, 60.0, 80.0, 28.0, 75.0, 6.8, 160.0], "sugarcane"),
      ([125.0, 65.0, 85.0, 29.0, 78.0, 7.0, 170.0], "sugarcane"),
      ([115.0, 55.0, 75.0, 27.0, 72.0, 6.6, 150.0], "sugarcane"),
      ([130.0, 62.0, 82.0, 30.0, 80.0, 6.9, 180.0], "sugarcane"),
      ([110.0, 58.0, 78.0, 26.5, 74.0, 6.5, 155.0], "sugarcane"),
      ([122.0, 60.0, 80.0, 28.5, 76.0, 6.8, 165.0], "sugarcane"),
      ([118.0, 57.0, 76.0, 27.5, 73.0, 6.7, 158.0], "sugarcane"),
      ([128.0, 64.0, 84.0, 29.5, 79.0, 7.0, 175.0], "sugarcane"),
      ([112.0, 56.0, 74.0, 26.0, 71.0, 6.6, 145.0], "sugarcane"),
      ([124.0, 61.0, 81.0, 28.2, 77.0, 6.9, 168.0], "sugarcane"),
      ([116.0, 59.0, 77.0, 27.2, 74.5, 6.7, 152.0], "sugarcane"),
      ([126.0, 63.0, 83.0, 29.2, 78.5, 6.9, 172.0], "sugarcane"),
      ([114.0, 55.0, 75.0, 26.8, 72.5, 6.5, 148.0], "sugarcane"),
      ([121.0, 60.0, 79.0, 28.4, 76.5, 6.8, 162.0], "sugarcane"),
      ([127.0, 62.0, 82.0, 29.6, 79.5, 7.0, 176.0], "sugarcane"),
  ]

  for features, label in extra_samples:
    X.append(features)
    y.append(label)

  return X, y


if __name__ == "__main__":
  data_path = os.path.join("dataset", "Crop_recommendation.csv")
  X, y = load_dataset(data_path)

  random.seed(42)
  indices = list(range(len(X)))
  random.shuffle(indices)
  split = int(len(indices) * 0.8)

  train_idx = indices[:split]
  test_idx = indices[split:]

  X_train = [X[i] for i in train_idx]
  y_train = [y[i] for i in train_idx]
  X_test = [X[i] for i in test_idx]
  y_test = [y[i] for i in test_idx]

  rf = PureRandomForest(n_trees=25, max_depth=12, min_samples_split=2, seed=42)
  rf.fit(X_train, y_train)

  preds = rf.predict(X_test)
  correct = sum(1 for p, actual in zip(preds, y_test) if p == actual)
  acc = (correct / len(y_test)) * 100 if y_test else 0.0
  print(f"Total rows: {len(X)} | Model Accuracy: {acc:.2f}%")

  # Full dataset par final model train karein
  final_rf = PureRandomForest(
      n_trees=25, max_depth=12, min_samples_split=2, seed=42
  )
  final_rf.fit(X, y)

  os.makedirs("models", exist_ok=True)
  model_path = os.path.join("models", "crop_model.pkl")
  with open(model_path, "wb") as f:
    pickle.dump(final_rf, f)

  print(f"Success! Model saved at: {model_path}")