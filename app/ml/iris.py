"""A classic (non-LLM) ML model: a flower classifier trained with scikit-learn.

It shows the most common ML-serving pattern: load/train the model ONCE when the
server starts, keep it in memory, and only call .predict() per request.
In a real project you'd load a saved file instead (joblib.load("model.pkl")).
"""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


class IrisClassifier:
    def __init__(self):
        data = load_iris()
        self.species = list(data.target_names)  # ['setosa', 'versicolor', 'virginica']
        self.model = LogisticRegression(max_iter=1000).fit(data.data, data.target)

    def predict(self, sepal_length: float, sepal_width: float,
                petal_length: float, petal_width: float) -> dict[str, float]:
        """Returns the probability of each species."""
        probabilities = self.model.predict_proba(
            [[sepal_length, sepal_width, petal_length, petal_width]])[0]
        return {name: round(float(p), 4) for name, p in zip(self.species, probabilities)}
