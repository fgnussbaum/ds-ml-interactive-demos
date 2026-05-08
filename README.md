# Data Science and Machine Learning Interactive Demos

Browser-based, interactive visualizations for core machine learning concepts. Built for hands-on exploration.

## Available Demos

| Demo | Description |
|---|---|
| Gradient Descent | Step through gradient descent on a regression problem |
| Linear Regression | Draw your own points and fit a line in real time |
| Overfitting | Explore how polynomial degree affects model fit |
| Classification | Adjust a decision threshold and read the confusion matrix |
| Decision Tree | Visualize tree splits and compare with a random forest |

Demos use either synthetic data or seaborn datasets.

## Quickstart

```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

Then open [http://localhost:8000](http://localhost:8000).