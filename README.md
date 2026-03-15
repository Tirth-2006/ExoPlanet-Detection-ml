# Exoplanet Detection using Machine Learning

This project explores detecting **exoplanets** using machine learning techniques applied to **stellar light curve data**.

When a planet passes in front of a star, the brightness of the star slightly decreases.
This event is called a **transit**. By analyzing these small dips in brightness, machine learning models can identify potential exoplanets.

---

## Project Objective

The goal of this project is to build a **machine learning pipeline** capable of detecting potential exoplanets by analyzing patterns in stellar brightness data.

The project focuses on:

* Data preprocessing
* Feature preparation
* Model training
* Model evaluation
* Prediction of potential exoplanets

---

## Dataset

The project uses **light curve data** similar to datasets collected by space telescopes such as:

* NASA **Kepler Mission**
* NASA **TESS Mission**

A **light curve** represents how the brightness of a star changes over time.

If a planet crosses in front of the star, the brightness slightly dips. These dips can be detected and analyzed using machine learning models.

---

## Machine Learning Pipeline

The notebook follows a structured pipeline:

1. Load and explore the dataset
2. Data preprocessing and cleaning
3. Feature preparation
4. Train machine learning models
5. Evaluate model performance
6. Predict potential exoplanet candidates

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn

---

## Project Structure

```
ExoPlanet-Detection-ml
│
├── exoplanet_detection_pipeline.ipynb   # Main ML notebook
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
├── LICENSE
└── .gitignore
```

---

## Installation

Clone the repository:

```
git clone https://github.com/Tirth-2006/ExoPlanet-Detection-ml.git
```

Move into the project folder:

```
cd ExoPlanet-Detection-ml
```

Install required libraries:

```
pip install -r requirements.txt
```

Run the notebook:

```
jupyter notebook
```

---

## Results

The machine learning model is trained to classify whether a light curve corresponds to a potential **exoplanet transit**.

Evaluation metrics used include:

* Accuracy
* Precision
* Recall
* Confusion Matrix

These metrics help assess the model's ability to correctly identify planetary signals.

---

## Future Improvements

Possible improvements for the project include:

* Better feature engineering from light curves
* Hyperparameter optimization
* Deep learning models for time-series analysis
* Larger datasets from Kepler or TESS missions
* Automated transit detection algorithms

---

## Author

**Tirth**

Machine learning enthusiast exploring scientific applications of AI.
