Traffic Congestion Classification Using Random Forest

1. Overview:

This project focuses on predicting traffic congestion levels using machine learning, specifically a Random Forest Classifier. The dataset contains various traffic-related features such as speed, number of trips, and congestion indicators to classify traffic conditions.

2. Dataset:
The dataset is based on GTFS (General Transit Feed Specification) and contains key traffic parameters.
Features and Target:
	arrival_time – Scheduled arrival time (converted to numeric).
	speed – Speed of the vehicle at data capture.
	Number_of_trips – Number of recorded trips in a timeframe.
	SRI – Smooth Road Index (higher values indicate congestion).
	Degree_of_congestion – Target variable with categories like Smooth, Medium, Heavy, etc.



3. Technologies Used:
Python
Libraries: pandas, NumPy, scikit-learn, matplotlib, seaborn

4. Data Preprocessing:
- Converted arrival_time into a numerical format.
- Handled missing values using median imputation.
- Removed SRI feature to prevent overfitting, its high correlation with the target led to an unrealistic accuracy boost.
- Applied Label Encoding to categorical features. (trip_id and Degree_of_congestion)

5. Model Implementation:
- Random Forest Classifier with n_estimators=100.
- Train-Test Split: 80% training, 20% testing.
- Accuracy:
	With SRI feature: 99.78%
	Without SRI feature: 87.82%.
