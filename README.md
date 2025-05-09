````markdown
# Time Series Forecasting on Energy Consumption

This project involves time series forecasting on the "Hourly Energy Consumption" dataset. The goal is to predict energy consumption (MW) using historical data and XGBoost model. The dataset used for this project is the PJME hourly energy consumption data.

## Libraries Used
- `kagglehub`: For downloading datasets from Kaggle
- `pandas`: For data manipulation and analysis
- `numpy`: For numerical computations
- `matplotlib`: For data visualization
- `seaborn`: For statistical data visualization
- `xgboost`: For the machine learning model
- `sklearn`: For evaluation metrics

## Dataset

The dataset used in this project is "Hourly Energy Consumption" from Kaggle: [Link to Dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).

After downloading, the dataset is loaded into a pandas DataFrame for analysis and preprocessing.

### Steps:
1. **Data Loading & Exploration:**
   - Load the dataset using pandas.
   - Set the index of the DataFrame to the 'Datetime' column and convert it to `datetime` type.
   - Visualize the energy consumption over time.

2. **Train/Test Split:**
   - Split the dataset into training and test sets using the year 2015 as a cutoff.

3. **Feature Engineering:**
   - Generate time-based features like hour, day of the week, month, quarter, and year.

4. **Visualization:**
   - Visualize the relationship between features and target variable (`PJME_MW`) using boxplots.

5. **Modeling:**
   - Train an XGBoost regressor model using selected features.
   - Evaluate the model using the root mean square error (RMSE) score.

6. **Prediction & Evaluation:**
   - Predict energy consumption on the test set.
   - Visualize the predictions vs actual data.
   - Calculate prediction error and analyze the worst predicted days.

## Steps to Run the Code

1. Clone this repository and install necessary dependencies:
   ```bash
   pip install -r requirements.txt
````

2. Run the Jupyter notebook or Python script to load the dataset, perform feature engineering, train the model, and evaluate the predictions.

   ```python
   # Example to run the dataset download:
   import kagglehub
   path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
   print("Path to dataset files:", path)
   ```

3. The notebook will download the dataset and proceed with:

   * Plotting the energy consumption data.
   * Splitting the data into training and test sets.
   * Creating time-based features.
   * Training an XGBoost model and making predictions.
   * Plotting the prediction results and evaluating the model performance.

## Results

* **Visualization of Energy Consumption**: The dataset is plotted to show the energy consumption trends.
* **Feature Importance**: The features contributing the most to energy consumption predictions are shown.
* **Predictions vs Actual Data**: A comparison of predicted and actual energy consumption.

### RMSE Score

After testing the model, the RMSE score is calculated for the test set, giving an indication of the model's accuracy.

## Files in This Repository

* `EnergyConsumptionForecasting.ipynb`: The Jupyter notebook containing all the steps, from data loading to model training.
* `requirements.txt`: A list of required packages.
* `data/`: Folder where the dataset will be stored after downloading from Kaggle.

## Acknowledgements

* [Kaggle: Hourly Energy Consumption Dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
* [Time Series Forecasting in Machine Learning (Reference)](https://engineering.99x.io/time-series-forecasting-in-machine-learning-3972f7a7a467)

```
