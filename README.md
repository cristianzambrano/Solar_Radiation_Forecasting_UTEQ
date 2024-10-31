
# Solar Radiation Forecasting in UTEQ using Machine Learning

This repository contains a script for solar radiation forecasting at the State Technical University of Quevedo (UTEQ) using Machine Learning models. The code focuses on predicting solar radiation through various modeling techniques, with real-time visualization and weather data retrieval via an API.

This work has been presented at the following conference:

**Troncoso, J. A., Quijije, Á. T., Oviedo, B., & Zambrano-Vega, C. (2024). Solar Radiation Prediction in the UTEQ based on Machine Learning Models. 9th International Congress on Information and Communication Technology. London, United Kingdom. Lecture Notes in Networks and Systems, Springer, Singapore (in press).**

## Project Description

This project predicts solar radiation at hourly intervals using regression models such as K-Nearest Neighbors, Random Forest, Polynomial Regression, and Gradient Boosting. The system also allows comparing results with real data obtained from a pyranometer.

### Features

1. **Data Loading and Preprocessing**: Uses stored temperature and solar radiation data in CSV format.
2. **Weather API Integration**: Retrieves current temperature and temperature forecasts from Meteosource API.
3. **Modeling and Prediction**: Predicts solar radiation using multiple machine learning models.
4. **Visualization**: Utilizes `Altair` and `Matplotlib` for interactive charts of predictions and historical data.
5. **User Interface**: Built with `Streamlit`, it includes customizable settings to select models, data sources, and visualizations.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <REPOSITORY_URL>
   cd <REPOSITORY_NAME>
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the requirements**:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

This project requires the following Python libraries:

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `Pillow`
- `altair`
- `requests`

You can install these dependencies by running:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn Pillow altair requests
```

## Execution

To run the application in `Streamlit`, use the following command:

```bash
streamlit run <script_name>.py
```

Replace `<script_name>.py` with the actual name of the main script file.

## Usage

1. Run the `streamlit` command above.
2. In your browser, select the desired configuration:
   - **Action Type**: Choose between `Predict` or `Test`.
   - **Machine Learning Model**: Choose among K-Nearest Neighbors, Random Forest, Polynomial Regression, and Gradient Boosting.
   - **Data Source**: Select current data from Meteosource API or historical data.

## Contribution

Contributions are welcome! To make changes or add features, please open a pull request and describe your changes in detail.

## License

This project is licensed under the terms of the MIT License.
