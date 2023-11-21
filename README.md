Overview
This project aims to predict future sales based on historical data. The prediction is made using machine learning techniques, and the model is trained on a dataset containing information about past sales.

Technologies Used
Python
Jupyter Notebook
scikit-learn
pandas
matplotlib
seaborn
Dataset
The dataset used for training and testing the model contains the following features :

Date: Date of the sale
Product ID: Unique identifier for each product
Units Sold: Number of units sold on a given day
Price: Price of the product on that day
Promotion: Binary value indicating whether a promotion was active (1) or not (0)
Project Structure
data/: Contains the dataset used for training and testing.
notebooks/: Jupyter notebooks for data exploration, model training, and evaluation.
01_Data_Exploration.ipynb: Exploring the dataset.
02_Model_Training.ipynb: Training the machine learning model.
03_Model_Evaluation.ipynb: Evaluating the model performance.
src/: Python scripts for data preprocessing, model training, and utility functions.
data_preprocessing.py: Preprocesses the raw data.
model.py: Defines and trains the sales prediction model.
utils.py: Utility functions used throughout the project.
requirements.txt: List of required Python packages. Use pip install -r requirements.txt to install dependencies.
README.md: Project overview, setup instructions, and usage guide.
How to Use
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/sales-prediction-project.git
cd sales-prediction-project
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Explore the Jupyter notebooks in the notebooks/ directory to understand the data and model training process.

Run the model training notebook (02_Model_Training.ipynb)
