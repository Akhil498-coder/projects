
 
CIFAR-10 CNN Image Classification
This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. This project includes functionality to train the model on subsets of the dataset.

Features
Custom CNN architecture.

Training on partial CIFAR-10 datasets.

Balanced data subset creation.

GPU acceleration (if available).

Training progress visualization.

Model checkpointing.

CIFAR-10 Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes: plane, car, bird, cat, deer, dog, frog, horse, ship, and truck.

Getting Started
Prerequisites
Python 3.x

PyTorch

Torchvision

Matplotlib

NumPy

Install dependencies:

pip install torch torchvision matplotlib numpy

Running the Code
Clone this repository.

Open the project_image.ipynb (or your Python script).

Adjust configuration parameters like DATASET_FRACTION or TRAIN_SAMPLES as needed.

Run the script.

The script will download the dataset, train the model, and display results.

Project Structure


.
├── project_image.ipynb
└── cifar10_models/
    ├── best_model_partial.pth
    └── checkpoint_partial_epoch_X.pth

Results
Training output will show dataset usage, total time, and final accuracy.

Example:

TRAINING COMPLETE!
Dataset used: 10,000 train, 2,000 test samples
Percentage of full dataset: 20.0%
Total time: 1.4 minutes
Best test accuracy: 59.00%











#Titanic Dataset Analysis


This repository contains a Jupyter Notebook (statisticsly.ipynb) that performs an exploratory data analysis (EDA) and preprocessing pipeline on the famous Titanic dataset. The goal is to prepare the data for machine learning model training by handling missing values, engineering new features, and transforming existing ones.

Project Overview
The notebook covers the following key steps:

Initial Data Exploration:

Loading the dataset.

Displaying basic information (.info(), .head()).

Checking for missing values.

Reviewing column names and dataset shape.

Data Cleaning:

Age Imputation: Missing 'Age' values are imputed using the median age, grouped by 'Pclass' and 'Sex'. This strategy accounts for potential variations in age distribution across different passenger classes and genders.

Embarked Imputation: Missing 'Embarked' values are filled with the mode (most frequent port of embarkation).

'Deck' Column Handling: The 'Deck' column is dropped due to a high percentage of missing values, as extensive imputation would introduce significant bias.

Exploratory Data Analysis (EDA):

Visualizations to understand relationships between features and the 'Survived' target variable.

Analysis of survival rates by 'Sex' and 'Passenger Class'.

Feature Engineering:

Creation of new features to potentially improve model performance:

family_size: Combines 'SibSp' (siblings/spouses) and 'Parch' (parents/children) to represent the total number of family members.

fare_per_person: Calculates the fare divided by the family_size (or 1 if alone).

age_category: Categorizes 'Age' into bins (Child, Adult, Senior).

is_alone: A binary feature indicating if a passenger was traveling alone.

Data Preprocessing:

Categorical Encoding: One-hot encoding is applied to categorical features ('Sex', 'Embarked', 'Age_Category', 'Pclass', 'Who', 'Alone', 'Title').

Numerical Scaling: Numerical features ('Age', 'Fare', 'Family_Size', 'Fare_Per_Person') are standardized using StandardScaler.

Data Splitting: The dataset is split into training (80%) and testing (20%) sets for model development and evaluation.

Getting Started
Prerequisites
Python 3.x

Pandas

Seaborn

Matplotlib

NumPy

Scikit-learn

You can install the necessary libraries using pip:

pip install pandas seaborn matplotlib numpy scikit-learn

Running the Notebook
Clone the repository:

git clone https://github.com/YourUsername/your-repo-name.git
cd your-repo-name

(Replace your-repo-name and YourUsername with your actual GitHub details.)

Open the Jupyter Notebook:
Launch Jupyter Lab or Jupyter Notebook and open statisticsly.ipynb:

jupyter notebook statisticsly.ipynb

or

jupyter lab statisticsly.ipynb

Execute Cells:
Run all cells in the notebook sequentially. The output will guide you through each step of the analysis, cleaning, feature engineering, and preprocessing.


  # Main Jupyter Notebook for Titanic data analysis

Final Dataset Statistics
After running the notebook, the processed dataset will be ready for machine learning. Key statistics are printed at the end of the notebook:

FINAL DATASET STATISTICS:
Total samples: 891
Overall survival rate: 38.38%
Features for modeling: [Number of features after encoding]
Training samples: [Number of training samples]
Test samples: [Number of test samples]

License
This project is open-source and available under the MIT License.
Final test accuracy: 58.10%

License
This project is open-source under the MIT License.
