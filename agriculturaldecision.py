import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options for Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.3f}'.format)

data = pd.read_csv("crop.csv")

def explore_dataframe(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe())

# Explore the loaded DataFrame
explore_dataframe(data, head=2)

print("Average ratio of Nitrogen in Soil:{0: .2f}".format(data["N"].mean()))
print("Average ratio of Phosphorus in Soil:{0: .2f}".format(data["P"].mean()))
print("Average ratio of Potassium in Soil:{0: .2f}".format(data["K"].mean()))
print("Average ratio of Temperature in celcius:{0: .2f}".format(data["temperature"].mean()))
print("Average relative humdity in %:{0: .2f}".format(data["humidity"].mean()))
print("Average pH value of the Soil:{0: .2f}".format(data["ph"].mean()))
print("Average rainfall in mm:{0: .2f}".format(data["rainfall"].mean()))

def summary_stats(crop_name):
    # Check if the crop_name exists in the data
    if crop_name not in data['label'].unique():
        print(f"No data found for the crop: {crop_name}")
        return

    subset_data = data[data['label'] == crop_name]

    # Define the list of columns and their descriptions for which statistics are to be calculated
    columns_desc = {
        'N': 'Nitrogen',
        'P': 'Phosphorus',
        'K': 'Potassium',
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'ph': 'pH',
        'rainfall': 'Rainfall'
    }

    # Loop over the columns and print statistics
    for col, desc in columns_desc.items():
        print("---------------------------------------------")
        print(f"Statistics for {desc}")
        print(f"Minimum {desc} required: {subset_data[col].min():.2f}")
        print(f"Average {desc} required: {subset_data[col].mean():.2f}")
        print(f"Maximum {desc} required: {subset_data[col].max():.2f}")

# For usage in PyCharm or any other IDE, manually enter a crop name.
crop_name = input("Please enter a crop name: ")
summary_stats(crop_name)

def compare():
    conditions_list = ['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']

    # Prompt the user to choose a condition
    print("Please choose a condition from the following list:")

    for i, cond in enumerate(conditions_list, 1):
        print(f"{i}. {cond}")

    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))

            if 1 <= choice <= len(conditions_list):
                condition = conditions_list[choice - 1]

                above_avg_crops = data[data[condition] > data[condition].mean()]['label'].unique()
                below_avg_crops = data[data[condition] <= data[condition].mean()]['label'].unique()

                print(f"\nCrops which require greater than average {condition}:\n{' '.join(above_avg_crops)}")
                print("-------------------------------------")
                print(f"Crops which require less than average {condition}:\n{' '.join(below_avg_crops)}")
                break

            else:
                print("Invalid choice. Please select a number from the list.")

        except ValueError:
            print("Please enter a valid number from the list.")


# Call the function to execute
compare()

import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(subplot_position, data_column, color, xlabel):
    plt.subplot(2, 4, subplot_position)
    sns.histplot(data[data_column], kde=True, color=color, bins=30)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(axis='y')
    plt.tight_layout()

# Setting up figure size
plt.figure(figsize=(18, 10))

# Using a consistent color palette
palette = sns.color_palette("coolwarm", 7)

# Plotting each agricultural condition
plot_distribution(1, 'N', palette[0], 'Ratio of Nitrogen')
plot_distribution(2, 'P', palette[1], 'Ratio of Phosphorous')
plot_distribution(3, 'K', palette[2], 'Ratio of Potassium')
plot_distribution(4, 'temperature', palette[3], 'Temperature')
plot_distribution(5, 'rainfall', palette[4], 'Rainfall')
plot_distribution(6, 'humidity', palette[5], 'Humidity')
plot_distribution(7, 'ph', palette[6], 'pH Level')

# Adding a title
plt.suptitle('Distribution for Agricultural Conditions', fontsize=20, y=1.05)
plt.show()

def print_crops_with_conditions(condition_column, threshold, comparison='greater', descriptor=''):
    """Prints crops based on given conditions"""

    if comparison == 'greater':
        crops = data[data[condition_column] > threshold]['label'].unique()
        print(f"Crops requiring {descriptor} {threshold} of {condition_column}:", crops)
    elif comparison == 'lesser':
        crops = data[data[condition_column] < threshold]['label'].unique()
        print(f"Crops requiring {descriptor} {threshold} of {condition_column}:", crops)


print("Interesting Observations")
print("------------------------")

print_crops_with_conditions('N', 120, 'greater', 'more than')
print_crops_with_conditions('P', 100, 'greater', 'more than')
print_crops_with_conditions('K', 200, 'greater', 'more than')
print_crops_with_conditions('rainfall', 200, 'greater', 'more than')
print_crops_with_conditions('temperature', 10, 'greater', 'less than')
print_crops_with_conditions('temperature', 40, 'greater', 'more than')
print_crops_with_conditions('humidity', 20, 'greater', 'less than')
print_crops_with_conditions('ph', 4, 'lesser', 'less than')
print_crops_with_conditions('ph', 9, 'greater', 'more than')

def get_crops_based_on_conditions(temp_range=None, humidity_range=None, rainfall_range=None, season_name=""):
    """Get crops based on provided conditions."""

    conditions = True  # start with a boolean series of True values
    if temp_range:
        conditions &= data['temperature'].between(*temp_range)
    if humidity_range:
        conditions &= data['humidity'].between(*humidity_range)
    if rainfall_range:
        conditions &= data['rainfall'].between(*rainfall_range)

    unique_crops = data[conditions]['label'].unique()
    return unique_crops


# Define conditions for each season
season_conditions = {
    'Summer': ((30, 50), (50, 100), None),
    'Winter': ((0, 20), (30, 100), None),
    'Rainy': (None, (30, 100), (200, 300))
}

# Display crops for each season
for season, conditions in season_conditions.items():
    print(f"{season} Crops")
    print(get_crops_based_on_conditions(*conditions, season_name=season))
    print("-----------------------------------------")

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Drop the 'label' column to prepare data for clustering
features = data.drop('label', axis=1).values

# Determine the optimal number of clusters using the elbow method
wcss = []  # within-cluster sum of squares
clusters_range = range(1, 11)

for n_clusters in clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

# Visualize the elbow method to identify the optimal number of clusters
plt.figure(figsize=(10, 4))
plt.plot(clusters_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal Number of Clusters', fontsize=20)
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()

# K-Means Clustering Analysis
# Setting up KMeans model and predicting clusters
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(features)

# Merging cluster assignments with original labels
merged_data = pd.concat([pd.DataFrame(clusters, columns=['cluster']), data['label']], axis=1)

# Displaying crops in each cluster
print("\nResults from KMeans Clustering Analysis:\n")
for i in range(4):
    print(f"Crops in Cluster {i + 1}:", merged_data[merged_data['cluster'] == i]['label'].unique())
    print("----------------------------------------------------------------")

# Data Preparation

# Splitting the data into features and target variables
X = data.drop('label', axis=1)
y = data['label']

# Dividing the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("\nData Split Information:")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# ==============================================
# Predictive Modeling and Evaluation
# ==============================================

# Initializing and training the Logistic Regression model
logreg = LogisticRegression(max_iter=1000)  # Increased max_iter for better convergence
logreg.fit(X_train, y_train)

# Making predictions
predicted_values = logreg.predict(X_test)

# Displaying the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_test, predicted_values), annot=True, cmap='Wistia')
plt.title('Logistic Regression Confusion Matrix', fontsize=15)
plt.show()

# Showing the classification report
print("\nClassification Report:\n")
print(classification_report(y_test, predicted_values))

sample_data = np.array([[90, 40, 40, 20, 80, 7, 200]])
suggested_crop = logreg.predict(sample_data)
print(f"\nThe recommended crop for the given climatic condition is: {suggested_crop[0]}")

# Sampling
# Define three strategically distinct sample data points with different climatic conditions
samples = np.array([
    [20, 80, 10, 25, 50, 6, 100],  # Sample 1: This is skewed towards more humid conditions
    [90, 40, 80, 15, 60, 8, 210],  # Sample 2: This is skewed towards hot and dry conditions
    [65, 55, 50, 28, 90, 5.5, 80]  # Sample 3: This is a balanced sample
])

# Predict the most suitable crop for each sample using the trained logistic regression model
recommended_crops = logreg.predict(samples)

# Display the recommended crop for each sample
for i, crop in enumerate(recommended_crops, 1):
    print(f"For Sample {i}, the recommended crop is: {crop}")






