import pandas as pd
import numpy as np
from scipy.stats import kstest
import statsmodels.api as sm

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file = r'E:/1 PROJECT/MHIA/303Workshop/project/breast+cancer+wisconsin+original/wdbc.data'
bca = pd.read_csv(file, header=None)

# Rename columns
column_names = [
    'ID', 'Diagnosis',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

bca.columns = column_names

# Move the 'Diagnosis' column to the last
diagnosis_column = bca.pop('Diagnosis')
bca['Diagnosis'] = diagnosis_column

# Add a new column 'unique ID' with unique values
bca.insert(0, 'PID', range(1, len(bca) + 1))

# Remove the 'Patient ID' column
bca.drop('ID', axis=1, inplace=True)

# Check for duplicate rows
duplicate_rows = bca[bca.duplicated()]

# OUTLIERS
# Selecting all continuous variables for outlier detection
continuous_columns = bca.columns[1:31]

# Create individual horizontal box plots for each continuous variable
plt.figure(figsize=(15, 20))
for i, column in enumerate(continuous_columns, 1):
    plt.subplot(6, 5, i)
    sns.boxplot(y=bca[column], width=0.3, color='lightblue', linewidth=2)
    plt.title(f'Box Plot for {column}')

outlier_threshold = 1.5
# Calculate the percentage of outliers for each column
outliers_percentage = ((bca[continuous_columns] < bca[continuous_columns].quantile(0.25) - outlier_threshold * (bca[continuous_columns].quantile(0.75) - bca[continuous_columns].quantile(0.25))) |
                       (bca[continuous_columns] > bca[continuous_columns].quantile(0.75) + outlier_threshold * (bca[continuous_columns].quantile(0.75) - bca[continuous_columns].quantile(0.25)))).mean() * 100

# Calculate statistics for each continuous variable
statistics = []

for column in continuous_columns:
    q1 = bca[column].quantile(0.25)
    q3 = bca[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    min_value = bca[column].min()
    max_value = bca[column].max()

    statistics.append({
        'Variable': column,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Minimum Value': min_value,
        'Maximum Value': max_value
    })

# Create a DataFrame from the list of dictionaries
statistics_df = pd.DataFrame(statistics)


# OUTLIERS CORRECTION
# Create a copy of the original DataFrame
bca_capped = bca.copy()
start_column = 1
end_column = 31

# Iterate over each variable in columns start_column to end_column
for column in bca_capped.columns[start_column:end_column + 1]:
    # Check if the column contains numeric data
    if pd.api.types.is_numeric_dtype(bca_capped[column]):
        q1 = bca[column].quantile(0.25)
        q3 = bca[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        
        # Cap values above the upper bound to the upper bound value
        bca_capped[column] = np.clip(bca_capped[column], lower_bound, upper_bound)


# Check normality using Kolmogorov–Smirnov test and QQ plot
for column in continuous_columns:
    # Kolmogorov–Smirnov test
    kstest_result = kstest(bca[column], 'norm')
    p_value_formatted = "{:.3f}".format(kstest_result.pvalue)
    print(f'Kolmogorov–Smirnov test for {column}: p-value = {kstest_result.pvalue}')

# Combine QQ plots into subplots
num_plots = len(continuous_columns)
num_cols = 6
num_rows = 5

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 15))

for i, column in enumerate(continuous_columns):
    row = i // num_cols
    col = i % num_cols

    # QQ plot
    sm.qqplot(bca[column], line='s', ax=axes[row, col])
    axes[row, col].set_title(f'QQ Plot for {column}')

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Calculate the Spearman correlation matrix
correlation_matrix = bca[continuous_columns].corr(method='spearman')

# Plot the heatmap
plt.figure(figsize=(30, 35))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Spearman Correlation Heatmap for Continuous Variables')
plt.show()


# Create individual box plots for each continuous variable in the corrected DataFrame

plt.figure(figsize=(15, 20))
for i, column in enumerate(continuous_columns, 1):
    plt.subplot(6, 5, i)
    sns.boxplot(y=bca_capped[column], width=0.3, color='lightblue', linewidth=2)
    plt.title(f'Box Plot for {column}')

plt.tight_layout()
plt.show()

# Missing values
# Check for missing values
missing_values = bca.isnull().sum()

"""DATA SCALLING"""
# Create a copy of the DataFrame for feature scaling
bca_capped_scaled = bca_capped.copy()

# Select only the numeric columns for feature scaling
numeric_columns = bca_capped_scaled.select_dtypes(include='number').columns

# Instantiate StandardScaler
scaler = StandardScaler()

# Perform feature scaling on selected columns
bca_capped_scaled[continuous_columns] = scaler.fit_transform(bca_capped_scaled[continuous_columns])


# Create individual box plots for each continuous variable in the scaled DataFrame

plt.figure(figsize=(15, 20))
for i, column in enumerate(continuous_columns, 1):
    plt.subplot(6, 5, i)
    sns.boxplot(y=bca_capped_scaled[column], width=0.3, color='lightblue', linewidth=2)
    plt.title(f'Box Plot for {column}')

#SMOTE
# Separate features and target variable
X = bca_capped_scaled.drop('Diagnosis', axis=1)
y = bca_capped_scaled['Diagnosis']

# SMOTE
smote = SMOTE(random_state=43)

# Fit and apply SMOTE on the dataset
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine the resampled data into a new DataFrame
bca_capped_scaled_balanced = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='Diagnosis')], axis=1)

# Display the class distribution after oversampling
print("Class distribution before oversampling:")
print(bca_capped_scaled['Diagnosis'].value_counts())

print("Class distribution after oversampling:")
print(bca_capped_scaled_balanced['Diagnosis'].value_counts())

plt.figure(figsize=(10, 6))
sns.countplot(x='Source', data=pd.concat([bca_capped_scaled.assign(Source='Before SMOTE'), bca_capped_scaled_balanced.assign(Source='After SMOTE')]),
              hue='Diagnosis', palette='Set1')
plt.title('Class Distribution Before and After SMOTE Oversampling')
plt.xlabel('SMOTE Status')
plt.ylabel('Count')
plt.legend(title='Diagnosis')
plt.show()

# Create a copy of the DataFrame for mapping and modification
bca_processed = bca_capped_scaled_balanced.copy()

# Map 'M' to 1 and 'B' to 0 in the 'Diagnosis' column
bca_processed['Diagnosis'] = bca_processed['Diagnosis'].map({'M': 1, 'B': 0})

# Create a new column 'dx' with 1 for 'M' and 0 for 'B'
bca_processed['Malignant'] = (bca_processed['Diagnosis'] == 1).astype(int)

# Drop the original 'Diagnosis' column
bca_processed = bca_processed.drop('Diagnosis', axis=1)


"""SPLITTING DATA"""
# Perform train-test split (80:20) using bca_balanced_2
X_train, X_test, y_train, y_test = train_test_split(bca_processed.drop('Malignant', axis=1),
                                                    bca_processed['Malignant'],
                                                    test_size=0.2,
                                                    random_state=43)

# Create DataFrames for training and testing sets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# # Specify filenames for training and testing CSV files
# train_csv_filename = '/content/bca_train.csv'
# test_csv_filename = '/content/bca_test.csv'

# # Save training and testing sets to CSV files
# train_data.to_csv(train_csv_filename, index=False)
# test_data.to_csv(test_csv_filename, index=False)




#libraries for RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# libraries for RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score


#libraries for Elastic Net
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet

# libraries to plot Venn diagram
from matplotlib_venn import venn3


train_data = train_data.drop('PID', axis=1)
columns_to_check_cont = train_data.columns[0:30]

# RF feature importance
# Extract features and target variable
columns_to_check_cont = train_data.columns[0:30]
X = train_data[columns_to_check_cont]
y = train_data['Malignant']

# Initialize RandomForest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)

# 5-Fold Cross-Validation using AUC as the scoring metric
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
cross_val_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')

# Fit the Random Forest model to get feature importances
rf.fit(X, y)

# Get feature importances
feature_importances = rf.feature_importances_

# Create a DataFrame to display feature importance
feature_importance_df = pd.DataFrame({'Feature': columns_to_check_cont, 'Importance': feature_importances})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Cross-Validation AUC Scores:", cross_val_scores)
print("Average AUC Score:", np.mean(cross_val_scores))
print("\nFeature Importances:\n", feature_importance_df)

# Sort the DataFrame by importance in ascending order for the plot
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)

plt.title('Feature Importance Ranked by Gini Impurity')
plt.xlabel('Gini Impurity Decrease')
plt.ylabel('Features')

plt.show()

# RFE feature selection
X = train_data[columns_to_check_cont]
y = train_data['Malignant']

# Initialize Logistic Regression
logistic_model = LogisticRegression()

# Initialize StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=5)

# Initialize RFE with Logistic Regression
rfe = RFE(estimator=logistic_model)

# Lists to store results
num_features_list = []
auc_list = []

# Iterate over different numbers of features
for num_features in range(1, len(X.columns) + 1):
    rfe.n_features_to_select = num_features
    
    # Initialize list for cross-validation results
    cv_auc_list = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        rfe.fit(X_train, y_train)
        RFE_selected_features = X_train.columns[rfe.support_]
        
        # Train and evaluate the model on the test set
        logistic_model.fit(X_train[RFE_selected_features], y_train)
        y_pred = logistic_model.predict_proba(X_test[RFE_selected_features])[:, 1]
        
        # Calculate AUC
        auc_score = roc_auc_score(y_test, y_pred)
        cv_auc_list.append(auc_score)
    
    # Calculate average AUC across folds
    avg_auc = np.mean(cv_auc_list)
    
    # Append results to lists
    num_features_list.append(num_features)
    auc_list.append(avg_auc)

# Find the index corresponding to the maximum AUC
optimal_num_features_index = np.argmax(auc_list)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(num_features_list, auc_list, marker='o')
plt.title('Number of Features vs Average AUC (RFE with Logistic Regression, 5-fold Stratified Cross-Validation)')
plt.xlabel('Number of features (n)')
plt.ylabel('Average AUC')

# Set x-axis ticks at the interval of 1 integer
plt.xticks(range(min(num_features_list), max(num_features_list) + 1, 1))

plt.axvline(num_features_list[optimal_num_features_index], linestyle='--', color='red', label='Optimal Number of Features')
plt.legend()
plt.grid(True)
plt.show()

# # Set the optimal number of features (automated)
# optimal_num_features = num_features_list[optimal_num_features_index]

# Set the optimal number of features (to manual set features)
optimal_num_features = 12

# Initialize RFE with the optimal number of features
rfe.n_features_to_select = optimal_num_features
rfe.fit(X, y)

# Get the selected features
RFE_selected_features_withHighAUC = X.columns[rfe.support_]

# Elastic Net feature selection
target_variable = 'Malignant'
X_train = train_data.drop(['Malignant'], axis=1)
y_train = train_data[target_variable]

# Split the training data into train and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=43)

# Create Elastic Net model with cross-validation
elastic_net = ElasticNetCV(
    l1_ratio=np.linspace(0.01, 1.0, 100),
    alphas=np.logspace(-4, 2, 100),  
    cv=5,
    random_state=42,
    max_iter=10000  # Increase max_iter if needed
)

# Fit the Elastic Net model to the training data
elastic_net.fit(X_train_split, y_train_split)

# Get the best alpha and rho
best_alpha = elastic_net.alpha_
best_rho = elastic_net.l1_ratio_

# Print the best alpha and rho
print(f'Best Alpha: {best_alpha}')
print(f'Best Rho (l1_ratio): {best_rho}')

# Predict the probability of positive class for validation set
y_val_pred_proba = elastic_net.predict(X_val_split)

# Train Elastic Net model with best parameters
final_elastic_net = ElasticNetCV(
    l1_ratio=best_rho,
    alphas=[best_alpha],
    cv=5,
    random_state=42,
    max_iter=10000
)

final_elastic_net.fit(X_train, y_train)

# Select features based on the best Elastic Net model
feature_selector = SelectFromModel(final_elastic_net, threshold='mean')
X_train_selected = feature_selector.fit_transform(X_train, y_train)

# Get the selected feature names
EN_selected_feature = X_train.columns[feature_selector.get_support()]

# Print the selected feature names
print(f'Selected Features: {EN_selected_feature}')

# Store coefficients in a DataFrame
EN_selected_feature_coef = pd.DataFrame(index=elastic_net.alphas_)

# Plot the coefficients of the selected features with respect to alpha values
plt.figure(figsize=(15, 8))

# Iterate over the selected features
for feature in EN_selected_feature:
    # Extract the coefficients using a single feature at a time
    coef_values = []
    for alpha in elastic_net.alphas_:
        elastic_net_per_alpha = ElasticNet(alpha=alpha, l1_ratio=best_rho, max_iter=10000)
        elastic_net_per_alpha.fit(X_train[[feature]], y_train)
        coef_values.append(elastic_net_per_alpha.coef_[0])
    
    # Plot the coefficients against the alpha values
    plt.plot(elastic_net.alphas_, coef_values, label=feature)
    
    # Store coefficients in the DataFrame
    EN_selected_feature_coef[feature] = coef_values

# Plot a vertical line at the position of the best alpha
plt.axvline(x=best_alpha, color='r', linestyle='--', label='Best Alpha')

# Customize the plot
plt.xscale('log')  # Use log scale for better visualization
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Elastic Net Coefficients for Selected Features')
plt.legend()
plt.show()

# Venn diagram for feature selection 
# Sample lists of features. 
RF_selected_features = [
    "perimeter_worst",
    "radius_worst",
    "concave_points_mean",
    "concave_points_worst",
    "area_worst",
    "concavity_mean",
    "perimeter_mean",
    "area_mean",
    "concavity_worst",
    "area_se",
    "radius_mean",
    "radius_se",
    "texture_mean",
    "texture_worst",
    "smoothness_worst"
]

EN_selected_features = ['radius_mean', 'concavity_mean', 'fractal-dimension_mean', 'radius_se', 'texture_se', 'area_se', 'texture_worst', 'smoothness_worst', 'concave_points_worst', 'fractal_dimension_worst']
RFE_selected_features_withHighAUC = ['area_mean', 'concavity_mean', 'concave_points_mean', 'area_se',
       'compactness_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
       'area_worst', 'smoothness_worst', 'concave_points_worst',
       'symmetry_worst']
    
# Create a Venn diagram
plt.figure(figsize=(8, 8))
venn3([set(RF_selected_features), set(EN_selected_features), set(RFE_selected_features_withHighAUC)], 
      set_labels=('Random Forest Important Features', 'Elastic Net Features', 'RFE Features'))
plt.title("Venn Diagram of Feature Selection Methods")
plt.show()

# Convert lists to sets
RF_set = set(RF_selected_features)
EN_set = set(EN_selected_features)
RFE_set = set(RFE_selected_features_withHighAUC)

# Find 
common_features = RF_set.intersection(EN_set, RFE_set)
print("Common features across all three methods:", common_features)


# Find unique and shared features
shared_RF_EN = RF_set.intersection(EN_set) - RFE_set
shared_RF_RFE = RF_set.intersection(RFE_set) - EN_set
shared_EN_RFE = EN_set.intersection(RFE_set) - RF_set

# Print results
print("Shared between Random Forest and Elastic Net:", shared_RF_EN)
print("Shared between Random Forest and RFE:", shared_RF_RFE)
print("Shared between Elastic Net and RFE:", shared_EN_RFE)


