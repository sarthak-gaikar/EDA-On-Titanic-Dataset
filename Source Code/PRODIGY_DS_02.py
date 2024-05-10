import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Reading CSV File
train_data = pd.read_csv('train.csv')

# Creating List for Columns
Pclass = train_data['Pclass']
Survived = train_data['Survived']
Sex = train_data['Sex']
Age = train_data['Age']
SibSp = train_data['SibSp']
Parch = train_data['Parch']
Embarked = train_data['Embarked']

# Checking Missing Values
# print(train_data.isnull().sum())

# Filling Missing Values
train_data.fillna({'Age': train_data['Age'].mean()}, inplace=True)
train_data.fillna({'Embarked': train_data['Embarked'].mode()[0]}, inplace=True)
# print(train_data.isnull().sum())

# Summary Statistics
# print(train_data.describe(include='all'))

# Data Visulization

# Age Distribution
age_bins = np.arange(0, 100, 10)
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=age_bins, right=False)

plt.figure(figsize=(8, 6))
plt.hist(train_data['Age'], bins=age_bins, edgecolor='white', color='c')
plt.title('Age Distribution of Passengers on Titanic')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

# Class v/s Survived
plt.figure(figsize=(8, 6))
sns.countplot(data=train_data, x='Pclass', hue='Survived', palette=['r', 'g'], edgecolor='white')
plt.title('Class vs. Survival')
plt.xlabel('Class')
plt.ylabel('Count')
plt.legend(labels=['Not Survived', 'Survived'], loc='upper left')
plt.show()

# Sex v/s Survived
plt.figure(figsize=(8, 6))
sns.countplot(data=train_data, x='Sex', hue='Survived', palette=['r', 'g'], edgecolor='white')
plt.title('Sex vs. Survival')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(labels=['Not Survived', 'Survived'], loc='upper right')
plt.show()


# Embarked v/s Survived
plt.figure(figsize=(8, 6))
sns.countplot(data=train_data, x='Embarked', hue='Survived', palette=['r', 'g'], edgecolor='white')
plt.title('Embarked vs. Survival')
plt.xlabel('Embarked Point')
plt.ylabel('Count')
plt.legend(labels=['Not Survived', 'Survived'], loc='upper right')
plt.show()

# Age v/s Survived
plt.figure(figsize=(10, 6))
survival_rates = train_data.groupby('AgeGroup', observed=True)['Survived'].mean()
survival_rates.plot(kind='bar', color='c', edgecolor='white')
plt.title('Survival Rates by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.show()

# Plot age distribution for survived and not-survived passengers
plt.figure(figsize=(10, 6))
sns.histplot(train_data[train_data['Survived'] == 0]['Age'], bins=age_bins, color='r', label='Not Survived', kde=True, edgecolor='white')
sns.histplot(train_data[train_data['Survived'] == 1]['Age'], bins=age_bins, color='g', label='Survived', kde=True, edgecolor='white')
plt.title('Age Distribution of Survived and Not Survived Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()

# SibSp v/s Survived
plt.figure(figsize=(8, 6))
sns.countplot(data=train_data, x='SibSp', hue='Survived', palette=['r', 'g'], edgecolor='white')
plt.title('SibSp vs. Survival')
plt.xlabel('SibSp')
plt.ylabel('Count')
plt.legend(labels=['Not Survived', 'Survived'], loc='upper right')
plt.show()

# Parch v/s Survived
plt.figure(figsize=(8, 6))
sns.countplot(data=train_data, x='Parch', hue='Survived', palette=['r', 'g'], edgecolor='white')
plt.title('Parch vs. Survival')
plt.xlabel('Parch')
plt.ylabel('Count')
plt.legend(labels=['Not Survived', 'Survived'], loc='upper right')
plt.show()

# EDA

# Function to calculate survival rate
def calculate_survival_rates(df, group_by_column):
    # Calculate survival rates
    survival_rates = df.groupby(group_by_column, observed=True)['Survived'].mean() * 100
    
    # Format survival rates to two decimal places and add '%' symbol
    formatted_survival_rates = survival_rates.apply(lambda x: f"{x:.2f}%")
    
    return formatted_survival_rates

# Analysis and visualization for each factor
# 1. Gender vs. Survival
gender_survival_rates = calculate_survival_rates(train_data, 'Sex')
print("Gender vs. Survival:")
print(gender_survival_rates)

# 2. Age Group vs. Survival
age_group_survival_rates = calculate_survival_rates(train_data, 'AgeGroup')
print("\nAge Group vs. Survival:")
print(age_group_survival_rates)

# 3. Class (Pclass) vs. Survival
class_survival_rates = calculate_survival_rates(train_data, 'Pclass')
print("\nClass vs. Survival:")
print(class_survival_rates)

# 4. SibSp (Siblings/Spouse count) vs. Survival
sibsp_survival_rates = calculate_survival_rates(train_data, 'SibSp')
print("\nSibSp vs. Survival:")
print(sibsp_survival_rates)

# 5. Parch (Parents/Children count) vs. Survival
parch_survival_rates = calculate_survival_rates(train_data, 'Parch')
print("\nParch vs. Survival:")
print(parch_survival_rates)

# 6. Embarked (Embarkation point) vs. Survival
embarked_survival_rates = calculate_survival_rates(train_data, 'Embarked')
print("\nEmbarked vs. Survival:")
print(embarked_survival_rates)