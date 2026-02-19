# Import NumPy
import numpy as np

# 1. Create a NumPy array of numbers from 1 to 20
arr = np.arange(1, 21)
print("Original Array:")
print(arr)

# 2. Reshape the array into 4x5 matrix
matrix = arr.reshape(4, 5)
print("\n4x5 Matrix:")
print(matrix)

# 3. Find mean, median, and standard deviation
mean = np.mean(arr)
median = np.median(arr)
std_dev = np.std(arr)

print("\nMean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)

# 4. Extract all even numbers
even_numbers = arr[arr % 2 == 0]
print("\nEven Numbers:")
print(even_numbers)

# 5. Generate random 5x5 matrix and compute transpose
random_matrix = np.random.rand(5, 5)
transpose_matrix = random_matrix.T

print("\nRandom 5x5 Matrix:")
print(random_matrix)

print("\nTranspose of Matrix:")
print(transpose_matrix)
# Import Pandas
import pandas as pd

# 1. Load CSV file
df = pd.read_csv("student_data.csv")

# 2. Display first 5 and last 5 records
print("First 5 Records:")
print(df.head())

print("\nLast 5 Records:")
print(df.tail())

# 3. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values (fill with 0)
df.fillna(0, inplace=True)

# 4. Calculate average marks
df['Average'] = df[['Math', 'Science', 'English']].mean(axis=1)

print("\nData with Average Marks:")
print(df)

# 5. Student with highest average
top_student = df.loc[df['Average'].idxmax()]
print("\nStudent with Highest Average:")
print(top_student)

# 6. Students scoring more than 75 in Math
high_math = df[df['Math'] > 75]
print("\nStudents with Math > 75:")
print(high_math)

# 7. Add Result column (Pass/Fail)
df['Result'] = df['Average'].apply(lambda x: "Pass" if x >= 40 else "Fail")

print("\nFinal Data:")
print(df)
# Import Matplotlib
import matplotlib.pyplot as plt

# 1. Bar chart (Student Name vs Average Marks)
plt.figure()
plt.bar(df['Name'], df['Average'])
plt.title("Student vs Average Marks")
plt.xlabel("Student Name")
plt.ylabel("Average Marks")
plt.xticks(rotation=45)
plt.show()

# 2. Pie chart (Pass vs Fail)
result_counts = df['Result'].value_counts()

plt.figure()
plt.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%')
plt.title("Pass vs Fail Distribution")
plt.show()

# 3. Line graph (Marks comparison)
plt.figure()
plt.plot(df['Name'], df['Math'], label="Math")
plt.plot(df['Name'], df['Science'], label="Science")
plt.plot(df['Name'], df['English'], label="English")

plt.title("Marks Comparison")
plt.xlabel("Student Name")
plt.ylabel("Marks")
plt.legend()
plt.xticks(rotation=45)
plt.show()
