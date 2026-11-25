# Predicting final grades of students
The final grades in the subject "Portuguese" of portuguese students were predicted according to available data, such as previous intermediate grades and individual, family or school-related factors.

An attempt to predict "quitting" was also performed, even though data on quitting school were inferred from the dataset.

# Description
The main goals were:
* predict the final grade (G3) without using previous intermediate grades (G1 and/or G2)
* predict the final grade (G3) with G2 only, and with G2 + other features
* predict when the students quit (G3 <= 1)

*Dataset* = data obtained from the public dataset https://uci-ics-mlr-prod.aws.uci.edu/dataset/320/student%2Bperformance


*Tools* = Python

# Example analysis
## Python
One-Hot Encoding of categorical non-ordinal variables
```python
def one_hot_encode(df, columns):

    '''
    Does not modify original DataFrame
    Works only on selected columns
    Creates n-1 columns per categorical column, where n = original categories of original column 
    Transforms categories into numbers
    '''

    df = df.copy()

    df = pd.get_dummies(
        df,
        columns=columns,
        drop_first=True,
        dtype=int
    )

    return df

categorical_columns = ['Mjob', 'Fjob', 'reason', 'guardian']

df = one_hot_encode(df, categorical_columns)
```

Binary Encoding of categorical text variables
```python
def binary_encoding(df):

    '''
    Does not modify original DataFrame
    Selects only categorical string columns
    Detects unique values (excluding NaN)
    If there are only 2, assigns 0 to the first and 1 to the second
    Keeps track of binary transformed columns in the list "binary columns"
    Keeps track of the correspondence between strings and 0, 1 in the list "mappings"
    '''
    df = df.copy()

    binary_columns = []
    mappings = []

    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_values = df[col].dropna().unique()

        if len(unique_values) == 2:
            mapping = {
                unique_values[0]: 0,
                unique_values[1]: 1
            }

            df[col] = df[col].map(mapping).astype("Int64")
            binary_columns.append(col)

            for original_value, encoded_value in mapping.items():
                mappings.append({
                    'variable': col,
                    'original_value': original_value,
                    'encoded_value': encoded_value
                })

    mappings_df = pd.DataFrame(mappings)

    return df, binary_columns, mappings_df

df, binary_columns, mappings_df = binary_encoding(df_raw)
mappings_df.to_csv("documentation/binary_encoding.csv", index=False)

print("Binary encoding:")
print(binary_columns)
display(mappings_df)
```

Simple Linear Regression
```python
df = df_raw.copy()
independent_variable = "G2"
dependent_variable = "G3"
df = df[[independent_variable, dependent_variable]]
df.rename(columns={
    "G2" : "x",
    "G3" : "y"
    }, inplace=True)

X = df['x']
y = df['y']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())
```


# Insights
The main results (summarized in "Results.docx") include: 
* Models that exclude intermediate grades (G1 and/or G2) cannot explain more than 1/3rd of the variability of the data.
* Two models were considered satisfactory: simple linear regression with only G2 and multiple linear regression with G2 + other features
  * Both models explain almost 90% of the variability of the data and are statistically significant
  * For both models, some predictions show large errors (students who performed much better/worse than in intermediate grades)
* The simple linear regression predicts that the final grade is the 2nd period grade +1. 
* The other model predicts that the final grade is mostly dictated by the 2nd period grade, but also by:
  * age = older student will have a higher score (almost 0.1 points/year increase)
  * failures = students who failed previously will have lower scores (almost 0.2 points)
  * higher = student who want to take higher education will perform better (almost 0.4 points)
  * intercept = this time, there is no constant contribution
* "Quitting" events were too few for a stable model, so the logistic regression failed to predict quitters 



# How to
Data interpretation and encoding are in the folder "documentation":
* "documentation/documentation.xlsx" contains info on variables and their category, with a description
* "documentation/binary_encoding.csv" contains the correspondence between their binary values (0-1) and their original values.

Starting CSV data are in the file "raw_data/student-por.csv". In the same folder, there are also data on the corresponding mathematics grade, but those were not included in the present project.  

Processed CSV data are "processed_data/encoded.csv".

Jupyter Notebooks can run the tests again. They are ordered as they were intended to be used (from preprocessing to more and more complex models).

An overview of the main results is in "Results.docx".

## Requirements
Python 3.12.12

## Installation
* Clone the repository: 
    * git clone https://github.com/daniele-spd/regression_students_score.git
* Navigate to the project directory: 
    * cd REPOSITORY
* (Optional but recommended) Create a virtual environment: 
    * python -m venv venv
* Activate the environment: 
    * Windows: venv\Scripts\activate 
    * Linux / macOS: source venv/bin/activate
* Install the dependencies: 
    * pip install -r requirements.txt