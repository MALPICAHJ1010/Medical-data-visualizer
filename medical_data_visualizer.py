import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

'''

Add an overweight column to the data. To determine if a person is overweight,
first calculate their BMI by dividing their weight in kilograms by the square of their height in meters.
 If that value is > 25 then the person is overweight. Use the value 0 for NOT overweight and
   the value 1 for overweight.

Normalize the data by making 0 always good and 1 always bad. 
 If the value of cholesterol or gluc is 1, make the value 0.
   If the value is more than 1, make the value 1.

Convert the data into long format and create a chart that shows the value counts of the categorical features
 using seaborn's catplot(). The dataset should be split by 'Cardio'
   so there is one chart for each cardio value. The chart should look like examples/Figure_1.png.

Clean the data. Filter out the following patient segments that represent incorrect data:
 diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
  height is less than the 2.5th percentile
    (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
height is more than the 97.5th percentile
weight is less than the 2.5th percentile
weight is more than the 97.5th percentile
Create a correlation matrix using the dataset. Plot the correlation matrix using seaborn's heatmap().
 Mask the upper triangle. The chart should look like examples/Figure_2.png.
Any time a variable is set to None, make sure to set it to the correct code.

'''



# Import data
df = pd.read_csv('medical_examination.csv', header=0)
cols_names = df.columns
print(cols_names)


#Index(['id', 'age', 'sex', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol',
#       'gluc', 'smoke', 'alco', 'active', 'cardio'],
#      dtype='object')

'''
Add an overweight column to the data. To determine if a person is overweight,
first calculate their BMI by dividing their weight in kilograms by the square of their height in meters.
 If that value is > 25 then the person is overweight. Use the value 0 for NOT overweight and
   the value 1 for overweight.'''
# Add 'overweight' column
# weight kg, height cm
df['overweight'] = [1 if k>25 else 0 for k in [i/((j/100)**2) for i,j in zip (df['weight'],df['height'])]]

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1,
#  make the value 0. If the value is more than 1, make the value 1.

df['cholesterol'] = [0 if i<=1 else 1 for i in df['cholesterol']]
df['gluc']= [0 if i<=1 else 1 for i in df['gluc']]

# Draw Categorical Plot
def draw_cat_plot(df=df):
    # Create DataFrame for cat plot using `pd.melt` using just the values from 
    # 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.

    df_cat = pd.melt(df,id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. 
    # You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio','variable', 'value']).size().reset_index(name='counts')

    # Draw the catplot with 'sns.catplot()'

    barplot =    sns.catplot(df_cat, kind='bar',x="variable", y="counts", hue="value", col="cardio")
    # Get the figure for the output

    fig = barplot.fig
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map(df=df):
    # Clean the data

    '''height is less than the 2.5th percentile
    (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
    height is more than the 97.5th percentile
    weight is less than the 2.5th percentile
    weight is more than the 97.5th percentile'''

    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))
                 ]
    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots()

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr,mask=mask,cbar=0, linewidths=2, square=True, cmap='Blues')

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
