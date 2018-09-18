
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import f
from scipy.stats import norm

def calculate_subplot_config(num_x):
    n_rows = 1
    n_cols = 1
    
    while n_rows*n_cols < num_x:
        if n_rows == n_cols:
            n_rows +=1
        else:
            n_cols +=1
    
    return (n_cols, n_rows)

def unique_values_dict(df):
    unique_df = {}
    for column in df.columns:
        unique_df[column] = df[column].unique()
    
    return unique_df

def Two_factor_ANOVA(data):
    """Creates an ANVOA table for a 3-factor factorial experiment. Requires at least one repeat (i.e. 2 measurements) for each combination of factors."""
    #Edit column names
    data.columns = ['A', 'B', 'Response']
    
    #Determine the number of levels in each factor and how many repeats
    unique_dict = unique_values_dict(data)
    a = len(unique_dict['A'])
    b = len(unique_dict['B'])
    n = len(data)/(a*b)
    
    sum_y = data.iloc[:,-1].sum()
    
    #Main effects
    SS_A = (1/(b*n)) * (data.groupby('A').sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*n)
    SS_B = (1/(a*n)) * (data.groupby('B').sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*n)
    
    #2-factor interactions
    SS_Subtotals_AB = (1/(n)) * (data.groupby(['A', 'B']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*n)

    SS_AB = SS_Subtotals_AB - SS_A - SS_B

    #Total
    SS_T = (data.iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*n)
    
    #Error
    SS_E = SS_T - SS_Subtotals_AB
    
    #Setup ANOVA table from calculated sum of squareds (SS_...)
    ANOVA_table = pd.DataFrame()
    ANOVA_table['Source of Variation'] = ['A', 'B', 'AB', 'Error', 'Total']
    ANOVA_table.index = ANOVA_table['Source of Variation']
    ANOVA_table.drop(columns = ['Source of Variation'], inplace=True)
    ANOVA_table['Sum of Squares'] = [SS_A, SS_B, SS_AB, SS_E, SS_T]
    ANOVA_table['Degrees of Freedom'] = [a-1, b-1, (a-1)*(b-1), a*b*(n-1), a*b*n - 1]
    ANOVA_table['Mean Square'] = ANOVA_table['Sum of Squares']/ANOVA_table['Degrees of Freedom']
    ANOVA_table.loc['Total', 'Mean Square'] = None
    ANOVA_table['F0'] = ANOVA_table['Mean Square']/ANOVA_table.loc['Error', 'Mean Square']
    ANOVA_table.loc['Error', 'F0'] = None
    f_function = f(n, a*b)
    ANOVA_table['P-Value'] = f_function.sf(ANOVA_table['F0'])
    
    return ANOVA_table


def Three_factor_ANOVA(data):
    """Creates an ANVOA table for a 3-factor factorial experiment. Requires at least one repeat (i.e. 2 measurements) for each combination of factors."""
    #Edit column names
    data.columns = ['A', 'B', 'C', 'Response']
    #Determine the number of levels in each factor and how many repeats
    unique_dict = unique_values_dict(data)
    a = len(unique_dict['A'])
    b = len(unique_dict['B'])
    c = len(unique_dict['C'])
    n = len(data)/(a*b*c)
    
    sum_y = data.iloc[:,-1].sum()
    
    #Main effects
    SS_A = (1/(b*c*n)) * (data.groupby('A').sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*n)
    SS_B = (1/(a*c*n)) * (data.groupby('B').sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*n)
    SS_C = (1/(a*b*n)) * (data.groupby('C').sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*n)
    
    #2-factor interactions
    SS_Subtotals_AB = (1/(c*n)) * (data.groupby(['A', 'B']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*n)
    SS_Subtotals_AC = (1/(b*n)) * (data.groupby(['A', 'C']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*n)
    SS_Subtotals_BC = (1/(a*n)) * (data.groupby(['B', 'C']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*n)
    SS_AB = SS_Subtotals_AB - SS_A - SS_B
    SS_AC = SS_Subtotals_AC - SS_A - SS_C
    SS_BC = SS_Subtotals_BC - SS_B - SS_C
    
    #3-factor interations
    SS_Subtotals_ABC = (1/n) * (data.groupby(['A', 'B', 'C']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*n)
    SS_ABC = SS_Subtotals_ABC - SS_A - SS_B - SS_C - SS_AB - SS_AC - SS_BC
    
    #Total
    SS_T = (data.iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*n)
    
    #Error
    SS_E = SS_T - SS_Subtotals_ABC
    
    #Setup ANOVA table from calculated sum of squareds (SS_...)
    ANOVA_table = pd.DataFrame()
    ANOVA_table['Source of Variation'] = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC', 'Error', 'Total']
    ANOVA_table.index = ANOVA_table['Source of Variation']
    ANOVA_table.drop(columns = ['Source of Variation'], inplace=True)
    ANOVA_table['Sum of Squares'] = [SS_A, SS_B, SS_C, SS_AB, SS_AC, SS_BC, SS_ABC, SS_E, SS_T]
    ANOVA_table['Degrees of Freedom'] = [a-1, b-1, c-1, (a-1)*(b-1), (a-1)*(c-1), (b-1)*(c-1), (a-1)*(b-1)*(c-1), a*b*c*(n-1), a*b*c*n - 1]
    ANOVA_table['Mean Square'] = ANOVA_table['Sum of Squares']/ANOVA_table['Degrees of Freedom']
    ANOVA_table.loc['Total', 'Mean Square'] = None
    ANOVA_table['F0'] = ANOVA_table['Mean Square']/ANOVA_table.loc['Error', 'Mean Square']
    ANOVA_table.loc['Error', 'F0'] = None
    f_function = f(n, a*b*c)
    ANOVA_table['P-Value'] = f_function.sf(ANOVA_table['F0'])
    
    return ANOVA_table

def Four_factor_ANOVA(data):
    """Creates an ANVOA table for a 4-factor factorial experiment. Requires at least one repeat (i.e. 2 measurements) for each combination of factors."""
    #Edit column names
    data.columns = ['A', 'B', 'C', 'D', 'Response']
    
    #Determine the number of levels in each factor and how many repeats
    unique_dict = unique_values_dict(data)
    a = len(unique_dict['A'])
    b = len(unique_dict['B'])
    c = len(unique_dict['C'])
    d = len(unique_dict['D'])
    n = len(data)/(a*b*c*d)
    
    #Sum of all data points
    sum_y = data.iloc[:,-1].sum()
    
    #Main effects
    SS_A = (1/(b*c*d*n)) * (data.groupby('A').sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    SS_B = (1/(a*c*d*n)) * (data.groupby('B').sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    SS_C = (1/(a*b*d*n)) * (data.groupby('C').sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    SS_D = (1/(a*b*c*n)) * (data.groupby('D').sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    
    #2-factor interactions
    SS_Subtotals_AB = (1/(c*d*n)) * (data.groupby(['A', 'B']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    SS_Subtotals_AC = (1/(b*d*n)) * (data.groupby(['A', 'C']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    SS_Subtotals_AD = (1/(b*c*n)) * (data.groupby(['A', 'D']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    SS_Subtotals_BC = (1/(a*d*n)) * (data.groupby(['B', 'C']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    SS_Subtotals_BD = (1/(a*c*n)) * (data.groupby(['B', 'D']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    SS_Subtotals_CD = (1/(a*b*n)) * (data.groupby(['C', 'D']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    
    SS_AB = SS_Subtotals_AB - SS_A - SS_B
    SS_AC = SS_Subtotals_AC - SS_A - SS_C
    SS_AD = SS_Subtotals_AD - SS_A - SS_D
    SS_BC = SS_Subtotals_BC - SS_B - SS_C
    SS_BD = SS_Subtotals_BD - SS_B - SS_D
    SS_CD = SS_Subtotals_CD - SS_C - SS_D
    
    #3-factor interactions
    SS_Subtotals_ABC = (1/(d*n)) * (data.groupby(['A', 'B', 'C']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    SS_Subtotals_ABD = (1/(c*n)) * (data.groupby(['A', 'B', 'D']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    SS_Subtotals_ACD = (1/(b*n)) * (data.groupby(['A', 'C', 'D']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    SS_Subtotals_BCD = (1/(a*n)) * (data.groupby(['B', 'C', 'D']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    
    SS_ABC = SS_Subtotals_ABC - SS_A - SS_B - SS_C - SS_AB - SS_AC - SS_BC
    SS_ABD = SS_Subtotals_ABD - SS_A - SS_B - SS_D - SS_AB - SS_AD - SS_BD
    SS_ACD = SS_Subtotals_ACD - SS_A - SS_C - SS_D - SS_AC - SS_AD - SS_CD
    SS_BCD = SS_Subtotals_BCD - SS_B - SS_C - SS_D - SS_BC - SS_BD - SS_CD
    
    #4-factor interactions
    SS_Subtotals_ABCD = (1/(n)) * (data.groupby(['A', 'B', 'C', 'D']).sum().iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    
    SS_ABCD = SS_Subtotals_ABCD - SS_A - SS_B - SS_C - SS_D - SS_AB - SS_AC - SS_AD - SS_BC - SS_BD - SS_CD - SS_ABC - SS_ABD - SS_ACD - SS_BCD
    
    #Total
    SS_T = (data.iloc[:,-1]**2).sum() - (sum_y**2)/(a*b*c*d*n)
    
    #Error
    SS_E = SS_T - SS_Subtotals_ABCD
    
    #Setup ANOVA table from calculated sum of squareds (SS_...)
    ANOVA_table = pd.DataFrame()
    ANOVA_table['Source of Variation'] = ['A', 'B', 'C', 'D', 'AB', 'AC', 'AD', 'BC', 'BD', 'CD', 'ABC', 'ABD', 'ACD', 'BCD', 'ABCD', 'Error', 'Total']
    ANOVA_table.index = ANOVA_table['Source of Variation']
    ANOVA_table.drop(columns = ['Source of Variation'], inplace=True)
    ANOVA_table['Sum of Squares'] = [SS_A, SS_B, SS_C, SS_D, SS_AB, SS_AC, SS_AD, SS_BC, SS_BD, SS_CD, SS_ABC, SS_ABD, SS_ACD, SS_BCD, SS_ABCD, SS_E, SS_T]
    ANOVA_table['Degrees of Freedom'] = [a-1, b-1, c-1, d-1, (a-1)*(b-1), (a-1)*(c-1), (a-1)*(d-1), (b-1)*(c-1), (b-1)*(d-1), (c-1)*(d-1), (a-1)*(b-1)*(c-1), (a-1)*(b-1)*(d-1), (a-1)*(c-1)*(d-1), (b-1)*(c-1)*(d-1), (a-1)*(b-1)*(c-1)*(d-1), a*b*c*d*(n-1), a*b*c*d*n - 1]
    ANOVA_table['Mean Square'] = ANOVA_table['Sum of Squares']/ANOVA_table['Degrees of Freedom']
    ANOVA_table.loc['Total', 'Mean Square'] = None
    ANOVA_table['F0'] = ANOVA_table['Mean Square']/ANOVA_table.loc['Error', 'Mean Square']
    ANOVA_table.loc['Error', 'F0'] = None
    f_function = f(n, a*b*c*d)
    ANOVA_table['P-Value'] = f_function.sf(ANOVA_table['F0'])
    
    return ANOVA_table

def residual_plot(data, ANOVA_table):
    """Makes a normal probability plot of residuals"""
    columns = list(data.columns[:-1])
    tmp_data = data.set_index(list(data.columns[:-1]))
    sigma = np.sqrt(ANOVA_table.loc['Error', 'Mean Square'])
    residuals = (tmp_data - tmp_data.groupby(columns).mean()).iloc[:, -1].values/sigma
    residuals.sort()
    df = pd.DataFrame(columns=['Residuals'], data=residuals)
    df['Position'] = df.index + 1
    df['f'] = (df.Position - 0.375)/(len(df) + 0.25)
    df['z'] = norm.ppf(df.f)
    
    sns.regplot(x='Residuals', y='z', data=df)

def normal_plot(data):
    """Makes a normal probability plot of the response"""
    tmp_data = data.iloc[:, -1].values
    tmp_data.sort()
    df = pd.DataFrame(columns=['Response'], data=tmp_data)
    df['Position'] = df.index + 1
    df['f'] = (df.Position - 0.375)/(len(df) + 0.25)
    df['z'] = norm.ppf(df.f)
    
    sns.regplot(x='Response', y='z', data=df)

#----------------------Import data and data manipulation-----------------------
#Last column must be dependant variable.
#data = pd.read_csv('example_data.csv')
data = pd.read_csv('test_data.csv')
#data.drop(columns='order', inplace=True)

x_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]
num_x = len(x_data.columns)

#-------------------------------------Boxplot----------------------------------
sns.boxplot(y = data.columns[-1], data=data)

#-----------------------Scatter plot of each variable--------------------------
subplot_config = calculate_subplot_config(num_x)
cr = str(subplot_config[0]) + str(subplot_config[1])

plt.figure()

for i, x_col in enumerate(x_data.columns):
    plt.subplot(int(cr + str(i + 1)))
    plt.scatter(x_data[x_col], y_data)
    plt.ylabel(data.columns[-1])
    plt.xlabel(x_col)
    
plt.tight_layout()


#----------------------------Boxplots of each variable-------------------------
test = pd.melt(data, id_vars = data.columns[-1], value_vars=data.columns[:-1])
#sns.catplot(x='variable', y = test.columns[0], hue = 'value', data = test, kind='swarm')
sns.boxplot(x='variable', y = test.columns[0], hue = 'value', data = test)
#sns.violinplot(x='variable', y = test.columns[0], hue = 'value', split=True, data = test)

#---------------------------------Plot of means--------------------------------
unique = unique_values_dict(x_data)

means = {}
for column in x_data.columns:
    temp = []
    for value in unique[column]:
        temp.append(data.iloc[:,-1][data[column] == value].mean())
    means[column] = temp

fig, ax = plt.subplots(1, num_x, sharey=True)

for i, col in enumerate(ax):
    column = x_data.columns[i]
    col.plot(unique[column], means[column])
    col.set(xlabel=x_data.columns[i])

plt.tight_layout()

#----------------------------------Not really useful---------------------------
for column in x_data.columns:
    plt.plot(unique[column], means[column])
