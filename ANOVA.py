import pandas as pd
import itertools
from scipy.stats import f
import numpy as np

class ANOVA:
    """Analyse DOE experiments using ANOVA. NB: n > 1 for the code to work, where n is the number of repeats."""
    def __init__(self, data):
        #Initialise variables and define simple statistical values
        self.data = data
        self.num_factors = len(self.data.columns) - 1
        self.factors = list(self.data.columns[:-1])
        self.sum_y = data.iloc[:,-1].sum()
        self.unique_dict = self.unique_values_dict(self.data)
        self.levels = {}
        
        #Determine all interactions between factors
        sources_of_variation = []
        for interaction_level in range(self.num_factors):
            combos = itertools.combinations(self.factors, interaction_level + 1)
            for combo in combos: 
                sources_of_variation.append(self.make_interaction_name(combo))
        
        sources_of_variation.append('Error')        
        sources_of_variation.append('Total')
        
        #Create ANOVA table
        self.table = pd.DataFrame(columns =['Sum of Squares', 'Degrees of Freedom', 'Mean Square', 'F0', 'P-Value'], index=sources_of_variation)
        
        #Needed for functions later, even though the data ends up in the table.
        #Code is designed like this because initally more dictionaries were used instead of pandas dataframe.
        self.sum_of_squares = [{}]*self.num_factors
        
        #Determine number of repeats. Must be the same for all measurements.
        total = 1
        for factor in self.factors:
            level = len(self.unique_dict[factor])
            self.levels[factor] = level
            total = total*level
        
        self.n = len(self.data)/total
        self.total = len(self.data)
        
        #Most of the complicated equations are contained within this loop/function
        for interaction_level in range(self.num_factors):
            self.calculate_interactions(interaction_level + 1)
        
        #Create the table from component parts
        #Sum of squares
        self.table['Sum of Squares'] = pd.DataFrame(self.sum_of_squares).max()
        self.table.loc['Total', 'Sum of Squares'] = (data.iloc[:,-1]**2).sum() - (self.sum_y**2)/(self.total)
        prefactor = self.make_prefactor(self.factors)
        final_subtotal = (1/(prefactor*self.n)) * (self.data.groupby(self.factors).sum().iloc[:,-1]**2).sum() - (self.sum_y**2)/self.total
        self.table.loc['Error', 'Sum of Squares']= self.table.loc['Total', 'Sum of Squares'] - final_subtotal
        
        #Degrees of freedom
        self.table.loc['Total', 'Degrees of Freedom'] = self.total - 1
        self.table.loc['Error', 'Degrees of Freedom'] = (self.total/self.n) * (self.n - 1)
        
        #Mean square
        self.table['Mean Square'] = self.table['Sum of Squares']/self.table['Degrees of Freedom']
        
        #F0
        self.table['F0'] = self.table['Mean Square']/self.table.loc['Error', 'Mean Square']
        
        #P-value
        self.f_function = f(self.n, self.total/self.n)
        self.table['P-Value'] = self.f_function.sf(list(self.table['F0']))
        
        #Remove values which have no meaning. Only calculated in the first place because it was simpler to code.
        self.table.iloc[-2:, -2:] = np.NaN
        self.table.iloc[-1, -3] = np.NaN
        self.table.iloc[:, :-1] = self.table.iloc[:, :-1].astype(float)
        
        #F0 for statistical significance P<0.05
        self.calculate_F0_significance_level()
       
    def calculate_interactions(self, interaction_level):
        """Calculates sum of squares and degrees of freedom for a specified interaction level and saves them in the self.table dataframe.
        interaction_level = 1 ---> Main factors
        interaction_level = 2 ---> 2-factor interactions
        interaction_level = 3 ---> 3-factor interactions
        ..."""
        combinations = itertools.combinations(self.factors, interaction_level)       
        subtotals = {}
        effects = {}
        for combo in combinations:
            interaction_factors = list(combo)
            interaction = self.make_interaction_name(interaction_factors)
            prefactor = self.make_prefactor(interaction_factors)
            self.table.loc[interaction, 'Degrees of Freedom'] = self.calculate_degrees_of_freedom(interaction_factors)
            
            subtotals[interaction] = (1/(prefactor*self.n)) * (self.data.groupby(interaction_factors).sum().iloc[:,-1]**2).sum() - (self.sum_y**2)/self.total
            effects[interaction] = subtotals[interaction]
            
            for level in range(interaction_level - 1) :
                factor_combos = itertools.combinations(combo, level + 1)
                for factor_combo in factor_combos:
                    name = self.make_interaction_name(factor_combo)
                    effects[interaction] += -self.sum_of_squares[level][name]
             
        self.sum_of_squares[interaction_level - 1] = effects 
            
    def calculate_degrees_of_freedom(self, interaction_factors):
        dof = 1
        for factor in interaction_factors:
            dof = (self.levels[factor] - 1) * dof
        
        return dof
        
    def unique_values_dict(self, df):
        unique_dict = {}
        for column in df.columns:
            unique_dict[column] = df[column].unique()
        
        return unique_dict
        
    def make_prefactor(self, interaction_factors):
        #Determine prefactor. Multiply all factor levels together which aren't the main factor
        prefactor = 1
        for factor in self.factors:
            if factor not in interaction_factors:
                prefactor = prefactor * self.levels[factor]
        
        return prefactor
    
    def make_interaction_name(self, interaction_factors):
        interaction = ''
        for factor in interaction_factors:
            interaction = interaction + ':' + factor
        interaction = interaction[1:]
        
        return interaction
    
    def calculate_F0_significance_level(self, sig=0.05):
        self.significance = self.f_function.isf(sig)
