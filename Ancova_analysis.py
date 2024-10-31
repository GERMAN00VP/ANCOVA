# Ancova analysis
# By Germán Vallejo Palma

# Import the required packages

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from scipy.stats import levene
from scipy.stats import shapiro
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns




def generate_formula(target,categorical_var,covars, interactions=None):

    covars_str = " + ".join(covars)

    formula = f"{target} ~ C({categorical_var}) + {covars_str}"

    if not interactions is None:

        if interactions=="ALL":
            interactions = " + ".join(["*".join(list(interaction)) for interaction in obtener_combinaciones([categorical_var]+covars)])
    
        elif type(interactions)==list:
            interactions = " + ".join(["*".join(list(interaction)) for interaction in interactions])
        
        formula = formula+ " + " + interactions

    return formula

def obtener_combinaciones(lista):
    resultado = []
    for r in range(2, len(lista) + 1):  # r es el tamaño de las combinaciones
        resultado.extend(combinations(lista, r))
    return resultado

def remove_unwanted_chars(df,unwanted_chars_dict={},revert_dict={},revert=False):

    df1 = df.copy()
    if revert:
        revert_dict.update({"signopos":"+","_espacio_":" ","signoneg":"-","7barra7":"/","_CORCHETE1_":"[","_CORCHETE2_":"]"})

        for wanted_char in revert_dict.keys():
            df1.columns = df1.columns.str.replace(wanted_char,revert_dict[wanted_char],regex=True)

        return df1

    unwanted_chars_dict.update({"\+":"signopos"," ":"_espacio_","-":"signoneg","/":"7barra7","\[":"_CORCHETE1_","\]":"_CORCHETE2_"})
    for unwanted_char in unwanted_chars_dict.keys():
        df1.columns = df1.columns.str.replace(unwanted_char,unwanted_chars_dict[unwanted_char],regex=True)

    return df1



def generar_diccionario_colores(valores):
    # Genera un colormap que cubre una cantidad de colores igual a la longitud de la lista
    colores = plt.cm.tab10(range(len(valores)))  # Usa colormap 'tab10', que tiene 10 colores distintos
    
    # Crear el diccionario con cada valor de la lista como clave y su color asignado como valor
    diccionario_colores = {valor: colores[i] for i, valor in enumerate(valores)}
    return diccionario_colores


    


def interactions_translator(interactions,original_columns,translated_columns):
    columns_dict=dict(zip(original_columns,translated_columns))
    new_interactions =  [(columns_dict[term] for term in interaction )for interaction in interactions]
    return new_interactions


def do_ancova(data:pd.DataFrame,interactions:list|str=None,
              plot:bool=False, save_plot:bool|str=False,covariate_to_plot:str=None,palette:dict=None ):
    
    """ Function that allows you to make parametrical or non-parametrical ancovas.

    Parameters:

        - data: pd.Dataframe. It contains the info for the ancova in the following order:
            -> Column_1: The response variable, the target.
            -> Column_2: The cathegorical independent variable of your study (now allows up to 3 levels).
            -> Column_3 to end: The rest of the continous variables.       

        - interactions(list|str): defaults to None.
            -> If "ALL": computes all interactions.
            -> If list: expects a list of tuples (one for each interaction) with the interacting variables (columns) names.
        
        - plot(bool):if True performs a lm_plot data_pl with a boxplot. Defaults to False.

        - save_plot(bool|str): Defaults to false. If a path (str) is given, saves the plot there.

        - covariate_to_plot(str): the continuos covariate that is plotted.

        -palette(dict): Optional. A dictionary with:
                -> keys= leves of cathegorical independent variable
                -> values= colors
        

    Returns:
        ancova_results: A dictionary with the main results and job parameters.
    """

    
    # Store the data for plottiong
    data_pl = data.dropna().copy()

    # Extract the variable names from the df
    target = data_pl.columns[0]
    categorical_var = data_pl.columns[1]
    covars = data_pl.columns[2:].tolist()


    # Format the data for statmodels error handling
    data = remove_unwanted_chars(data).dropna()

    N = data.shape[0]

    # Format the interactions statmodels error handling
    if not interactions is None:
        new_interactions = interactions_translator(interactions,data_pl.columns,data.columns)
    else:
        new_interactions = interactions
        

    # For the report
    pretty_formula = generate_formula(target,categorical_var,covars, interactions=interactions)

    #For the analysis
    formula = generate_formula(data.columns[0],data.columns[1],data.columns[2:].tolist(), interactions=new_interactions)

    # Ajuste de modelo lineal con ambos factores
    model = ols(formula, data=data).fit()

    # Extrae los residuos
    residuals = model.resid

    # Extrae las agrupaciones de los residuos
    residual_groups = [residuals[data[data.columns[1]]==level] for level in data[data.columns[1]].unique()]

    if len(residual_groups)==2:
        # Prueba de Levene sobre los residuos (agrupados por categoria)
        levene_test = levene(residual_groups[0],residual_groups[1])
    elif len(residual_groups)==3:
        # Prueba de Levene sobre los residuos (agrupados por categoria)
        levene_test = levene(residual_groups[0],residual_groups[1],residual_groups[2])
    elif len(residual_groups)>3:
        print("Not implemented levene test for more than 3 groups")


    shapiro_test = shapiro(residuals).pvalue>0.05
    levene_test = levene_test.pvalue > 0.05

    if shapiro_test and levene_test:
        # Fit normal anova to the model
        anova_results = anova_lm(model, typ=2)

    else:
        print("Data not normal or not homoscedastic, doing the analyisis in the ranked data:")
        # Convert the response to ranked response
        data[data.columns[0]] = data[data.columns[0]].rank()
        model = ols(formula, data=data).fit()
        anova_results = anova_lm(model, typ=2)

        
    results_dict = {"Target":target, "Categorical condition":categorical_var, "Co-variables":covars, "Res-Normality":shapiro_test,
                    "Res-Homoscedasticity":levene_test,"Formula":pretty_formula, "N":N}
    
    # Do a combined plot with the results
    if plot:

        # Define the optional values if not provided
        if palette is None:
            palette = generar_diccionario_colores(data[data.columns[1]].unique().tolist())
                                                  
        if covariate_to_plot is None:
            covariate_to_plot=data.columns[2]


        # Crear subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Añadir líneas de regresión para cada grupo en hue
        for group in data_pl[categorical_var].unique():
            subset = data_pl[data_pl[categorical_var] == group]
            sns.regplot(data=subset, x=covars[0], y=target, ax=axs[0], scatter=True, label=f'{group}',color=palette[group])

        # EXTRACT THE COVARIABLE PVAL
        pval_covariable = anova_results.loc[f"{covariate_to_plot}"]["PR(>F)"]

        if pval_covariable<0.001:
            pval_covariable = "<0.001"
        else:
            pval_covariable = round(pval_covariable,3)


        plot_int_sig = False

        
        covariate_in_interaction =  data_pl.columns[2]         

        # Interaction, pval:
        if interactions is None:
            None  


        elif  (categorical_var,covariate_in_interaction) in interactions:
            pval_interaction = anova_results.loc[f"{categorical_var}:{covariate_to_plot}"]["PR(>F)"]
            plot_int_sig = True

        elif  (covariate_in_interaction,categorical_var) in interactions:

            pval_interaction = anova_results.loc[f"{covariate_to_plot}:{categorical_var}"]["PR(>F)"]
            plot_int_sig = True

        if plot_int_sig:

            if pval_interaction<0.001:
                pval_interaction = "<0.001"
            else:
                pval_interaction = round(pval_interaction,3)

            # ADD SIGNIFICANCE AS TITLE with interaction
            axs[0].set_title(f"P-value: {pval_covariable}| Interaction P-value: {pval_interaction}")

        else: 
            # ADD SIGNIFICANCE AS TITLE
            axs[0].set_title(f"P-value: {pval_covariable}")

        axs[0].legend()

        # The cathegorical plot

        sns.boxplot(data=data_pl,y=target,x=categorical_var,
                    palette=palette,ax=axs[1])
        

        pval_categorical = anova_results.loc[f"C({categorical_var})"]["PR(>F)"]

        if pval_categorical<0.001:
            pval_categorical = "<0.001"
        else:
            pval_categorical = round(pval_categorical,3)

        # ADD SIGNIFICANCE AS TITLE
        axs[1].set_title(f"P-value: {pval_categorical}")

    
        # Guardar la figura
        if save_plot:
            plt.savefig(save_plot,bbox_to_inches="tight")

        # Mostrar la figura
        plt.show()


    return pd.DataFrame(results_dict),anova_results
