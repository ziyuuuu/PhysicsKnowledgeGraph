#!/usr/bin/env python
# coding: utf-8

# Here, we attempt to create a knowledge graph to represent knowledge in Physics equations from the Feynman lectures in Physics. The full list of Physics equations is provided by at https://space.mit.edu/home/tegmark/aifeynman/FeynmanEquations.csv
# 
# Importing the equations using a function and creating some helper functions to clean the equations:

# In[1]:


import csv
import pandas as pd
import numpy as np
import networkx as nx
import sympy
import math
import matplotlib.pyplot as plt 

def add_slash(i):
  if i != i: return
  greek = ['alpha','beta', 'chi', 'delta', 'epsilon', 'gamma', 'mu', 'lambd', 'rho', 'omega', 'sigma', 'theta', 'tau', 'pi', 'kappa']
  for symbol in greek:
    if symbol in i: 
      if symbol == 'lambd': return i.replace("lambd", "\lambda") # change to lambda here, only affects the latex form not the original node
      return i.replace("%s" %symbol, "\%s" %symbol)
    else: continue
  return i

def add_brackets_to_underscores(i):
  index = i.find("_")
  # print("1", i)
  if index == -1: return i
  else: 
    # print("1", i[0:index+1] +"{" + i[index+1:] + "}")
    return i[0:index+1] +"{" + i[index+1:] + "}"

def to_relationship(ls):
  for i in ls:
    return '$R('+','.join(list(map(add_slash, [add_brackets_to_underscores(x) for x in ls if x==x])))+')$'


def import_equations(source = "https://space.mit.edu/home/tegmark/aifeynman/FeynmanEquations.csv"):
    all_mysteries = pd.read_csv(source, header = 0, index_col= 'Filename', na_values=["nan", ""])
    all_mysteries = all_mysteries.dropna(how = "all")


    # all_mysteries['Relevant_variables'] = all_mysteries[['Output']+['v%s_name' %i for i in range(1,10)]].values.tolist()

    # all_mysteries['Relationship'] = all_mysteries['Relevant_variables'].apply(to_relationship)
    return all_mysteries

# print(import_equations())


# Importing the list of all variables using a function, on which the equations are based on. We also create helper functions to clean the data for the variables:

# In[2]:


#processing the data for all variables

def import_variables(source = "https://space.mit.edu/home/tegmark/aifeynman/units.csv"):
    all_variables = pd.read_csv(source, header = 0, na_values=["nan", ""])
    all_variables = all_variables.dropna(how = "all")
    return all_variables


def to_mathtext(i):
  return "r'%s'" %i

#     def add_slash(i):
#       # print(i)
#       greek = ['alpha','beta', 'chi', 'delta', 'epsilon', 'gamma', 'mu', 'lambd', 'rho', 'omega', 'sigma', 'theta', 'tau', 'pi', 'kappa']
#       for symbol in greek:
#         if symbol in i: 
#           if symbol == 'lambd': return i.replace("lambd", "\lambda") # change to lambda here, only affects the latex form not the original node
#           return i.replace("%s" %symbol, "\%s" %symbol)
#         else: continue
#       return i



# print(import_variables())


# Some helper functions to break equations down into sub equations:

# In[3]:


#creating a list of variables: 

def split_by_brackets(formula):
    #takes in a formula and returns a list of all the subequations of the formula
    i = 0
    j = len(formula)
    counter = 0
    subeqns = []
    if formula.find("(") <0: return subeqns+ ["("+formula+")"]
    while formula[i:j].find("(") >0:
#         while formula[i:j].find("(") >0:
        i = formula[i:j].rfind("(")
        j = formula[i:j].find(")")+i+1
        #at this point, have found the variables in a subequation
        #check if there is a function in front of it
        while formula[i-1].isalpha():
            i -= 1
        subeqns.append(formula[i:j])
        formula = formula[:i]+"EQN"+str(counter)+formula[j:]
        i = 0
        j = len(formula)
        counter += 1 
    return subeqns+ ["("+formula+")"]

def find_variables_in_subeqns(subeqn, mathtext):
    #takes in a subequation and returns a list of variables that are in the subequation
    variables_in_subeqn = []
    subeqn = "(" + str(subeqn) + ")"
    for i in mathtext:
        index = subeqn.find(i)
        if ((index > 0 and (subeqn[index-1] in ["*", "+", "/", "-", "(", ")"]) 
            and (subeqn[index+len(i)] in ["*", "+", "/", "-", "(", ")"]))):
            variables_in_subeqn.append(i)
    return variables_in_subeqn
    

# print(split_by_brackets(all_mysteries['Formula'][2]))
# print(find_variables_in_subeqns(split_by_brackets(all_mysteries['Formula'][2])[2], mathtext))


# Creating the knowledge graph, a directed graph with attributes in both the nodes and edges. 
# 
# We start by creating nodes of all the variables, and the nodes for SI units. These nodes have the attributes which are their types. Each variable has a unique name, while each SI unit is a tuple of length five, with a 1 representing the quantity of the SI unit and four 0s eg (0,1,0,0,0).
# 
# Then we create nodes for unit vectors, which are just tuples, but may involve more than one quantity eg (0,1,1,0,-1). The type of these nodes is "unit vector".
# 
# We create edges to match variables to unit vectors and unit vectors to SI units. Each variable has one unit vector with the label of these edges as "has unit". For SI units to unit vectors, only some unit vectors are also SI units. These edges are labelled "is also"
# 
# Lastly we break every Physics equation from the Feynman lectures into subequations. Each subequation is comprised of variables and possibly other subequations. Every Physics equation can be traced back to its variables by following its subequations. 

# In[4]:


def generate_graph(all_mysteries, all_variables):
    mathtext = all_variables['Variable'].values
    G = nx.DiGraph()

    #creating nodes for all variables:
    for i in all_variables['Variable']:
      G.add_node(i, type = "Variable", latex = "$%s$" %add_brackets_to_underscores(add_slash(i)))

    all_variables['non-zero'] = all_variables[['m','s','kg','T','V']].apply(np.count_nonzero,axis = 1)
    all_variables['sum'] = all_variables[['m','s','kg','T','V']].apply(np.sum,axis = 1)
    SI_unit_variables = all_variables[(all_variables['non-zero']==1) & (all_variables['sum']==1)]['Variable']
    all_variables['Unit_vector'] = all_variables[['m','s','kg','T','V']].apply(list,axis = 1)

    #creating nodes for SI units:
    # print(np.unique(all_variables[(all_variables['non-zero']==1) & (all_variables['sum']==1)]['Units']))
    for i in np.unique(all_variables[(all_variables['non-zero']==1) & (all_variables['sum']==1)]['Units']): 
      #this adds all variables with SI units unit vector similar to SI unit
      G.add_node(i, type = "SI_Unit", latex = "$%s$" %i)

    #creating nodes for unit vectors:
    for i in np.unique(all_variables['Unit_vector']): #this adds all variables with SI units unit vector similar to SI unit
      G.add_node(tuple(i), type = "Unit_vector", latex = "$%s$" %i)

    #creating edges to match variables to their unit vectors (Variable -> has_unit -> unit_vector):
    number_of_nodes = G.number_of_nodes()
    for i in all_variables["Variable"][:]:
      G.add_edge(i, tuple(all_variables[all_variables["Variable"] == i]['Unit_vector'].values[0]), HAS_UNIT = True)
    if number_of_nodes != G.number_of_nodes(): print("ERROR: NODES ACCIDENTALLY ADDED WHILE ADDING EDGES (HAS_UNIT)")

    #creating edges to match unit vectors to SI units (unit vectors -> is_also -> SI_Unit):
    number_of_nodes = G.number_of_nodes()
    for i in np.unique(all_variables[(all_variables['non-zero']==1) & (all_variables['sum']==1)]['Units']):
      # print(all_variables[all_variables["Variable"] == i]['Units'].values)
      G.add_edge(tuple(all_variables[all_variables["Units"] == i]['Unit_vector'].values[0]), i, IS_ALSO = True)
    if number_of_nodes != G.number_of_nodes(): print("ERROR: NODES ACCIDENTALLY ADDED WHILE ADDING EDGES (IS_ALSO)")

    #creating nodes and edges for the subequations of a formula
    for f in range(len(all_mysteries['Formula'])):
    #     print(all_mysteries['Formula'][i])
        subequations = split_by_brackets("("+all_mysteries['Formula'][f]+")")
        for i in subequations:
            if i[0] != "(": i = "("+i+")"
            if i.find("EQN") < 0: #if it has no other equation inside it
                variables = find_variables_in_subeqns(i, mathtext)
                G.add_node(i, type = "Subequation", latex = "$%s$" %i)
                number_of_nodes = G.number_of_nodes()
                for j in variables:
                    G.add_edge(j, i, IS_VARIABLE_IN = True)
                if number_of_nodes != G.number_of_nodes(): print("ERROR: NODES ACCIDENTALLY ADDED WHILE ADDING EDGES (IS_IN)")
            else: 
                variables = find_variables_in_subeqns(i, mathtext)
                contains = []
                while i.find("EQN") > 0:
                    index = i.find("EQN")+3
                    eqn_no = i[index]
                    i = i.replace("EQN"+str(eqn_no), subequations[int(eqn_no)])
                    contains.append(subequations[int(eqn_no)])
                G.add_node(i, type = "Subequation", latex = "$%s$" %i)
                for k in set(contains): 
                    while k.find("EQN") > 0:
                        index = k.find("EQN")+3
                        eqn_no = k[index]
                        k = k.replace("EQN"+str(eqn_no), subequations[int(eqn_no)])
                    number_of_nodes = G.number_of_nodes()
                    if k[0] is not "(": k = "("+k+")"
                    G.add_edge(k, i, IS_SUBEQUATION_IN = True)
                    if number_of_nodes != G.number_of_nodes(): print("ERROR: NODES ACCIDENTALLY ADDED WHILE ADDING EDGES (IS_IN)")
                number_of_nodes = G.number_of_nodes()
                for j in variables:
                    G.add_edge(j, i, IS_VARIABLE_IN = True)
                if number_of_nodes != G.number_of_nodes(): print("ERROR: NODES ACCIDENTALLY ADDED WHILE ADDING EDGES (IS_IN)")
        number_of_nodes = G.number_of_nodes()
        G.add_edge(i, all_mysteries['Output'][f], IS_EQUAL = True)
        if number_of_nodes != G.number_of_nodes(): print("ERROR: NODES ACCIDENTALLY ADDED WHILE ADDING EDGES (IS_IN)")

    return G


# demonstrate the creation of the graph, by printing all its nodes and edges

# In[5]:


# G = generate_graph(import_equations(), import_variables())
# print([x for x in G.nodes()])
# print([(x,y,z) for x,y,z in G.edges(data = True)])







