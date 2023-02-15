"""
Perhaps place in here some ways of 
"""


import classes

X = classes.DT_IFP()
Y = classes.CT_stat_IFP()
Z = classes.CT_nonstat_IFP()

X_list = [method for method in dir(X) if method.startswith('__') is False]
Y_list = [method for method in dir(Y) if method.startswith('__') is False]
Z_list = [method for method in dir(Z) if method.startswith('__') is False]

with open(r'DT_IFP_methods.txt', 'w') as fp:
    for item in X_list:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'CT_stat_IFP_methods.txt', 'w') as fp:
    for item in Y_list:
        # write each item on a new line
        fp.write("%s\n" % item)

with open(r'CT_nonstat_IFP_methods.txt', 'w') as fp:
    for item in Z_list:
        # write each item on a new line
        fp.write("%s\n" % item)
