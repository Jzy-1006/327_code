import numpy as np

# hole_num: 空穴数目, layer_num: 层数, Norb: 轨道数目
hole_num = 9
layer_num = 3

Sz_list = [1/2]
if_coupled = 0
if_save_coupled_uid = 1
if_basis_change_type = 'd_double'

Norb = 5
Mc = 2
pressures = ('low', 'high')

A = 6.
As = [5., 6., 7.]
B = 0.15
C = 0.58
Upp = 4.
Uoo = 4.
Upps = [4.0]
Uoos = [4.0]

# 三层数据
eds = {'high': ({'d3z2r2': 0.102, 'dx2y2': 0.},
                {'d3z2r2': -0.1, 'dx2y2': 0.},
                {'d3z2r2': 0.102, 'dx2y2': 0.}),
        'low': ({'d3z2r2': 0.170, 'dx2y2': 0.},
                {'d3z2r2': 0.02, 'dx2y2': 0.},
                {'d3z2r2': 0.170, 'dx2y2': 0.})}
eps = {'high': (3.224, 3.121, 3.224),
           'low': (3.093, 2.875, 3.093)}
eos = {'high': (2.834, 2.834),
           'low': (2.760, 2.760)}

tpds = {'high': np.array([1.373, 1.377, 1.373]),
            'low': np.array([1.187, 1.285, 1.187])}
tpps = {'high': np.array([0.521, 0.497, 0.521]),
            'low': np.array([0.478, 0.435, 0.478])}
tdos = {'high': np.array([1.557, 1.502, 1.557]),
            'low': np.array([1.266, 1.210, 1.266])}
tpos = {'high': np.array([0.456, 0.467, 0.456]),
            'low': np.array([0.403, 0.435, 0.403])}


# 两层数据
# eds = {'0': ({'d3z2r2': 0.046, 'dx2y2': 0.},
#              {'d3z2r2': 0.046, 'dx2y2': 0.}),
#        '4': ({'d3z2r2': 0.054, 'dx2y2': 0.},
#              {'d3z2r2': 0.054, 'dx2y2': 0.}),
#        '8': ({'d3z2r2': 0.060, 'dx2y2': 0.},
#              {'d3z2r2': 0.060, 'dx2y2': 0.}),
#        '16': ({'d3z2r2': 0.072, 'dx2y2': 0.},
#               {'d3z2r2': 0.072, 'dx2y2': 0.}),
#        '29.5': ({'d3z2r2': 0.095, 'dx2y2': 0.},
#                 {'d3z2r2': 0.095, 'dx2y2': 0.})}
# eps = {'0': (2.47, 2.47),
#        '4': (2.56, 2.56),
#        '8': (2.62, 2.62),
#        '16': (2.75, 2.75),
#        '29.5': (2.9, 2.9)}
# eos = {'0': (2.94, 2.94),
#        '4': (3.03, 3.03),
#        '8': (3.02, 3.02),
#        '16': (3.14, 3.14),
#        '29.5': (3.24, 3.24)}
#
# tpds = {'0': np.array([1.38, 1.38]),
#         '4': np.array([1.43, 1.43]),
#         '8': np.array([1.46, 1.46]),
#         '16': np.array([1.52, 1.52]),
#         '29.5': np.array([1.58, 1.58])}
# tpps = {'0': np.array([0.537, 0.537]),
#         '4': np.array([0.548, 0.548]),
#         '8': np.array([0.554, 0.554]),
#         '16': np.array([0.566, 0.566]),
#         '29.5': np.array([0.562, 0.562])}
# tdos = {'0': np.array([1.48, 1.48]),
#         '4': np.array([1.53, 1.53]),
#         '8': np.array([1.55, 1.55]),
#         '16': np.array([1.61, 1.61]),
#         '29.5': np.array([1.66, 1.66])}
# tpos = {'0': np.array([0.445, 0.445]),
#         '4': np.array([0.458, 0.458]),
#         '8': np.array([0.468, 0.468]),
#         '16': np.array([0.484, 0.484]),
#         '29.5': np.array([0.487, 0.487])}
# tz_a1a1 = 0.028
# tz_b1b1 = 0.047

if_tz_exist = 2
if_bond = 0
val_num = 1    # 要有足够的Neval才能计算val_num！！！
Neval = 5

if Norb == 5:
    Ni_orbs = ['dx2y2', 'd3z2r2']
    O1_orbs = ['px']
    O2_orbs = ['py']
    Oap_orbs = ['apz']
else:
    print('Norb is error')
O_orbs = O1_orbs + O2_orbs
O_orbs.sort()
O1_orbs.sort()
O2_orbs.sort()
Oap_orbs.sort()
