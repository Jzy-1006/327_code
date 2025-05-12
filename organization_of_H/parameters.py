import numpy as np

# 空穴数目: 2*hole_num, layer_num: 层数, Norb: 轨道数目
hole_num = 2
layer_num = 2
# Sz = 'All_Sz'时, 考虑的所有自旋的情况
# Sz_list = ['All_Sz']
Sz_list = [0]
if_coupled = 0
if_save_coupled_uid = 1
if_basis_change_type = 'd_double'

Norb = 5
Mc = 2
pressure_list = (0, 4, 8, 16, 29.5)

A = 6.
A_list = [5., 6., 7.]
B = 0.15
C = 0.58
Upp = 4.
Uoo = 4.
Upps = [4.0]
Uoos = [4.0]

ed_list = ({'d3z2r2': 0.02, 'dx2y2': 0.},
           {'d3z2r2': 0.1, 'dx2y2': 0.})
ep_list = (2.872, 3.121)
eo_list = (2.762, 2.870)

tpd_list = (1.133, 1.377)
tpp_list = (0.435, 0.497)
tdo_list = (1.210, 1.502)
tpo_list = (0.435, 0.467)
tz_a1a1 = 0.0
tz_b1b1 = 0.0

if_tz_exist = 2
if_bond = 0
val_num = 1    # 要有足够的Neval才能计算val_num！！！
Neval = 30

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
