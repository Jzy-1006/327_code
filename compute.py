import os
import time
import numpy as np
import pandas as pd

import basis_change
import parameters as pam
import variational_space as vs
import hamiltonian as ham
import ground_state as gs


def compute_Aw_main(A, Uoo, Upp, ed, ep, eo, tpd, tpp, tdo, tpo):
    """
    计算La3Ni4O10的主程序
    :param A:
    :param Uoo:
    :param Upp:
    :param ed:
    :param ep:
    :param eo:
    :param tpd:
    :param tpp:
    :param tdo:
    :param tpo:
    :return:
    """
    # 生成Tpd和Tpp矩阵
    tpd_nn_hop_dir, tpd_nn_hop_fac, tpp_nn_hop_dir, tpp_nn_hop_fac = ham.set_tpd_tpp(tpd, tpp)
    Tpd = ham.create_tpd_nn_matrix(VS, tpd_nn_hop_dir, tpd_nn_hop_fac)
    Tpp = ham.create_tpp_nn_matrix(VS, tpp_nn_hop_dir, tpp_nn_hop_fac)
    # 生成Tdo和Tpo矩阵
    tdo_nn_hop_dir, tdo_nn_hop_fac, tpo_nn_hop_dir, tpo_nn_hop_fac = ham.set_tdo_tpo(tdo, tpo)
    Tdo = ham.create_tdo_nn_matrix(VS, tdo_nn_hop_dir, tdo_nn_hop_fac)
    Tpo = ham.create_tpo_nn_matrix(VS, tpo_nn_hop_dir, tpo_nn_hop_fac)
    # 生成Tz矩阵, 层间杂化
    tz_fac = ham.set_tz(if_tz_exist, tz_a1a1, tz_b1b1)
    Tz = ham.create_tz_matrix(VS, tz_fac)
    # 生成Esite矩阵
    Esite = ham.create_Esite_matrix(VS, A, ed, ep, eo)
    # 跳跃部分
    H0 = Tpd + Tpp + Tdo + Tpo + Tz + Esite

    H = H0
    # 依次变换到不同Ni(d8)上的耦合表象
    for position, U_Ni in multi_U_Ni.items():
        U_Ni_d = (U_Ni.conjugate()).transpose()
        H_new = U_Ni_d @ H @ U_Ni
        d_state_idx = multi_d_state_idx[position]
        d_hole_idx = multi_d_hole_idx[position]
        S_val = multi_S[position]
        Sz_val = multi_Sz[position]
        Hint = ham.create_interaction_matrix_d8(VS, d_state_idx, d_hole_idx, S_val, Sz_val, A)
        H = H_new + Hint
    Hint_po = ham.create_interaction_matrix_po(VS, p_idx_pair, apz_idx_pair, Upp, Uoo)
    H = H + Hint_po

    print(f"\nA = {A}, Uoo = {Uoo}, Upp = {Upp}\ned = {ed}, ep = {ep}, eo = {eo}\n"
          f"tpd = {tpd}, tpp = {tpp}, tdo = {tdo}, tpo = {tpo}\n")
    if pam.layer_num == 2:
        H_bond = U_bond_d @ H @ U_bond
        df = gs.get_ground_state(H_bond, VS, multi_S, multi_Sz, bonding_val=bonding_val)
    else:
        df = gs.get_ground_state(H, VS, multi_S, multi_Sz)

    return df


def type_weight():
    """
    计算state_type_weight随参数的变化
    :return: df, 列分别是state_type, 参数1下的state_type_weight, 参数2...
    """
    tpd = tpd_list[0]
    tdo = tpd*1.05
    dfs = compute_Aw_main(A, Uoo, Upp, ed, ep, eo, tpd, tpp, tdo, tpo)
    dfs.rename(columns={"type_weight":f"tpd={tpd}"}, inplace=True)

    for tpd in tpd_list[1:]:
        tdo = 1.05*tpd
        df = compute_Aw_main(A, Uoo, Upp, ed, ep, eo, tpd, tpp, tdo, tpo)
        df.rename(columns={"type_weight":f"tpd={tpd}"}, inplace=True)
        dfs = pd.merge(dfs, df, on='state_type', how='inner')
    dfs.to_csv('./data/type_weight.csv', index=False)


if __name__ == '__main__':
    t0 = time.time()
    # 创建data的文件夹
    os.makedirs('data', exist_ok=True)
    # 计算前, 清空文件夹所有文件的内容
    for filename in os.listdir('data'):
        file_path = os.path.join('data', filename)
        with open(file_path, 'w') as file:
            file.truncate(0)

    tz_a1a1 = pam.tz_a1a1
    tz_b1b1 = pam.tz_b1b1
    if_tz_exist = pam.if_tz_exist

    VS = vs.VariationalSpace()
    multi_d_state_idx, multi_d_hole_idx, p_idx_pair, apz_idx_pair = ham.get_double_occ_list(VS)

    # 变换到Ni(d8)上耦合表象下的矩阵
    multi_U_Ni = {}
    multi_S = {}
    multi_Sz = {}
    # 遍历所有的Ni
    for position, d_state_idx in multi_d_state_idx.items():
        d_hole_idx = multi_d_hole_idx[position]
        U_Ni, S_val, Sz_val = basis_change.create_singlet_triplet_basis_change_matrix_d8(VS, d_state_idx, d_hole_idx)
        multi_U_Ni[position] = U_Ni
        multi_S[position] = S_val
        multi_Sz[position] = Sz_val

    # bonding, anti_bonding变换矩阵
    if pam.layer_num == 2:
        U_bond, bonding_val = basis_change.create_bonding_anti_bonding_basis_change_matrix(VS)
        U_bond_d = (U_bond.conjugate()).transpose()

    A = pam.A
    Uoo = pam.Uoos[0]
    Upp = pam.Upps[0]

    ed = pam.ed_list[4]
    ep = pam.ep_list[4]
    eo = pam.eo_list[4]

    tpd_list = np.linspace(0.9*1.58, 1.1*1.58, num=3)
    # tpd = pam.tpd_list[4]
    tpp = pam.tpp_list[4]
    # tdo = pam.tdo_list[4]
    tpo = pam.tpo_list[4]
    type_weight()

    t1 = time.time()
    print('compute cost time', t1-t0)
