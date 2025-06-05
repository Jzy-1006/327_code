import os
import time
import pandas as pd
import numpy as np
import scipy.sparse as sps
from scipy.sparse import linalg

import basis_change
import parameters as pam
import variational_space as vs
import hamiltonian as ham
import ground_state as gs
import lattice as lat


def compute_Aw_main(A=pam.A, Uoo=pam.Uoo, Upp=pam.Upp,
                    ed=pam.eds['29.5'], ep=pam.eps['29.5'], eo=pam.eos['29.5'],
                    tpd=pam.tpds['29.5'], tpp=pam.tpps['29.5'],
                    tdo=pam.tdos['29.5'], tpo=pam.tpos['29.5']):
    """
    计算一层Ni2O9的主程序
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
    def create_H0_matrix(VS):
        """
        设置hopping和onsite部分
        :return:
        """
        # 生成Tpd和Tpp矩阵
        Tpd = ham.create_tpd_nn_matrix(VS, tpd_nn_hop_dir, tpd_nn_hop_fac)
        Tpp = ham.create_tpp_nn_matrix(VS, tpp_nn_hop_dir, tpp_nn_hop_fac)
        # 生成Tdo和Tpo矩阵
        Tdo = ham.create_tdo_nn_matrix(VS, tdo_nn_hop_dir, tdo_nn_hop_fac)
        Tpo = ham.create_tpo_nn_matrix(VS, tpo_nn_hop_dir, tpo_nn_hop_fac)
        # 生成Tz矩阵, 层间杂化
        # tz_fac = ham.set_tz(if_tz_exist, tz_a1a1, tz_b1b1)
        # Tz = ham.create_tz_matrix(VS, tz_fac)
        # 跳跃部分
        H0 = Tpd + Tpp + Tdo + Tpo
        return H0

    t1 = time.time()
    # hopping和onsite部分
    tpd_nn_hop_dir, tpd_nn_hop_fac, tpp_nn_hop_dir, tpp_nn_hop_fac = ham.set_tpd_tpp(tpd, tpp)
    tdo_nn_hop_dir, tdo_nn_hop_fac, tpo_nn_hop_dir, tpo_nn_hop_fac = ham.set_tdo_tpo(tdo, tpo)

    if up_n == 0 and dn_n == 0:
        print('up_n and dn_n are not all 0')
        up_VS, dn_VS = None, None
        S_vals, Sz_vals = None, None
        H = None
    elif up_n != 0 and dn_n == 0:
        up_VS = vs.VariationalSpace(up_n)
        dn_VS = None
        print(f"hole_num = {pam.hole_num}, Sz = {Sz}, VS.dim = {up_VS.dim}\n")
        up_H0 = create_H0_matrix(up_VS)
        Esite = ham.create_Esite_matrix(up_VS, dn_VS, A, ed, ep, eo)
        H = up_H0 + Esite

        up_d_idx, up_p_idx, up_apz_idx = ham.get_double_occ_list(up_VS)
        S_vals = {}
        Sz_vals = {}
        Ni_num = len(lat.Ni_position)
        for i in range(Ni_num):
            up_di_idx = {key[1:]: item for key, item in up_d_idx.items() if key[0] == i}
            Hint, S_val, Sz_val = ham.create_interaction_matrix_d8(up_VS, dn_VS, up_di_idx, None, A)
            H = H + Hint
            S_vals[i] = S_val
            Sz_vals[i] = Sz_val

    elif up_n == 0 and dn_n != 0:
        up_VS = None
        dn_VS = vs.VariationalSpace(dn_n)
        print(f"hole_num = {pam.hole_num}, Sz = {Sz}, VS.dim = {dn_VS.dim}\n")
        dn_H0 = create_H0_matrix(dn_VS)
        Esite = ham.create_Esite_matrix(up_VS, dn_VS, A, ed, ep, eo)
        H = dn_H0 + Esite

        dn_d_idx, dn_p_idx, dn_apz_idx = ham.get_double_occ_list(dn_VS)
        S_vals = {}
        Sz_vals = {}
        Ni_num = len(lat.Ni_position)
        for i in range(Ni_num):
            dn_di_idx = {key[1:]: item for key, item in dn_d_idx.items() if key[0] == i}
            Hint, S_val, Sz_val = ham.create_interaction_matrix_d8(up_VS, dn_VS, None, dn_di_idx, A)
            H = H + Hint
            S_vals[i] = S_val
            Sz_vals[i] = Sz_val

    else:
        up_VS = vs.VariationalSpace(up_n)
        dn_VS = vs.VariationalSpace(dn_n)
        print(f"hole_num = {pam.hole_num}, Sz = {Sz}, up_VS.dim = {up_VS.dim}, "
              f"dn_VS.dim = {dn_VS.dim}, VS.dim = {up_VS.dim * dn_VS.dim}\n")

        up_H0 = create_H0_matrix(up_VS)
        up_I = sps.identity(up_VS.dim, format='csr')
        dn_H0 = create_H0_matrix(dn_VS)
        dn_I = sps.identity(dn_VS.dim, format='csr')

        Esite = ham.create_Esite_matrix(up_VS, dn_VS, A, ed, ep, eo)
        H = sps.kron(dn_I, up_H0) + sps.kron(dn_H0, up_I) + Esite

        up_d_idx, up_p_idx, up_apz_idx = ham.get_double_occ_list(up_VS)
        dn_d_idx, dn_p_idx, dn_apz_idx = ham.get_double_occ_list(dn_VS)
        # 依次变换到不同Ni上的耦合表象
        S_vals = {}
        Sz_vals = {}
        Ni_num = len(lat.Ni_position)
        for i in range(Ni_num):
            up_di_idx = {key[1:]: item for key, item in up_d_idx.items() if key[0] == i}
            dn_di_idx = {key[1:]: item for key, item in dn_d_idx.items() if key[0] == i}
            U_Ni = basis_change.create_singlet_triplet_basis_change_matrix_d8(up_VS, up_di_idx, dn_VS, dn_di_idx)
            H = U_Ni @ H @ U_Ni.T
            Hint, S_val, Sz_val = ham.create_interaction_matrix_d8(up_VS, dn_VS, up_di_idx, dn_di_idx, A)

            H = H + Hint
            S_vals[i] = S_val
            Sz_vals[i] = Sz_val
        del U_Ni, Hint, S_val, Sz_val

        Hint_po = ham.create_interaction_matrix_po(up_VS, dn_VS, up_n, dn_n, up_p_idx, dn_p_idx, up_apz_idx, dn_apz_idx, Upp, Uoo)
        H = H + Hint_po
        del Hint_po

        # 变换到耦合表象
        if pam.if_coupled:
            for i in range(Ni_num):
                up_di_idx = {key[1:]: item for key, item in up_d_idx.items() if key[0] == i}
                dn_di_idx = {key[1:]: item for key, item in dn_d_idx.items() if key[0] == i}
                U_Ni = basis_change.create_singlet_triplet_basis_change_matrix_d8(up_VS, up_di_idx, dn_VS, dn_di_idx)
                H = U_Ni.T @ H @ U_Ni
            U_coupled, S_val, Sz_val = basis_change.create_coupled_representation_matrix(up_VS, dn_VS)
            H = U_coupled @ H @ U_coupled.T

    t2 = time.time()
    print(f"build H time: {(t2 - t1) // 60 // 60}h, {(t2 - t1) // 60 % 60}min, {(t2 - t1) % 60}s\n")
    print(f"A = {A}, Uoo = {Uoo}, Upp = {Upp}\ned = {ed}, ep = {ep}, eo = {eo}\n"
          f"tpd = {tpd}, tpp = {tpp}, tdo = {tdo}, tpo = {tpo}\n")
    vals, vecs = linalg.eigsh(H, k=pam.Neval, which='SA')
    del H
    t3 = time.time()
    print(f"determine the eigenvalues of H time {(t3 - t2) // 60 // 60}h, {(t3 - t2) // 60 % 60}min, {(t3 - t2) % 60}s\n")

    if pam.if_coupled and up_n and dn_n:
        dL_weights, dL_orb_weights = gs.get_ground_state(up_VS, dn_VS, vals, vecs, S_vals, Sz_vals, S_val=S_val, Sz_val=Sz_val)
    else:
        dL_weights, dL_orb_weights = gs.get_ground_state(up_VS, dn_VS, vals, vecs, S_vals, Sz_vals)

    return vals, dL_weights, dL_orb_weights

def state_type_weight():
    """
    计算state_type_weight随参数的变化
    :return:
    """
    DFT_tpd = pam.tpds['29.5'][0]
    dL_types = ['d8-d7L-d8', 'd8-d8L-O-d8', 'd8-d7-O-d8', 'd8-d8L-d8L', 'd8-d7-d8L', 'd8-d8L-d7']
    orb_types = {'d8-d8L-O-d8': 'dx2dz2_dx2dz2L_apz_dx2dz2', 'd8-d8L-d8L': 'dx2dz2_dx2dz2L_dx2dz2L'}
    # dL_types = ['d8L-d9L', 'd8-d8L', 'd7-d8', 'd8-d9L2']
    # orb_types = ['dx2dz2_apz_dx2dz2', 'dx2dz2_dx2dz2L']
    dL_weight_tpd = {key: [] for key in dL_types}
    dL_weight_tpd['tpd'] = []
    orb_weight_tpd = {key: [] for key in orb_types}
    orb_weight_tpd['tpd'] = []
    for tpd in np.linspace(DFT_tpd * 0.8, DFT_tpd * 1.2, num=21, endpoint=True):
        # 高压 tdo/tpd
        tdo = 1.134 * tpd
        # # 低压 tdo/tpd
        # tdo = 0.942 * tpd
        dL_weight_tpd['tpd'].append(tpd)
        orb_weight_tpd['tpd'].append(tpd)
        # 高压 内层/外层
        _, dL_weights, dL_orb_weights = compute_Aw_main(tpd=np.array([tpd, tpd, tpd]),
                                                        tdo=np.array([tdo, 0.965*tdo, tdo]))
        # 低压 内层/外层
        # _, dL_weights, dL_orb_weights = compute_Aw_main(tpd=np.array([tpd, 1.083*tpd, tpd]),
        #                                                 tdo=np.array([tdo, 0.956*tdo, tdo]))

        for dL in dL_types:
            weight = dL_weights[dL]
            dL_weight_tpd[dL].append(weight)

        for dL_type, orb_type in orb_types.items():
            weight = dL_orb_weights[dL_type][orb_type]
            orb_weight_tpd[orb_type].append(weight)

    dL_weight_tpd = pd.DataFrame(dL_weight_tpd)
    dL_weight_tpd.to_csv('./data/dL_type.csv', index=False)

    orb_weight_tpd = pd.DataFrame(orb_weight_tpd)
    orb_weight_tpd.to_csv('./data/orb_type.csv', index=False)


def get_val_tpd():
    """
    得到同一Sz下, 不同tpd的本征值
    :return:
    """
    tpd_DFT = pam.tpds['29.5'][0]
    val_tpd = {'tpd': [], 'val': []}
    for tpd in np.linspace(tpd_DFT*0.8, tpd_DFT*1.2, num=21, endpoint=True):
        # 高压 tdo/tpd
        tdo = 1.134 * tpd
        # # 低压 tdo/tpd
        # tdo = 0.942 * tpd
        
        # 高压 内层/外层
        vals, _, _ = compute_Aw_main(tpd=np.array([tpd, tpd, tpd]),
                                    tdo=np.array([tdo, 0.965*tdo, tdo]))
        # # 低压 内层/外层
        # vals, _, _ = compute_Aw_main(tpd=np.array([tpd, 1.083*tpd, tpd]),
        #                              tdo=np.array([tdo, 0.956*tdo, tdo]))
        val = vals[0]
        val_tpd['tpd'].append(tpd)
        val_tpd['val'].append(val)
    val_tpd = pd.DataFrame(val_tpd)
    val_tpd.to_csv(f'./data/Sz={Sz}val_tpd.csv', index=False)


def get_val_pressure():
    """
    计算不同压力下的基态能量
    :return:
    """
    val_pressure = {'A': [], 'pressure': [], 'val': []}
    for A in pam.As:
        for pressure in pam.pressures:
            ed = pam.eds[pressure]
            ep = pam.eps[pressure]
            eo = pam.eos[pressure]

            tpd = pam.tpds[pressure]
            tpp = pam.tpps[pressure]
            tdo = pam.tdos[pressure]
            tpo = pam.tpos[pressure]

            vals, _, _ = compute_Aw_main(A=A, ed=ed, ep=ep, eo=eo, tpd=tpd, tpp=tpp, tdo=tdo, tpo=tpo)
            val_pressure['A'].append(A)
            val_pressure['pressure'].append(pressure)
            val_pressure['val'].append(vals[0])

    val_pressure = pd.DataFrame(val_pressure)
    val_pressure.to_csv('./data/val_pressure.csv', index=False)


def get_max_type(val):
    """
    得到最大的态类型
    :param: val = {parameter1: value1, parameter2: value2}
    :return: max_type, max_orb_type
    """
    _, dL_weights, dL_orb_weights = compute_Aw_main(**val)
    max_dL = next(iter(dL_weights))
    max_weight = dL_weights[max_dL]

    for dL, weight in dL_weights.items():
        dL_list = dL.split('-')
        if dL_list == dL_list[::-1]:
            weight1 = weight
        else:
            weight1 = weight * 2
        if weight1 > max_weight:
            max_dL = dL
            max_weight = weight1

    max_orb = next(iter(dL_orb_weights[max_dL]))
    max_orb_weight = dL_orb_weights[max_dL][max_orb]
    for orb, weight in dL_orb_weights[max_dL].items():
        orb_list = orb.split('_')
        if orb_list == orb_list[::-1]:
            weight1 = weight
        else:
            weight1 = weight * 2
        if weight1 > max_orb_weight:
            max_orb = orb
            max_orb_weight = weight1

    # 标准化max_dL, max_orb
    max_dL_list = max_dL.split('-')
    if max_dL_list > max_dL_list[::-1]:
        max_dL = '-'.join(max_dL_list[::-1])

    max_orb_list = max_orb.split('_')
    if max_orb_list > max_orb_list[::-1]:
        max_orb = '_'.join(max_orb_list[::-1])

    return max_dL, max_orb


class BinaryTreeNode:
    def __init__(self, bounds, depth=0):
        self.bounds = bounds    # (x_min, x_max)
        self.children = []  # 子节点
        self.depth = depth  # 当前深度
        self.is_same = False    # 两端类型是否相同


def evaluate_node(node, min_size, max_depth, global_cache, fix_var, fix_val, binary_var):
    """
    利用二分法, 判断(x_min, x_max)两端态的类型是否相同, 若不同取中点x_mid,
    生成子节点(x_min, x_mid)和(x_mid, x_max), 递归生成二叉树
    :param node:节点
    :param min_size: 最小区域, x_max - x_min
    :param max_depth: 递归深度
    :param global_cache: 存储计算过的端点, 避免重复计算
    :param fix_var: 固定的变量名
    :param fix_val: 固定的变量值
    :param binary_var: 被二分的变量名
    """
    state_orb = []
    bounds = node.bounds
    x_min, x_max = bounds
    for x in (x_min, x_max):
        key = round(x, 5)
        if key in global_cache:
            state_orb.append(global_cache[key])
        else:
            # 高压, 固定tdo, 二分tpd, 内层/外层
            state_type, orb_type = get_max_type({fix_var: np.array([fix_val, 0.965*fix_val, fix_val]), binary_var: np.array([x, x, x])})
            
            # # 低压, 固定tdo, 二分tpd, 内层/外层
            # state_type, orb_type = get_max_type({fix_var: np.array([fix_val, 0.956*fix_val, fix_val]), binary_var: np.array([x, 1.083*x, x])})

            state_orb.append((state_type, orb_type))
            global_cache[key] = (state_type, orb_type)
    if state_orb[0] == state_orb[1]:
        node.is_same = True
    else:
        if (x_max - x_min) <= min_size or node.depth >= max_depth:
            node.is_same = False
        else:
            x_mid = (x_min + x_max) / 2

            # 生成子节点
            child_bounds = [(x_min, x_mid), (x_mid, x_max)]
            node.children = [BinaryTreeNode(b, node.depth + 1) for b in child_bounds]

            # 递归评估子节点
            for child in node.children:
                evaluate_node(child, min_size, max_depth, global_cache, fix_var, fix_val, binary_var)


def collect_boundary(node, segments):
    """
    找到满足两端类型不同的部分(二叉树树枝), 取中间值作为边界点
    :param node: 节点
    :param segments: 树枝片段
    """
    if node.children:
        for child in node.children:
            collect_boundary(child, segments)
    else:
        if not node.is_same:
            x_min, x_max = node.bounds
            segments.append((x_min + x_max) / 2)


def phase_diagram(region, fix_var):
    """
    找到相图的边界
    :region: 扫描区域
    :fix_var: 固定的变量名
    :step_size: 移动步长
    :return:
    """
    # 先固定一个变量不动, 二分另外一个变量
    boundary = {key: [] for key in region.keys()}
    fix_range = region[fix_var]
    binary_var = None
    for key in region.keys():
        if key != fix_var:
            binary_var = key
    for x1 in fix_range:
        # 生成二叉树
        root = BinaryTreeNode(bounds=region[binary_var])
        global_cache = {}   # 存储计算过的端点, 避免重复计算
        evaluate_node(root, 0.01,10, global_cache, fix_var, x1, binary_var)


        # 打印计算过的state_type和orb_type
        sorted_keys = sorted(global_cache.keys())
        print("x: state_type, orb_type")
        current_type = None
        for x in sorted_keys:
            state_type, orb_type = global_cache[x]

            # 当(state_type, orb_type)与之前不同时, 打一空行, 来区分
            if (state_type, orb_type) != current_type:
                print()
                current_type = (state_type, orb_type)

            print(f"{x}: {state_type}, {orb_type}")
        print()
        # 找出边界点
        boundary_segments = []
        collect_boundary(root, boundary_segments)
        for x2 in boundary_segments:
            boundary[fix_var].append(x1)
            boundary[binary_var].append(x2)

    boundary = pd.DataFrame(boundary)
    boundary.to_csv('./data/boundary.csv', index=False)


if __name__ == '__main__':
    t0 = time.time()
    # 创建data的文件夹
    os.makedirs('data', exist_ok=True)
    # 计算前, 清空文件夹所有文件的内容
    for filename in os.listdir('data'):
        file_path = os.path.join('data', filename)
        with open(file_path, 'w') as file:
            file.truncate(0)

    # tz_a1a1 = pam.tz_a1a1
    # tz_b1b1 = pam.tz_b1b1
    # if_tz_exist = pam.if_tz_exist
    for Sz in pam.Sz_list:
        hole_num = pam.hole_num
        up_n = (hole_num + 2 * Sz) // 2
        up_n = int(up_n)
        dn_n = (hole_num - 2 * Sz) // 2
        dn_n = int(dn_n)

        assert up_n >= 0 and dn_n >=0, 'Sz is error'
        assert hole_num == (up_n + dn_n), 'Sz is error'

        compute_Aw_main()
        # state_type_weight()
        # get_val_tpd()
        # get_val_pressure()
        # phase_diagram({'tpd': (0.1, 4.2), 'tdo': np.linspace(2.4, 4.2, 1)}, 'tdo')

        t_end = time.time()
        print(f'total time {(t_end-t0)//60//60}h, {(t_end-t0)//60%60}min, {(t_end-t0)%60}s\n')
