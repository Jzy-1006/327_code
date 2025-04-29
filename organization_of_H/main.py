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
# from memory_profiler import profile


def compute_Aw_main(A=pam.A, Uoo=pam.Uoo, Upp=pam.Upp,
                    ed=pam.ed_list[0], ep=pam.ep_list[0], eo=pam.eo_list[0],
                    tpd=pam.tpd_list[0], tpp=pam.tpp_list[0],
                    tdo=pam.tdo_list[0], tpo=pam.tpo_list[0]):
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
    t1 = time.time()
    # 生成Tpd和Tpp矩阵
    tpd_nn_hop_dir, tpd_nn_hop_fac, tpp_nn_hop_dir, tpp_nn_hop_fac = ham.set_tpd_tpp(tpd, tpp)
    Tpd = ham.create_tpd_nn_matrix(VS, tpd_nn_hop_dir, tpd_nn_hop_fac)
    Tpp = ham.create_tpp_nn_matrix(VS, tpp_nn_hop_dir, tpp_nn_hop_fac)
    # 生成Tdo和Tpo矩阵
    tdo_nn_hop_dir, tdo_nn_hop_fac, tpo_nn_hop_dir, tpo_nn_hop_fac = ham.set_tdo_tpo(tdo, tpo)
    Tdo = ham.create_tdo_nn_matrix(VS, tdo_nn_hop_dir, tdo_nn_hop_fac)
    Tpo = ham.create_tpo_nn_matrix(VS, tpo_nn_hop_dir, tpo_nn_hop_fac)
    # 生成Tz矩阵, 层间杂化
    # tz_fac = ham.set_tz(if_tz_exist, tz_a1a1, tz_b1b1)
    # Tz = ham.create_tz_matrix(VS, tz_fac)
    # 生成Esite矩阵
    Esite = ham.create_Esite_matrix(VS, A, ed, ep, eo)
    # 跳跃部分
    H0_up = Tpd + Tpp + Tdo + Tpo
    del Tpd, Tpp, Tdo, Tpo
    I = sps.identity(VS.dim, format='csr')

    H = sps.kron(H0_up, I) + sps.kron(I, H0_up) + Esite
    del H0_up, Esite

    # 依次变换到不同Ni上的耦合表象
    S_vals = {}
    Sz_vals = {}
    Ni_num = len(lat.Ni_position)
    for i in range(Ni_num):
        di_idx = {key[1:]: item for key, item in d_idx.items() if key[0] == i}
        U_Ni = basis_change.create_singlet_triplet_basis_change_matrix_d8(VS, di_idx)
        H = U_Ni @ H @ U_Ni.T
        Hint, S_val, Sz_val = ham.create_interaction_matrix_d8(VS, di_idx, A)
        H = H + Hint
        S_vals[i] = S_val
        Sz_vals[i] = Sz_val
    del U_Ni, Hint, S_val, Sz_val

    Hint_po = ham.create_interaction_matrix_po(VS, p_idx, apz_idx, Upp, Uoo)
    H = H + Hint_po
    del Hint_po

    t2 = time.time()
    print(f"build H time: {(t2 - t1) // 60 // 60}h, {(t2 - t1) // 60 % 60}min, {(t2 - t1) % 60}s\n")
    print(f"A = {A}, Uoo = {Uoo}, Upp = {Upp}\ned = {ed}, ep = {ep}, eo = {eo}\n"
          f"tpd = {tpd}, tpp = {tpp}, tdo = {tdo}, tpo = {tpo}\n")

    # 变换到耦合表象
    # if Sz == 'All_Sz' and pam.if_coupled == 1:
    #     for _, U in multi_U_Ni.items():
    #         U_d = (U.conjugate()).transpose()
    #         H = U @ H @ U_d
    #     H_coupled = U_coupled_d @ H @ U_coupled
    #     vals, df = gs.get_ground_state(H_coupled, VS, multi_S, multi_Sz, S_val=S_val, Sz_val=Sz_val)
    # else:
    #     vals, df = gs.get_ground_state(H, VS)
    vals, vecs = linalg.eigsh(H, k=pam.Neval, which='SA')
    del H
    t3 = time.time()
    print(f"determine the eigenvalues of H time {(t3 - t2) // 60 // 60}h, {(t3 - t2) // 60 % 60}min, {(t3 - t2) % 60}s\n")
    gs.get_ground_state(VS, vals, vecs, S_vals, Sz_vals)
    # return vals, df


def state_type_weight():
    """
    计算state_type_weight随参数的变化
    :return: df_types, 列分别是state_type, 参数1下的state_type_weight, 参数2...
    """
    df_types = {'tpd': []}
    df_orb_types = {'tpd': []}
    for tpd in np.linspace(1.58*0.5, 1.58*1.1, num=7, endpoint=True):
        tdo = 1.05*tpd
        _, df = compute_Aw_main(tpd=tpd, tdo=tdo)

        # 提取态类型和type_weight, 并除去重复列
        df_type = df[['state_type', 'type_weight']].drop_duplicates(subset='state_type')
        for _, row in df_type.iterrows():
            state_type = row['state_type']
            type_weight = row['type_weight']
            if state_type in df_types:
                df_types[state_type].append(type_weight)
            else:
                df_types[state_type] = [type_weight]
        df_types['tpd'].append(tpd)

        # 提取orb_type和orb_type_weight, 并除去重复列
        df_orb_type = df[['orb_type', 'orb_type_weight']].drop_duplicates(subset='orb_type')
        for _, row in df_orb_type.iterrows():
            orb_type = row['orb_type']
            orb_type_weight = row['orb_type_weight']
            if orb_type in df_orb_types:
                df_orb_types[orb_type].append(orb_type_weight)
            else:
                df_orb_types[orb_type] = [orb_type_weight]
        df_orb_types['tpd'].append(tpd)

    df_types = pd.DataFrame(df_types)

    df_types.to_csv('./data/state_type.csv', index=False)

    df_orb_types = pd.DataFrame(df_orb_types)
    df_orb_types.to_csv('./data/orb_type.csv', index=False)


def get_val_tpd():
    """
    得到同一Sz下, 不同tpd的本征值
    :return:
    """
    tpd_list = np.linspace(1.58*0.5, 1.58*1.1, num=7, endpoint=True)
    val_tpd = {'tpd': [], 'val': []}
    for tpd in tpd_list:
        tdo = 1.05*tpd
        vals, _ = compute_Aw_main(tpd=tpd, tdo=tdo)
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
    for A in pam.A_list:
        for i, pressure in enumerate(pam.pressure_list):
            ed = pam.ed_list[i]
            ep = pam.ep_list[i]
            eo = pam.eo_list[i]

            tpd = pam.tpd_list[i]
            tpp = pam.tpp_list[i]
            tdo = pam.tdo_list[i]
            tpo = pam.tpo_list[i]

            vals, _ = compute_Aw_main(A=A, ed=ed, ep=ep, eo=eo, tpd=tpd, tpp=tpp, tdo=tdo, tpo=tpo)
            val_pressure['A'].append(A)
            val_pressure['pressure'].append(pressure)
            val_pressure['val'].append(vals[0])

    val_pressure = pd.DataFrame(val_pressure)
    val_pressure.to_csv('./data/val_pressure.csv', index=False)


def get_max_type(val):
    """
    得到最大的态类型
    :param: val = {variable1: value1, variable2: value2}
    :return: max_type, max_orb_type
    """
    _, df = compute_Aw_main(**val)
    # 提取态的类型
    df_type = df[['state_type', 'type_weight']].drop_duplicates(subset='state_type')
    # 将type_weight相同的分在同一组，并进行求和
    df_type['group_key'] = (df_type['type_weight'].diff().abs() > 1e-5).cumsum()
    df_type = df_type.groupby('group_key', as_index=False).agg({'state_type': 'min', 'type_weight': 'sum'})
    # 对type_weight进行降序
    df_type.sort_values(by='type_weight', ascending=False, inplace=True)
    # 提取第一行
    max_type = df_type['state_type'].iloc[0]

    # 提取轨道类型
    df_orb = df[df['state_type']==max_type]
    df_orb = df_orb[['orb_type', 'orb_type_weight']].drop_duplicates(subset='orb_type')
    df_orb.sort_values(by=['orb_type_weight', 'orb_type'], ascending=[False, True], inplace=True)
    # 提取第一行
    max_orb_type = df_orb['orb_type'].iloc[0]

    return max_type, max_orb_type


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

            state_type, orb_type = get_max_type({fix_var: fix_val, binary_var: x})
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
    Sz = pam.Sz_list[0]

    VS = vs.VariationalSpace()
    d_idx, p_idx, apz_idx = ham.get_double_occ_list(VS)

    # if Sz == 'All_Sz' and pam.if_coupled == 1:
    #     U_coupled, S_val, Sz_val = basis_change.create_coupled_representation_matrix(VS)
    #     U_coupled_d = (U_coupled.conjugate()).transpose()

    compute_Aw_main()
    # state_type_weight()
    # get_val_tpd()
    # get_val_pressure()
    # phase_diagram({'tpd': (0, 4.2), 'tdo': np.linspace(0.3, 4.2, 1)}, 'tdo')

    t_end = time.time()
    print(f'total time {(t_end-t0)//60//60}h, {(t_end-t0)//60%60}min, {(t_end-t0)%60}s\n')
