import time
import numpy as np
import scipy.sparse as sps
from itertools import product, combinations
from bisect import bisect_left
from collections import defaultdict
from sympy import Rational
from sympy.physics.quantum.cg import CG

import lattice as lat
import variational_space as vs
import parameters as pam


def set_singlet_triplet_matrix_element(VS, state_idx, hole_idx1, hole_idx2,
                                       row, col, data, S_double_val, Sz_double_val, count_list):
    """
    设置单态三重态的变换矩阵元
    """
    state = vs.get_state(VS.lookup_tbl[state_idx])
    orb1, s1 = state[hole_idx1][-2:]
    orb2, s2 = state[hole_idx2][-2:]

    # 当两个空穴自旋相同时
    if s1 == s2:
        row.append(state_idx)
        col.append(state_idx)
        data.append(np.sqrt(2))
        S_double_val[state_idx] = 1
        if s1 == 'up':
            Sz_double_val[state_idx] = 1
        else:
            Sz_double_val[state_idx] = -1

    # 当两个空穴自旋不同是, 分为轨道相同和不同两个部分
    else:
        if orb1 == orb2:
            data.append(np.sqrt(2))
            row.append(state_idx)
            col.append(state_idx)
            S_double_val[state_idx] = 0
            Sz_double_val[state_idx] = 0
        else:
            # 交换两个自旋
            partner_state = [list(hole) for hole in state]
            partner_state[hole_idx1][-1], partner_state[hole_idx2][-1] = \
                partner_state[hole_idx2][-1], partner_state[hole_idx1][-1]
            partner_state = [tuple(hole) for hole in partner_state]
            partner_state, _ = vs.make_state_canonical(partner_state)
            # 找到对应的索引
            partner_idx = VS.get_index(partner_state)
            count_list.append(partner_idx)

            # 将state_idx设为单态 = 1/sqrt(2)(|up, dn> - |dn, up>)
            # 注意在这里也有可能是1/sqrt(2)(|dn, up> - |up, dn>)
            data.append(1.)
            row.append(state_idx)
            col.append(state_idx)

            data.append(-1.)
            row.append(partner_idx)
            col.append(state_idx)

            S_double_val[state_idx] = 0
            Sz_double_val[state_idx] = 0

            # 将partner_idx设为三重态 = 1/sqrt(2)(|up, dn> + |dn, up>)
            data.append(1.)
            row.append(state_idx)
            col.append(partner_idx)

            data.append(1.)
            row.append(partner_idx)
            col.append(partner_idx)

            S_double_val[partner_idx] = 1
            Sz_double_val[partner_idx] = 0


def create_singlet_triplet_basis_change_matrix_d8(VS, d_idx):
    """
    对d8的单态三重态变换矩阵
    :param VS:
    :param d_idx:{}
    :return:
    """
    # 整体框架
    # 1.生成单位矩阵
    # 2.生成非对角部分
    # 3.非1对角元的修正
    t0 = time.time()
    dim = VS.dim
    # 1.单位矩阵
    out_diag = sps.eye(dim*dim, format='coo')
    # 2.非1对角元
    diag_i = []
    diag_val= []
    # 3.非对角项
    off_i = []
    off_j = []
    off_val = []
    # 存储非1对角元, 非对角项
    ph = -1
    for i1_idx, i in enumerate(d_idx[(1, 0)]):
        i1 = d_idx[(0, 1)][i1_idx]
        for j1_idx, j in enumerate(d_idx[(0, 1)]):
            j1 = d_idx[(1, 0)][j1_idx]
            k_updn = i * dim + j
            k_dnup = i1 * dim + j1

            # 1/sqrt(2)(|up, dn> + |dn, up>), U @ H0 @ U.T
            diag_val.append(1/np.sqrt(2))
            diag_i.append(k_updn)

            off_val.append(ph/np.sqrt(2))
            off_i.append(k_updn)
            off_j.append(k_dnup)

            # 1/sqrt(2)(-|dn, up> + |up, dn>)
            diag_val.append(-ph/np.sqrt(2))
            diag_i.append(k_dnup)

            off_val.append(1/np.sqrt(2))
            off_i.append(k_dnup)
            off_j.append(k_updn)

    # 非1对角修正
    out_diag_correction = sps.coo_matrix((np.array(diag_val)-1., (diag_i, diag_i)), shape=(dim*dim, dim*dim))
    # 非对角项
    out_offdiag = sps.coo_matrix((off_val, (off_i, off_j)), shape=(dim*dim, dim*dim))
    t1 = time.time()
    out = out_diag + out_diag_correction + out_offdiag
    print(f'singlet_triplet_double basis change time {(t1 - t0) // 60 // 60}h, {(t1 - t0) // 60 % 60}min, {(t1 - t0) % 60}s')

    return out.tocsr()


def create_singlet_triplet_basis_change_matrix(VS, d_state_idx, d_hole_idx, position):
    """
    单边的单态三重态变换矩阵
    :param VS:
    :param d_state_idx:[state_idx1, state_idx2, ...]
    :param d_hole_idx:[hole_idx1, hole_idx2, ...]
    :param position: Ni的位置
    :return:
    """
    t0 = time.time()
    dim = VS.dim
    row = []
    col = []
    data = []

    # 标记该态是单态或者三重态
    S_Ni_val = {}
    Sz_Ni_val = {}
    # 存储partner state的索引, 避免重复
    count_list = []

    # 遍历态空间
    for state_idx in range(dim):
        if state_idx in count_list:
            continue

        # d8
        if state_idx in d_state_idx:
            # 找到d8对应的空穴索引
            idx = d_state_idx.index(state_idx)
            hole_idx1, hole_idx2 = d_hole_idx[idx]
            set_singlet_triplet_matrix_element(VS, state_idx, hole_idx1, hole_idx2,
                                               row, col, data, S_Ni_val, Sz_Ni_val, count_list)
        else:
            state = vs.get_state(VS.lookup_tbl[state_idx])
            side1, side2 = lat.get_Ni_side_num(state)
            # 一边是两个空穴的情况. 为防止重复, 先只做一边的变换
            if (len(side1) == 2 and position[2] == 0) or (len(side2) == 2 and position[2] == 2):
                # 要交换的两个空穴索引
                if position[2] == 0:
                    hole_idx1, hole_idx2 = side1
                else:
                    hole_idx1, hole_idx2 = side2
                set_singlet_triplet_matrix_element(VS, state_idx, hole_idx1, hole_idx2,
                                                   row, col, data, S_Ni_val, Sz_Ni_val, count_list)
            else:
                data.append(np.sqrt(2))
                row.append(state_idx)
                col.append(state_idx)

    t1 = time.time()
    print(f'singlet_triplet_double basis change time {(t1-t0)//60//60}h, {(t1-t0)//60%60}min, {(t1-t0)%60}s')

    return sps.coo_matrix((data, (row, col)), shape=(dim, dim)) / np.sqrt(2), S_Ni_val, Sz_Ni_val


def create_bonding_anti_bonding_basis_change_matrix(VS):
    """

    :param VS:
    :return:
    """
    t0 = time.time()
    dim = VS.dim
    data = []
    row = []
    col = []
    count_list = []     # 防止重复
    bonding_val = {}

    # 遍历所有态
    for i in range(dim):
        state = vs.get_state(VS.lookup_tbl[i])
        hole_num = len(state)

        # if_Ni0_dz2和if_Ni2_dz2判断态里面是否有(0, 0, 0)位置的dz2和(0, 0, 2)位置的dz2
        if_Ni0_dz2 = False
        if_Ni2_dz2 = False

        # 遍历整个态, 通过上下两层对称变换得到partner_state, 同时判断判断态里面是否有(0, 0, 0)位置的dz2和(0, 0, 2)位置的dz2
        partner_state = list(state)
        for hole_idx in range(hole_num):
            x, y, z, orb, s = state[hole_idx]
            if z == 0 and orb == 'd3z2r2':
                if_Ni0_dz2 = True
            if z == 2 and orb == 'd3z2r2':
                if_Ni2_dz2 = True
            partner_state[hole_idx] = (x, y, 2-z, orb, s)
        partner_state, ph = vs.make_state_canonical(partner_state)
        j = VS.get_index(partner_state)

        # 如果对称对角线位置设为sqrt(2)
        if j == i:
            data.append(np.sqrt(2))
            row.append(i)
            col.append(i)
        # 如果
        elif if_Ni0_dz2 and if_Ni2_dz2:
            if i not in count_list:
                # 记录对应的j, 防止在循环时重复
                count_list.append(j)
                # i为bonding
                data.append(1.)
                row.append(i)
                col.append(i)

                data.append(ph)
                row.append(j)
                col.append(i)
                bonding_val[i] = 1

                # j为anti_bonding
                data.append(1.)
                row.append(i)
                col.append(j)

                data.append(-ph)
                row.append(j)
                col.append(j)
                bonding_val[j] = -1
        else:
            data.append(np.sqrt(2))
            row.append(i)
            col.append(i)

    out = sps.coo_matrix((data, (row, col)), shape=(dim,dim)) / np.sqrt(2)
    t1 = time.time()
    print(f'bonding_anti_boding_basis_change time {(t1-t0)//60//60}h, {(t1-t0)//60%60}min, {(t1-t0)%60}s')

    return out, bonding_val


def coupling_representation(j1_list, j2_list, j1m1_list, j2m2_list, expand1_list, expand2_list):
    """
    This function is used to expand two states in the coupled representation into the uncoupled representation.
    :param j1_list:particle1's spin quantum numbers
    :param j2_list:
    :param j1m1_list:particle1's spin quantum numbers and its corresponding magnetic quantum number
    :param j2m2_list:
    :param expand1_list:
    :param expand2_list:
    :return:
    """
    # j1m1_list和j2m2_list中的(j1, m1)转化为sympy.Rational类型, 方便符号运算和查找
    j1m1_list = [(Rational(j1), Rational(m1)) for j1, m1 in j1m1_list]
    j2m2_list = [(Rational(j2), Rational(m2)) for j2, m2 in j2m2_list]

    # 1. 计算耦合自旋量子数j
    cou_j_list = []
    jm_list = []
    expand_list = []
    start_idx1 = 0  # 用于j1m1_list的起始索引
    for j1 in j1_list:
        start_idx2 = 0  # 用于j2m2_list的起始索引
        for j2 in j2_list:
            j_list = np.arange(abs(j1 - j2), j1 + j2 + 1)
            for j in j_list:
                cou_j_list.append(j)

                # 2. 计算耦合磁量子数m
                for m in np.arange(-j, j + 1):
                    expand = {}

                    # 3. 计算耦合磁量子数m1和m2
                    for m1 in np.arange(-j1, j1 + 1):
                        m2 = m - m1
                        if m2 < -j2 or m2 > j2:
                            continue

                        # 4. 计算CG系数并将|j1, m1> = expand1和|j2, m2> = expand2代入cg|j1, m1>|j2, m2> + ...中
                        # 计算结果存储在expand中
                        j1, m1, j2, m2 = Rational(j1), Rational(m1), Rational(j2), Rational(m2)
                        j, m = Rational(j), Rational(m)
                        cg = CG(j1, m1, j2, m2, j, m).doit()
                        # (j1, m1)有重复出现，需要加入索引范围start_idx1到start_idx1+2*j1+1，在2*j1+1(m1的个数)个里面找
                        idx1 = j1m1_list.index((j1, m1), start_idx1, start_idx1 + 2 * j1 + 1)
                        idx2 = j2m2_list.index((j2, m2), start_idx2, start_idx2 + 2 * j2 + 1)
                        for factor1, coef1 in expand1_list[idx1].items():
                            for factor2, coef2 in expand2_list[idx2].items():
                                factor = factor1 + factor2
                                if factor in expand:
                                    expand[factor] += cg * coef1 * coef2  # 如果项因子factor已经出现过，则累加
                                else:
                                    expand[factor] = cg * coef1 * coef2

                    # 将|j, m>存储在jm_list，将展开式右边存储在expand_list
                    jm_list.append((j, m))
                    expand_list.append(expand)

            start_idx2 += 2 * j2 + 1  # 更新j2m2_list的起始索引
        start_idx1 += 2 * j1 + 1  # 更新j1m1_list的起始索引

    return cou_j_list, jm_list, expand_list


def create_coupled_representation_matrix(VS):
    """
    Construct the transformation matrix to the coupled representation.
    :param VS:all state
    :return:sps.coc_matrix, 变换矩阵; S_val, 总自旋; Sz_val, 总自旋的z分量
    """
    t0 = time.time()
    # 单个空穴
    half = Rational(1, 2)
    j1_list = [half]
    j1m1_list = [(half, -half), (half, half)]
    expand1_list = [{(-half,): 1}, {(half,): 1}]

    dim = VS.dim
    row = list(range(dim*dim))
    col = list(range(dim*dim))
    data = [1.0] * (dim * dim)
    S_val = {}
    Sz_val = {}

    # 1. 读取coupled_uid文件中需要耦合变换的uid
    b_hole = len(lat.position) * pam.Norb
    coupled_istate = defaultdict(list)
    with open("coupled_uid", 'r') as file:
        for line in file:
            # double, single分别表示在同一个空穴上的数目是2, 1
            double_uid, single_uid = line.split(',')
            double_uid = int(double_uid.strip())
            single_uid = int(single_uid.strip())
            # 提取double_uid和single_uid每个进制上的hole_uid
            double_hole_uids = set()
            single_hole_uids = set()
            while double_uid:
                hole_uid = double_uid % b_hole
                double_hole_uids.add(hole_uid)
                double_uid //= b_hole
            while single_uid:
                hole_uid = single_uid % b_hole
                single_hole_uids.add(hole_uid)
                single_uid //= b_hole

            # 单占据的个数
            single_num = len(single_hole_uids)
            # 选定一个标准的位置轨道顺序
            canonical_pos_orb_uids = sorted(single_hole_uids)

            # 由标准的位置轨道顺序, 找到其他所有可能组合, 存储为coupled_group = {自旋朝向的分布ss: 态索引istate}
            coupled_group = {}
            for single_up_uids in combinations(canonical_pos_orb_uids, int(single_num/2)):
                up_uids = set(single_up_uids) | double_hole_uids
                dn_uids = single_hole_uids - set(single_up_uids) | double_hole_uids
                up_uids = list(up_uids)
                up_uids.sort()
                dn_uids = list(dn_uids)
                dn_uids.sort()
                # 将空穴uid按进制b_hole, 转为只用一个数字存储
                up_uid = 0
                dn_uid = 0
                for idx, hole_uid in enumerate(up_uids):
                    up_uid += hole_uid * (b_hole ** idx)
                for idx, hole_uid in enumerate(dn_uids):
                    dn_uid += hole_uid * (b_hole ** idx)
                # 根据VS中态的查询列表lookup_tbl, 找到对应的索引
                iup = bisect_left(VS.lookup_tbl, up_uid)
                idn = bisect_left(VS.lookup_tbl, dn_uid)
                istate = iup * dim + idn

                # 自旋朝向分布
                ss = []
                for pos_orb_uid in canonical_pos_orb_uids:
                    s = half if pos_orb_uid in up_uids else -half
                    ss.append(s)
                ss = tuple(ss)

                ph = 1
                hole_uids = up_uids + dn_uids
                uid_num = len(hole_uids)
                for i in range(1, uid_num):
                    behind_uid = hole_uids[i]
                    for front_uid in hole_uids[:i]:
                        if front_uid > behind_uid:
                            ph *= -1
                coupled_group[ss] = (istate, ph)
                data[istate] = 0.
            coupled_istate[single_num].append(coupled_group)

    for coupled_num, coupled_groups in coupled_istate.items():
        # 2.根据耦合的空穴数目, 生成耦合表象在非耦合表象下的展开式
        j_list = j1_list
        jm_list = j1m1_list
        expand_list = expand1_list
        for _ in range(coupled_num-1):
            j_list, jm_list, expand_list = coupling_representation(j_list, j1_list, jm_list, j1m1_list,
                                                                       expand_list, expand1_list)
        # 调整展开式的顺序, 按照m, j升序排列
        jm_idx = [i for i in range(len(jm_list)) if float(jm_list[i][1]) == 0]
        jm_list = [jm_list[i] for i in jm_idx]
        expand_list = [expand_list[i] for i in jm_idx]

        sorted_jm_idx = sorted(range(len(jm_list)), key=lambda i: (jm_list[i][1], jm_list[i][0]))
        jm_list = [jm_list[i] for i in sorted_jm_idx]
        expand_list = [expand_list[i] for i in sorted_jm_idx]

        # 3.设置耦合变换的矩阵元
        for coupled_group in coupled_groups:
            row_idxs = [istate for istate, _ in coupled_group.values()]
            row_idxs.sort()
            for jm_idx, expand in enumerate(expand_list):
                j, m = jm_list[jm_idx]
                row_idx = row_idxs[jm_idx]
                S_val[row_idx] = j
                Sz_val[row_idx] = m

                for factor, coef in expand.items():
                    coef = float(coef)
                    col_idx, ph = coupled_group[factor]

                    row.append(row_idx)
                    col.append(col_idx)
                    data.append(ph*coef)

    out = sps.coo_matrix((data, (row, col)), shape=(dim*dim, dim*dim))
    t1 = time.time()
    print(f'coupled representation time {(t1-t0)//60//60}h, {(t1-t0)//60%60}min, {(t1-t0)%60}s\n')

    return out, S_val, Sz_val
