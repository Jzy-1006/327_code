import time
import numpy as np
import scipy.sparse as sps
import parameters as pam
import lattice as lat
import variational_space as vs

directions_to_vecs = {'UR': (1, 1, 0), 'UL': (-1, 1, 0), 'DL': (-1, -1, 0), 'DR': (1, -1, 0),
                      'L': (-1, 0, 0), 'R': (1, 0, 0), 'U': (0, 1, 0), 'D': (0, -1, 0),
                      'T': (0, 0, 1), 'B': (0, 0, -1),
                      'L2': (-2, 0, 0), 'R2': (2, 0, 0), 'U2': (0, 2, 0), 'D2': (0, -2, 0), 'T2': (0, 0, 2), 'B2': (0, 0, -2),
                      'pzL': (-1, 0, 1), 'pzR': (1, 0, 1), 'pzU': (0, 1, 1), 'pzD': (0, -1, 1),
                      'mzL': (-1, 0, -1), 'mzR': (1, 0, -1), 'mzU': (0, 1, -1), 'mzD': (0, -1, -1)}


def set_tpd_tpp(tpd, tpp):
    """
    设置通过tpd, tpp跳跃的轨道和对应的方向, 以及对应的值
    :param tpd: p, d轨道的跳跃值
    :param tpp: p, p轨道的跳跃值
    :return: tpd_nn_hop_dir, tpd_nn_hop_fac, tpp_nn_hop_fac
    """
    if pam.Norb == 5 or pam.Norb == 8:
        tpd_nn_hop_dir = {'d3z2r2': ['L', 'R', 'U', 'D'],
                          'dx2y2': ['L', 'R', 'U', 'D']}

        tpd_nn_hop_fac = {('d3z2r2', 'L', 'px'): -tpd/np.sqrt(3),
                          ('d3z2r2', 'R', 'px'): tpd/np.sqrt(3),
                          ('d3z2r2', 'U', 'py'): tpd/np.sqrt(3),
                          ('d3z2r2', 'D', 'py'): -tpd/np.sqrt(3),
                          ('dx2y2', 'L', 'px'): tpd,
                          ('dx2y2', 'R', 'px'): -tpd,
                          ('dx2y2', 'U', 'py'): tpd,
                          ('dx2y2', 'D', 'py'): -tpd}

        tpp_nn_hop_dir = ['UR', 'UL', 'DL', 'DR']
        # 注意字典顺序
        tpp_nn_hop_fac = {('UR', 'px', 'py'): -tpp,
                          ('UL', 'px', 'py'): tpp,
                          ('DL', 'px', 'py'): -tpp,
                          ('DR', 'px', 'py'): tpp}
    else:
        print('the Norb is error')

    return tpd_nn_hop_dir, tpd_nn_hop_fac, tpp_nn_hop_dir, tpp_nn_hop_fac


def set_tdo_tpo(tdo, tpo):
    """
    设置通过tdo, tpo跳跃的轨道和对应的方向, 以及跳跃值
    :param tdo: d, pz轨道的跳跃值
    :param tpo: p, pz轨道的跳跃值
    :return:
    """
    if pam.Norb == 5 or pam.Norb == 8:
        # 设置tdo的跳跃轨道和方向, 跳跃值
        tdo_nn_hop_dir = {'apz': ['T', 'B']}
        tdo_nn_hop_fac = {('apz', 'B', 'd3z2r2'): -tdo,
                          ('apz', 'T', 'd3z2r2'): tdo}

        # 设置tpo的跳跃轨道和方向, 和跳跃值
        tpo_nn_hop_dir = {'apz': ['pzL', 'pzR', 'mzL', 'mzR', 'pzU', 'pzD', 'mzU', 'mzD']}
        tpo_nn_hop_fac = {('apz', 'mzR', 'px'): tpo,
                          ('apz', 'mzL', 'px'): -tpo,
                          ('apz', 'pzR', 'px'): -tpo,
                          ('apz', 'pzL', 'px'): tpo,
                          ('apz', 'mzD', 'py'): -tpo,
                          ('apz', 'mzU', 'py'): tpo,
                          ('apz', 'pzD', 'py'): tpo,
                          ('apz', 'pzU', 'py'): -tpo}
    else:
        tdo_nn_hop_dir = None
        tdo_nn_hop_fac = None

        tpo_nn_hop_dir = None
        tpo_nn_hop_fac = None

    return tdo_nn_hop_dir, tdo_nn_hop_fac, tpo_nn_hop_dir, tpo_nn_hop_fac


def set_tz(if_tz_exist, tz_a1a1, tz_b1b1):
    """

    :param if_tz_exist:
    :param tz_a1a1:
    :param tz_b1b1:
    :return:
    """
    if pam.Norb ==5:
        if if_tz_exist == 0:
            tz_fac = {('px', 'px'): tz_b1b1,
                      ('py', 'py'): tz_b1b1,
                      ('d3z2r2', 'd3z2r2'): tz_a1a1,
                      ('dx2y2', 'dx2y2'): tz_b1b1}
        if if_tz_exist == 1:
            tz_fac = {('d3z2r2', 'd3z2r2'): tz_a1a1,
                      ('dx2y2', 'dx2y2'): tz_b1b1}
        if if_tz_exist == 2:
            tz_fac = {('d3z2r2', 'd3z2r2'): tz_a1a1}
        else:
            tz_fac = None
    else:
        tz_fac = None

    return tz_fac


def get_interaction_mat(A, sym):
    """
    根据对称性, 设置d8相互作用的矩阵元
    :param A:相互作用的一个参数
    :param sym: 对称性
    :return: state_order, interaction_mat, Stot, Sz_set
    """
    B = pam.B
    C = pam.C
    if sym == '1A1':
        Stot = 0
        Sz_set = [0]
        state_order = {('d3z2r2', 'd3z2r2'): 0,
                       ('dx2y2', 'dx2y2'): 1}
        interaction_mat = [[A+4.*B+3.*C, 4.*B+C],
                           [4.*B+C, A+4.*B+3.*C]]

    if sym == '1B1':
        Stot = 0
        Sz_set = [0]
        state_order = {('d3z2r2', 'dx2y2'): 0}
        interaction_mat = [[A+2.*C]]

    if sym == '3B1':
        Stot = 1
        Sz_set = [-1, 0, 1]
        state_order = {('d3z2r2', 'dx2y2'): 0}
        interaction_mat = [[A-8.*B]]

    else:
        Stot = None
        Sz_set = None
        state_order = None
        interaction_mat = None

    return Stot, Sz_set, state_order, interaction_mat


def create_tpd_nn_matrix(VS, tpd_nn_hop_dir, tpd_nn_hop_fac):
    """
    创建Tpd哈密顿矩阵, 只用遍历d到p轨道的跳跃,
    而p轨道到d轨道的跳跃只需将行列交换, 值不变
    :param VS:类, 含有lookup_tbl(存储要计算的态), 函数get_state_uid, get_state, get_index
    :param tpd_nn_hop_dir: 跳跃轨道和方向
    :param tpd_nn_hop_fac: 跳跃的值
    :return:
    """
    t0 = time.time()
    dim = VS.dim
    # tpd_orbs = [orbital 1, ...], tpd_keys = (orbital 1, direction, orbital 2)
    tpd_orbs = tpd_nn_hop_dir.keys()
    tpd_keys = tpd_nn_hop_fac.keys()
    data = []
    row = []
    col = []

    # 遍历整个态空间
    for row_idx in range(dim):
        state = VS.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb, s = hole
            # 根据轨道决定是否跳跃, 求出跳跃后的坐标
            if orb in tpd_orbs:
                for direction in tpd_nn_hop_dir[orb]:
                    vx, vy, vz = directions_to_vecs[direction]
                    hop_x, hop_y, hop_z = x + vx, y + vy, z + vz
                    # 由跳跃后的坐标得出跳跃后的轨道, 自旋不变
                    hop_orbs = lat.get_unit_cell_rep(hop_x, hop_y, hop_z)
                    if hop_orbs == ['NotOnSublattice']:
                        continue
                    for hop_orb in hop_orbs:
                        orb12 = (orb, direction, hop_orb)
                        if orb12 in tpd_keys:

                            # 跳跃后的空穴
                            hop_hole = (hop_x, hop_y, hop_z, hop_orb, s)
                            if hop_hole not in state:       # 检验是否满足Pauli不相容原理
                                # 将其中的一个空穴换成是跳跃后的空穴
                                hop_state = list(state)
                                hop_state[hole_idx] = hop_hole
                                hop_state, ph = vs.make_state_canonical(hop_state)
                                col_idx = VS.get_index(hop_state)
                                if col_idx is not None:
                                    value = tpd_nn_hop_fac[orb12] * ph
                                    data.extend((value, value))
                                    row.extend((row_idx, col_idx))
                                    col.extend((col_idx, row_idx))

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print('Tpd cost time', t1-t0)

    return out


def create_tpp_nn_matrix(VS, tpp_nn_hop_dir, tpp_nn_hop_fac):
    """
    设置Tpp哈密顿矩阵
    :param VS: 态空间
    :param tpp_nn_hop_dir: p轨道之间的跳跃方向
    :param tpp_nn_hop_fac: p轨道之间的跳跃值
    :return:out(coo_matrix), Tpp哈密顿矩阵
    """
    t0 = time.time()
    dim = VS.dim
    # tpp_keys = (direction, orb1, orb2)...
    tpp_keys = tpp_nn_hop_fac.keys()
    data = []
    row = []
    col = []

    # 遍历整个态空间
    for row_idx in range(dim):
        state = VS.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb, s = hole
            # 根据轨道决定是否跳跃, 求出跳跃后的坐标
            if orb in pam.O_orbs:
                for direction in tpp_nn_hop_dir:
                    vx, vy, vz = directions_to_vecs[direction]
                    hop_x, hop_y, hop_z = x+vx, y+vy, z+vz
                    # 由跳跃后的坐标得出跳跃后的轨道, 自旋不变
                    hop_orbs = lat.get_unit_cell_rep(hop_x, hop_y, hop_z)
                    if hop_orbs == ['NotOnSublattice']:
                        continue
                    if hop_orbs != pam.O1_orbs and hop_orbs != pam.O2_orbs:
                        continue
                    for hop_orb in hop_orbs:
                        # 注意字典顺序
                        orb12 = sorted([orb, direction, hop_orb])
                        orb12 = tuple(orb12)
                        if orb12 in tpp_keys:

                            # 跳跃后的空穴
                            hop_hole = (hop_x, hop_y, hop_z, hop_orb, s)
                            if hop_hole not in state:       # 检验是否满足Pauli不相容原理
                                # 将其中的一个空穴换成是跳跃后的空穴
                                hop_state = list(state)
                                hop_state[hole_idx] = hop_hole
                                hop_state, ph = vs.make_state_canonical(hop_state)
                                col_idx = VS.get_index(hop_state)
                                if col_idx is not None:
                                    value = tpp_nn_hop_fac[orb12] * ph
                                    data.append(value)
                                    row.append(row_idx)
                                    col.append(col_idx)

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print('Tpp cost time', t1 - t0)

    return out


def create_tdo_nn_matrix(VS, tdo_nn_hop_dir, tdo_nn_hop_fac):
    """
    设置Tdo哈密顿矩阵
    :param VS: 态空间
    :param tdo_nn_hop_dir: apz的跳跃方向
    :param tdo_nn_hop_fac: apz往不同方向对应的跳跃值
    :return: out(coo_matrix), Tdo哈密顿矩阵
    """
    t0 = time.time()
    dim = VS.dim
    tdo_orbs = tdo_nn_hop_dir.keys()
    tdo_keys = tdo_nn_hop_fac.keys()
    data = []
    row = []
    col = []

    # 遍历整个态空间
    for row_idx in range(dim):
        state = VS.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb, s = hole
            # 根据轨道决定是否跳跃, 求出跳跃后的坐标
            if orb in tdo_orbs:
                for direction in tdo_nn_hop_dir[orb]:
                    vx, vy, vz = directions_to_vecs[direction]
                    hop_x, hop_y, hop_z = x + vx, y + vy, z + vz
                    # 由跳跃后的坐标得出跳跃后的轨道, 自旋不变
                    hop_orbs = lat.get_unit_cell_rep(hop_x, hop_y, hop_z)
                    if hop_orbs == ['NotOnSublattice']:
                        continue
                    for hop_orb in hop_orbs:
                        orb12 = (orb, direction, hop_orb)
                        if orb12 in tdo_keys:

                            # 跳跃后的空穴
                            hop_hole = (hop_x, hop_y, hop_z, hop_orb, s)
                            if hop_hole not in state:  # 检验是否满足Pauli不相容原理
                                # 将其中的一个空穴换成是跳跃后的空穴
                                hop_state = list(state)
                                hop_state[hole_idx] = hop_hole
                                hop_state, ph = vs.make_state_canonical(hop_state)
                                col_idx = VS.get_index(hop_state)
                                if col_idx is not None:
                                    value = tdo_nn_hop_fac[orb12] * ph
                                    data.extend((value, value))
                                    row.extend((row_idx, col_idx))
                                    col.extend((col_idx, row_idx))

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print('Tdo cost time', t1 - t0)

    return out


def create_tpo_nn_matrix(VS, tpo_nn_hop_dir, tpo_nn_hop_fac):
    """
    设置Tpo哈密顿矩阵
    :param VS: 态空间
    :param tpo_nn_hop_dir: apz的跳跃方向
    :param tpo_nn_hop_fac: apz往不同方向对应的跳跃值
    :return: out(coo_matrix), Tpo哈密顿矩阵
    """
    t0 = time.time()
    dim = VS.dim
    tpo_orbs = tpo_nn_hop_dir.keys()
    tpo_keys = tpo_nn_hop_fac.keys()
    data = []
    row = []
    col = []

    # 遍历整个态空间
    for row_idx in range(dim):
        state = VS.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb, s = hole
            # 根据轨道决定是否跳跃, 求出跳跃后的坐标
            if orb in tpo_orbs:
                for direction in tpo_nn_hop_dir[orb]:
                    vx, vy, vz = directions_to_vecs[direction]
                    hop_x, hop_y, hop_z = x + vx, y + vy, z + vz
                    # 由跳跃后的坐标得出跳跃后的轨道, 自旋不变
                    hop_orbs = lat.get_unit_cell_rep(hop_x, hop_y, hop_z)
                    if hop_orbs == ['NotOnSublattice']:
                        continue
                    for hop_orb in hop_orbs:
                        orb12 = (orb, direction, hop_orb)
                        if orb12 in tpo_keys:

                            # 跳跃后的空穴
                            hop_hole = (hop_x, hop_y, hop_z, hop_orb, s)
                            if hop_hole not in state:  # 检验是否满足Pauli不相容原理
                                # 将其中的一个空穴换成是跳跃后的空穴
                                hop_state = list(state)
                                hop_state[hole_idx] = hop_hole
                                hop_state, ph = vs.make_state_canonical(hop_state)
                                col_idx = VS.get_index(hop_state)
                                if col_idx is not None:
                                    value = tpo_nn_hop_fac[orb12] * ph
                                    data.extend((value, value))
                                    row.extend((row_idx, col_idx))
                                    col.extend((col_idx, row_idx))

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print('Tpo cost time', t1 - t0)

    return out


def create_Esite_matrix(VS, A, ed, ep, eo):
    """
    创建Onsite_energy哈密顿矩阵, 并计算dn, 能量设为A + abs(n - 8) * A / 2
    :param VS:
    :param A:
    :param ed
    :param ep:
    :param eo:
    :return:
    """
    t0 = time.time()
    dim = VS.dim
    data = []
    row = []
    col = []
    for row_idx in range(dim):
        state = VS.get_state(VS.lookup_tbl[row_idx])
        diag_el = 0.
        Ni_num = {}
        for x, y, z, orb, _ in state:
            # 计算d, p, apz轨道上的在位能
            if orb in pam.Ni_orbs:
                diag_el += ed[orb]
            elif orb in pam.O_orbs:
                diag_el += ep
            elif orb in pam.Oap_orbs:
                diag_el += eo

            # 统计在相同Ni上的个数
            if orb in pam.Ni_orbs:
                if (x, y, z) in Ni_num:
                    Ni_num[(x, y, z)] += 1
                else:
                    Ni_num[(x, y, z)] = 1

        # dn的能量, 比d8高A/2 * abs(n - 8)
        for num in Ni_num.values():
            diag_el += A / 2 * abs(num - 2)

        data.append(diag_el)
        row.append(row_idx)
        col.append(row_idx)

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print('Esite cost time', t1 - t0)

    return out


def create_tz_matrix(VS, tz_fac):
    """
    设置d轨道之间的杂化
    :param VS:
    :param tz_fac:
    :return: out(coo_matrix)
    """
    t0 = time.time()
    dim = VS.dim
    data = []
    row = []
    col = []
    # tz_keys = (orbital1, orbital2)
    tz_keys = tz_fac.keys()
    # 只选择被夹的那一层(含有Ni)
    z_list = range(2, 2*pam.layer_num-1, 2)

    # 遍历整个态空间
    for row_idx in range(dim):
        state = VS.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb, s = hole
            if z in z_list and (orb, orb) in tz_keys:     # 夹层并且满足选定的轨道
                hop_z = z - 2   # 往下一层
                hop_hole = (x, y, hop_z, orb, s)
                if hop_hole in state:       # 是否符合Pauli不相容原理
                    # 将其中的一个空穴换成是跳跃后的空穴
                    hop_state = list(state)
                    hop_state[hole_idx] = hop_hole
                    hop_state, ph = vs.make_state_canonical(hop_state)
                    col_idx = VS.get_index(hop_state)
                    if col_idx is not None:
                        value = tz_fac[(orb, orb)] * ph
                        data.extend((value, value))
                        row.extend((row_idx, col_idx))
                        col.extend((col_idx, row_idx))

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print('Tz cost time', t1 - t0)

    return out


def get_double_occ_list(VS):
    """
    找出态中有两个空穴是在同一位置
    :param VS: 态空间
    :return: d_state_idx, d_hole_idx, p_idx_pair, apz_idx_pair
    """
    d_state_idx = {}
    d_hole_idx = {}
    p_idx_pair = []
    apz_idx_pair = []

    dim = VS.dim
    # 遍历整个态空间
    for i in range(dim):
        state = VS.get_state(VS.lookup_tbl[i])
        # 统计在相同Ni上空穴的索引和在相同O上空穴数目
        Ni_idx = {}
        O_num = {}      # 相同层间O上空穴数目
        Oap_num = {}        #相同层内O上空穴数目
        hole_num = len(state)
        for hole_idx in range(hole_num):
            x, y, z, orb, s = state(hole_idx)
            if orb in pam.Ni_orbs:
                if (x, y, z) in Ni_idx:
                    Ni_idx[(x, y, z)] += [hole_idx]
                else:
                    Ni_idx[(x, y, z)] = [hole_idx]
            elif orb in pam.O_orbs:
                if (x, y, z) in O_num:
                    O_num[(x, y, z)] += 1
                else:
                    O_num[(x, y, z)] = 1
            else:
                if (x, y, z) in Oap_num:
                    Oap_num[(x, y, z)] += 1
                else:
                    Oap_num[(x, y, z)] = 1

        # 记录在Ni上的空穴数目是2的态索引和空穴索引
        for position, idx_list in Ni_idx.items():
            Ni_num = len(idx_list)
            if Ni_num == 2:
                if position in d_state_idx:
                    d_state_idx[position] += [i]
                    d_hole_idx[position] += [idx_list]
                else:
                    d_state_idx[position] = [i]
                    d_hole_idx[position] = [idx_list]

        # 记录在层内O上空穴数目大于1的态索引和空穴对
        p_pair = 0
        for num in O_num.values():
            if num > 1:
                p_pair += num * (num - 1) / 2
        if p_pair > 0:
            p_idx_pair.append((i, p_pair))

        # 记录在层间O上空穴数目大于1的
        apz_pair = 0
        for num in Oap_num.values():
            if num > 1:
                apz_pair += num * (num - 1) / 2
        if apz_pair > 0:
            apz_idx_pair.append((i, apz_pair))

    return d_state_idx, d_hole_idx, p_idx_pair, apz_idx_pair
