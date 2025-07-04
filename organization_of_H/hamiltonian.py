import time
from scipy.special import comb
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

    elif sym == '1B1':
        Stot = 0
        Sz_set = [0]
        state_order = {('d3z2r2', 'dx2y2'): 0}
        interaction_mat = [[A+2.*C]]

    elif sym == '3B1':
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
        state = vs.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb = hole
            layer_idx = z // 2
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
                            hop_hole = (hop_x, hop_y, hop_z, hop_orb)
                            if hop_hole not in state:       # 检验是否满足Pauli不相容原理
                                # 将其中的一个空穴换成是跳跃后的空穴
                                hop_state = list(state)
                                hop_state[hole_idx] = hop_hole
                                hop_state, ph = vs.make_state_canonical(hop_state)
                                col_idx = VS.get_index(hop_state)
                                if col_idx is not None:
                                    value = tpd_nn_hop_fac[orb12][layer_idx] * ph
                                    data.extend((value, value))
                                    row.extend((row_idx, col_idx))
                                    col.extend((col_idx, row_idx))

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print(f"Tpd time {(t1-t0)//60//60}h {(t1-t0)//60%60}min {(t1-t0) % 60}s")

    return out.tocsr()


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
        state = vs.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb = hole
            layer_idx = z // 2
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
                            hop_hole = (hop_x, hop_y, hop_z, hop_orb)
                            if hop_hole not in state:       # 检验是否满足Pauli不相容原理
                                # 将其中的一个空穴换成是跳跃后的空穴
                                hop_state = list(state)
                                hop_state[hole_idx] = hop_hole
                                hop_state, ph = vs.make_state_canonical(hop_state)
                                col_idx = VS.get_index(hop_state)
                                if col_idx is not None:
                                    value = tpp_nn_hop_fac[orb12][layer_idx] * ph
                                    data.append(value)
                                    row.append(row_idx)
                                    col.append(col_idx)

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print(f'Tpp time {(t1-t0)//60//60}h {(t1-t0)//60%60}min {(t1-t0) % 60}s')

    return out.tocsr()


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
        state = vs.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb = hole
            # 根据轨道决定是否跳跃, 求出跳跃后的坐标
            if orb in tdo_orbs:
                for direction in tdo_nn_hop_dir[orb]:
                    vx, vy, vz = directions_to_vecs[direction]
                    hop_x, hop_y, hop_z = x + vx, y + vy, z + vz
                    # 根据跳跃后在第几层, 确定tdo值
                    layer_idx = hop_z // 2
                    # 由跳跃后的坐标得出跳跃后的轨道, 自旋不变
                    hop_orbs = lat.get_unit_cell_rep(hop_x, hop_y, hop_z)
                    if hop_orbs == ['NotOnSublattice']:
                        continue
                    for hop_orb in hop_orbs:
                        orb12 = (orb, direction, hop_orb)
                        if orb12 in tdo_keys:

                            # 跳跃后的空穴
                            hop_hole = (hop_x, hop_y, hop_z, hop_orb)
                            if hop_hole not in state:  # 检验是否满足Pauli不相容原理
                                # 将其中的一个空穴换成是跳跃后的空穴
                                hop_state = list(state)
                                hop_state[hole_idx] = hop_hole
                                hop_state, ph = vs.make_state_canonical(hop_state)
                                col_idx = VS.get_index(hop_state)
                                if col_idx is not None:
                                    value = tdo_nn_hop_fac[orb12][layer_idx] * ph
                                    data.extend((value, value))
                                    row.extend((row_idx, col_idx))
                                    col.extend((col_idx, row_idx))

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print(f'Tdo time {(t1-t0)//60//60}h {(t1-t0)//60%60}min {(t1-t0) % 60}s')

    return out.tocsr()


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
        state = vs.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb = hole
            # 根据轨道决定是否跳跃, 求出跳跃后的坐标
            if orb in tpo_orbs:
                for direction in tpo_nn_hop_dir[orb]:
                    vx, vy, vz = directions_to_vecs[direction]
                    hop_x, hop_y, hop_z = x + vx, y + vy, z + vz
                    # 根据跳跃后在第几层, 确定tdo值
                    layer_idx = hop_z // 2
                    # 由跳跃后的坐标得出跳跃后的轨道, 自旋不变
                    hop_orbs = lat.get_unit_cell_rep(hop_x, hop_y, hop_z)
                    if hop_orbs == ['NotOnSublattice']:
                        continue
                    for hop_orb in hop_orbs:
                        orb12 = (orb, direction, hop_orb)
                        if orb12 in tpo_keys:

                            # 跳跃后的空穴
                            hop_hole = (hop_x, hop_y, hop_z, hop_orb)
                            if hop_hole not in state:  # 检验是否满足Pauli不相容原理
                                # 将其中的一个空穴换成是跳跃后的空穴
                                hop_state = list(state)
                                hop_state[hole_idx] = hop_hole
                                hop_state, ph = vs.make_state_canonical(hop_state)
                                col_idx = VS.get_index(hop_state)
                                if col_idx is not None:
                                    value = tpo_nn_hop_fac[orb12][layer_idx] * ph
                                    data.extend((value, value))
                                    row.extend((row_idx, col_idx))
                                    col.extend((col_idx, row_idx))

    out = sps.coo_matrix((data, (row, col)), shape=(dim, dim))
    t1 = time.time()
    print(f'Tpo time {(t1-t0)//60//60}h {(t1-t0)//60%60}min {(t1-t0) % 60}s')

    return out.tocsr()


def create_Esite_matrix(up_VS, dn_VS, A, ed, ep, eo):
    """
    创建Onsite_energy哈密顿矩阵, 并计算d_num, 能量设为A + abs(n - 8) * A / 2
    :param up_VS:
    :param dn_VS:
    :param A:
    :param ed:
    :param ep:
    :param eo:
    :return:
    """
    def get_onsite_and_Ni_num(VS):
        """
        得到onsite_energy和每个态在Ni1, Ni2, Ni3, ...上的空穴数目
        :param VS:
        :return:
        """
        dim = VS.dim
        onsite = []
        state_Ni_num = []

        # diag_Ni_num表示的是在Ni上的数目
        for row_idx in range(dim):
            state = vs.get_state(VS.lookup_tbl[row_idx])
            Ni_num = {position: 0 for position in lat.Ni_position}
            diag_el = 0
            for x, y, z, orb in state:
                # 计算d, p, apz轨道上的在位能
                if orb in pam.Ni_orbs:
                    layer_idx = z // 2
                    diag_el += ed[layer_idx][orb]
                elif orb in pam.O_orbs:
                    layer_idx = z // 2
                    diag_el += ep[layer_idx]
                elif orb in pam.Oap_orbs:
                    layer_idx = (z - 1) // 2
                    diag_el += eo[layer_idx]

                # 统计在相同Ni上的个数
                if orb in pam.Ni_orbs:
                    Ni_num[(x, y, z)] += 1

            onsite.append(diag_el)

            # dn(n != 8)的能量, 比d8高A/2 * abs(n - 8)
            for position in lat.Ni_position:
                state_Ni_num.append(Ni_num[position])

        onsite = np.array(onsite)
        state_Ni_num = np.array(state_Ni_num)
        state_Ni_num = state_Ni_num.reshape(-1, len(lat.Ni_position))

        return onsite, state_Ni_num

    t0 = time.time()
    # 创建对角矩阵
    if up_VS is None and dn_VS is None:
        print("please set up_VS or dn_VS")
    elif up_VS is not None and dn_VS is None:
        up_onsite, up_state_Ni_num = get_onsite_and_Ni_num(up_VS)
        d_num_energy = (up_state_Ni_num !=2 ) * A + abs(up_state_Ni_num - 2) * A / 2
        d_num_energy = d_num_energy.sum(axis=1)
        out = sps.diags(up_onsite + d_num_energy, format='csr')
    elif up_VS is None and dn_VS is not None:
        dn_onsite, dn_state_Ni_num = get_onsite_and_Ni_num(dn_VS)
        d_num_energy = (dn_state_Ni_num != 2) * A + abs(dn_state_Ni_num - 2) * A / 2
        d_num_energy = d_num_energy.sum(axis=1)
        out = sps.diags(dn_onsite + d_num_energy, format='csr')
    else:
        up_I = np.ones(up_VS.dim)
        up_onsite, up_state_Ni_num = get_onsite_and_Ni_num(up_VS)
        dn_I = np.ones(dn_VS.dim)
        dn_onste, dn_state_Ni_num =get_onsite_and_Ni_num(dn_VS)
        onsite = np.kron(dn_I, up_onsite) + np.kron(dn_onste, up_I)

        up_I = np.ones((up_VS.dim, 1))
        dn_I = np.ones((dn_VS.dim, 1))
        state_Ni_num = np.kron(dn_I, up_state_Ni_num) + np.kron(dn_state_Ni_num, up_I)
        d_num_energy = (state_Ni_num != 2) * A + abs(state_Ni_num - 2) * A / 2
        d_num_energy = d_num_energy.sum(axis=1)

        out = sps.diags(onsite + d_num_energy, format='csr')

    t1 = time.time()
    print(f'Esite time {(t1-t0)//60//60}h {(t1-t0)//60%60}min {(t1-t0) % 60}s\n')

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
    z_list = list(z_list)

    # 遍历整个态空间
    for row_idx in range(dim):
        state = vs.get_state(VS.lookup_tbl[row_idx])
        hole_num = len(state)

        # 其中一个空穴跳跃, 其他空穴不动
        for hole_idx in range(hole_num):
            hole = state[hole_idx]
            x, y, z, orb = hole
            if z in z_list and (orb, orb) in tz_keys:     # 夹层并且满足选定的轨道
                hop_z = z - 2   # 往下一层
                hop_hole = (x, y, hop_z, orb)
                if hop_hole not in state:       # 是否符合Pauli不相容原理
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
    print(f'Tz time {(t1-t0)//60//60}h {(t1-t0)//60%60}min {(t1-t0) % 60}s')

    return out.tocsr()


def get_double_occ_list(VS):
    """
    找出态中有两个空穴是在同一位置
    :param VS: 态空间
    :return: d_idx, p_idx, apz_idx
    """
    t0 = time.time()
    dim = VS.dim

    Ni_position = lat.Ni_position
    Ni_num = len(Ni_position)
    d_idx = {(i, if_dz2, if_dx2): [] for i in range(Ni_num)
             for if_dz2, if_dx2 in [(0, 0), (1, 0), (0, 1), (1, 1)]}
    # 找到自选向上（向下）态中Ni原子的dz2与dx2所有被占据的可能并保存为字典
    #i代表第几层的Ni，比如i=0代表Ni的坐标为(0, 0, 0), i=1为(0, 0, 2)为什么不是（0， 0， 1）因为这是O的pz轨道坐标
    for i, position in enumerate(Ni_position):
        for j in range(dim):
            state = vs.get_state(VS.lookup_tbl[j])
            if_dx2 = 0
            if_dz2 = 0
            for x, y, z, orb in state:
                if (x, y, z) == position:
                    if orb == 'd3z2r2':
                        if_dz2 = 1
                    elif orb == 'dx2y2':
                        if_dx2 = 1
            d_idx[(i, if_dz2, if_dx2)].append(j)

    p_idx = {}
    O_position = lat.O_position
    #找到不同px，py位置被占据的索引并保存为字典
    for i, position in enumerate(O_position):
        point = 0
        for j in range(dim):
            state = vs.get_state(VS.lookup_tbl[j])
            for x, y, z, _ in state:
                if (x, y, z) == position:
                    p_idx[i, point] = j
                    point += 1
                    break

    apz_idx = {}
    Oap_position = lat.Oap_position
    # 找到不同pz位置被占据的索引并保存为字典
    for i, position in enumerate(Oap_position):
        point = 0
        for j in range(dim):
            state = vs.get_state(VS.lookup_tbl[j])
            for x, y, z, _ in state:
                if (x, y, z) == position:
                    apz_idx[i, point] = j
                    point += 1
                    break


    t1 = time.time()
    print(f'double_occ time {(t1-t0)//60//60}h {(t1-t0)//60%60}min {(t1-t0) % 60}s')

    return d_idx, p_idx, apz_idx


def create_interaction_matrix_d8(up_VS, dn_VS, up_di_idx, dn_di_idx, A):
    """
    设置d8相互作用矩阵
    :param up_VS:
    :param dn_VS:
    :param up_di_idx:
    :param dn_di_idx:
    :param A:
    :return: out
    """
    t0 = time.time()
    B = pam.B
    C = pam.C
    data = []
    row = []
    col = []

    S_val = {}
    Sz_val = {}

    if up_VS is None and dn_VS is None:
        print("please set up_VS or dn_Vs")
        out = None
    elif up_VS is not None and dn_VS is None:
        for i in up_di_idx[(1, 1)]:
            dim = up_VS.dim
            data.append(A - 8 * B)
            row.append(i)
            col.append(i)

            S_val[i] = 1
            Sz_val[i] = 1

            out = sps.csr_matrix((data, (row, col)), shape=(dim, dim))
            out.tocsr()
    elif up_VS is None and dn_VS is not None:
        for i in dn_di_idx[(1, 1)]:
            dim = dn_VS.dim
            data.append(A - 8 * B)
            row.append(i)
            col.append(i)

            S_val[i] = 1
            Sz_val[i] = -1

            out = sps.csr_matrix((data, (row, col)), shape=(dim, dim))
    else:
        up_dim = up_VS.dim
        dn_dim = dn_VS.dim
        # d8相互作用
        # (1).1A1, S = 0, Sz = [0]
        # (dz2, up, dz2, dn)和(dz2, up, dz2, dn)
        for i in up_di_idx[(1, 0)]:
            for j in dn_di_idx[(1, 0)]:
                r = j * up_dim + i
                data.append(A + 4 * B + 3 * C)
                row.append(r)
                col.append(r)

                S_val[r] = 0
                Sz_val[r] = 0
        # (dx2, up, dx2, dn)和(dx2, up, dx2, dn)
        for i in up_di_idx[(0, 1)]:
            for j in dn_di_idx[(0, 1)]:
                r = j * up_dim + i
                data.append(A + 4 * B + 3 * C)
                row.append(r)
                col.append(r)

                S_val[r] = 0
                Sz_val[r] = 0
        # (dz2, up, dz2, dn)和(dx2, up, dx2, dn)
        for i1_idx, i in enumerate(up_di_idx[(1, 0)]):
            i1 = up_di_idx[(0, 1)][i1_idx]  # 保证除dz2, dx2部分空穴不同之外, 其余相同
            for j1_idx, j in enumerate(dn_di_idx[(1, 0)]):
                j1 = dn_di_idx[(0, 1)][j1_idx]
                r1 = j * up_dim + i
                r2 = j1 * up_dim + i1
                data.extend([4 * B + C, 4 * B + C])
                row.extend([r1, r2])
                col.extend([r2, r1])

                S_val[r1] = 0
                Sz_val[r1] = 0
                S_val[r2] = 0
                Sz_val[r2] = 0

        # (2).1B1, S=0, Sz=[0]
        # (dz2, up, dx2, dn)
        for i in up_di_idx[(1, 0)]:
            for j in dn_di_idx[(0, 1)]:
                r = j * up_dim + i
                data.append(A+2*C)
                row.append(r)
                col.append(r)

                S_val[r] = 0
                Sz_val[r] = 0

        # (3).3B1, S=1, Sz=[-1, 0, 1]
        # (dx2, up, dz2, dn), S=1, Sz=0
        for i in up_di_idx[(0, 1)]:
            for j in dn_di_idx[(1, 0)]:
                r = j * up_dim + i
                data.append(A-8*B)
                row.append(r)
                col.append(r)

                S_val[r] = 1
                Sz_val[r] = 0

        # (dz2, up, dx2, up), S=1, Sz=1
        for i in up_di_idx[(1, 1)]:
            for j in dn_di_idx[(0, 0)]:
                r = j * up_dim + i
                data.append(A - 8 * B)
                row.append(r)
                col.append(r)

                S_val[r] = 1
                Sz_val[r] = 1
        # (dz2, dn, dx2, dn), S=1, Sz=-1
        for i in up_di_idx[(0, 0)]:
            for j in dn_di_idx[(1, 1)]:
                r = j * up_dim + i
                data.append(A - 8 * B)
                row.append(r)
                col.append(r)

                S_val[r] = 1
                Sz_val[r] = -1

        out = sps.csr_matrix((data, (row, col)), shape=(dn_dim*up_dim, dn_dim*up_dim))
    t1 = time.time()
    print(f'create_interaction_matrix_d8 time {(t1-t0)//60//60}h {(t1-t0)//60%60}min {(t1-t0) % 60}s')

    return out, S_val, Sz_val


def create_interaction_matrix_po(up_VS, dn_VS, up_num, dn_num, up_p_idx, dn_p_idx, up_apz_idx, dn_apz_idx, Upp, Uoo):
    """
    设置p, pz轨道的相互作用
    :param up_VS:
    :param dn_VS:
    :param up_num:
    :param dn_num:
    :param up_p_idx: [(p_idx, point): state_idx, ...]
    :param dn_p_idx:
    :param up_apz_idx: [(p_idx, point): state_idx, ...]
    :param dn_apz_idx:
    :param Upp: p,p轨道的相互作用
    :param Uoo: pz, pz轨道的相互作用
    :return:
    """
    t0 = time.time()
    up_dim = up_VS.dim
    dn_dim = dn_VS.dim
    p_num = len(lat.O_position)
    apz_num = len(lat.Oap_position)
    orb_num = 2 * len(lat.Ni_position) + p_num + apz_num
    data = []
    diag_i = []

    up_kl_num = comb(orb_num - 1, up_num - 1, exact=True)
    dn_kl_num = comb(orb_num - 1, dn_num - 1, exact=True)
    # p, p轨道相互作用矩阵
    if Upp != 0:
        for i in range(p_num):
            for k in range(up_kl_num):
                for l in range(dn_kl_num):
                    r = dn_p_idx[i, l] * up_dim + up_p_idx[i, k]
                    data.append(Upp)
                    diag_i.append(r)

    # pz, pz轨道相互作用矩阵
    if Uoo != 0:
        for i in range(apz_num):
            for k in range(up_kl_num):
                for l in range(dn_kl_num):
                    r = dn_apz_idx[i, l] * up_dim + up_apz_idx[i, k]
                    data.append(Uoo)
                    diag_i.append(r)

    out = sps.coo_matrix((data, (diag_i, diag_i)), shape=(dn_dim*up_dim, dn_dim*up_dim))
    t1 = time.time()
    print(f'create_interaction_matrix_po time {(t1-t0)//60//60}h {(t1-t0)//60%60}min {(t1-t0) % 60}s')

    return out.tocsr()
