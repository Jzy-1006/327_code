import parameters as pam

# 将轨道, 自旋与数字一一对应, 用来生成每个态的数字uid
if pam.Norb == 5:
    orb_int = {'d3z2r2': 0,
               'dx2y2': 1,
               'px': 2,
               'py': 3,
               'apz': 4}
    int_orb = {value: key for key, value in orb_int.items()}
spin_int = {'up': 1, 'dn': 0}
int_spin = {value: key for key, value in spin_int.items()}

# 各种元素对应的位置坐标
Ni_position = [(0, 0, z) for z in range(0, 2*pam.layer_num-1, 2) ]

O1_xy = [(-1, 0), (1, 0)]
O1_position = [(x, y, z) for x, y in O1_xy for z in range(0, 2*pam.layer_num-1, 2)]
O2_xy = [(0, 1), (0, -1)]
O2_position = [(x, y, z) for x, y in O2_xy for z in range(0, 2*pam.layer_num-1, 2)]
O_position = O1_position + O2_position

Oap_position = [(0, 0, z) for z in range(1, 2*pam.layer_num-2, 2)]

def get_unit_cell_rep(x, y, z):
    """
    确定需要计算的晶格, 根据坐标确定轨道
    :return:orbs
    """
    if (x, y, z) in Ni_position:
        return pam.Ni_orbs
    elif (x, y, z) in O1_position:
        return pam.O1_orbs
    elif (x, y, z) in O2_position:
        return pam.O2_orbs
    elif (x, y, z) in Oap_position:
        return pam.Oap_orbs
    else:
        return ['NotOnSublattice']


def get_state_type(state):
    """
    按照Ni, 层内O, 层间O每层的数量给态分类
    :param state: state = ((x1, y1, z1, orb1, s1), ...)
    :return:state_type
    """
    # 统计每一层Ni, 层内O, 层间O的数量
    Ni_num = {}
    O_num = {}
    Oap_num = {}
    for hole in state:
        _, _, z, orb, _ = hole
        # Ni
        if orb in pam.Ni_orbs:
            if z in Ni_num:
                Ni_num[z] += 1
            else:
                Ni_num[z] = 1
        # 层内O
        if orb in pam.O_orbs:
            if z in O_num:
                O_num[z] += 1
            else:
                O_num[z] = 1
        # 层间O
        if orb in pam.Oap_orbs:
            if z in Oap_num:
                Oap_num[z] += 1
            else:
                Oap_num[z] = 1

    # 根据每一层Ni, 层内O和层间O的空穴数量, 生成态的类型
    state_type = {}
    # Ni
    for z, num in Ni_num.items():
        state_type[z] = f'd{10 - num}'
    # 层内O
    for z, num in O_num.items():
        if num == 1:
            if z in state_type:
                state_type[z] += f'L'
            else:
                state_type[z] = f'L'
        else:
            if z in state_type:
                state_type[z] += f'L{num}'
            else:
                state_type[z] = f'L{num}'
    # 层间O
    for z, num in Oap_num.items():
        if num == 1:
            state_type[z] = f'O'
        else:
            state_type[z] = f'O{num}'

    sorted_z = sorted(state_type.keys())
    state_type = [state_type[z] for z in sorted_z]
    state_type = '-'.join(state_type)

    return state_type


def get_orb_type(state):
    """
    将具体的state_type细化, 输出具体的d轨道
    :param state:
    :return:
    """
    simple_orbs = {'d3z2r2': 'dz2', 'dx2y2': 'dx2'} # 简化坐标表示
    Ni_orb = {}
    Oap_orb = {}    # 收集层间O的轨道数目
    L_orb = {}  # 收集层内O的轨道数目
    for hole in state:
        _, _, z, orb, _ = hole
        if orb in pam.Ni_orbs:
            if orb in simple_orbs:
                orb = simple_orbs[orb]
            if z in Ni_orb:
                Ni_orb[z] += [orb]
            else:
                Ni_orb[z] = [orb]
        elif orb in pam.Oap_orbs:
            if z in Oap_orb:
                Oap_orb[z] += 1
            else:
                Oap_orb[z] = 1
        else:
            if z in L_orb:
                L_orb[z] += 1
            else:
                L_orb[z] = 1

    orb_type = {}
    for z, orbs in Ni_orb.items():
        orbs.sort()
        orb_type[z] = ''.join(orbs)
    for z, num in L_orb.items():
        if z in orb_type:
            if num == 1:
                orb_type[z] += f'L'
            else:
                orb_type[z] += f'L{num}'
        else:
            if num == 1:
                orb_type[z] = f'L'
            else:
                orb_type[z] = f'L{num}'
    for z, num in Oap_orb.items():
        if z in orb_type:
            if num == 1:
                orb_type[z] += f'apz'
            else:
                orb_type[z] += f'apz{num}'
        else:
            if num == 1:
                orb_type[z] = f'apz'
            else:
                orb_type[z] = f'apz{num}'
    sorted_z = sorted(orb_type.keys())
    orb_type = [orb_type[z] for z in sorted_z]
    orb_type = '_'.join(orb_type)

    return orb_type


def get_Ni_side_num(state):
    """
    得到和Ni同一边的空穴个数
    改动该函数, 对应basis_change.py中的
    create_singlet_triplet_basis_change_matrix也要一起修改
    :param state:
    :return:
    """
    side1_idx = []
    side2_idx = []
    for idx, hole in enumerate(state):
        if hole[2] == 0:
            side1_idx.append(idx)
        elif hole[2] == 2:
            side2_idx.append(idx)
    return side1_idx, side2_idx
