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
Ni_position = [(0, 0, z) for z in range(0, 2*pam.layer_num-1, 2) ]
Ni_position = tuple(Ni_position)


def get_unit_cell_rep(x, y, z):
    """
    确定需要计算的晶格, 根据坐标确定轨道
    :return:orbs
    """
    layer_num = pam.layer_num
    if z < 0 or z > 2*layer_num-2:
        print('z is error')
    z_Ni = range(0, 2 * layer_num, 2)
    z_Ni = tuple(z_Ni)
    if layer_num > 1:
        z_Oap = range(1, 2*layer_num-1, 2)
        z_Oap = tuple(z_Oap)
    else:
        z_Oap = ()
    if x == 0 and y == 0 and z in z_Ni:
        return pam.Ni_orbs
    elif x == 0 and y == 0 and z in z_Oap:
        return pam.Oap_orbs
    elif abs(x) % 2 == 1 and abs(y) % 2 == 0 and z in z_Ni:
        return pam.O1_orbs
    elif abs(x) % 2 == 0 and abs(y) % 2 == 1 and z in z_Ni:
        return pam.O2_orbs
    else:
        return ['NotOnSublattice']


def get_state_type(state):
    """
    按照Ni, 层内O, 层间O每层的数量给态分类
    :param state: state = ((x1, y1, z1, orb1, s1), ...)
    :return:state_type
    """
    layer_num = pam.layer_num
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
                state_type[z] = f'L{num}'
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
