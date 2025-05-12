import parameters as pam
import lattice as lat
import time
from itertools import combinations
import bisect


def get_hole_uid(hole):
    """
    将空穴信息以进制的规则转化为数字信息
    :param hole: hole = ('x', 'y', 'z', 'orb'), orb轨道
    :return: hole_uid, 空穴所对应的数字
    """
    x, y, z, orb = hole

    # 依次将position, orb的个数求出来, 并作为进制
    position = lat.position
    b_orb = pam.Norb

    # 将position, orb依次转为数字
    i_position = position.index((x, y, z))
    # 轨道和自旋转为数字
    i_orb = lat.orb_int[orb]

    # 按将这些数, 按进制规则转为一个大数, 从低位到高位分别是orb, position = (x, y, z)
    hole_uid = i_orb + i_position * b_orb
    assert hole == get_hole(hole_uid), 'check hole and get_hole are tuple'

    return hole_uid


def get_hole(hole_uid):
    """
    将空穴的数字信息转化为空穴信息
    :param hole_uid: 空穴的数字信息
    :return: hole = ('x', 'y', 'z', 'orb'), orb轨道
    """
    # 依次将position, orb的个数求出来, 并作为进制
    position = lat.position
    b_orb = pam.Norb

    # 提取每个进制上的数
    # 依次提取i_orb, i_position
    i_orb = hole_uid % b_orb
    hole_uid //= b_orb
    i_position = hole_uid

    # 将这些数转为对应的空穴信息 hole = (x, y, z, orb)
    x, y, z = position[i_position]
    orb = lat.int_orb[i_orb]
    hole = (x, y, z, orb)

    return hole


def get_state_uid(state):
    """
    将态信息转化为数字信息
    :param state: state = (hole1, hole2, ....)
    :return: uid, 态对应的数字
    """
    # 计算存储一个空穴信息需要多大的进制, 并记录在b_hole
    b_hole = len(lat.position) * pam.Norb

    # 将每个空穴数字, 按b_hole进制转成一个大数
    uid = 0
    for idx, hole in enumerate(state):
        i_hole = get_hole_uid(hole)
        uid += i_hole * (b_hole ** idx)
    assert state == get_state(uid), 'check state and get_state are tuple'

    return uid


def get_state(uid):
    """
    将态的数字信息转为态信息
    :param uid: 态的数字信息
    :return: state = (hole1, hole2, ....)
    """
    # 计算存储一个空穴信息需要多大的进制, 并记录在b_hole
    b_hole = len(lat.position) * pam.Norb

    # 将大数uid按照b_hole进制, 提取每个进制上的数
    state = []
    while uid:
        hole_uid = uid % b_hole
        hole = get_hole(hole_uid)
        state.append(hole)
        uid //= b_hole
    state = tuple(state)

    return state


def count_inversion(state):
    """
    计算一个态, 需要经过多少次交换才能得到规范化的顺序
    :param state: state是一个嵌套元组, state = (hole1, hole2, ...)
    :return: inversion, 交换次数
    """
    # 将state中的每个空穴转为hole_uid
    uid_state = map(get_hole_uid, state)  # 注意map是惰性求解器
    uid_state = tuple(uid_state)

    inversion = 0
    hole_num = len(uid_state)
    for i in range(1, hole_num):
        behind_uid = uid_state[i]
        for front_uid in uid_state[:i]:
            if front_uid > behind_uid:
                inversion += 1
    return inversion


def make_state_canonical(state):
    """
    将态中的每个空穴按照hole_uid升序排列,
    :param state: state = (hole1, hole2, ...)
    :return: canonical_state and phase
    canonical_state, 按照hole_uid升序后的态
    phase, 每交换一次顺序相位乘以-1
    """
    inversion = count_inversion(state)
    phase = 1 if inversion % 2 == 0 else -1
    canonical_state = sorted(state, key=get_hole_uid)
    canonical_state = tuple(canonical_state)
    return canonical_state, phase


class VariationalSpace:
    def __init__(self):
        self.lookup_tbl = self.create_lookup_tbl()
        self.dim = len(self.lookup_tbl)
        print(f'VS_up.dim = {self.dim}, VS.dim = {self.dim ** 2}')

    def create_lookup_tbl(self):
        """
        找出所有可能的态，并根据能量的大小，砍去一部分态后存储在列表中
        :return: lookup_tbl
        """
        t0 = time.time()
        hole_num = pam.hole_num
        position = lat.position
        hole_uids = []
        for x, y, z in position:
            orbs = lat.get_unit_cell_rep(x, y, z)
            for orb in orbs:
                hole_uid = get_hole_uid((x, y, z, orb))
                hole_uids.append(hole_uid)

        hole_uids.sort()
        lookup_tbl = []
        b_hole = len(lat.position) * pam.Norb
        for hole_uid_tuple in combinations(hole_uids, hole_num):
            state_uid = 0
            for idx, hole_uid in enumerate(hole_uid_tuple):
                state_uid += hole_uid * (b_hole ** idx)
            lookup_tbl.append(state_uid)

        lookup_tbl.sort()       # 一定要有这一步, 这会影响get_index函数
        t1 = time.time()
        print(f"VS time {(t1-t0)//60//60}h, {(t1-t0)//60%60}min, {(t1-t0)%60}s")
        return lookup_tbl


    def get_index(self, state):
        """
        根据lookup_tbl, 找到state对应的索引
        :param state: state = ((x1, y1, z1, orb1, s1)...)
        :return: index, state对应在lookup_tbl中的索引
        """
        uid = get_state_uid(state)
        index = bisect.bisect_left(self.lookup_tbl, uid)

        # 判断索引是否超出lookup_tbl
        if index < self.dim:
            if self.lookup_tbl[index] == uid:
                return index
            else:
                return None
        else:
            return None
