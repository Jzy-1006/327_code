import time
import numpy as np
from collections import Counter, defaultdict

import parameters as pam
import variational_space as vs
import lattice as lat


def get_state(istate, up_VS, dn_VS):
    """
    get state from istate
    :param istate: index of VS
    :param up_VS: space of up state
    :param dn_VS: space of dn state
    :return:
    """
    iup = istate % up_VS.dim
    idn = istate // up_VS.dim
    state_up = vs.get_state(up_VS.lookup_tbl[iup])
    state_up = [(x, y, z, orb, 'up') for x, y, z, orb in state_up]
    state_dn = vs.get_state(dn_VS.lookup_tbl[idn])
    state_dn = [(x, y, z, orb, 'dn') for x, y, z, orb in state_dn]
    state = tuple(state_up + state_dn)

    return state


def get_position_orb_uid(istate, up_VS, dn_VS):
    """
    得到将位置轨道进行排序后的uid
    :param istate:
    :param up_VS:
    :param dn_VS:
    :return:
    """
    iup = istate % up_VS.dim
    idn = istate // up_VS.dim
    uid_up, uid_dn = up_VS.lookup_tbl[iup], dn_VS.lookup_tbl[idn]
    b_hole = len(lat.position) * pam.Norb

    uids = []
    while uid_up:
        uid = uid_up % b_hole
        uids.append(uid)
        uid_up //= b_hole
    while uid_dn:
        uid = uid_dn % b_hole
        uids.append(uid)
        uid_dn //= b_hole
    double = []
    single = []
    for uid, same_num in Counter(uids).items():
        if same_num > 1:
            double.append(uid)
        else:
            single.append(uid)

    if len(single) < 2:
        return None
    double.sort()
    single.sort()
    double_uid = 0
    single_uid = 0
    for idx, uid in enumerate(double):
        double_uid += uid * (b_hole ** idx)
    for idx, uid in enumerate(single):
        single_uid += uid * (b_hole ** idx)

    return double_uid, single_uid


def get_ground_state(up_VS, dn_VS, vals, vecs, S_vals, Sz_vals, **kwargs):
    """
    求解矩阵的本征值和本征向量, 并对求解结果进行整理
    :param up_VS: 自旋向上的所有态
    :param dn_VS: 自旋向下的所有态
    :param vals: 能谱
    :param vecs: 本征向量构成的矩阵
    :param S_vals:
    :param Sz_vals:
    :return:
    """
    t0 = time.time()
    print('lowest eigenvalue of H from np.linalg.eigsh = ')
    print(vals)

    # 计算不同的本征值第一次出现的索引，并存储在degen_idx中
    # 对应的简并度即为degen_idx[i+1] - degen_idx[i]
    val_num = pam.val_num
    degen_idx = [0]
    for _ in range(val_num):
        for idx in range(degen_idx[-1] + 1, pam.Neval):
            if abs(vals[idx] - vals[degen_idx[-1]]) > 1e-4:
                degen_idx.append(idx)
                break

    coupled_uid = set()
    for i in range(val_num):
        print(f'Degeneracy of {i}th state is {degen_idx[i + 1] - degen_idx[i]}')
        print('val = ', vals[degen_idx[i]])
        weight_average = np.average(abs(vecs[:, degen_idx[i]:degen_idx[i + 1]]) ** 2, axis=1)
        ilead = np.argsort(-weight_average).astype(np.int32)
        dL_weights = defaultdict(float)
        dL_orb_weights = defaultdict(lambda: defaultdict(float))
        dL_orb_i = defaultdict(lambda: defaultdict(list))
        for istate in ilead:
            weight = weight_average[istate]
            state = get_state(istate, up_VS, dn_VS)
            state_type = lat.get_state_type(state)
            orb_type = lat.get_orb_type(state)

            dL_weights[state_type] += weight
            dL_orb_weights[state_type][orb_type] += weight
            if weight > 1e-3:
                dL_orb_i[state_type][orb_type].append(istate)

        sorted_dL = sorted(dL_weights.keys(), key=lambda dL_key: dL_weights[dL_key], reverse=True)
        for dL in sorted_dL:
            dL_weight = dL_weights[dL]
            if dL_weight < 1e-2:
                continue
            print(f"{dL} == {dL_weight}\n")
            orb_weights = dL_orb_weights[dL]
            sorted_orb = sorted(orb_weights.keys(), key=lambda orb_key: orb_weights[orb_key], reverse=True)
            for orb_type in sorted_orb:
                orb_weight = orb_weights[orb_type]
                if orb_weight < 5e-3:
                    continue
                print(f"{orb_type} : {orb_weight}\n")
                istates = dL_orb_i[dL][orb_type]
                for istate in istates:
                    # 只对dL_weight > 0.05的态进行耦合变换
                    if pam.if_save_coupled_uid and i == 0 and dL_weight > 0.05:
                        uid = get_position_orb_uid(istate, up_VS, dn_VS)
                        if uid is not None:
                            coupled_uid.add(uid)

                    state = get_state(istate, up_VS, dn_VS)
                    weight = weight_average[istate]

                    # 将态转为字符串
                    state_string = []
                    for hole in state:
                        x, y, z, orb, s = hole
                        hole_string = f'({x}, {y}, {z}, {orb}, {s})'
                        state_string.append(hole_string)

                    # 将字符串列表分成四个一组
                    chunks = [state_string[i: i+4] for i in range(0, len(state_string), 4)]
                    # 每个组内用', '连接, 并把组与组之间用'\n\t'连接
                    state_string = '\n\t'.join([', '.join(chunk) for chunk in chunks])

                    other_string = []
                    # 自旋信息转为字符串
                    for Ni_i, position in enumerate(lat.Ni_position):
                        if istate in S_vals[Ni_i]:
                            other_string.append(f'S,Sz{position} = {S_vals[Ni_i][istate]},{Sz_vals[Ni_i][istate]}')
                    if 'S_val' in kwargs:
                        if istate in kwargs['S_val']:
                            other_string.append(f"S_tot, Sz_tot = {kwargs['S_val'][istate]}, {kwargs['Sz_val'][istate]}")
                    # 串联字符串
                    other_string = '; '.join(other_string)

                    # 打印输出
                    print(f"\t{state_string}\n\t{other_string}\n\tweight = {weight}\n")

    if pam.if_save_coupled_uid:
        with open("coupled_uid", 'w') as file:
            for double_uid, single_uid in coupled_uid:
                file.write(f"{double_uid}, {single_uid}\n")
    t1 = time.time()
    print(f'gs time {(t1-t0)//60//60}h, {(t1-t0)//60%60}min, {(t1-t0)%60}s\n')
