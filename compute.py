import os
import time
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

    gs.get_ground_state(H0, VS)


if __name__ == '__main__':
    t0 = time.time()
    # 创建data的文件夹
    os.makedirs('data', exist_ok=True)
    # 计算前, 清空文件夹所有文件的内容
    for filename in os.listdir('data'):
        file_path = os.path.join('data', filename)
        with open(file_path, 'w') as file:
            file.truncate(0)

    A = pam.A
    Uoo = pam.Uoos[0]
    Upp = pam.Upps[0]

    ed = pam.ed_list[4]
    ep = pam.ep_list[4]
    eo = pam.eo_list[4]

    tpd = pam.tpd_list[4]
    tpp = pam.tpp_list[4]
    tdo = pam.tdo_list[4]
    tpo = pam.tpd_list[4]
    tz_a1a1 = pam.tz_a1a1
    tz_b1b1 = pam.tz_b1b1
    if_tz_exist = pam.if_tz_exist

    VS = vs.VariationalSpace()
    d_state_idx, d_hole_idx, p_idx_pair, apz_idx_pair = ham.get_double_occ_list(VS)

    compute_Aw_main(A, Uoo, Upp, ed, ep, eo, tpd, tpp, tdo, tpo)
    t1 = time.time()
    print('compute cost time', t1-t0)
