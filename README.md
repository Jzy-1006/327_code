# La4Ni3O10
1.主程序: compute.py
2.如何检验代码
(1). 在parameters.py文件中, 令layer_num = 2, hole_num =4, 5 or 6, 
同时令variational_space里的max_energy = 100(目的是要计算全部的态)
与旧的Ni2O9的4hole, 5hole以及6hole程序对比。
(2). 如果写完一个Hamiltonian, 比如Tpd, 可以将主程序里的gs.get_ground_state(参数1, ...)
中的参数1改成Tpd，只算Tpd, 和旧的Ni2O9程序对比能谱和weight
(3). 如果写完相互作用矩阵, 可以令参数A, B, C不为0, 其他为0, 检查能谱是否正确
