# La4Ni3O10
1. 主程序: compute.py
2. 如何检验代码
   (1)在parameters.py文件中, 令layer_num = 2, hole_num =4, 5 or 6, 
同时令variational_space里的max_energy = 100(目的是要计算全部的态)
与旧的Ni2O9的4hole, 5hole以及6hole程序对比。
   (2)如果写完一个Hamiltonian, 比如Tpd, 可以将主程序里的gs.get_ground_state(参数1, ...)
中的参数1改成Tpd，只算Tpd, 和旧的Ni2O9程序对比能谱和weight
   (3)如果写完相互作用矩阵, 可以令参数A, B, C不为0, 其他为0, 检查能谱是否正确
3. 注意事项:
   (1) 当layer_num = 3, hole_num = 8时, 一定得调节max_energy(cut_off)
   (2) 让layer_num = 2, hole_num = 4, 5 or 6时, 可以回到双层Ni2O9, 可以和之前Ni2O9的计算结果进行对比
   (3) 在改变lattice的时候, 要注意d10这种态, 会影响variational_space.py中的get_atomic_energy和get_state_type, 
hamiltonian.py中的create_Esite_matrix
   (4) 注意顺序问题, 一个是get_interaction_mat中的state_order, ('d3z2r2', 'dx2y2'), d3z2r2的字典序在dx2y2的前面;
另外一个是由于态中空穴的排列方式带来phase(make_state_canonical函数)的问题
   (5) 另外代码还没加入dxy, dxz, dyz轨道
   (6) hamiltonian里, 要if, elif, elif, 最后是else; 或者是if if if, 不要else
