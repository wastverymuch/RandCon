# 主代码
from imports import *
from utils import *
# remodel = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=1000)
# remodel.fit(tc[0].T)
# states_show(remodel.covars_)
# a=remodel.predict(tc[0].T)
# metrics.adjusted_rand_score(a,index[0])
# @profile
def train_and_evaluate_model(tc, ground_truth, index, t_clusters):  # 输入的是被试数，roi，时间点的格式
    result = []
    # cony = conv(tc, kernal_shape='31', out_channels=10)
    # cony33 = conv(tc, kernal_shape='33', out_channels=10)
    # cony13 = conv(tc, kernal_shape='13', out_channels=10)
    # cony31128 = conv(tc, kernal_shape='31', out_channels=128)
    # mmy = mm_optimized(tc)
    mtdoy = mtds_official(tc)
    mtdoy = np.insert(mtdoy, mtdoy.shape[1], mtdoy[:, -1], 1)  # 插一行让时间点对齐
    # cony33128 = conv(tc, kernal_shape='33', out_channels=128)
    cony13128 = conv(tc, kernal_shape='13', out_channels=128)  # 卷积模型
    swy = sliding_windows(tc, 3)  # 滑窗
    swy = np.insert(swy, swy.shape[1], swy[:, -1], 1)  # 复制最后一行的结果两遍，对齐时间点
    swy = np.insert(swy, swy.shape[1], swy[:, -1], 1)
    # hmmy = HMM_cluster(tc,ground_truth,index,nstate=state)
    phasey = phase(tc)  # 相位同步的算法
    # models = [(cony, '31'), (cony13, '13'), (cony33, '33'),(cony128,'31128'),(cony13128,'13128'),(cony33128,'33128'),
    #           (mmy, 'mm'), (mtdoy, 'mtd'), (swy, 'sw')]
    models = [(mtdoy, 'mtd'),(phasey, 'phase'),(cony13128,'13conv'), (swy, 'sw')]  # size:n_sub,trs,rois,rois
    for model, name in models:
        a, b = faisscluster(model, t_clusters,state)
        best = davies_compare(a,b,t_clusters,model)
        ARI, kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd = compare(a[best], b[best], ground_truth, index)
        result.append((ARI, kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd ))
    # result.insert(3,hmmy)
    return result

if __name__ == '__main__':
    data_path = '/longlab/duan/pycharm_programs/formal_project/osl data/experiment/600subs_321seed_10den/'
    # conv_data_path = '/root/formal_project/osl/conv/'
    # conv_data_path = 'D:/Pycharm code/formal_project/osl/conv/'

    rois = [30,60,90]
    rois = [90]
    states = [4,6,8]
    states = [4]
    trs = [600,1200]
    trs = [1200]
    # noise_levels = 20
    num_subs = 20
    num_groups = np.arange(30)
    noises = ['randn','random','poisson']
    noises = np.arange(5,11) / 10
    size = 10
    t_clusters = 100
    # noises = ['random', 'poisson']
    models = ['mtd','phase','13conv','sw']
    # models = ['31conv', 'mm', 'mtd', 'phase', '13conv', 'sw']

    # 根据参数组合建表
    combinations = list(itertools.product(num_groups,rois, states, trs, noises,models))
    df = pd.DataFrame(combinations, columns=['sub_groups','roi', 'nstates', 'trs', 'noise', 'model'])
    df = df.assign(ARI='', dwell_time='', occupancy='', MSE='',cosine = '',Edis = '', Mdis = '', PCC = '',PRD = '')
    # df = pd.read_excel('test1.xlsx')
    # df.to_excel("exp_10noise1.xlsx", index=False)
    # result_save = np.zeros((len(sizes), len(degrees), noise_levels, 9, 3))
    for roi in rois:
        for state in states:
            for tr in trs:
                for noise in noises:
                    for num_group in num_groups:
                    # if pd.isna(df.loc[(df['ROIs'] == roi) & (df['nStates'] == state) & (df['TRs'] == tr) & (df['Noise_types'] == noise) & (df['Model'] == '31conv') ,'ARI'].values):
                    #     if df.loc[(df['ROIs'] == roi) & (df['nStates'] == state) & (df['TRs'] == tr) & (df['Noise_types'] == noise) & (df['Model'] == '31conv') ,'ARI'].values == '':
                            data = np.load((data_path + f'{size}size_{state}states_{roi}_{tr}.npz'))
                            tcn = data['tc'][num_subs * num_group:num_subs * (num_group + 1)]
                            index = data['index'][num_subs * num_group:num_subs * (num_group + 1)]
                            ground_truth = data['ground_truth']
                            # if noise == 'randn':
                            tc = tcn + noise * np.random.randn(num_subs, roi, tr)
                            # elif noise == 'random':
                            #     tc = tcn + 0.6 * np.random.random((num_subs,roi,tr))
                            # else:
                            #     tc = tcn + 0.6 * np.random.poisson(1,(num_subs,roi,tr))
                            print(f'开始,第{num_group}组被试处理中')
                            # 所有模型的训练和评估
                            result = train_and_evaluate_model(tc, ground_truth, index, t_clusters)
                            # 数据记录到表格上
                            for j, (ARI,kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd) in enumerate(result):
                                # result_save[sizes.index(size), degrees.index(degree), i, j, 0] = res1
                                # result_save[sizes.index(size), degrees.index(degree), i, j, 1] = res2
                                # result_save[sizes.index(size), degrees.index(degree), i, j, 2] = res3
                                df.loc[(df['sub_groups'] == num_group) &(df['roi'] == roi) & (df['nstates'] == state) & (df['trs'] == tr) & (df['noise'] == noise) & (df['model'] == models[j]) , ['ARI','dwell_time','occupancy','MSE','cosine','Edis','Mdis','PCC','PRD']] = ARI,kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd
                                # df.to_excel("/longlab/duan/pycharm_programs/formal_project/results/exp_20sub_30group_noises.xlsx", index=False)