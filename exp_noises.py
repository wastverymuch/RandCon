# 主代码
from imports import *
# remodel = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=1000)
# remodel.fit(tc[0].T)
# states_show(remodel.covars_)
# a=remodel.predict(tc[0].T)
# metrics.adjusted_rand_score(a,index[0])
def train_and_evaluate_model(tc, ground_truth, index, num_subs):
    result = []
    # cony = conv(tc, kernal_shape='31', out_channels=10)
    # cony33 = conv(tc, kernal_shape='33', out_channels=10)
    # cony13 = conv(tc, kernal_shape='13', out_channels=10)
    cony31128 = conv(tc, kernal_shape='31', out_channels=128)
    mmy = mm_optimized(tc)
    mtdoy = mtds_official(tc)
    mtdoy = np.insert(mtdoy, mtdoy.shape[1], mtdoy[:, -1], 1)
    # cony33128 = conv(tc, kernal_shape='33', out_channels=128)
    cony13128 = conv(tc, kernal_shape='13', out_channels=2048)
    swy = sliding_windows(tc, 3)
    swy = np.insert(swy, swy.shape[1], swy[:, -1], 1)
    swy = np.insert(swy, swy.shape[1], swy[:, -1], 1)
    hmmy = HMM_cluster(tc,ground_truth,index,nstate=state)
    phasey = phase(tc)
    # models = [(cony, '31'), (cony13, '13'), (cony33, '33'),(cony128,'31128'),(cony13128,'13128'),(cony33128,'33128'),
    #           (mmy, 'mm'), (mtdoy, 'mtd'), (swy, 'sw')]
    models = [(cony31128,'31conv'),(mmy, 'mm'), (mtdoy, 'mtd'),(phasey, 'phase'),(cony13128,'13conv'), (swy, 'sw')]# size:n_sub,trs,rois,rois
    for model, name in models:
        a, b = faisscluster(model, 20,state)
        ARI, kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd = compare(a, b, ground_truth, index, 20)
        result.append((ARI, kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd ))
    result.insert(3,hmmy)
    return result

if __name__ == '__main__':
    data_path = '/longlab/duan/pycharm_programs/formal_project/osl data/experiment/20subs_321seed/'
    # conv_data_path = '/root/formal_project/osl/conv/'
    # conv_data_path = 'D:/Pycharm code/formal_project/osl/conv/'
    rois = [30,60,90]
    state = 4
    tr = 600
    noise_levels = np.arange(4,15) / 10
    num_subs = 10
    noises = ['randn','random','poisson']
    noises = ['randn']
    models = ['31conv','mm','mtd','hmm','phase','13conv','sw']
    combinations = list(itertools.product(noises,rois,noise_levels,models))
    df = pd.DataFrame(combinations, columns=['Noise_types','Rois','Noise_levels', 'Model'])# 生成上述列表的组合并建表
    df = df.assign(ARI='', dwell_time='', occupancy='', MSE='',cosine = '',Edis = '', Mdis = '', PCC = '',PRD = '')
    # df = pd.read_excel('test1.xlsx')
    # df.to_excel("exp_10noise1.xlsx", index=False)
    # result_save = np.zeros((len(sizes), len(degrees), noise_levels, 9, 3))
    for noiselevel in noise_levels:
        for noise in noises:
            for roi in rois:
                # if pd.isna(df.loc[(df['ROIs'] == roi) & (df['nStates'] == state) & (df['TRs'] == tr) & (df['Noise_types'] == noise) & (df['Model'] == '31conv') ,'ARI'].values):
                if df.loc[(df['Noise_levels'] == noiselevel) & (df['Rois'] == roi) &(df['Noise_types'] == noise) & (df['Model'] == '31conv') ,'ARI'].values == '':
                    data = np.load((data_path + f'4states_{roi}_600.npz'))
                    tcn = data['tc'][:num_subs]
                    index = data['index'][:num_subs]
                    ground_truth = data['ground_truth']
                    if noise == 'randn':
                        tc = tcn + noiselevel * np.random.randn(num_subs, roi, tr)
                    elif noise == 'random':
                        tc = tcn + noiselevel * np.random.random((num_subs,roi,tr))
                    else:
                        tc = tcn + noiselevel * np.random.poisson(1,(num_subs,roi,tr))
                    print(f'开始,{noise},{roi},{noiselevel}')
                    result = train_and_evaluate_model(tc, ground_truth, index, num_subs)
                    for j, (ARI,kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd) in enumerate(result):
                        # result_save[sizes.index(size), degrees.index(degree), i, j, 0] = res1
                        # result_save[sizes.index(size), degrees.index(degree), i, j, 1] = res2
                        # result_save[sizes.index(size), degrees.index(degree), i, j, 2] = res3
                        df.loc[(df['Noise_levels'] == noiselevel) &  (df['Noise_types'] == noise) & (df['Rois'] == roi) & (df['Model'] == models[j]) , ['ARI','dwell_time','occupancy','MSE','cosine','Edis','Mdis','PCC','PRD']] = ARI,kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd
                        df.to_excel("exp_noiselevel_4_roi_600.xlsx", index=False)