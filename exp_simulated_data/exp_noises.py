from imports import *
from utils import *

def train_and_evaluate_model(tc, ground_truth, index, t_clusters):
    result = []
    cony = conv(tc,  out_channels=2048)
    mtdy = mtd(tc,3)
    mtdy = np.insert(mtdy, mtdy.shape[1], mtdy[:, -1], 1)
    mtdy = np.insert(mtdy, mtdy.shape[1], mtdy[:, -1], 1)
    mtdy = np.insert(mtdy, mtdy.shape[1], mtdy[:, -1], 1)
    swy = sliding_windows(tc, 3)
    swy = np.insert(swy, swy.shape[1], swy[:, -1], 1)
    swy = np.insert(swy, swy.shape[1], swy[:, -1], 1)
    phasey = phase(tc)
    models = [(mtdy, 'mtd'),(phasey, 'phase'),(cony,'conv'), (swy, 'sw')]
    for model, name in models:
        a, b = faisscluster(model, t_clusters,state)
        best,_ = davies_compare(a,b,t_clusters,model)
        np.save(f'{name}.npy',b[best])
        ARI, kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd = compare(a[best], b[best], ground_truth, index)
        result.append((ARI, kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd))
    return result
if __name__ == '__main__':
    data_path = 'osl data/experiment/600subs_321seed_10den/'
    rois = [90]
    states = [4]
    trs = [1200]
    num_subs = 20
    num_groups = np.arange(1)
    noises = np.arange(6,7) / 10
    size = 10
    t_clusters = 100
    models = ['mtd','phase','conv','sw']
    combinations = list(itertools.product(num_groups,rois, states, trs, noises,models))
    df = pd.DataFrame(combinations, columns=['sub_groups','roi', 'nstates', 'trs', 'noise', 'model'])
    df = df.assign(ARI='', dwell_time='', occupancy='', MSE='',cosine = '',Edis = '', Mdis = '', PCC = '',PRD = '')
    for roi in rois:
        for state in states:
            for tr in trs:
                for noise in noises:
                    for num_group in num_groups:
                            data = np.load((data_path + f'{size}size_{state}states_{roi}_{tr}.npz'))
                            tcn = data['tc'][num_subs * num_group:num_subs * (num_group + 1)]
                            index = data['index'][num_subs * num_group:num_subs * (num_group + 1)]
                            ground_truth = data['ground_truth']
                            ground_truth = convert_to_lower_triangular_no_diagonal(ground_truth.reshape(ground_truth.shape[0],-1))
                            tc = tcn + noise * np.random.randn(num_subs, roi, tr)
                            result = train_and_evaluate_model(tc, ground_truth, index, t_clusters)
                            for j, (ARI,kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd) in enumerate(result):
                                df.loc[(df['sub_groups'] == num_group) &(df['roi'] == roi) & (df['nstates'] == state)
                                       & (df['trs'] == tr) & (df['noise'] == noise) & (df['model'] == models[j]) ,
                                ['ARI','dwell_time','occupancy','MSE','cosine','Edis','Mdis','PCC','PRD']] = (
                                    ARI,kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd)