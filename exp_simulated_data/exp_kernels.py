from imports import *
from utils import *

def train_and_evaluate_model(tc, ground_truth, index, t_clusters):
    cony5 = conv(tc,  out_channels=5)
    cony10 = conv(tc,  out_channels=10)
    cony20 = conv(tc, out_channels=20)
    cony40 = conv(tc,  out_channels=40)
    cony80 = conv(tc, out_channels=80)
    cony160 = conv(tc, out_channels=160)
    models = [(cony5, '5'),(cony10, '10'),(cony20,'20'), (cony40, '40'), (cony80, '80'), (cony160, '160')]  # size:n_sub,trs,rois,rois
    for model, name in models:
        a, b = faisscluster(model, t_clusters,state)
        best = davies_compare(a,b,t_clusters,model)
        ARI, kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd = compare(a[best], b[best], ground_truth, index)
        result.append((ARI, kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd ))
    return result

if __name__ == '__main__':
    data_path = 'osl_data/'
    rois = [90]
    states = [4]
    trs = [1200]
    num_subs = 20
    num_groups = np.arange(30)
    noises = [0.6]
    size = 10
    t_clusters = 100
    models = ['5 kernels','10 kernels','20 kernels','40 kernels', '80 kernels','160 kernels']
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
                            ground_truth = convert_to_lower_triangular_no_diagonal(
                                ground_truth.reshape(ground_truth.shape[0], -1))
                            tc = tcn + noise * np.random.randn(num_subs, roi, tr)
                            result = train_and_evaluate_model(tc, ground_truth, index, t_clusters)
                            for j, (ARI,kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd) in enumerate(result):
                                df.loc[(df['sub_groups'] == num_group) &(df['roi'] == roi) & (df['nstates'] == state) &
                                       (df['trs'] == tr) & (df['noise'] == noise) & (df['model'] == models[j]) ,
                                ['ARI','dwell_time','occupancy','MSE','cosine','Edis','Mdis','PCC','PRD']] = \
                                    (ARI,kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd)
                                df.to_excel("results/kernels/results.xlsx", index=False)