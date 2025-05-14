from imports import *
from utils import *
def train_and_evaluate_model(tc, ground_truth, index):
    result = []
    cony = conv(tc,  out_channels=2048,windows_length=windows_length,padding = False)
    swy = sliding_windows(tc, window_length=windows_length)
    models = [(cony,'conv'), (swy, 'sw')]    # size:n_sub,trs,rois,rois
    for model, name in models:
        a, b = faisscluster(model, 100,state)
        best = davies_compare_old(a, b, 100, model)
        kl,mse, cosine_sim,eud,mand,cor,prd = compare_diff_time(a[best], b[best], ground_truth, index)
        result.append((kl,mse, cosine_sim,eud,mand,cor,prd))
    return result

if __name__ == '__main__':
    data_path = 'osl data/experiment/600subs_321seed_10den/'
    rois = [90]
    states = [4]
    trs = [1200]
    num_subs = 20
    num_groups = np.arange(30)
    windows_lengths = np.arange(4,10)
    noises = [0.8,1.0]
    size = 10
    t_clusters = 100
    models = ['conv','sw']
    combinations = list(itertools.product(num_groups,rois, states, trs, noises,windows_lengths,models))
    df = pd.DataFrame(combinations, columns=['sub_groups','ROIs', 'nStates', 'TRs', 'Noise_levels','Window_length', 'Model'])
    df = df.assign(dwell_time='', MSE='',cosine = '',Edis = '', Mdis = '', PCC = '',PRD = '')
    for roi in rois:
        for state in states:
            for tr in trs:
                for noise in noises:
                    for windows_length in windows_lengths:
                        for num_group in num_groups:
                            if pd.isna(df.loc[(df['sub_groups'] == num_group) &(df['ROIs'] == roi) & (df['nStates'] == state) & (df['TRs'] == tr) & (df['Noise_levels'] == noise)& (df['Window_length'] == windows_length) & (df['Model'] == '13conv') , ['dwell_time']].values):
                                data = np.load((data_path + f'{size}size_{state}states_{roi}_{tr}.npz'))
                                tcn = data['tc'][num_subs * num_group:num_subs * (num_group + 1)]
                                index = data['index'][num_subs * num_group:num_subs * (num_group + 1)]
                                ground_truth = data['ground_truth']
                                ground_truth = convert_to_lower_triangular_no_diagonal(
                                    ground_truth.reshape(ground_truth.shape[0], -1))
                                tc = tcn + noise * np.random.randn(num_subs, roi, tr)
                                result = train_and_evaluate_model(tc, ground_truth, index, num_subs)
                                for j, (kl,mse, cosine_sim,eud,mand,cor,prd) in enumerate(result):
                                    df.loc[(df['sub_groups'] == num_group) &(df['ROIs'] == roi) & (df['nStates'] == state)
                                           & (df['TRs'] == tr) & (df['Noise_levels'] == noise)& (df['Window_length'] == windows_length)
                                           & (df['Model'] == models[j]) , ['dwell_time','MSE','cosine','Edis','Mdis','PCC','PRD']] =\
                                        kl,mse, cosine_sim,eud,mand,cor,prd