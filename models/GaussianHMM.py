import time

from imports import *
from utils import *


def HMM_single(all_data,ground_truth,index,nstate,repeat = 20):
    if all_data.shape.__len__() == 2:
        all_data = all_data[None,:]
    n_sub,n_roi,n_tr = all_data.shape
    all_data = all_data.transpose(0,2,1)
    subindexs = []
    substates = []
    subaris = []
    for sub in range(n_sub):
        indexs = []
        states = []
        aris = []
        for _ in range(repeat):
            remodel = hmm.GaussianHMM(n_components=nstate, covariance_type="full", n_iter=40)
            # t = time.time()
            try:
                remodel.fit(all_data[sub])
            except ValueError:
                continue
            # states_show(remodel.covars_)
            try:
                a = remodel.predict(all_data[sub])
            except ValueError:
                continue
            indexs.append(a)
            states.append(remodel.covars_)
            aris.append(metrics.adjusted_rand_score(a, index[sub]))
        subindexs.append(indexs[aris.index(max(aris))])
        substates.append(states[aris.index(max(aris))])
        subaris.append(max(aris))
    best_sub = subaris.index(max(subaris))
    best_index = subindexs[best_sub]
    best_states = substates[best_sub].reshape(nstate,n_roi**2)
    ARI = metrics.adjusted_rand_score(best_index,index[best_sub])
    kl = KL(index[best_sub],best_index,nstate)
    overlap_ratio = overlap(index[best_sub],best_index,nstate)
    # mse, cosine_sim = align_and_calculate_metrics(best_states, ground_truth.reshape(nstate,n_roi**2))
    mse, cosine_sim, eud, mand, cor, prd = align_and_calculate_metrics(best_states, ground_truth.reshape(nstate,n_roi**2))
    # cosine_sim = np.diag(cosine_sim).mean()
    return ARI,kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd
def HMM_cluster(all_data,ground_truth,index,nstate = 4,repeat = 20):
    if all_data.shape.__len__() == 2:
        all_data = all_data[None,:]
    n_sub,n_roi,n_tr = all_data.shape
    all_data = all_data.transpose(0,2,1).reshape(-1,n_roi)
    index = index.reshape(-1)
    subindexs = []
    substates = []
    subaris = []
    # for sub in range(n_sub):
    indexs = []
    states = []
    aris = []
    for _ in range(repeat):
        remodel = hmm.GaussianHMM(n_components=nstate, covariance_type="full", n_iter=10)
        # t = time.time()
        try:
            remodel.fit(all_data)
        except ValueError:
            continue
        # states_show(remodel.covars_)
        try:
            a = remodel.predict(all_data)
        except ValueError:
            continue
        subindexs.append(a)
        substates.append(remodel.covars_)
        subaris.append(metrics.adjusted_rand_score(a, index))
        # subindexs.append(indexs[aris.index(max(aris))])
        # substates.append(states[aris.index(max(aris))])
        # subaris.append(max(aris))
    best_iter = subaris.index(max(subaris))
    best_index = subindexs[best_iter]
    best_states = substates[best_iter].reshape(nstate,n_roi**2)
    ARI = metrics.adjusted_rand_score(best_index,index)
    kl = KL(index,best_index,nstate)
    overlap_ratio = overlap(index,best_index,nstate)
    # mse, cosine_sim = align_and_calculate_metrics(best_states, ground_truth.reshape(nstate,n_roi**2))
    mse, cosine_sim, eud, mand, cor, prd = align_and_calculate_metrics(best_states, ground_truth.reshape(nstate,n_roi**2))
    # cosine_sim = np.diag(cosine_sim).mean()
    return ARI,kl,overlap_ratio ,mse, cosine_sim,eud,mand,cor,prd

if __name__ == '__main__':
    conv_data_path = '/longlab/duan/pycharm_programs/formal_project/osl data/'
    data = np.load((conv_data_path + '10size_10_200sub_90_1200_321seed.npz'))
    tc = data['tc'][:10]
    index = data['index'][:10]
    # tc += 0.1* np.random.randn(12000,90)
    ground_truth = data['ground_truth']
    t=time.time()
    x = HMM_cluster(tc,ground_truth,index)
    print((time.time() - t))
    print(x)

    # 10被试，90*1200，4states，20次重复，7分钟不到

    #
    # aris=[]
    # t=time.time()
    # for i in range(10):
    #     remodel = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
    #     # t = time.time()
    #     try:
    #         remodel.fit(tc)
    #     except ValueError:
    #         continue
    #     # states_show(remodel.covars_)
    #     try:
    #         a=remodel.predict(tc)
    #     except ValueError:
    #         continue
    #     # print(time.time() - t)
    #     aris.append(metrics.adjusted_rand_score(a,index))
    #     np.savetxt('ariss.txt', np.stack(aris))
    # print(np.stack(aris))
    # print(max(aris))
    # print(time.time() - t)
    # pass
    #
