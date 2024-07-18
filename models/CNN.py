import torch.nn.functional as F
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from utils import *

start_time = time.time()
from sklearn import metrics

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class CNN(nn.Module):
    def __init__(self, num_kernels):
        super(CNN, self).__init__()
        # 卷积层，假设输入数据是单通道的，所以in_channels=1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size=3, stride=1, padding=1)
        # 计算扁平化后的大小
        # self.flattened_size = 1200 * num_kernels
        # 全连接层
        # self.fc1 = nn.Linear(self.flattened_size, 1200)
        # self.manylinears = nn.ModuleList([nn.Linear(num_kernels,1) for _ in range(1200)])
        # 卷积回去
        self.conv2 = nn.Conv2d(in_channels=num_kernels, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv31=nn.Conv2d(1,10,(3,1),1,(1,0),1,1,0).to(device)

    def forward(self, x):
        # 应用卷积层
        y = F.relu(self.conv1(x).squeeze())
        t = F.relu(self.conv31(x).squeeze())
        # # 平均池化
        # z = torch.mean(y,dim=1)
        # 卷积回来
        z = self.conv2(y).squeeze()
        # # 1200个全连接层
        # z = torch.zeros_like(x)
        # for i in range(1200):
        #     z[:,:,:,i] = self.manylinears[i](y[:,:,:,i].transpose(1,2)).transpose(1,2)
        # z = z.squeeze()
        # # 一个大的全连接层
        # z = y.transpose(1,2).reshape(x.shape[0],90,self.flattened_size)
        # # 全连接层
        # z = self.fc1(z)
        # 重构输出尺寸
        return y,t
def tmp(test,m=4):
    _,y = model(test)   # y是n_sub,c,roi,tr
    if y.shape.__len__() < 4:
        y=y[None,:]
    need = corshow(y.transpose(-1,-3).reshape(-1,90,10).detach().cpu())

    need = np.nan_to_num(need, 1e-10)
    a,b = cluster(need,m)
    # print('ARI = {}'.format(metrics.adjusted_rand_score(b, index)))
    # states_show(a,title='ARI = {}'.format(metrics.adjusted_rand_score(b, index)))
    return metrics.adjusted_rand_score(b, index.reshape(-1))

if __name__ == '__main__':
    noise = -0.1
    noise_level = 40
    times_of_cluster = 200
    ari_noise = np.zeros(noise_level)
    aris = np.zeros(times_of_cluster)
    for i in range(noise_level):
        noise += 0.1
        ari = np.zeros(paras.epochs)

        # for j in range(20):
        test = np.load('/longlab/duan/pycharm_programs/formal_project/osl/1sub0noise90roi1200tr_TC.npy')[:,None]
        # test = np.random.random((1,1,90,1200))
        test = test + noise * np.random.randn(1,1,90,1200)
        test = torch.from_numpy(test).float().to(device)
        index = np.load('/longlab/duan/pycharm_programs/formal_project/osl/1sub0noise90roi1200tr_index.npy').squeeze()
        # 设定卷积核数量
        num_kernels = 10
        data = np.load(paras.all_tc)
        data_n = data + noise * np.random.random(data.shape)
        data_n = torch.from_numpy(data_n).float().to(device)
        data = torch.from_numpy(data).float().to(device)

        if data.shape.__len__() == 3:
            data = data.unsqueeze(dim=1)
        if data_n.shape.__len__() == 3:
            data_n = data_n.unsqueeze(dim=1)
        # 创建模型实例
        model = CNN(num_kernels=num_kernels).to(device)
        # 测试
        aris33 = []
        aris31 = []
        arin33 = []
        arin31=[]
        testn = test + 0 * torch.randn(1, 1, 90, 1200).to(device)
        for i in range(128):
            conv = nn.Conv2d(1,i+1,(3,3),1,(1,1),1,1,0).to(device)
            y=conv(testn)
            need = corshow(y.transpose(-1, -3).reshape(-1, 90, i+1).detach().cpu())
            need = np.nan_to_num(need, 1e-10)







            for _ in range(200):
                a, b = cluster(need, 4)
                # print('ARI = {}'.format(metrics.adjusted_rand_score(b, index)))
                # states_show(a,title='ARI = {}'.format(metrics.adjusted_rand_score(b, index)))
                aris33.append(metrics.adjusted_rand_score(b, index.reshape(-1)))
            arin33.append(np.max(aris33))
        for n in range(30):
            noise += 0.1
            aris33 = []
            testn =test +  noise * torch.randn(1, 1, 90, 1200).to(device)
            y, t = model(testn)
            need = corshow(y.transpose(-1, -3).reshape(-1, 90, 10).detach().cpu())
            need = np.nan_to_num(need, 1e-10)
            for i in range(200):
                a, b = cluster(need, 4)
                # print('ARI = {}'.format(metrics.adjusted_rand_score(b, index)))
                # states_show(a,title='ARI = {}'.format(metrics.adjusted_rand_score(b, index)))
                aris33.append(metrics.adjusted_rand_score(b, index.reshape(-1)))
            arin33.append(np.max(aris33))

            aris31 = []
            need = corshow(t.transpose(-1, -3).reshape(-1, 90, 10).detach().cpu())
            need = np.nan_to_num(need, 1e-10)
            for i in range(200):
                a, b = cluster(need, 4)
                # print('ARI = {}'.format(metrics.adjusted_rand_score(b, index)))
                # states_show(a,title='ARI = {}'.format(metrics.adjusted_rand_score(b, index)))
                aris31.append(metrics.adjusted_rand_score(b, index.reshape(-1)))
            arin31.append(np.max(aris31))
        # 测试结束
        # model.load_state_dict(torch.load('./model.pkl')['model_state_dict'])
        criterion = nn.MSELoss()
        # for j in range(times_of_cluster):
        #     aris[j] = tmp(test)
        # ari_noise[i] = aris.mean()
        # test += 0.1 * torch.randn_like(test)
        # np.save('./ari_noise.npy',ari_noise)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # kernals = []
        for epoch in range(paras.epochs):
            model.train()
            outputs,_ = model(data_n)
            # test = np.load('/longlab/duan/pycharm_programs/formal_project/osl/1sub0noise90roi1200tr_TC.npy')
            # index = np.load('/longlab/duan/pycharm_programs/formal_project/osl/1sub0noise90roi1200tr_index.npy')
            # ari = metrics.adjusted_rand_score(b, index)
            loss = criterion(data.squeeze(),outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch[{epoch + 1}/{paras.epochs}], Training Loss: {loss.item():.10f}')
        print(f'第{i}轮CNN结束，开始聚类')
        for x in range(times_of_cluster):
            aris[x] = tmp(test)
            print(aris[x])
        ari_noise[i] = aris.mean() - ari_noise[i]
        # ari[epoch] = aris.mean()
            # checkpoint = {"model_state_dict": model.state_dict(),
            #   "optimizer_state_dict": optimizer.state_dict(),
            #   "epoch": paras.epochs}
            # torch.save(checkpoint,'./model.pkl')
            # if epoch % 100 == 0:
            #     states_show(model.conv1.weight.data.detach().cpu(),title=f'{epoch}epoch')
            #     kernals.append(model.conv1.weight.data.detach().cpu())
        pass
        # print(ari[j])
        # np.save('./ari_{}noise.npy'.format(noise),ari)
        # checkpoint = {"model_state_dict": model.state_dict(),
        #   "optimizer_state_dict": optimizer.state_dict(),
        #   "epoch": paras.epochs}
        # torch.save(checkpoint,'./model.pkl')
    np.save('./ari_noise_cnn.npy', ari_noise)
    print('共用时{}秒'.format(time.time()- start_time))
    pass