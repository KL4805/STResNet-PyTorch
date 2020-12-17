import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import torch.optim as optim
import time

class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, lng, lat):
        # It takes in a four dimensional input (B, C, lng, lat)
        super(ResUnit, self).__init__()
        # self.ln1 = nn.LayerNorm(normalized_shape = (lng, lat))
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        # self.ln2 = nn.LayerNorm(normalized_shape = (lng, lat))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    
    def forward(self, x):
        # z = self.ln1(x)
        z = self.bn1(x)
        z = F.relu(z)
        z = self.conv1(z)
        # z = self.ln2(z)
        z = self.bn2(z)
        z = F.relu(z)
        z = self.conv2(z)
        return z + x


class STResNet(nn.Module):
    def __init__(self,
                 learning_rate=0.0001,
                 epoches=50,
                 batch_size=32,
                 len_closeness=3,
                 len_period=1,
                 len_trend=1,
                 external_dim=28,
                 map_heigh=32,
                 map_width=32,
                 nb_flow=2,
                 nb_residual_unit=2,
                 data_min = -999, 
                 data_max = 999):
        super(STResNet, self).__init__()
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.len_closeness = len_closeness
        self.len_period = len_period
        self.len_trend = len_trend
        self.external_dim = external_dim
        self.map_heigh = map_heigh
        self.map_width = map_width
        self.nb_flow = nb_flow
        self.nb_residual_unit = nb_residual_unit
        self.logger = logging.getLogger(__name__)
        self.data_min = data_min
        self.data_max = data_max
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu = torch.device("cuda:0")
        self._build_stresnet()
        self.save_path = "L%d_C%d_P%d_%T%d/" % (self.nb_residual_unit, self.len_closeness, self.len_period, self.len_trend)
        self.best_rmse = 999
        self.best_mae = 999


    def _build_stresnet(self, ):
        branches = ['c', 'p', 't']
        self.c_net = nn.ModuleList([
            nn.Conv2d(self.len_closeness * self.nb_flow, 64, kernel_size=3, stride = 1, padding = 1), 
        ])
        for i in range(self.nb_residual_unit):
            self.c_net.append(ResUnit(64, 64, self.map_heigh, self.map_width))
        self.c_net.append(nn.Conv2d(64, self.nb_flow, kernel_size=3, stride = 1, padding = 1))

        self.p_net = nn.ModuleList([
            nn.Conv2d(self.len_period * self.nb_flow, 64, kernel_size=3, stride = 1, padding = 1), 
        ])
        for i in range(self.nb_residual_unit):
            self.p_net.append(ResUnit(64, 64, self.map_heigh, self.map_width))
        self.p_net.append(nn.Conv2d(64, self.nb_flow, kernel_size=3, stride = 1, padding = 1))

        self.t_net = nn.ModuleList([
            nn.Conv2d(self.len_trend * self.nb_flow, 64, kernel_size=3, stride = 1, padding = 1), 
        ])
        for i in range(self.nb_residual_unit):
            self.t_net.append(ResUnit(64, 64, self.map_heigh, self.map_width))
        self.t_net.append(nn.Conv2d(64, self.nb_flow, kernel_size=3, stride = 1, padding = 1))

        self.ext_net = nn.Sequential(
            nn.Linear(self.external_dim, 10), 
            nn.ReLU(inplace = True),
            nn.Linear(10, self.nb_flow * self.map_heigh * self.map_width)
        )
        self.w_c = nn.Parameter(torch.rand((self.nb_flow, self.map_heigh, self.map_width)), requires_grad=True)
        self.w_p = nn.Parameter(torch.rand((self.nb_flow, self.map_heigh, self.map_width)), requires_grad=True)
        self.w_t = nn.Parameter(torch.rand((self.nb_flow, self.map_heigh, self.map_width)), requires_grad=True)



    def forward_branch(self, branch, x_in):
        for layer in branch:
            x_in = layer(x_in)
        return x_in

    def forward(self, xc, xp, xt, ext):
        c_out = self.forward_branch(self.c_net, xc)
        p_out = self.forward_branch(self.p_net, xp)
        t_out = self.forward_branch(self.t_net, xt)
        ext_out = self.ext_net(ext).view([-1, self.nb_flow, self.map_heigh, self.map_width])
        # FUSION
        res = self.w_c.unsqueeze(0) * c_out + \
                self.w_p.unsqueeze(0) * p_out + \
                self.w_t.unsqueeze(0) * t_out
        res += ext_out
        return torch.tanh(res)

    def train_model(self, train_loader, test_x, test_y):
        optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        start_time = time.time()
        epoch_loss = []
        for ep in range(self.epoches):
            self.train()
            for i, (xc, xp, xt, ext, y) in enumerate(train_loader):
                if self.gpu_available:
                    xc = xc.to(self.gpu)
                    xp = xp.to(self.gpu)
                    xt = xt.to(self.gpu)
                    ext = ext.to(self.gpu)
                    y = y.to(self.gpu)
                ypred = self.forward(xc, xp, xt, ext)
                loss = ((ypred - y) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                if i % 50 == 0:
                    print("[%.2fs] ep %d it %d, loss %.4f"%(time.time() - start_time, ep, i, loss.item()))
            print("[%.2fs] ep %d, loss %.4f"%(time.time() - start_time, ep, np.mean(epoch_loss)))
            epoch_loss = []
            test_rmse, test_mae = self.evaluate(test_x, test_y)
            print("[%.2fs] ep %d test rmse %.4f, mae %.4f" % (time.time() - start_time, ep, test_rmse, test_mae))
            if test_rmse < self.best_rmse:
                self.save_model("best")
                self.best_rmse = test_rmse
                self.best_mae = test_mae

    def evaluate(self, X_test, Y_test):
        """
        X_test: a quadruplle: (xc, xp, xt, ext)
        y_test: a label
        mmn: minmax scaler, has attribute _min and _max
        """
        self.eval()
        if self.gpu_available:
            for i in range(4):
                X_test[i] = X_test[i].to(self.gpu)
            Y_test = Y_test.to(self.gpu)
        with torch.no_grad():
            ypred = self.forward(X_test[0], X_test[1], X_test[2], X_test[3])
            rmse = ((ypred - Y_test) **2).mean().pow(1/2)
            mae = ((ypred - Y_test).abs()).mean()
            return rmse * (self.data_max - self.data_min) / 2, mae * (self.data_max - self.data_min) / 2
    
    def save_model(self, name):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.state_dict(), self.save_path + name + ".pt")
    
    def load_model(self, name):
        if not name.endswith(".pt"):
            name += ".pt"
        self.load_state_dict(torch.load(self.save_path + name))
            
