import argparse
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpunum", default="0", type=str, help="GPU number to train the model.")
parser.add_argument("-d", "--dataset", help="Name of the dataset.")
parser.add_argument("-k", "--rank_k", default=10, type=int, help="number of ranks of perturbation for covariance.")
parser.add_argument("-z", "--hidden_dim", default=500, type=int, help="number of hidden units for the encoder network.")
parser.add_argument("-m", "--sample_m", default=10, type=int,
                    help="number of samples used to construct the lower bound of ELBO.")
parser.add_argument("-b", "--nbits", type=int, help="Number of bits of the embedded vector.")
parser.add_argument("-s", "--seed", default=123, type=int, help="random seed")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout probability (0 means no dropout)")
parser.add_argument("--train_batch_size", default=100, type=int, help="training batch size")
parser.add_argument("--test_batch_size", default=100, type=int, help="testing batch size")
parser.add_argument("--num_epochs", default=100, type=int, help="num of epochs to run")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpunum
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils import load_dataset, TextDataset, retrieve_topk, compute_precision_at_k, straightThrough, load_dataset_old, \
    preprocess

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if not args.gpunum:
    parser.error("Need to provide the GPU number.")

if not args.dataset:
    parser.error("Need to provide the dataset.")

if not args.nbits:
    parser.error("Need to provide the number of bits.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if args.dataset in ['reuters', 'tmc']:
    single_label_flag = False
else:
    single_label_flag = True


class CorrSH(nn.Module):

    def __init__(self, feature_dim, latent_dim, device, slope=1.0, rank=1, hidden_dim=500, dropoutProb=0.1):
        super(CorrSH, self).__init__()

        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.dropoutProb = dropoutProb
        self.device = device
        self.slope = slope
        self.noise_slope = 1.0
        self.k = rank
        self.encoder = nn.Sequential(nn.Linear(self.feature_dim, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=dropoutProb),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=dropoutProb))

        self.h_to_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.h_to_std = nn.Linear(self.hidden_dim, self.latent_dim)
        self.h_to_U = nn.ModuleList([nn.Linear(self.hidden_dim, self.latent_dim) for i in range(rank)])
        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, self.feature_dim),
                                     nn.LogSoftmax(dim=1))

    def encode(self, doc_mat):
        # output the parameters (mean, std, u) for multivariate Gaussian,
        # where u is used for covariance perturbation.
        h = self.encoder(doc_mat)
        mu = self.h_to_mu(h)
        std = F.softplus(self.h_to_std(h))
        rs = []
        for i in range(self.k):
            rs.append((1 / self.k) * torch.tanh(self.h_to_U[i](h)))
        u_pert = tuple(rs)
        u_perturbation = torch.stack(u_pert, 2)
        return mu, std, u_perturbation

    def reparameterize(self, mu, std, u_pert):
        eps = torch.randn_like(mu)
        eps_corr = torch.randn((u_pert.shape[0], u_pert.shape[2], 1)).to(self.device)
        # Reparameterisation trick for low-rank-plus-diagonal Gaussian
        z = mu + eps * std + torch.matmul(u_pert, eps_corr).squeeze()
        
        # Transform to Bernoulli random variable with ST estimator
        # Here self.slope is used to accelerate training: It will approach step function as slope is increasing.
        # NOTE: through experiments we set slope to be constant and found it worked well.
        s = straightThrough(self.slope * z, is_logit=True, stochastic=True)
        return s

    def forward(self, document_mat):
        mu, std, u_pert = self.encode(document_mat)
        s = self.reparameterize(mu, std, u_pert)
        
        # inject Gaussian noise with annealing scale
        # However, with fixed scale we found it also worked well.
        eps = torch.randn_like(mu)
        logprob_w = self.decoder(s + eps * self.noise_slope)
        return s, mu, std, u_pert, logprob_w

    def set_slope(self, slope, noise_slope):
        self.slope = slope
        self.noise_slope = noise_slope

    def _get_hashcode(self, document_mat):
        mu, std, u_pert = self.encode(document_mat)
        # get deterministic hash codes for evaluation.
        s = straightThrough(self.slope * mu, is_logit=True, stochastic=False)
        return s

    def get_all_hash_code(self, train, test):
        train_zy = [(self._get_hashcode(xb.to(self.device)), yb) for xb, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self._get_hashcode(xb.to(self.device)), yb) for xb, yb in test]
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)

        train_b = train_z.type(torch.cuda.ByteTensor)
        test_b = test_z.type(torch.cuda.ByteTensor)
        del train_z
        del test_z

        return train_b, test_b, train_y, test_y

################################################################
##                                                            ##
##                      Prepare dataset                       ##
##                                                            ##
################################################################

data_dict = load_dataset(args.dataset)
num_train = data_dict["n_trains"]
num_test = data_dict["n_tests"]
feature_dim = data_dict["n_features"]
num_tags = data_dict["n_tags"]
train_dataset = TextDataset(data_dict["x_train"], data_dict["y_train"])
test_dataset = TextDataset(data_dict["x_test"], data_dict["y_test"])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
print("Training...")
print("feature dimensions:       {}".format(feature_dim))
print("number of tags:           {}".format(num_tags))
print("number of training cases: {}".format(num_train))
print("number of testing cases:  {}".format(num_test))

################################################################
##                                                            ##
##                 Load Model and Optimizer                   ##
##                                                            ##
################################################################

model = CorrSH(feature_dim, args.nbits, rank=args.rank_k, dropoutProb=args.dropout, device=device)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=10000, gamma=0.96)


################################################################
##                                                            ##
##                Set Hyper-Parameters                        ##
##                                                            ##
################################################################

# initial KL weight and noise scale
kl_weight = 0.
noise_slope = 1.0

# annealing rate
kl_step = 1 / (50 * args.rank_k + 50 * args.sample_m)
noise_step = 1 / (50 * args.rank_k + 50 * args.sample_m)

# slope for straight through estimator, here we set it to be constant (i.e., 1)
# but one can also use annealing techniques, as done in https://arxiv.org/abs/1609.01704 
slope = 1.0

m_samples = args.sample_m
eps = 1e-8
best_precision = 0
best_precision_epoch = 0


################################################################
##                                                            ##
##                Construct the lower bound                   ##
##                                                            ##
################################################################

def construct_multiple_samples(mean, std, u_pert, m_samples, slope):
    eps = torch.randn((mean.shape[0], m_samples, mean.shape[1])).to(device)
    eps_corr = torch.randn((u_pert.shape[0], m_samples, u_pert.shape[2], 1)).to(device)
    _z = mean.unsqueeze(1) + eps * std.unsqueeze(1) + torch.matmul(u_pert.unsqueeze(1), eps_corr).squeeze()
    z = slope * _z
    classes = torch.multinomial(torch.Tensor([[1 / m_samples] * m_samples]), mean.shape[0], replacement=True).to(device)
    ind = classes.repeat(mean.shape[1], 1).T.unsqueeze(1)
    selected_z = torch.gather(z, 1, ind)
    sample = straightThrough(selected_z, is_logit=True, stochastic=True)
    return sample, z


################################################################
##                                                            ##
##                 Training & Evaluation                      ##
##                                                            ##
################################################################

for epoch in range(args.num_epochs):
    avg_loss = []
    
    # noise slope annealing
    # kl weight annealing
    noise_slope = max(noise_slope - noise_step, 0.05)
    kl_weight = min(kl_weight + kl_step, 1.)
    model.set_slope(slope, noise_slope)
    constant = torch.log(torch.tensor(m_samples + 0.)).to(device)

    for step, (xb, yb) in enumerate(train_loader):
        model.train()
        xb = xb.to(device)
        yb = yb.to(device)
        s, mu, std, u_pert, logprob_w = model(xb)
        s_prime, z_k_samples = construct_multiple_samples(mu, std, u_pert, m_samples, slope)
        
        # compute expected log log likelihood
        logqs_r = torch.sum(z_k_samples * s_prime - F.softplus(z_k_samples), dim=2)
        masked = xb.clone()
        masked[masked != 0.] = 1.
        logpD_s = torch.sum(logprob_w * masked, dim=1)
                
        # We re-arrange the final objective such that our loss function consists of 
        # the expected log likelihood E_s[log p(x|s)], and
        # a negative KL divergence between constructed mixture h_k(s) and prior p(s).
        kl_div_h_p = torch.logsumexp(logqs_r - constant, dim=1) + s.shape[1] * np.log(2.)

        loss = -torch.mean(logpD_s - kl_weight * kl_div_h_p)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        avg_loss.append(loss.item())

    print('epoch:{} Training ELBO:{:.4f} Best Precision:({}){:.4f}'.format(epoch + 1, -np.mean(avg_loss),
                                                                           best_precision_epoch, best_precision))

    with torch.no_grad():
        model.eval()
        train_b, test_b, train_y, test_y = model.get_all_hash_code(train_loader, test_loader)
        retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
        prec = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100,
                                      is_single_label=single_label_flag)
        print("precision at 100: {:.4f}".format(prec.item()))

        # record best precision for early stopping or other techniques.
        if prec.item() > best_precision:
            best_precision = prec.item()
            best_precision_epoch = epoch + 1

with open('log.txt', 'a') as handle:
    handle.write(
        '{},{},{},{},{:.4f},{},{},{}\n'.format("CorrSH",
                                                  args.dataset,
                                                  args.nbits,
                                                  best_precision_epoch,
                                                  best_precision,
                                                  args.rank_k,
                                                  args.sample_m,
                                                  args.dropout
                                                  ))
