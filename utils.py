import numpy as np
import scipy.io
import torch
import torch.utils.data
from torch import sigmoid
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm
import scipy.sparse
import pickle
import numpy as np
import os

def load_dataset(dataset_name):
    dataset = scipy.io.loadmat("dataset/" + dataset_name+".tfidf.mat")
    x_train = dataset['train']
    x_test = dataset['test']
    x_cv = dataset['cv']
    y_train = dataset['gnd_train']
    y_test = dataset['gnd_test']
    y_cv = dataset['gnd_cv']
    train = x_train.toarray().astype(np.float32)
    test = x_test.toarray().astype(np.float32)
    cv = x_cv.toarray().astype(np.float32)

    gnd_train = y_train.astype(np.float32)
    gnd_test = y_test.astype(np.float32)
    gnd_cv = y_cv.astype(np.float32)
    dataset = {"n_trains": train.shape[0], "n_tests": test.shape[0], "n_cv": cv.shape[0],
               "n_tags": gnd_train.shape[1], "n_features": train.shape[1], "x_train": train, "y_train": gnd_train,
               "x_test": test, "y_test": gnd_test, "x_cv": cv, "y_cv": gnd_cv}
    return dataset


class TextDataset(Dataset):

    def __init__(self, x_train, y_train):
        self.x_train = torch.from_numpy(x_train)
        self.y_train = torch.from_numpy(y_train)

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return (self.x_train[idx], self.y_train[idx])


def retrieve_topk(query_b, doc_b, topK, batch_size=100):
    n_bits = doc_b.size(1)
    n_train = doc_b.size(0)
    n_test = query_b.size(0)

    topScores = torch.cuda.ByteTensor(n_test, topK + batch_size).fill_(n_bits + 1)
    topIndices = torch.cuda.LongTensor(n_test, topK + batch_size).zero_()

    testBinmat = query_b.unsqueeze(2)

    for batchIdx in tqdm(range(0, n_train, batch_size), ncols=0, leave=False):
        s_idx = batchIdx
        e_idx = min(batchIdx + batch_size, n_train)
        numCandidates = e_idx - s_idx

        trainBinmat = doc_b[s_idx:e_idx]
        trainBinmat.unsqueeze_(0)
        trainBinmat = trainBinmat.permute(0, 2, 1)
        trainBinmat = trainBinmat.expand(testBinmat.size(0), n_bits, trainBinmat.size(2))

        testBinmatExpand = testBinmat.expand_as(trainBinmat)

        scores = (trainBinmat ^ testBinmatExpand).sum(dim=1)
        indices = torch.arange(start=s_idx, end=e_idx, step=1).type(torch.cuda.LongTensor).unsqueeze(0).expand(n_test,
                                                                                                               numCandidates)

        topScores[:, -numCandidates:] = scores
        topIndices[:, -numCandidates:] = indices

        topScores, newIndices = topScores.sort(dim=1)
        topIndices = torch.gather(topIndices, 1, newIndices)

    return topIndices


def compute_precision_at_k(retrieved_indices, query_labels, doc_labels, topK, is_single_label):
    n_test = query_labels.size(0)

    Indices = retrieved_indices[:, :topK]
    # if is_single_label:
    #     test_labels = query_labels.unsqueeze(1).expand(n_test, topK, query_labels.size(-1))
    #     topTrainLabels = [torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0) for idx in range(0, n_test)]
    #     topTrainLabels = torch.cat(topTrainLabels, dim=0)
    #     relevances = (test_labels == topTrainLabels).sum(dim=2)
    #     relevances = (relevances == query_labels.size(-1)).type(torch.cuda.FloatTensor)
    # else:
    topTrainLabels = [torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze(0) for idx in range(0, n_test)]
    topTrainLabels = torch.cat(topTrainLabels, dim=0).type(torch.cuda.ShortTensor)
    test_labels = query_labels.unsqueeze(1).expand(n_test, topK, topTrainLabels.size(-1)).type(
        torch.cuda.ShortTensor)
    relevances = (topTrainLabels & test_labels).sum(dim=2)
    relevances = (relevances > 0).type(torch.cuda.FloatTensor)

    true_positive = relevances.mean(dim=1)
    prec_at_k = torch.mean(true_positive)
    return prec_at_k


def straightThrough(logit, is_logit=True, stochastic=False):
    shape = logit.size()
    if is_logit:
        prob = sigmoid(logit)
    else:
        prob = logit
    if stochastic:
        random = torch.rand_like(prob)
        output_binary = prob.data.new(*shape).zero_().add(((prob - random) > 0.).float())
    else:
        output_binary = prob.data.new(*shape).zero_().add((prob > 0.5).float())
    output = Variable(output_binary - prob.data) + prob
    return output
