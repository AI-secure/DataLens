import numpy as np
import math
import sys

import torch
from sklearn.preprocessing import normalize
from pate_core import *
from numpy import linalg as LA

EPS = sys.float_info.epsilon


# Algorithm 1 in 'Scalable Private Learning with PATE'
def gnmax_thresh_aggregator(counts, thresh_cnt, sigma_thresh, sigma, orders):
    log_pr_answered = compute_logpr_answered(thresh_cnt, sigma_thresh, counts)
    rdp_budget = compute_rdp_threshold(log_pr_answered, sigma_thresh, orders)
    # print("Threshold budget:" + str(rdp_budget))

    if np.random.normal(np.max(counts), sigma_thresh) >= thresh_cnt:
        logq = compute_logq_gaussian(counts, sigma)
        res = np.argmax(np.random.normal(counts, sigma))
        g_rdp_budget = rdp_gaussian(logq, sigma, orders)
        rdp_budget += g_rdp_budget
    else:
        # do not return result if teacher models do not agree
        res = -1

    return res, rdp_budget


def gnmax_aggregator(counts, sigma, orders):
    logq = compute_logq_gaussian(counts, sigma)
    dir_index = np.argmax(np.random.normal(counts, sigma))
    rdp_budget = rdp_gaussian(logq, sigma, orders)
    return dir_index, rdp_budget


def rdp_percentile(arr_list, q, orders, vmin, vmax, lmbd, axis=0):
    arr_length = len(arr_list)
    arr_size = arr_list[0].size
    input_shape = arr_list[0].shape
    arr_reshaped = np.vstack([arr.reshape([1, arr_size]) for arr in arr_list])

    arr_ordered = np.sort(arr_reshaped, axis=0)
    arr_ordered = arr_ordered.clip(min=vmin, max=vmax)

    arr_ordered_new = np.vstack([np.ones([1, arr_size]) * vmin, arr_ordered, np.ones([1, arr_size]) * vmax])
    arr_ordered_new[np.abs(arr_ordered_new) < sys.float_info.epsilon] = 0

    n_teachers, n_feature = arr_reshaped.shape
    arr_prob = np.zeros([n_teachers + 1, n_feature])

    for i in range(arr_length + 1):
        diff = arr_ordered_new[i + 1, :] - arr_ordered_new[i, :]
        diff = diff.clip(min=0)
        arr_prob[i] = diff * np.exp(-0.5 / lmbd * abs(i - q / 100 * arr_length))
        # arr_prob[i] = np.exp(np.log(diff) - 0.5/lmbd * abs(i - q/100 * arr_length))

    # arr_prob = normalize(arr_prob, norm='l1', axis=0)

    if np.min(arr_prob) < 0:
        print(arr_prob)
        exit()

    low = np.zeros([1, arr_size])
    high = np.zeros([1, arr_size])

    for i in range(arr_size):
        prob = arr_prob[:, i] / np.sum(arr_prob[:, i])
        rindex = np.random.choice(arr_length + 1, p=prob)
        # print(rindex)

        low[0, i] = arr_ordered_new[rindex, i]
        high[0, i] = arr_ordered_new[rindex + 1, i]

    output_q = np.random.uniform(low=low, high=high, size=[1, arr_size])
    output_q = output_q.reshape(input_shape)

    rdp_budget = arr_size * np.multiply(
        1 / (orders - 1),
        np.log(
            np.multiply(np.divide(orders, 2 * orders - 1), np.exp((orders - 1) / lmbd)) \
            + np.multiply(np.divide(orders - 1, 2 * orders - 1), np.exp(-orders / lmbd))
        )
    )

    return output_q, rdp_budget


def rdp_winsorized_mean(arr_list, step_size, sigma_mean, sigma_percentile, orders, pca_mat=None):
    vmin = -step_size
    vmax = step_size

    flatten_arr = np.asarray([arr.flatten() for arr in arr_list])
    n_teachers, n_features = flatten_arr.shape

    if pca_mat is not None:
        # project to principal components
        flatten_arr = np.matmul(flatten_arr, pca_mat)
        n_features = flatten_arr.shape[1]

    q25, q25_budget = rdp_percentile(flatten_arr, 25, orders, vmin=vmin, vmax=vmax, lmbd=sigma_percentile)
    q75, q75_budget = rdp_percentile(flatten_arr, 75, orders, vmin=vmin, vmax=vmax, lmbd=sigma_percentile)

    arr_mean = np.mean(flatten_arr.clip(min=q25, max=q75), axis=0)

    arr_mean[np.sign(q75) != np.sign(q25)] = 0

    # when 75 percentile is smaller, update the model with the average of 75 and 25 percentile
    # quantile_mean = (q75 + q25) / 2
    arr_mean[q75 < q25] = 0

    update_index = np.nonzero(np.logical_and(np.sign(q75) == np.sign(q25), q75 > q25))
    q_range = q75 - q25

    sensitivity = LA.norm(q_range[update_index] / len(arr_list))

    gaussian_noise, mean_budget = gaussian_rdp(arr_mean[update_index], sensitivity, orders, sigma_mean)
    arr_mean[update_index] += gaussian_noise
    arr_mean[update_index] = arr_mean[update_index].clip(min=q25[update_index], max=q75[update_index])

    # for testing only
    # update_ratio = gaussian_noise.size / arr_mean.size
    # print("Update ratio: %.8f, norm: %.8f" % (update_ratio, sensitivity))

    rdp_budget = q25_budget + q75_budget + mean_budget

    if pca_mat is not None:
        # project res direction back to original axis
        arr_mean = np.matmul(arr_mean, np.transpose(pca_mat))

    return arr_mean.reshape(arr_list[0].shape), rdp_budget


def gradient_voting_nonprivate(output_list, step_size, nbins=10):
    n = len(output_list)
    flatten_arr = np.asarray([arr.flatten() for arr in output_list])
    n_teachers, n_features = flatten_arr.shape

    flatten_arr = flatten_arr.clip(min=-step_size, max=step_size)

    bins = np.arange(-step_size, step_size, (step_size * 2 / nbins))
    bins = np.hstack([bins, step_size])
    result = np.zeros([1, n_features])

    for i in range(n_features):
        votes_arr, _ = np.histogram(flatten_arr[:, i], bins)
        res_idx = np.argmax(votes_arr)
        result[:, i] = (bins[res_idx] + bins[res_idx + 1]) / 2

    return result.reshape(output_list[0].shape)


def gradient_voting_rdp(output_list, step_size, sigma, sigma_thresh, orders, pca_mat=None, nbins=10, thresh=0.9):
    import time
    st = time.time()
    n = len(output_list)
    use_gpu = False  # turn it on if you are running a huge matrix and the bottleneck lies on CPU matmul
    if use_gpu:
        # have to use torch==1.2.0 and torchvision==0.4.0 to run tensorflow-gpu==1.4.0
        import torch
        flatten_arr = torch.tensor([arr.flatten() for arr in output_list], device='cuda:0')
    else:
        flatten_arr = np.asarray([arr.flatten() for arr in output_list])
    n_teachers, n_features = flatten_arr.shape

    if pca_mat is not None:
        # project to principal components
        if use_gpu:
            pca_mat_tensor = torch.from_numpy(pca_mat).float().to('cuda:0')
            flatten_arr = torch.matmul(flatten_arr, pca_mat_tensor)
            flatten_arr = flatten_arr.cpu().numpy()
        else:
            flatten_arr = np.matmul(flatten_arr, pca_mat)
        n_features = flatten_arr.shape[1]

    flatten_arr = flatten_arr.clip(min=-step_size, max=step_size)

    bins = np.arange(-step_size, step_size, (step_size * 2 / nbins))
    bins = np.hstack([bins, step_size])
    result = np.zeros([1, n_features])

    rdp_budget = 0
    skipped_cnt = 0
    for i in range(n_features):
        votes_arr, _ = np.histogram(flatten_arr[:, i], bins)
        print(votes_arr)
        res_idx, cur_budget = gnmax_thresh_aggregator(votes_arr, thresh * n_teachers, sigma_thresh, sigma, orders)
        rdp_budget += cur_budget
        if res_idx < 0:
            skipped_cnt += 1
        else:
            result[:, i] = (bins[res_idx] + bins[res_idx + 1]) / 2
    print("Skipped %d feaatures out of %d" % (skipped_cnt, n_features))


    if pca_mat is not None:
        # project res direction back to original axis
        result = np.matmul(result, np.transpose(pca_mat))
    return result.reshape(output_list[0].shape), rdp_budget


def convert2topk(grad, topk):
    """
    :param grad: original gradient
    :param topk: topk we want to choose
    :return: voting array (+1/0/-1)
    """
    topk_ind = np.argpartition(np.abs(grad), -topk)[:, -topk:]
    votes = np.zeros_like(grad)
    sign_grad = np.sign(grad)
    for i in range(len(grad)):
        votes[i, topk_ind[i]] = 1
        votes[i] = sign_grad[i] * votes[i]
    return votes


def convert2topk_gpu(grad, topk):
    """
    :param grad:  sign grad (torch.tensor.cuda())
    :param topk:  topk value (int)
    :return: voted sign grad (torch.tensor.cuda())
    """
    topk_ind = torch.topk(torch.abs(grad), k=topk)[1]
    sign_grad = torch.sign(grad)
    votes = torch.zeros_like(grad).cuda()
    votes.scatter_(1, topk_ind, 1)
    votes = sign_grad * votes
    print(votes.type())
    return votes

def stachastic_convert2topk(grad, topk, b=None):
    abs_grad = np.abs(grad)
    topk_ind = np.argpartition(abs_grad, -topk)[:, -topk:]
    if b is None:
        # b = np.max(abs_grad, axis=0) ## DP proof assumes all the teachers to be independent
        b = np.max(abs_grad, axis=1)
    else:
        b = np.max(abs_grad, axis=1).clip(max=b)
    prob = 1/2 + (grad.T / b).T / 2  # prob of positive sign
    rand = np.random.rand(*prob.shape)
    sign_grad = np.ones_like(grad)
    sign_grad[rand > prob] = -1

    votes = np.zeros_like(grad)
    for i in range(len(votes)):
        votes[i, topk_ind[i]] = 1
        votes[i] = sign_grad[i] * votes[i]
    return votes


def stochastic_klevel(grad, b, k_level):
    from scipy.stats import special_ortho_group
    from tqdm import tqdm
    # for i in tqdm(range(len(grad))): ## extremely slow (1h to finish on CPU)
    #     clipped_grad = grad[i]  # clipping factor = 1
    #     rotation_matrix= special_ortho_group.rvs(len(grad[i]))
    #     grad[i] = rotation_matrix @ clipped_grad # gradient rotation

    interval = 2 * b / (k_level - 1)
    lower = -b
    lower_grad = - k_level / 2
    rand = np.random.rand(*grad.shape)
    votes = np.zeros_like(grad)

    for i in range(1, k_level):
        upper = lower + interval
        upper_grad = lower_grad + 1
        if i == 1:
            mask = (grad <= upper)
        elif i == k_level - 1:
            mask = (grad >= lower)
        else:
            mask = (grad <= upper) & (grad >= lower)

        print(f"level {i}: {np.sum(mask)}")
        print(f"lower_grad : {lower_grad}")
        print(f"interval: {interval}")
        prob = (grad[mask] - lower) / (upper - lower)
        prob_grad = np.full_like(prob, lower_grad)
        prob_grad[rand[mask] <= prob] = upper_grad
        votes[mask] = prob_grad

        lower = upper
        lower_grad = upper_grad

    return votes

def ablation_test_on_alpha_k(grad, topk):
    abs_grad = torch.abs(grad)
    topk_ind = torch.topk(abs_grad, k=topk)[1]

    votes = torch.zeros_like(grad).cuda()
    votes.scatter_(1, topk_ind, 1)
    masked_topk_grad = grad * votes

    residual_grad_norm = torch.norm(grad - masked_topk_grad, dim=1)
    grad_norm = torch.norm(grad, dim=1)

    alpha_k = 1 - residual_grad_norm / grad_norm
    return alpha_k.cpu().numpy()

def ablation_test_on_different_k(output_list, ckpt_dir='', epoch=0):
    grad = torch.tensor([arr.flatten() for arr in output_list]).cuda()
    dim = grad.shape[-1]
    alpha_k = []
    for i in range(5):
        topk = int(dim / 2**(i))
        alpha_k.append(ablation_test_on_alpha_k(grad, topk))
    alpha_k = np.vstack(alpha_k)
    import joblib
    import os
    save_dir = os.path.join(ckpt_dir, 'alpha_k_epoch_' + str(epoch) + '.pkl')
    joblib.dump(alpha_k, save_dir)

def stochastic_klevel_gpu(grad, b, k_level):
    # from scipy.stats import special_ortho_group
    # rotation_matrix = torch.tensor(special_ortho_group.rvs(len(grad[0])), dtype=torch.float32).cuda()
    # grad = torch.matmul(rotation_matrix, grad.T).T # gradient rotation

    interval = 2 * b / (k_level - 1)
    lower = -b
    lower_grad = - k_level / 2
    rand = torch.rand_like(grad).cuda()
    votes = torch.zeros_like(grad).cuda()

    for i in range(1, k_level):
        upper = lower + interval
        upper_grad = lower_grad + 1
        if i == 1:
            mask = (grad <= upper)
        elif i == k_level - 1:
            mask = (grad >= lower)
        else:
            mask = (grad <= upper) & (grad >= lower)

        print(f"level {i}: {torch.sum(mask)}")
        print(f"lower_grad : {lower_grad}")
        print(f"interval: {interval}")
        prob = (grad[mask] - lower) / (upper - lower)
        prob_grad = torch.full_like(prob, lower_grad)
        prob_grad[rand[mask] <= prob] = upper_grad
        votes[mask] = prob_grad

        lower = upper
        lower_grad = upper_grad

    return votes


def stachastic_convert2topk_gpu(grad, topk, b=None):
    """
    :param grad:  sign grad (torch.tensor.cuda())
    :param topk:  topk value (int)
    :return: voted sign grad (torch.tensor.cuda())
    """
    abs_grad = torch.abs(grad)
    topk_ind = torch.topk(abs_grad, k=topk)[1]
    if b is None:
        b = torch.max(abs_grad, dim=1)[0]
    else:
        b = torch.max(abs_grad, dim=1)[0].clamp(max=b)

    prob = 1/2 + (grad.T / b).T / 2  # prob of positive sign
    rand = torch.rand_like(prob).cuda()
    sign_grad = torch.ones_like(grad).cuda()
    sign_grad[rand > prob] = -1

    votes = torch.zeros_like(grad).cuda()
    votes.scatter_(1, topk_ind, 1)
    votes = sign_grad * votes

    sign_sgd = torch.sign(grad)
    biased_votes = torch.zeros_like(grad).cuda()
    biased_votes.scatter_(1, topk_ind, 1)
    biased_votes = sign_sgd * biased_votes
    print("prob topk", prob[0][topk_ind[0]])
    print("grad topk", grad[0][topk_ind[0]])
    print("sto sign topk", votes[0][topk_ind[0]])
    print("biased sign topk", biased_votes[0][topk_ind[0]])
    print("not agreed all sign:", torch.sum(votes != biased_votes))
    print("not agreed one sign:", torch.sum(votes[0][topk_ind[0]] != biased_votes[0][topk_ind[0]]))
    return votes


def stochastic_sketch_topk_gpu(grad, topk, b=None):
    """
    :param grad:  sign grad (torch.tensor.cuda())
    :param topk:  topk value (int)
    :return: voted sign grad (torch.tensor.cuda())
    """
    abs_grad = torch.abs(grad)
    if b is None:
        b = torch.max(abs_grad, dim=1)[0]
    else:
        b = torch.max(abs_grad, dim=1)[0].clamp(max=b)

    prob = 1/2 + (grad.T / b).T / 2  # prob of positive sign
    rand = torch.rand_like(prob).cuda()
    sign_grad = torch.ones_like(grad).cuda()
    sign_grad[rand > prob] = -1

    d = sign_grad.shape[1]
    c = 10000
    r = 20
    from csvec import CSVec
    sketch = CSVec(d, c, r)
    for grad in sign_grad:
        sketch.accumulateVec(grad)
    votes = sketch.unSketch(topk)
    print(votes.shape)
    return votes


def signsgd_aggregate(output_list, sigma, orders, topk, beta=0.1, alpha=1e-3, stochastic=False, b=None):
    use_gpu = True
    if not use_gpu:
        nteacher = len(output_list)
        flatten_grad = np.asarray([arr.flatten() for arr in output_list])
        if stochastic:
            voted_arr = np.sum(stachastic_convert2topk(flatten_grad, topk, b=b), axis=0)
        else:
            voted_arr = np.sum(convert2topk(flatten_grad, topk), axis=0)
    else:
        nteacher = len(output_list)
        flatten_grad = torch.tensor([arr.flatten() for arr in output_list]).cuda()
        if stochastic:
            voted_arr = torch.sum(stachastic_convert2topk_gpu(flatten_grad, topk, b=b), dim=0).cpu().numpy()
        else:
            voted_arr = torch.sum(convert2topk_gpu(flatten_grad, topk), dim=0).cpu().numpy()

    voted_arr = np.random.normal(voted_arr, sigma)
    logq = compute_logq_gaussian(voted_arr, sigma)
    rdp_budget = rdp_gaussian(logq, sigma / ((2*topk) ** 0.5), orders)
    sign_grad = np.zeros_like(voted_arr)

    sign_grad[voted_arr > beta * nteacher] = 1
    sign_grad[voted_arr < -beta * nteacher] = -1
    print("Agreed Dimension: " + str(np.sum(abs(sign_grad))))

    return alpha * sign_grad.reshape(output_list[0].shape), rdp_budget


def signsgd_aggregate_no_thresh(output_list, sigma, orders, topk, beta=0.1, alpha=1e-3, stochastic=False, b=None):
    use_gpu = True
    if not use_gpu:
        nteacher = len(output_list)
        flatten_grad = np.asarray([arr.flatten() for arr in output_list])
        if stochastic:
            voted_arr = np.sum(stachastic_convert2topk(flatten_grad, topk, b=b), axis=0)
        else:
            voted_arr = np.sum(convert2topk(flatten_grad, topk), axis=0)
    else:
        nteacher = len(output_list)
        flatten_grad = torch.tensor([arr.flatten() for arr in output_list]).cuda()
        if stochastic:
            voted_arr = torch.sum(stachastic_convert2topk_gpu(flatten_grad, topk, b=b), dim=0).cpu().numpy()
        else:
            voted_arr = torch.sum(convert2topk_gpu(flatten_grad, topk), dim=0).cpu().numpy()

    # noise, rdp_budget2 = gaussian_rdp(voted_arr, (4 * topk) **0.5, orders, sigma)
    # print("before adding noise:", voted_arr)
    voted_arr = np.random.normal(voted_arr, sigma)
    # print("after adding noise:", voted_arr)
    # voted_arr += noise
    logq = compute_logq_gaussian(voted_arr, sigma)
    ## l2-sensitivity is (4k)**0.5, while GNMax sensitivity is 2**0.5. Hence the factor is 2k**0.5
    rdp_budget = rdp_gaussian(logq, sigma / ((2*topk) ** 0.5), orders)
    # print(rdp_budget)
    # sign_grad = np.zeros_like(voted_arr)

    # sign_grad[voted_arr > beta * nteacher] = 1
    # sign_grad[voted_arr < -beta * nteacher] = -1
    # print("Agreed Dimension: " + str(np.sum(abs(sign_grad))))

    return alpha * voted_arr.reshape(output_list[0].shape) / topk, rdp_budget



def sketchtopk_aggregate(output_list, sigma, orders, topk, beta=0.1, alpha=1e-3, stochastic=False, b=None):
    nteacher = len(output_list)
    flatten_grad = torch.tensor([arr.flatten() for arr in output_list]).cuda()
    voted_arr = stochastic_sketch_topk_gpu(flatten_grad, topk, b=b).cpu().numpy()

    # noise, rdp_budget2 = gaussian_rdp(voted_arr, (4 * topk) **0.5, orders, sigma)
    # print("before adding noise:", voted_arr)
    voted_arr = np.random.normal(voted_arr, sigma)
    # print("after adding noise:", voted_arr)
    # voted_arr += noise
    logq = compute_logq_gaussian(voted_arr, sigma)
    ## l2-sensitivity is (4k)**0.5, while GNMax sensitivity is 2**0.5. Hence the factor is 2k**0.5
    rdp_budget = rdp_gaussian(logq, sigma / ((2*topk) ** 0.5), orders)
    # print(rdp_budget)
    sign_grad = np.zeros_like(voted_arr)

    sign_grad[voted_arr > beta * nteacher] = 1
    sign_grad[voted_arr < -beta * nteacher] = -1
    print("Agreed Dimension: " + str(np.sum(abs(sign_grad))))

    return alpha * sign_grad.reshape(output_list[0].shape), rdp_budget

def k_level_sgd_aggregate(output_list, sigma, orders, k_level, beta=0.1, alpha=1e-3, b=None):
    use_gpu = False
    if not use_gpu:
        nteacher = len(output_list)
        flatten_grad = np.asarray([arr.flatten() for arr in output_list])
        voted_arr = np.sum(stochastic_klevel(flatten_grad, k_level=k_level, b=b), axis=0)
    else:
        nteacher = len(output_list)
        flatten_grad = torch.tensor([arr.flatten() for arr in output_list]).cuda()
        voted_arr = torch.sum(stochastic_klevel_gpu(flatten_grad, k_level=k_level, b=b), dim=0).cpu().numpy()

    # noise, rdp_budget2 = gaussian_rdp(voted_arr, (4 * topk) **0.5, orders, sigma)
    # print("before adding noise:", voted_arr)
    voted_arr = np.random.normal(voted_arr, sigma)
    # print("after adding noise:", voted_arr)
    # voted_arr += noise
    logq = compute_logq_gaussian(voted_arr, sigma)
    dim = voted_arr.shape[0]
    ## l2-sensitivity is (k_level^2 * dim)**0.5, while GNMax sensitivity is 2**0.5. Hence the factor is 2k**0.5
    rdp_budget = rdp_gaussian(logq, sigma / ((k_level**2 * dim) ** 0.5), orders)
    # print(rdp_budget)
    sign_grad = np.zeros_like(voted_arr)

    sign_grad[voted_arr > beta * nteacher] = 1
    sign_grad[voted_arr < -beta * nteacher] = -1
    print("Agreed Dimension: " + str(np.sum(abs(sign_grad))))

    return alpha * sign_grad.reshape(output_list[0].shape), rdp_budget


def signsgd_aggregate_dept(output_list, sigma, orders, topk, beta=0.1, alpha=1e-3, stochastic=False, b=None):
    use_gpu = True
    if not use_gpu:
        nteacher = len(output_list)
        flatten_grad = np.asarray([arr.flatten() for arr in output_list])
        if stochastic:
            voted_arr = np.sum(stachastic_convert2topk(flatten_grad, topk, b=b), axis=0)
        else:
            voted_arr = np.sum(convert2topk(flatten_grad, topk), axis=0)
    else:
        nteacher = len(output_list)
        flatten_grad = torch.tensor([arr.flatten() for arr in output_list]).cuda()
        if stochastic:
            voted_arr = torch.sum(stachastic_convert2topk_gpu(flatten_grad, topk, b=b), dim=0).cpu().numpy()
        else:
            voted_arr = torch.sum(convert2topk_gpu(flatten_grad, topk), dim=0).cpu().numpy()

    # noise, rdp_budget2 = gaussian_rdp(voted_arr, (4 * topk) **0.5, orders, sigma)
    # print("before adding noise:", voted_arr)
    voted_arr = np.random.normal(voted_arr, sigma)
    # print("after adding noise:", voted_arr)
    # voted_arr += noise
    logq = compute_logq_gaussian(voted_arr, sigma)
    ## l2-sensitivity is (4k)**0.5, while GNMax sensitivity is 2**0.5. Hence the factor is 2k**0.5
    # rdp_budget = rdp_gaussian(logq, sigma / ((2*topk) ** 0.5), orders)
    rdp_budget, dept_rdp_budget = double_rdp_gaussian(logq, sigma / ((2*topk) ** 0.5), orders)
    # print(rdp_budget)
    sign_grad = np.zeros_like(voted_arr)

    sign_grad[voted_arr > beta * nteacher] = 1
    sign_grad[voted_arr < -beta * nteacher] = -1
    print("Agreed Dimension: " + str(np.sum(abs(sign_grad))))

    return alpha * sign_grad.reshape(output_list[0].shape), rdp_budget, dept_rdp_budget



def gradient_voting_rdp_multiproj(output_list, step_size, sigma, sigma_thresh, orders, pca_mats=None, nbins=10, thresh=0.9):
    n = len(output_list)
    flatten_arr = np.asarray([arr.flatten() for arr in output_list])
    n_teachers, n_features = flatten_arr.shape
    print("flatten arr shape", flatten_arr.shape)

    if pca_mats is not None:
        # project to principal components
        split_flatten_arr = np.split(flatten_arr, len(pca_mats), axis=1)
        reduced_flatten_arr = []
        for pca_mat, arr in zip(pca_mats, split_flatten_arr):
            print("arr shape", arr.shape)
            print("pca shape", pca_mat.shape)
            arr = np.matmul(arr, pca_mat)
            reduced_flatten_arr.append(arr)
        flatten_arr = np.concatenate(reduced_flatten_arr, axis=1)
        n_features = flatten_arr.shape[1]

    flatten_arr = flatten_arr.clip(min=-step_size, max=step_size)

    bins = np.arange(-step_size, step_size, (step_size * 2 / nbins))
    bins = np.hstack([bins, step_size])
    result = np.zeros([1, n_features])

    rdp_budget = 0
    skipped_cnt = 0
    for i in range(n_features):
        votes_arr, _ = np.histogram(flatten_arr[:, i], bins)
        print(votes_arr)
        res_idx, cur_budget = gnmax_thresh_aggregator(votes_arr, thresh * n_teachers, sigma_thresh, sigma, orders)
        rdp_budget += cur_budget
        if res_idx < 0:
            skipped_cnt += 1
        else:
            result[:, i] = (bins[res_idx] + bins[res_idx + 1]) / 2

    print("Skipped %d feaatures out of %d" % (skipped_cnt, n_features))

    if pca_mat is not None:
        # project res direction back to original axis
        split_results = np.split(result, len(pca_mats), axis=1)
        final_results = []
        for split_result, pca_mat in zip(split_results, pca_mats):
            final_results.append(np.matmul(split_result, np.transpose(pca_mat)))
        final_results = np.concatenate(final_results, axis=1)
    return final_results.reshape(output_list[0].shape), rdp_budget


def gradient_sign_rdp(output_list, step_size, sigma, sigma_thresh, orders, pca_mat=None, thresh=0.9):
    n = len(output_list)
    flatten_arr = np.asarray([arr.flatten() for arr in output_list])
    n_teachers, n_features = flatten_arr.shape

    if pca_mat is not None:
        # project to principal components
        flatten_arr = np.matmul(flatten_arr, pca_mat)
        n_features = flatten_arr.shape[1]

    # first line for positive votes, second line for negative votes
    votes_arr = np.zeros([2, n_features])
    votes_sign = np.sign(flatten_arr)
    # counts for positive votes
    votes_arr[0, :] = np.sum(votes_sign[votes_sign > 0], axis=0)
    # counts for negative votes 
    votes_arr[1, :] = -np.sum(votes_sign[votes_sign < 0], axis=0)

    res_dir = np.zeros([1, n_features])

    rdp_budget = 0
    skipped_cnt = 0
    for i in range(n_features):
        dir_index, cur_budget = gnmax_thresh_aggregator(votes_arr[:, i], thresh * n_teachers, sigma_thresh, sigma,
                                                        orders)
        if dir_index == 0:
            res_dir[0, i] = step_size
        elif dir_index == 1:
            res_dir[0, i] = -step_size
        else:
            skipped_cnt += 1
        rdp_budget += cur_budget

    print("Skipped %d feaatures out of %d" % (skipped_cnt, n_features))

    if pca_mat is not None:
        # project res direction back to original axis
        res_dir = np.matmul(res_dir, np.transpose(pca_mat))

    return res_dir.reshape(output_list[0].shape), rdp_budget


def gradient_rdp(output_list, step_size, sigma, orders, pca_mat=None, thresh=None, sigma_thresh=1):
    n = len(output_list)
    flatten_arr = np.asarray([arr.flatten() for arr in output_list])
    n_teachers, n_features = flatten_arr.shape

    if pca_mat is not None:
        # project to principal components
        flatten_arr = np.matmul(flatten_arr, pca_mat)
        n_features = flatten_arr.shape[1]

    # first half votes for positive direction, second half votes for negative direction
    votes_arr = np.zeros([n_teachers, n_features * 2])
    max_index = np.argmax(np.abs(flatten_arr), axis=1)

    for i in range(n_teachers):
        if flatten_arr[i, max_index[i]] > 0:
            votes_arr[i, max_index[i]] = 1
        else:
            votes_arr[i, max_index[i] + n_features] = 1

    votes_count = np.sum(votes_arr, axis=0)

    if thresh is None:
        dir_index, rdp_budget = gnmax_aggregator(votes_count, sigma, orders)
    else:
        dir_index, rdp_budget = gnmax_thresh_aggregator(votes_count, thresh * n_teachers, sigma_thresh, sigma, orders)

    max_votes = np.max(votes_count)
    selected_votes = votes_count[dir_index]
    # print("Max cnt: %d, selected cnt: %d" % (max_votes, selected_votes))

    res_dir = np.zeros([1, n_features])

    if dir_index < n_features and dir_index >= 0:
        res_dir[0, dir_index] = step_size
    elif dir_index >= n_features:
        res_dir[0, dir_index - n_features] = -step_size
    else:
        print("Teachers don't agree. Skip...")

    if pca_mat is not None:
        # project res direction back to original axis
        res_dir = np.matmul(res_dir, np.transpose(pca_mat))

    return res_dir.reshape(output_list[0].shape), rdp_budget


def gaussian_rdp(arr, sensitivity, orders, sigma):
    gaussian_noise = np.random.normal(loc=np.zeros(arr.shape), scale=sigma * sensitivity, size=arr.shape)

    # Table 2 @ https://arxiv.org/pdf/1702.07476.pdf
    rdp_budget = [o / ((2 * sigma) ** 2) for o in orders]

    return gaussian_noise, rdp_budget
