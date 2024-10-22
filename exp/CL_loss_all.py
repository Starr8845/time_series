# 这里的cl的处理方法和memory network一样，把所有的variable一视同仁，展开为bs * variable
import torch as torch


def get_positive_mask(multi_y, device, threshold=0.1, y_sim_metric="dim1"):
    n, y_len = multi_y.shape
    
    if y_sim_metric == "dim1":
        gt = multi_y[:, 0:1]
        mask_positives = ((torch.abs(gt.sub(gt.T)) < threshold)).float().to(device)
    elif y_sim_metric == "weighted_sum":
        temp = torch.abs(multi_y.reshape(n, 1, y_len)-multi_y.reshape(1, n, y_len))
        weight = torch.exp(torch.arange(-1*y_len+1, 1, dtype=torch.float32)).reshape(1,1,y_len)
        sim = (temp*weight).sum(dim=2)
        mask_positives = (sim < 0.2).float().to(device)
    elif y_sim_metric == "mse":
        temp = torch.abs(multi_y.reshape(n, 1, y_len)-multi_y.reshape(1, n, y_len))
        mse = torch.mean(torch.square(temp), dim=-1)
        mask_positives = (mse < 0.1).float().to(device) 
    elif y_sim_metric == "IC":
        correlation = torch.corrcoef(multi_y)
        mask_positives = (correlation > 0.3).float().to(device)
    elif y_sim_metric == "IC_mse":
        correlation = torch.corrcoef(multi_y)
        mask_positives1 = (correlation > 0.3).float().to(device)
        temp = torch.abs(multi_y.reshape(n, 1, y_len)-multi_y.reshape(1, n, y_len))
        mse = torch.mean(torch.square(temp), dim=-1)
        mask_positives2 = (mse < 0.1).float().to(device) 
        mask_positives = mask_positives1 * mask_positives2
    # 返回的mask_positives的大小: 预期为[n, n]
    return mask_positives        


def multi_y_contrast_loss_(features, multi_y, tau=1, loss_weight="all1", y_sim_metric="dim1",
                          multi_y_repres=None, threshold=0.1,
                          sim=None, prior_mask = None, all_negative=False, cos=False):
    # features: [bs, var_num, d]
    features = features.reshape(-1, features.shape[2])
    # multi_y: [bs, y_len, var_num]
    multi_y = multi_y.permute(0,2,1)  # [var_num, bs, y_len]
    multi_y = multi_y.reshape(-1, multi_y.shape[2])
    gt = multi_y[:, 0:1]
    device = features.device
    batch_size = features.shape[0]
    
    mask_positives = get_positive_mask(multi_y, device, threshold, y_sim_metric=y_sim_metric)
    if prior_mask is not None:
        mask_positives = mask_positives * prior_mask
    if all_negative:
        mask_negatives = torch.ones_like(mask_positives, dtype=torch.float, device=device)
    else:
        mask_negatives = (torch.abs(gt.sub(gt.T)) > threshold+0.1).float().to(device)   # 这些样本作为negatives
    positive_num, negative_num = mask_positives.sum(), mask_negatives.sum()
    mask_neutral = mask_positives + mask_negatives
    if cos:
        anchor_dot_contrast = torch.div(pair_wise_cos(features,features),tau)
    else:
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), tau)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    logits_mask = torch.scatter(
        torch.ones_like(mask_positives), 1,
        torch.arange(batch_size).view(-1, 1).to(device), 0) * mask_neutral
    mask_positives = mask_positives * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)
    
    if loss_weight=='all1':
        mean_log_prob_pos = (mask_positives * log_prob).sum(1) / (mask_positives.sum(1) + 1e-20)
    elif loss_weight=="param":
        sim = torch.exp(torch.mm(multi_y_repres, multi_y_repres.T))  # 参数化的weight
        mean_log_prob_pos = (mask_positives * log_prob* sim).sum(1) / ((mask_positives*sim).sum(1) + 1e-20)
    elif loss_weight=="rule_gaze":
        sim = -1*torch.log(torch.abs(gt.sub(gt.T))+1e-2)  # 根据相似度定义的权重        
        mean_log_prob_pos = (mask_positives * log_prob* sim).sum(1) / ((mask_positives*sim).sum(1) + 1e-20)
    elif loss_weight=="rule_proxy": # 使用高斯核来定义权重
        sim = torch.exp(-1*torch.square(gt.sub(gt.T))/2)  # 根据相似度定义的权重        
        mean_log_prob_pos = (mask_positives * log_prob* sim).sum(1) / ((mask_positives*sim).sum(1) + 1e-20)

    elif loss_weight=="x_sim":
        mean_log_prob_pos = (mask_positives * log_prob* sim).sum(1) / ((mask_positives*sim).sum(1) + 1e-20)
    
    loss = -1 * mean_log_prob_pos
    loss = loss.view(1, batch_size).mean()
    return loss, (positive_num, negative_num)

# 如果存在多个表征，对每个表征分别进行类似的处理
def multi_y_contrast_loss(features, multi_y, tau=1, loss_weight="all1", y_sim_metric="dim1",
                          multi_y_repres=None, threshold=0.1,
                          sim=None, prior_mask = None, all_negative=False, cos=False):
    if type(features)==tuple:
        loss, positive_num, negative_num = 0., 0., 0.
        for i in range(len(features)):
            loss_, (pos_num, neg_num) = multi_y_contrast_loss_(features[i], multi_y, tau, loss_weight, y_sim_metric,
                          multi_y_repres, threshold, sim, prior_mask[i] if type(prior_mask)==tuple else prior_mask, all_negative, cos)
            loss += loss_
            pos_num += positive_num
            neg_num += negative_num
        return loss, (positive_num, negative_num)
    else:
        return multi_y_contrast_loss_(features, multi_y, tau, loss_weight, y_sim_metric,
                          multi_y_repres, threshold, sim, prior_mask, all_negative, cos)
    

def pair_wise_cos(a,b):
    a_norm = a/a.norm(dim=1)[:, None]
    b_norm = b/b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    return res



if __name__=="__main__":
    bs = 64
    var_num = 21
    
    repres = torch.rand(bs, var_num, 128)
    multi_y_truth = torch.rand(bs, 24, var_num) # 本身在在这个场景下，就是多步时间预测
    
    d = 128
    features = torch.randn(bs, var_num, d)

    for y_sim_metric in ["dim1", "weighted_sum", "mse", "IC"]:
        # mask_positives = get_positive_mask(multi_y_truth, multi_y_truth.device, threshold=0.2, y_sim_metric="dim1")
        # print(y_sim_metric, mask_positives.shape)
        cl_loss, (pos_num, neg_num) = multi_y_contrast_loss(features, multi_y_truth, tau=1, y_sim_metric=y_sim_metric, loss_weight="all1")
        print(cl_loss, pos_num/(bs*var_num), neg_num/(bs*var_num))



