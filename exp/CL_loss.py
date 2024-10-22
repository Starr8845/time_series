import torch as torch


# 这里不同的metric 需要看一下分布
def get_positive_mask(multi_y, device, threshold=0.2, y_sim_metric="dim1"):
    #multi_y: [bs, y_len, var_num]
    multi_y = multi_y.permute(2,0,1) # [var_num, bs, y_len]
    
    var_num, bs, y_len = multi_y.shape
    if y_sim_metric == "dim1":
        gt = multi_y[:, :, 0:1]
        mask_positives = ((torch.abs(gt.sub(gt.permute(0,2,1))) < threshold)).float().to(device)
    elif y_sim_metric == "weighted_sum":
        temp = torch.abs(multi_y.reshape(var_num, bs, 1, y_len)-multi_y.reshape(var_num, 1, bs, y_len))
        weight = torch.exp(torch.arange(-1*y_len+1, 1, dtype=torch.float32)).reshape(1,1,y_len)
        sim = (temp*weight).sum(dim=3)
        mask_positives = (sim < 2).float().to(device)
    elif y_sim_metric == "mse":
        temp = torch.abs(multi_y.reshape(var_num, bs, 1, y_len)-multi_y.reshape(var_num, 1, bs, y_len))
        mse = torch.mean(torch.square(temp), dim=-1)
        mask_positives = (mse < 0.05).float().to(device) # 先简单这么写，后面再改成topK等
    elif y_sim_metric == "IC":
        mean = multi_y.mean(dim=2, keepdim=True)  # 计算均值
        std = multi_y.std(dim=2, keepdim=True)    # 计算标准差
        normalized_data = (multi_y - mean) / std  # 标准化
        correlation = torch.matmul(normalized_data, normalized_data.transpose(1, 2)) / (multi_y.size(2) - 1)
        mask_positives = (correlation > 0.1).float().to(device)
    # 返回的mask_positives的大小: 预期为[var_num, bs, bs]
    return mask_positives


def pair_wise_sim(a,b,cos=True):
    # a,b: [bs, var_num, d]
    if cos:
        a = a/a.norm(dim=2, keepdim=True)
        b = b/b.norm(dim=2, keepdim=True)
    res = torch.einsum("bvd,Bvd->vbB", a, b)
    # res: [var_num, bs, bs]
    return res


# y的相似度 可以直接用第一个维度的距离；可以直接计算mse；可以计算IC；可以用表征的相似度

def multi_y_contrast_loss(features, multi_y, tau, y_sim_metric="dim1", loss_weight="all1", 
                          multi_y_repres=None, threshold=0.2, 
                          sim=None, prior_mask = None, all_negative=True, cos=False):
    # features: [bs, var_num, d]
    device = features.device
    batch_size, var_num, _ = features.shape
    
    mask_positives = get_positive_mask(multi_y, device, threshold, y_sim_metric=y_sim_metric) #[var_num, bs, bs]
    # [var_num, bs, bs]
    if prior_mask is not None:
        mask_positives = mask_positives * prior_mask
    if all_negative:
        mask_negatives = torch.ones_like(mask_positives, dtype=torch.float, device=device)
    else:
        # 负样本选择 这里先留空
        pass
        #mask_negatives = (torch.abs(gt.sub(gt.T)) > threshold+0.1).float().to(device)   # 这些样本作为negatives
    positive_num, negative_num = mask_positives.sum(), mask_negatives.sum()
    mask_neutral = mask_positives + mask_negatives
    
    anchor_dot_contrast = torch.div(pair_wise_sim(features,features,cos=cos),tau, )
    
    logits_max, _ = torch.max(anchor_dot_contrast, dim=2, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    # TODO: 怎么把对角线mask掉，暂时没写出来
    
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(2, keepdim=True) + 1e-20)
    
    if loss_weight=='all1':
        mean_log_prob_pos = (mask_positives * log_prob).sum(2) / (mask_positives.sum(2) + 1e-20)
    
    loss = -1 * mean_log_prob_pos
    loss = loss.view(var_num, batch_size).mean()
    return loss, (positive_num, negative_num)



if __name__=="__main__":
    bs = 1024
    var_num = 21
    # 多time series, 多
    repres = torch.rand(bs, var_num, 128)
    multi_y_truth = torch.rand(bs, 24, var_num) # 本身在在这个场景下，就是多步时间预测
    
    # cl_loss = multi_y_contrast_loss(repres, multi_y=multi_y_truth, tau=1, y_sim_metric="dim1")
    # print(cl_loss)
    # cl_loss = multi_y_contrast_loss(repres, multi_y=multi_y_truth, tau=1, y_sim_metric="weighted_sum")
    # print(cl_loss)
    # cl_loss = multi_y_contrast_loss(repres, multi_y=multi_y_truth, tau=1, y_sim_metric="mse")
    # print(cl_loss)
    # cl_loss = multi_y_contrast_loss(repres, multi_y=multi_y_truth, tau=1, y_sim_metric="IC")
    # print(cl_loss)
    
    
    d = 128
    features = torch.randn(bs, var_num, d)

    for y_sim_metric in ["dim1", "weighted_sum", "mse", "IC"]:
        mask_positives = get_positive_mask(multi_y_truth, multi_y_truth.device, threshold=0.2, y_sim_metric="dim1")
        print(y_sim_metric, mask_positives.shape)
        cl_loss, _ = multi_y_contrast_loss(features, multi_y_truth, tau=1, y_sim_metric=y_sim_metric, loss_weight="all1")
        print(cl_loss)


