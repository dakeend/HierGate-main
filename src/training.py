import random
import numpy as np
import torch
import torch.nn.functional as F

from src.loss import unbiased_curriculum_loss,unbiased_curriculum_loss_self
from torch.nn.functional import mse_loss
from torchmetrics.functional import pearson_corrcoef


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def CrossEntropy(outputs, targets, temperature=3.0):
    """KL散度损失函数 - 用于知识蒸馏"""
    log_softmax_outputs = F.log_softmax(outputs/temperature, dim=1)
    softmax_targets = F.softmax(targets/temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def self_distillation_loss(layer_predictions, layer_features_be, layer_features_af, targets, 
                          loss_coefficient=0.3, feature_loss_coefficient=0.03, 
                          temperature=3.0, criterion=None,args=None):
    """
    自蒸馏损失计算
    
    Args:
        layer_predictions: 所有层的预测结果列表 [pred_layer1, pred_layer2, ..., pred_final]
        layer_features_be: before特征列表
        layer_features_af: after特征列表  
        targets: 真实标签
        loss_coefficient: KL散度损失系数
        feature_loss_coefficient: 特征蒸馏损失系数
        temperature: 蒸馏温度
        criterion: 基础损失函数
    """
    if criterion is None:
        criterion = torch.nn.MSELoss()

    total_loss = torch.tensor(0.0, device=targets.device)
    
    # 最深层分类器的输出（teacher）
    teacher_output = layer_predictions[-1].detach()  # 最后一个是最深层
    
    # Loss Source 1: 最深层分类器的标准损失
    pre=layer_predictions[-1]
    # final_loss = unbiased_curriculum_loss_self(layer_predictions[-1], targets, args, criterion, scheduler='linear')
    final_loss = criterion(layer_predictions[-1], targets)
    total_loss += final_loss
    
    # 对浅层分类器计算损失
    for i in range(len(layer_predictions) - 1):  # 排除最深层
        student_output = layer_predictions[i]
        
        # Loss Source 1: 浅层分类器对真实标签的损失
        # ce_loss= unbiased_curriculum_loss_self(student_output, targets, args, criterion, scheduler='linear')
        ce_loss = criterion(student_output, targets)
        total_loss += ce_loss * (1 - loss_coefficient)
        
        # Loss Source 2: KL散度损失（浅层向深层学习）
        # 对于回归任务，我们可以使用MSE作为替代
        # 或者将回归输出转换为分布进行KL散度计算
        # kl_loss = unbiased_curriculum_loss_self(student_output, teacher_output, args, criterion, scheduler='linear')

        kl_loss = F.mse_loss(student_output, teacher_output)
        total_loss += kl_loss * loss_coefficient
        
        # Loss Source 3: L2特征损失（特征图对齐）
        # 这里我们需要确保特征维度一致

        # if i > 0 and layer_features_be and layer_features_af:  # 不对最浅层应用特征损失
        #     try:
        #         # 计算特征差异
        #         student_feat_be = layer_features_be[i] if i < len(layer_features_be) else layer_features_be[-1]
        #         teacher_feat_be = layer_features_be[-1]  # 最深层特征
        #         student_feat_af = layer_features_af[i] if i < len(layer_features_af) else layer_features_af[-1]
        #         teacher_feat_af = layer_features_af[-1]
        #
        #         student_feat_be = F.normalize(student_feat_be, p=2, dim=1)
        #         teacher_feat_be = F.normalize(teacher_feat_be, p=2, dim=1)
        #         student_feat_af = F.normalize(student_feat_af, p=2, dim=1)
        #         teacher_feat_af = F.normalize(teacher_feat_af, p=2, dim=1)
        #         # 特征蒸馏损失
        #         feat_loss_be = torch.dot(student_feat_be.view(-1), teacher_feat_be.view(-1))
        #         feat_loss_af = torch.dot(student_feat_af.view(-1), teacher_feat_af.view(-1))
        #         feat_loss = (feat_loss_be + feat_loss_af) /4000
        #
        #         total_loss += feat_loss * feature_loss_coefficient
        #     except Exception as e:
        #         # 如果特征维度不匹配，跳过特征损失
        #         print(f"特征损失计算出错: {e}")
        #         pass
    
    return total_loss


def train(args, epoch, model, train_loader, valid_loader, device, criterion, optimizer):
    # 检查是否使用自蒸馏
    use_self_distill = hasattr(model, 'self_distill') and model.self_distill
    
    args.epoch = epoch
    model.train()
    total_train_loss = 0
    train_data_size = 0

    abs_correct_rate = [0, 0, 0]
    re_correct_rate = [0, 0, 0]
    curri1 = []
    curri = []
    encodings, labels = [], []

    for data in train_loader:
        data = data.to(device)
        
        if use_self_distill:
            # 自蒸馏模式
            layer_predictions, layer_features_be, layer_features_af = model(data, epoch)
            
            # 计算自蒸馏损失
            loss = self_distillation_loss(
                layer_predictions=layer_predictions,
                layer_features_be=layer_features_be,
                layer_features_af=layer_features_af,
                targets=data.y,
                loss_coefficient=getattr(args, 'loss_coefficient', 0.3),
                feature_loss_coefficient=getattr(args, 'feature_loss_coefficient', 0.03),
                temperature=getattr(args, 'temperature', 3.0),
                criterion=criterion,
                args=args
            )
            # loss=criterion(layer_predictions[-1], data.y)
            
            # 用最深层的预测计算数据大小
            out = layer_predictions[-1]
            
        else:
            # 原始模式
            if args.fds:
                out, feature = model(data, epoch)
                encodings.extend(feature.data.cpu().numpy())
                labels.extend(data.y.data.cpu().numpy())
            elif args.contrast_curri:
                out, similarity = model(data, epoch)
            else:
                out = model(data)

            if args.loss == "WeightedMSELoss()":
                loss = criterion(out, data.y, data.wy)
            elif 'curri' in args.loss:
                if args.contrast_curri:
                    loss_list = []
                    diff_loss_list = []
                    diff_simi_list = []
                    for idx in range(out.shape[0]):
                        gt = abs(data.y[idx].item())
                        gt = 1 if gt < 1 else gt
                        temp_loss = criterion(out[idx], data.y[idx])
                        loss_list.append(temp_loss)
                        diff_loss_list.append(round(temp_loss.item() / gt, 3))
                        diff_simi_list.append(round(similarity[idx].item(), 3))
                    mean_simi, std_simi = np.mean(diff_simi_list), np.std(diff_simi_list)
                    mean_loss, std_loss = np.mean(diff_loss_list), np.std(diff_loss_list)
                    loss = 0
                    for loss_idx in range(len(loss_list)):
                        loss_value = loss_list[loss_idx]
                        if diff_simi_list[loss_idx] > mean_simi + args.std_coff * std_simi:
                            loss += linear(epoch, args.epochs) * loss_value
                        elif diff_loss_list[loss_idx] > mean_loss + args.std_coff * std_loss:
                            loss += linear(epoch, args.epochs) * loss_value
                        else:
                            loss += loss_value
                else:
                    loss = unbiased_curriculum_loss(out, data, args, criterion, scheduler='linear')
            else:
                loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_train_loss += loss * out.size(0)
        train_data_size += out.size(0)

    # FDS相关处理（如果需要）
    if args.fds and not use_self_distill:
        encodings, labels = torch.from_numpy(np.vstack(encodings)), torch.from_numpy(
            np.hstack(labels))
        model.FDS.update_last_epoch_stats(epoch)
        model.FDS.update_running_stats(encodings, labels, epoch)
        del encodings, labels
        
    train_loss = total_train_loss / train_data_size

    # 验证阶段
    model.eval()
    total_valid_loss = 0
    valid_data_size = 0
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            
            if use_self_distill:
                layer_predictions, layer_features_be, _ = model(data)
                # 使用最深层预测进行验证
                out = layer_predictions[-2]
                # out = layer_predictions[-1]
            elif args.fds or args.contrast_curri:
                out, _ = model(data)
            else:
                out = model(data)
                
            loss = mse_loss(out, data.y)
            total_valid_loss += loss * out.size(0)
            valid_data_size += out.size(0)
    features=layer_features_be
    labels = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5']

    # cluster_and_visualize(features, labels)

    valid_loss = total_valid_loss / valid_data_size
    return train_loss, valid_loss


def evaluate(args, model, loader, device, return_tensor=False):
    """评估函数 - 支持自蒸馏模型"""
    use_self_distill = hasattr(model, 'self_distill') and model.self_distill
    
    model.eval()
    auc_pred, auc_label = [], []
    pred = []
    y = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            if use_self_distill:
                layer_predictions, layer_features_be, _ = model(data)
                # 使用最深层预测
                # out = layer_predictions[-1]
                out = layer_predictions[-2]
            elif args.fds or args.contrast_curri:
                out, _ = model(data)
            else:
                out = model(data)
            out=out.view(-1)
            pred.append(out)
            y.append(data.y)
            auc_pred.extend(out.cpu().numpy().reshape(-1).tolist())
            auc_label.extend(data.y.cpu().numpy().reshape(-1).tolist())

        pred_tensor = torch.cat(pred)
        y_tensor = torch.cat(y)
        corr = pearson_corrcoef(pred_tensor, y_tensor)
        rmse = torch.sqrt(mse_loss(pred_tensor, y_tensor))
    features=layer_features_be
    labels = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5']

    # cluster_and_visualize(features, labels)
    if return_tensor:
        return pred_tensor, y_tensor
    else:
        return corr, rmse


# def metrics(pred_dir, pred_rev, y_dir, y_rev):
#     corr_dir = pearson_corrcoef(pred_dir, y_dir)
#     rmse_dir = torch.sqrt(mse_loss(pred_dir, y_dir))
#     corr_rev = pearson_corrcoef(pred_rev, y_rev)
#     rmse_rev = torch.sqrt(mse_loss(pred_rev, y_rev))
#     corr_dir_rev = pearson_corrcoef(pred_dir, pred_rev)
#     delta = torch.mean(pred_dir + pred_rev)
#
#     return corr_dir, rmse_dir, corr_rev, rmse_rev, corr_dir_rev, delta

def metrics(pred_all,pred_dir, pred_rev,y_all, y_dir, y_rev):
    corr_all=pearson_corrcoef(pred_all, y_all)
    rmse_all=torch.sqrt(mse_loss(pred_all, y_all))
    corr_dir = pearson_corrcoef(pred_dir, y_dir)
    rmse_dir = torch.sqrt(mse_loss(pred_dir, y_dir))
    corr_rev = pearson_corrcoef(pred_rev, y_rev)
    rmse_rev = torch.sqrt(mse_loss(pred_rev, y_rev))
    corr_dir_rev = pearson_corrcoef(pred_dir, pred_rev)
    delta = torch.mean(pred_dir + pred_rev)

    return corr_all,rmse_all,corr_dir, rmse_dir, corr_rev, rmse_rev, corr_dir_rev, delta

class EarlyStopping:
    def __init__(self, patience=10, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, score, model, goal="maximize"):
        if goal == "minimize":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        torch.save(model.state_dict(), self.path)
        self.best_score = score
