# === contrastive_losses.py ===
# This module provides multiple contrastive learning loss variants.
# You can keep it as a separate .py file to switch methods easily.
import torch
import torch.nn.functional as F

def mse_cluster_loss(vad_pred, labels, prototypes):
    """
    MSE cluster loss: pull predicted VADs toward each emotion's VAD prototype.
    """
    # vad_pred: [batch, seq_len, 3] -> [N, 3], each row is a [v,a,d] point
    vad_pred = vad_pred.view(-1, 3)

    # labels: emotion label (0-5) for each vad point
    labels = labels.view(-1)

    # mask out invalid labels (padding usually marked as -1)
    mask = (labels >= 0)
    vad_pred = vad_pred[mask]
    labels = labels[mask]

    # emotion_prototypes: [6, 3] -> map labels to corresponding prototypes
    centers = prototypes[labels]  # pick prototype [v,a,d] for each prediction

    # minimize MSE between prediction and its prototype
    loss = F.mse_loss(vad_pred, centers)
    return loss

def supcon_loss(vad_pred, labels, temperature=0.1):
    """
    Supervised Contrastive Loss: same-class positives, different-class negatives.
    """
    vad_pred = F.normalize(vad_pred.view(-1, 3), dim=-1)
    labels = labels.view(-1)
    mask = (labels >= 0)
    vad_pred = vad_pred[mask]
    labels = labels[mask]

    similarity_matrix = torch.matmul(vad_pred, vad_pred.T) / temperature
    labels = labels.unsqueeze(1)
    matches = torch.eq(labels, labels.T).float()
    logits_mask = 1 - torch.eye(matches.shape[0]).to(matches.device)
    matches = matches * logits_mask
    
    exp_sim = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (matches * log_prob).sum(1) / (matches.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss
    
    '''
def triplet_loss(vad_pred, labels, margin=1.0):
    vad_pred = vad_pred.view(-1, 3)
    labels = labels.view(-1)
    mask = (labels >= 0)
    vad_pred = vad_pred[mask]
    labels = labels[mask]
    loss = 0.0
    triplet_count = 0
    for i in range(len(vad_pred)):
        anchor, anchor_label = vad_pred[i], labels[i]
        pos = vad_pred[labels == anchor_label]
        neg = vad_pred[labels != anchor_label]
        if len(pos) > 1 and len(neg) > 0:
            positive = pos[torch.randint(0, len(pos), (1,))]
            negative = neg[torch.randint(0, len(neg), (1,))]
            dist_pos = F.pairwise_distance(anchor.unsqueeze(0), positive)
            dist_neg = F.pairwise_distance(anchor.unsqueeze(0), negative)
            loss += F.relu(dist_pos - dist_neg + margin)
            triplet_count += 1
    loss=loss / (triplet_count + 1e-8)
    return loss 
    '''
def triplet_loss(vad_pred, labels, margin=1.0):
    vad_pred = vad_pred.view(-1, 3)
    labels = labels.view(-1)
    mask = (labels >= 0)
    vad_pred, labels = vad_pred[mask], labels[mask]
    
    # vectorized implementation (replace Python loops)
    pairwise_dist = F.pairwise_distance(vad_pred.unsqueeze(1), vad_pred.unsqueeze(0))  # [N,N]
    
    # build triplet masks
    same_label = labels.unsqueeze(1) == labels.unsqueeze(0)
    pos_mask = same_label & ~torch.eye(len(labels), dtype=bool)
    neg_mask = ~same_label
    
    # compute loss over all valid triplets
    pos_dist = pairwise_dist[pos_mask].view(len(labels), -1)  # [N, num_pos]
    neg_dist = pairwise_dist[neg_mask].view(len(labels), -1)  # [N, num_neg]
    loss = F.relu(pos_dist[:, None] - neg_dist[None, :] + margin).mean()
    return loss
def info_nce_loss(vad_pred, labels, temperature=0.07):
    vad_pred = F.normalize(vad_pred.view(-1, 3), dim=-1)
    labels = labels.view(-1)
    mask = (labels >= 0)
    vad_pred = vad_pred[mask]
    labels = labels[mask]
    sim = torch.matmul(vad_pred, vad_pred.T) / temperature
    logits_mask = 1 - torch.eye(sim.size(0)).to(sim.device)
    labels = labels.unsqueeze(1)
    matches = torch.eq(labels, labels.T).float() * logits_mask
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (matches * log_prob).sum(1) / (matches.sum(1) + 1e-8)
    loss=-mean_log_prob_pos.mean()
    return loss

def circle_loss(vad_pred, labels, margin=0.25, gamma=256):
    vad_pred = F.normalize(vad_pred.view(-1, 3), dim=-1)
    labels = labels.view(-1)
    mask = labels >= 0
    vad_pred = vad_pred[mask]
    labels = labels[mask]
    sim = torch.matmul(vad_pred, vad_pred.T)
    logits_mask = 1 - torch.eye(sim.size(0)).to(sim.device)
    pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float() * logits_mask
    neg_mask = 1 - pos_mask
    pos_sim = sim * pos_mask
    neg_sim = sim * neg_mask
    alpha_p = torch.clamp_min(-pos_sim.detach() + 1 + margin, 0.)
    alpha_n = torch.clamp_min(neg_sim.detach() + margin, 0.)
    delta_p = 1 - margin
    delta_n = margin
    logit_p = -gamma * alpha_p * (pos_sim - delta_p)
    logit_n = gamma * alpha_n * (neg_sim - delta_n)
    loss = F.softplus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))
    return loss.mean()
def proto_supcon_loss(vad_pred, labels, prototypes, alpha=0.5, temperature=0.1):
    """
    Mixed loss: prototype MSE + supervised contrastive combined.
    Returns alpha * SupCon + (1-alpha) * MSE cluster loss.
    """
    # common
    vad_pred = vad_pred.view(-1, 3)
    labels = labels.view(-1)
    mask = (labels >= 0)
    vad_pred = vad_pred[mask]
    labels = labels[mask]
    
    # --- SupCon part ---
    vad_norm = F.normalize(vad_pred, dim=-1)
    sim = torch.matmul(vad_norm, vad_norm.T) / temperature
    logits_mask = 1 - torch.eye(sim.size(0)).to(sim.device)
    labels_ = labels.unsqueeze(1)
    matches = torch.eq(labels_, labels_.T).float() * logits_mask
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    supcon = -(matches * log_prob).sum(1) / (matches.sum(1) + 1e-8)
    supcon = supcon.mean()

    # --- prototype part ---
    centers = prototypes[labels]
    mse = F.mse_loss(vad_pred, centers)

    return alpha * supcon + (1 - alpha) * mse

# selectable via config in train.py
contrastive_loss_methods = {
    "mse": mse_cluster_loss,
    "supcon": supcon_loss,
    "triplet": triplet_loss,
    "infonce": info_nce_loss,
    "circle": circle_loss,
    "proto_supcon": proto_supcon_loss
}
