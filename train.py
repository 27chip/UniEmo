import numpy as np, random, math
from tqdm import tqdm
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from dataloader import DialogLoader
from model import JointModel, MaskedNLLLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import torch.nn.functional as F
import sys
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from contrastive_losses import contrastive_loss_methods
from sklearn.manifold import TSNE


# class weights
def create_class_weight(mu=1):
    # define the set of classes
    unique = [0, 1, 2, 3, 4, 5, 6]
    # define a label dictionary: key=class, value=sample count
    # iemocap
    labels_dict = {0: 1103, 1:1708, 2: 595, 3: 1084, 4: 1849, 5: 1044}
    # compute total number of samples across classes
    total = np.sum(list(labels_dict.values()))
    # initialize weight list
    weights = []
    # iterate over each class
    for key in unique:
        # compute weight for current class
        score = math.log(mu*total/labels_dict[key])
        # append weight to list
        weights.append(score)
    # return the weight list
    return weights
# set random seeds for reproducibility
def seed_everything(seed):
    # set Python random seed
    random.seed(seed)
    # set numpy random seed
    np.random.seed(seed)
    # set torch CPU seed
    torch.manual_seed(seed)
    # set current GPU seed
    torch.cuda.manual_seed(seed)
    # set all GPU seeds
    torch.cuda.manual_seed_all(seed)
    # disable cuDNN benchmark for deterministic behavior
    torch.backends.cudnn.benchmark = False
    # enable deterministic algorithms for cuDNN
    torch.backends.cudnn.deterministic = True

# prepare optimizer
def configure_optimizers(model, weight_decay, learning_rate, adam_epsilon):
    "Prepare optimizer"
    no_decay = ["bias", "LayerNorm.weight"]
    # grouping parameters
    optimizer_grouped_parameters = [
        {
            "params":  ([p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]),
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon) 
    return optimizer

 # prepare data loaders
def configure_dataloaders(dataset, classify, multitask, batch_size):
    "Prepare dataloaders"
    # define train mask file path
    train_mask = './datasets/' + dataset + '/' + dataset + '_train_loss_mask.tsv'
    # define validation mask file path
    valid_mask = './datasets/' + dataset + '/' + dataset + '_valid_loss_mask.tsv'
    # define test mask file path
    test_mask = './datasets/' + dataset + '/' + dataset + '_test_loss_mask.tsv'
    # create train loader
    train_loader = DialogLoader(
        # train utterances file path
        './datasets/' + dataset + '/' + dataset + '_train_utterances.tsv',  
        # train labels file path
        './datasets/' + dataset + '/' + dataset + '_train_' + classify + '.tsv',
        # train multi-task labels file path
        # './datasets/' + dataset + '/' + dataset + '_train_' + multitask + '_label.tsv',
        # train subtask01 index file path
        './datasets/' + dataset + '/' + dataset + '_train_subtask01_index.tsv',
        # train loss mask file path
        train_mask,
        # train speakers file path
        './datasets/' + dataset + '/' + dataset + '_train_speakers.tsv',
        './datasets/' + dataset + '/' + dataset + '_train_v_diff.tsv',
        './datasets/' + dataset + '/' + dataset + '_train_a_diff.tsv',
        './datasets/' + dataset + '/' + dataset + '_train_d_diff.tsv',
        './datasets/' + dataset + '/' + dataset + '_train_valence.tsv',
        './datasets/' + dataset + '/' + dataset + '_train_arousal.tsv',
        './datasets/' + dataset + '/' + dataset + '_train_dominance.tsv',
        # batch size
        batch_size,
        # shuffle
        shuffle=True
    )
    valid_loader = DialogLoader(
        # validation utterances file path
        './datasets/' + dataset + '/' + dataset + '_test_utterances.tsv',  
        # validation labels file path
        './datasets/' + dataset + '/' + dataset + '_test_' + classify + '.tsv',
        # validation multi-task labels file path
        # './datasets/' + dataset + '/' + dataset + '_test_' + multitask + '_label.tsv',
        # validation subtask01 index file path
        './datasets/' + dataset + '/' + dataset + '_test_subtask01_index.tsv',
        # validation loss mask file path
        test_mask,
        # validation speakers file path
        './datasets/' + dataset + '/' + dataset + '_test_speakers.tsv',
        './datasets/' + dataset + '/' + dataset + '_test_v_diff.tsv',
        './datasets/' + dataset + '/' + dataset + '_test_a_diff.tsv',
        './datasets/' + dataset + '/' + dataset + '_test_d_diff.tsv',

        './datasets/' + dataset + '/' + dataset + '_test_valence.tsv',
        './datasets/' + dataset + '/' + dataset + '_test_arousal.tsv',
        './datasets/' + dataset + '/' + dataset + '_test_dominance.tsv',
        # batch size
        batch_size,
        # shuffle
        shuffle=False
    )
    
    # create test loader
    test_loader = DialogLoader(
        # test utterances file path
        './datasets/' + dataset + '/' + dataset + '_test_utterances.tsv',  
        # test labels file path
        './datasets/' + dataset + '/' + dataset + '_test_' + classify + '.tsv',
        # test multi-task labels file path
        # './datasets/' + dataset + '/' + dataset + '_test_' + multitask + '_label.tsv',
        # test subtask01 index file path
        './datasets/' + dataset + '/' + dataset + '_test_subtask01_index.tsv',
        # test loss mask file path
        test_mask,
        # test speakers file path
        './datasets/' + dataset + '/' + dataset + '_test_speakers.tsv',
        './datasets/' + dataset + '/' + dataset + '_test_v_diff.tsv',
        './datasets/' + dataset + '/' + dataset + '_test_a_diff.tsv',
        './datasets/' + dataset + '/' + dataset + '_test_d_diff.tsv',

        './datasets/' + dataset + '/' + dataset + '_test_valence.tsv',
        './datasets/' + dataset + '/' + dataset + '_test_arousal.tsv',
        './datasets/' + dataset + '/' + dataset + '_test_dominance.tsv',
        # batch size
        batch_size,
        # shuffle
        shuffle=False
    )
    # return train, validation, and test loaders
    return train_loader, valid_loader, test_loader
def metric_helper(dataset, labels, preds, masks, losses, task_type):
    # if predictions list is not empty, concatenate; otherwise return zeros and empty lists
    if preds != []:
        preds = np.concatenate(preds)  # concatenate predictions
        labels = np.concatenate(labels)  # concatenate true labels
        masks = np.concatenate(masks)  # concatenate loss mask
    else:
        #return float('nan'), float('nan'), float('nan'), [], [], []
        return 0, 0, 0, [], [], []
    # compute average loss normalized by number of utterances
    avg_loss = round(np.sum(losses) / np.sum(masks), 4)  

    # compute weighted accuracy using sklearn's accuracy_score
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)

    # compute F1 score depending on dataset/task
    if dataset == 'iemocap':
        avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
        fscores = avg_fscore
    
    # return computed metrics (avg loss, accuracy, F1) and raw arrays (labels, preds, masks)
    return avg_loss, avg_accuracy, fscores, labels, preds, masks 
## train or evaluation routine
def train_or_eval_model(dataset, mode, model, main_loss_function, dataloader, epoch, acc_steps, optimizer=None, train=False, grad_acc=False):
    losses1, preds1,  labels1,  masks1,  = [], [], [], []
    v_preds, a_preds, d_preds = [], [], []
    delta_v_preds,delta_a_preds,delta_d_preds = [], [], []
    diff_v,diff_a,diff_d = [], [], []
    v_data , a_data, d_data = [], [], []
    v_datas , a_datas, d_datas = [], [], []
    vad_losses = []  # store total VAD losses
    contrastive_loss=[]

    assert not train or optimizer!=None        ## ensure optimizer provided when training
    if train:                                  ## choose train/eval mode
        model.train()
    else:
        model.eval()
    ## debugging
    ## conv_num = 0 
    features_list = []
    labels_list = []
    for conversations, label, subindex, loss_mask, speaker_mask,diff_valence,diff_arousal,diff_dominance,valence,arousal,dominance in tqdm(dataloader, leave=False): 
        # build umask and qmask
        ## number of utterances per conversation
        ## bsz elements: conv_len
        lengths = [len(item) for item in conversations]
        ## utterance mask: bsz, max_conv_len
        umask = torch.zeros(len(lengths), max(lengths)).long().cuda()
        
        for j in range(len(lengths)):
            ## [0,lengths[j])
            umask[j][:lengths[j]] = 1
        ## bsz, conv_len: for MELD elements are speaker identifiers like '0' or '1'
        qmask = speaker_mask
        qmask = torch.nn.utils.rnn.pad_sequence( [torch.tensor([int(speaker) for speaker in item]) for item in speaker_mask],
                                                batch_first=True).long().cuda()
        ## subindex: bsz, conv_len — index of previous utterance by same speaker
        ## convert to tensor: max_conv_len, bsz with padding value -1
        subindex = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in subindex],
                                                   batch_first=False, padding_value=-1).long().cuda()
        ## bsz, conv_len
        ## convert to tensor bsz, max_conv_len with padding value 0
        label1 = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in label],
                                               batch_first=True,padding_value=0).cuda()

        ## bsz, max_conv_len
        ## flatten to compute loss mask
        loss_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item).cuda() for item in loss_mask],
                                                    batch_first=True,padding_value=0).view(-1)  #  (batch_size * max_conv_len)

        # prepare VAD diffs and VAD targets
        diff_v= torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in diff_valence],
                        batch_first=True,padding_value=0).cuda()       #  (batch_size, max_conv_len) # fill value 0
        diff_a= torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in diff_arousal],
                        batch_first=True,padding_value=0).cuda()  #  (batch_size, max_conv_len)
        diff_d= torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in diff_dominance],
                        batch_first=True,padding_value=0).cuda()  #  (batch_size, max_conv_len) 

        v_data= torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in valence],
                        batch_first=True,padding_value=0).cuda()       #  (batch_size, max_conv_len) # fill value 0
        a_data= torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in arousal],
                        batch_first=True,padding_value=0).cuda()  #  (batch_size, max_conv_len)
        d_data= torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in dominance],
                        batch_first=True,padding_value=0).cuda()  #  (batch_size, max_conv_len) 
        if train:
            ## torch.Size([6, 1, 7])
            ## max_utt_num, bsz, label_num
            log_prob1,delta_v_pred,delta_a_pred,delta_d_pred,v_pred, a_pred, d_pred, tf_features = model(conversations, subindex, lengths, umask, qmask)
        else:
            with torch.no_grad():
                log_prob1,delta_v_pred,delta_a_pred,delta_d_pred, v_pred, a_pred, d_pred, tf_features = model(conversations, subindex, lengths, umask, qmask)
        # reshape log_prob1 to [utt_num, label_num]
        lp1_ = log_prob1.transpose(0, 1).contiguous().view(-1, log_prob1.size()[2]) ## utt_num, label_num   
        # reshape label1 to flat vector
        labels1_ = label1.view(-1)                                ## utt_num true labels
        # compute main loss
        loss1 = main_loss_function(lp1_, labels1_, loss_mask)     
        # get predicted labels
        pred1_ = torch.argmax(lp1_, 1)                            ## utt_num predicted labels
        # store predictions, labels and masks
        preds1.append(pred1_.data.cpu().numpy())
        labels1.append(labels1_.data.cpu().numpy())
        masks1.append(loss_mask.view(-1).cpu().numpy())
        losses1.append(loss1.item() * masks1[-1].sum())

        diff_v = diff_v.view(-1)  #  v_data shape [1, 29] to [29]
        diff_a = diff_a.view(-1)
        diff_d = diff_d.view(-1)

        # compute directional consistency loss for delta VAD
        pred_delta_vad = torch.stack([delta_v_pred, delta_a_pred, delta_d_pred], dim=-1)
        ideal_direction = torch.stack([diff_v, diff_a, diff_d], dim=-1)

        # normalize vectors to unit length
        pred_delta_vad = F.normalize(pred_delta_vad, dim=-1)
        ideal_direction = F.normalize(ideal_direction, dim=-1)
        # direction loss
        dir_loss = 1 - F.cosine_similarity(pred_delta_vad, ideal_direction).mean()
        
        # compute MSE loss (keep consistent dimensions)
        loss_v = F.mse_loss(delta_v_pred.view(-1), diff_v, reduction='sum')
        loss_a = F.mse_loss(delta_a_pred.view(-1), diff_a, reduction='sum')
        loss_d = F.mse_loss(delta_d_pred.view(-1), diff_d, reduction='sum')
        mse_loss = loss_v + loss_a +loss_d 
        total_vad_loss = 0.7*dir_loss + 0.3*mse_loss

        current_vad = torch.stack([v_pred, a_pred, d_pred], dim=-1)  # [bsz, seq, 3]
        labels_flat = label1.view(-1)  # align with vad_pred

        contrastive_loss = contrastive_func(current_vad, labels_flat)

        # total loss:
        loss = loss1  +0.5*contrastive_loss+ 0.5* total_vad_loss

        if train:                           ## backward pass
            if grad_acc:
                accumulation_steps = int(acc_steps)
                loss = loss/accumulation_steps
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # tf_features: [seq, batch, hidden_dim] -> [seq*batch, hidden_dim]
        tf_features_flat = tf_features.transpose(0,1).contiguous().view(-1, tf_features.size(-1))
        features_list.append(tf_features_flat.detach().cpu().numpy())
        labels_list.append(labels_flat.detach().cpu().numpy())
        # collect VAD predictions and delta (only during eval to save memory)
        if not train:
            v_preds.append(v_pred.view(-1).detach().cpu().numpy())
            a_preds.append(a_pred.view(-1).detach().cpu().numpy())
            d_preds.append(d_pred.view(-1).detach().cpu().numpy())
            delta_v_preds.append(delta_v_pred.view(-1).detach().cpu().numpy())
            delta_a_preds.append(delta_a_pred.view(-1).detach().cpu().numpy())
            delta_d_preds.append(delta_d_pred.view(-1).detach().cpu().numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    predictions = np.concatenate(preds1, axis=0)

    # save extracted features/predictions to npy directory (following train_nnime.py)
    base_npy_dir = './npy'
    features_dir = os.path.join(base_npy_dir, 'features')
    labels_dir = os.path.join(base_npy_dir, 'labels')
    predictions_dir = os.path.join(base_npy_dir, 'predictions')
    vad_data_dir = os.path.join(base_npy_dir, 'vad_data')
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(vad_data_dir, exist_ok=True)
    np.save(os.path.join(features_dir, f'features_epoch{epoch}.npy'), features)
    np.save(os.path.join(labels_dir, f'labels_true_epoch{epoch}.npy'), labels)
    np.save(os.path.join(predictions_dir, f'labels_pred_epoch{epoch}.npy'), predictions)
    # save VAD predictions and deltas only during evaluation (train==False)
    if not train:
        if len(v_preds) > 0:
            all_v = np.concatenate(v_preds, axis=0)
            all_a = np.concatenate(a_preds, axis=0)
            all_d = np.concatenate(d_preds, axis=0)
            np.save(os.path.join(vad_data_dir, f'v_pred_epoch{epoch}.npy'), all_v)
            np.save(os.path.join(vad_data_dir, f'a_pred_epoch{epoch}.npy'), all_a)
            np.save(os.path.join(vad_data_dir, f'd_pred_epoch{epoch}.npy'), all_d)
        if len(delta_v_preds) > 0:
            all_dv = np.concatenate(delta_v_preds, axis=0)
            all_da = np.concatenate(delta_a_preds, axis=0)
            all_dd = np.concatenate(delta_d_preds, axis=0)
            np.save(os.path.join(vad_data_dir, f'delta_v_pred_epoch{epoch}.npy'), all_dv)
            np.save(os.path.join(vad_data_dir, f'delta_a_pred_epoch{epoch}.npy'), all_da)
            np.save(os.path.join(vad_data_dir, f'delta_d_pred_epoch{epoch}.npy'), all_dd)

    # np.save(f'vad_init_epoch{epoch}.npy', vad_init)
    # # np.save(f'delta_vad_epoch{epoch}.npy', delta_vad)

    main_metrics = metric_helper(dataset, labels1, preds1, masks1, losses1, 'main')
    return main_metrics     
# result output helpers
def result_helper(valid_fscores, test_fscores, valid_losses, rf, lf, best_label, best_pred, best_mask, task_type):
    # convert valid and test fscore lists to numpy arrays and transpose
    print(f"test_fscores details: {test_fscores}")  
    print(f"valid_losses details: {valid_losses}")

    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()
    # check if test_fscores is scalar
    if np.isscalar(test_fscores):
        test_fscores = np.array([test_fscores])
    # check if valid_losses is scalar
    if np.isscalar(valid_losses):
        valid_losses = np.array([valid_losses])
    # output test performance
    print('Test performance.')
    if dataset == 'iemocap':
        score1 = test_fscores[np.argmin(valid_losses)]
        score2 = test_fscores[np.argmax(valid_fscores)]
        scores_val_loss = [score1]
        scores_val_f1 = [score2]
        loss_at_epoch = np.argmin(valid_losses) # epoch with lowest validation loss
        f1_at_epoch = np.argmax(valid_fscores)  # epoch with highest validation F1
        res1 = 'Scores: Weighted'
        res2 = 'F1@Best Valid Loss: {}'.format(scores_val_loss)
        res3 = 'F1@Best Valid F1: {}'.format(scores_val_f1)
        res4 = 'Lowest loss at epoch:' + str(loss_at_epoch)
        res5 = 'Highest F1 at epoch: ' + str(f1_at_epoch)
        print(res1)
        print(res2)
        print(res3)
        print(res4)
        print(res5)
        rf.write(res1 + '\n')
        rf.write(res2 + '\n')
        rf.write(res3 + '\n')
        rf.write(res4 + '\n')
        rf.write(res5 + '\n')
        lf.write(res1 + '\n')
        lf.write(res2 + '\n')
        lf.write(res3 + '\n')
        lf.write(res4 + '\n')
        lf.write(res5 + '\n')
        #labels = [0, 1, 2, 3, 4, 5]  # IEMOCAP 
        class_names = ['ang', 'neu', 'hap', 'sad', 'fru', 'exc']  # IEMOCAP class names
        best_loss_idx = np.argmin(valid_losses)  # best loss
        best_f1_idx = np.argmax(valid_fscores)   # best F1 

        # write classification report and confusion matrix
        if task_type == 'main':        
            rf.write('\n' + 'Classification Report at Best Loss Weighted F1' + '\n')
            rf.write('\n' + f'Best Valid Loss at epoch: {best_loss_idx}' + '\n')
            loss_classification_report = classification_report(best_label[best_loss_idx], best_pred[best_loss_idx],
                        sample_weight=best_mask[best_loss_idx], digits=4,target_names=class_names, zero_division=0)
            rf.write(str(loss_classification_report) + '\n')
            # compute & plot confusion matrix at best loss
            conf_matrix_loss = confusion_matrix(best_label[best_loss_idx], best_pred[best_loss_idx],
                        sample_weight=best_mask[best_loss_idx])
            rf.write(str(conf_matrix_loss) + '\n')
            rf.write('over' + '\n\n')
            save_dir = './images/'+ str(saved_model_number)
            os.makedirs(save_dir, exist_ok=True)  # ensure directory exists
            plot_confusion_matrix(conf_matrix_loss, "Confusion Matrix at Best Loss F1",os.path.join(save_dir, res4 +'_confusion_matrix_loss.png'))
            # classification report at best_F1 Weighted F1
            rf.write('\n' + 'Classification Report at Best F1 Weighted F1' + '\n')
            rf.write('\n' + f'Best F1 Score at epoch: {best_f1_idx}' + '\n')
            f1_classification_report = classification_report(best_label[best_f1_idx], best_pred[best_f1_idx],
                sample_weight=best_mask[best_f1_idx], digits=4,target_names=class_names, zero_division=0)
            rf.write(str(f1_classification_report) + '\n')
            # compute & plot confusion matrix at best F1
            conf_matrix_f1 = confusion_matrix( best_label[best_f1_idx], best_pred[best_f1_idx],
                            sample_weight=best_mask[best_f1_idx])
            rf.write(str(conf_matrix_f1) + '\n')
            rf.write('over' + '\n\n')
            plot_confusion_matrix(conf_matrix_f1, "Confusion Matrix at Best F1 Score",os.path.join(save_dir, res5 +'_confusion_matrix_f1.png'))
    # close result files   
    rf.close()
    lf.close()
# draw confusion matrix
def plot_vad_metrics(metric_name, train_values, valid_values, test_values):
    plt.figure(figsize=(10,5))
    plt.plot(train_values, label=f"Train {metric_name}")
    plt.plot(valid_values, label=f"Valid {metric_name}")
    plt.plot(test_values, label=f"Test {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric_name} MSE")
    plt.title(f"{metric_name} MSE Curve")
    plt.legend()
def plot_confusion_matrix(conf_matrix, title,save_path):
    class_names = ['ang', 'neu', 'hap', 'sad', 'fru', 'exc']
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(6,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(save_path)  # save the figure
    plt.close()  # close the figure
def plot_vad_clusters(vad_values, vad_labels, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    vad_2d = tsne.fit_transform(vad_values.cpu().numpy())
    plt.figure(figsize=(8,6))
    for label in sorted(set(vad_labels.cpu().numpy())):
        idx = vad_labels.cpu().numpy() == label
        plt.scatter(vad_2d[idx, 0], vad_2d[idx, 1], label=str(label), alpha=0.5)
    plt.legend()
    plt.title("VAD TSNE Cluster")
    plt.savefig(save_path)
    
# main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add learning rate argument (float)
    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR', help='learning rate')
    # add weight decay argument
    parser.add_argument('--weight_decay', default=1e-4, type=float, help="Weight decay if we apply some.")
    # add Adam epsilon argument
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    # add learning rate decay type
    parser.add_argument('--lr_decay_type', default='none', help="steplr|exlr")
    # add learning rate decay parameter (meaning depends on decay type)
    parser.add_argument('--lr_decay_param', default=0.5, type=float, help="steplr: 0.5|0.1;exlr:0.98|0.99|0.90")
    # add batch size argument
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS', help='batch size')
    # add number of epochs
    parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')
    # add class weight option
    parser.add_argument('--class_weight', default='none', help='cosmic|sklearn|none')
    # add class weight mu parameter
    parser.add_argument('--mu', type=float, default=0, help='class_weight_mu')
    # add transformer model family selection
    parser.add_argument('--model', default='bert', help='which model family bert|roberta|xlnet')
    # add transformer mode selection (size/type)
    parser.add_argument('--mode', default='0', help='which mode 0: bert or roberta base | 1: bert or roberta large; \
                                                     0, 1: bert base, large sentence transformer and 2, 3: roberta base, large sentence transformer')
    # add dataset name
    parser.add_argument('--dataset', default='meld')
    # add multitask option
    parser.add_argument('--multitask', default='subtask01', help='subtask01|subtask01Senti|subtask013Senti')
    # add gradient accumulation flag
    parser.add_argument('--grad_acc', action='store_true', default=False, help='use grad accumulation')
    # add gradient accumulation steps
    parser.add_argument('--acc_steps', default='1', help='1|2|4|8')
    # add random seed argument
    parser.add_argument('--seed', type=int, default=777, metavar='seed', help='seed')
    # add description field
    parser.add_argument('--describe', default='train.py')
    # add number of context encoder layers
    parser.add_argument('--context_encoder_layer', type=int, default=6)
    # add save-model flag
    parser.add_argument('--save_model', action='store_true', default=False, help='save model')
    # add no-lstm flag
    parser.add_argument('--no_lstm', action='store_true', default=False, help='no lstm')
    parser.add_argument('--contrastive', default='mse', help='mse|supcon|triplet|infonce')
    
    # parse command-line arguments
    args = parser.parse_args()
    contrastive_func = contrastive_loss_methods[args.contrastive]
    # print parsed args
    print(args)
    global dataset                        ## global: declare to modify outer-scope variable inside functions
    D_h = 512# lstm layer
    batch_size = args.batch_size
    n_epochs = args.epochs
    dataset = args.dataset
    classification_model = 'EBERC'
    transformer_model = args.model
    transformer_mode = args.mode
    multitask = args.multitask
    grad_acc = args.grad_acc
    acc_steps = args.acc_steps
    no_lstm = args.no_lstm
    context_encoder_layer = args.context_encoder_layer
    global train_weights_v,train_weights_a,train_weights_d,valid_weights_v,valid_weights_a,valid_weights_d,test_weights_v,test_weights_a,test_weights_d
    train_weights_v = []
    train_weights_a = []
    train_weights_d = []
    valid_weights_v = []
    valid_weights_a = []
    valid_weights_d = []
    test_weights_v = []
    test_weights_a = []
    test_weights_d = []
    global train_loss1_list, valid_loss1_list, test_loss1_list
    train_loss1_list = []
    valid_loss1_list = []
    test_loss1_list = []
    global train_total_vad_losses, valid_total_vad_losses, test_total_vad_losses
    train_total_vad_losses = []
    valid_total_vad_losses = []
    test_total_vad_losses = []
    global train_contrastive_losses,valid_contrastive_losses, test_contrastive_losses
    train_contrastive_losses = []
    valid_contrastive_losses = []
    test_contrastive_losses = []
    global seed
    train_dynamic_losses = []
    train_smooth_losses = []
    valid_dynamic_losses = []
    valid_smooth_losses = []
    test_dynamic_losses = []
    test_smooth_losses = []
    seed = args.seed
    seed_everything(seed)                 ## seed_everything: custom function; default seed=777  
    if dataset == 'iemocap':
        print ('Classifying emotion in iemocap.')
        n_classes  = 6       
    if multitask == 'subtask01' or multitask == 'subtask01Senti':  ## multitask default='subtask01' (emotion-shift three-class)
        n_subclasses = 3

    from model import JointModel    
    #transformer_model = args.model = Bert
    
    model = JointModel(D_h, classification_model, transformer_model, transformer_mode, n_classes, n_subclasses, context_encoder_layer, False, False, no_lstm)
    ## class weights inspired by COSMIC loss
    if args.class_weight == 'cosmic':                   ##  default='none'
        if args.mu > 0:
            loss_weights = torch.FloatTensor(create_class_weight(args.mu))
        else:   
            loss_weights = torch.FloatTensor([4, 0.3, 8, 8, 2, 4, 4])
        main_loss_function  = MaskedNLLLoss(loss_weights.cuda())
    elif args.class_weight == 'sklearn':
        # see calculate_class_weights.ipynb
        loss_weights = torch.FloatTensor([14.39460442,0.17191705,41.67503035,84.54761905,1.14173735,12.79242236,8.06982211])
        main_loss_function  = MaskedNLLLoss(loss_weights.cuda())
    elif args.class_weight == 'none':                                       
       main_loss_function = MaskedNLLLoss()                                 ## from model import MaskedNLLLoss()
    optimizer = configure_optimizers(model, args.weight_decay, args.lr, args.adam_epsilon)     
    if args.lr_decay_type == 'none':
        pass
    elif args.lr_decay_type == 'exlr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args.lr_decay_param)
    elif args.lr_decay_type == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.lr_decay_param)
    train_loader, valid_loader, test_loader = configure_dataloaders(dataset, 'emotion', multitask, batch_size)
    lf_main = open('./logs/' + dataset + '_' + transformer_model + '_mode_' + transformer_mode + '_' + classification_model
              + '_' + multitask + '_main.txt', 'a')
    rf_main = open('./results/' + dataset + '_' + transformer_model + '_mode_' + transformer_mode + '_' + classification_model
              + '_' + multitask + '_main.txt', 'a')
    lf_vad = open(f'./logs/{dataset}_{transformer_model}_mode_{transformer_mode}_{classification_model}_{multitask}_vad.txt', 'a')
    rf_vad = open(f'./results/{dataset}_{transformer_model}_mode_{transformer_mode}_{classification_model}_{multitask}_vad.txt', 'a')

    ## print('seed: ', seed)                                                          ## seed
    rf_main.write('\n' + str(args) + '\n')
    lf_main.write('\n' + str(args) + '\n')
    rf_vad.write('\n' + str(args) + '\n')
    lf_vad.write('\n' + str(args) + '\n')
    valid_losses1, valid_fscores1 = [], []
    test_fscores1,test_losses1= [], []
    train_losses1,train_fscores1= [],[]
    best_loss1 = None
    best_label1,  best_pred1,  best_mask1= [], [], []
    train_v_mse, valid_v_mse, test_v_mse = [], [], []
    train_a_mse, valid_a_mse, test_a_mse = [], [], []
    train_d_mse, valid_d_mse, test_d_mse = [], [], []
    # if args.save_model:
    saved_model_number = int(time.time())                                    ## generate a timestamp-based save model id
    ## print('saved_model_number is: ' + str(saved_model_number))
    rf_main.write('saved model number is: ' + str(saved_model_number) + '\n')
    lf_main.write('saved model number is: ' + str(saved_model_number) + '\n')
    rf_vad.write('saved model number is: ' + str(saved_model_number) + '\n')
    lf_vad.write('saved model number is: ' + str(saved_model_number) + '\n')
    # record VAD losses during training
    train_vad_losses = []
    valid_vad_losses = []
    test_vad_losses = []
    for e in range(n_epochs):
        start_time = time.time()                                               ## record start time                  
        print('---------train--------')                                        
        train_result = train_or_eval_model(dataset, 0, model, main_loss_function, train_loader, e, acc_steps, optimizer, True, grad_acc)
        
        print('-----------valid-----------')                                   
        valid_result = train_or_eval_model(dataset, 1, model, main_loss_function, valid_loader, e, acc_steps)
        
        print('-----------test-----------')
        test_result = train_or_eval_model(dataset, 2, model, main_loss_function,  test_loader, e, acc_steps)
        
        # main task result
        ## avg_loss, avg_accuracy, fscores, labels, preds, masks
        valid_losses1.append(valid_result[0])
        valid_fscores1.append(valid_result[2])
        test_losses1.append(test_result[0])
        test_fscores1.append(test_result[2])
        train_losses1.append(train_result[0])
        ## best valid loss
        if best_loss1 == None or best_loss1 > valid_result[0]:             ## update best loss
            best_loss1 = valid_result[0]
        best_label1.append(test_result[3])
        best_pred1.append(test_result[4])
        best_mask1.append(test_result[5])
        x1 = 'Epoch {}'.format(e) + '\n' + 'train_loss {} train_acc {} train_fscore {}'.format(train_result[0], train_result[1], train_result[2]) + '\n' + \
            'valid_loss {} valid_acc {} valid_fscore {}'.format(valid_result[0], valid_result[1], valid_result[2]) + '\n' + \
            'test__loss {} test__acc {} test__fscore {}'.format(test_result[0], test_result[1], test_result[2]) + '\n' + \
            'time {}'.format(round(time.time() - start_time, 2))
        print(x1)
        lf_main.write(x1 + '\n')
    
        if args.lr_decay_type != 'none':
            print("Epoch %d learning rate: %f" % (e, optimizer.param_groups[0]['lr']))
            scheduler.step()
        # save model  (train_nnime style)
        if args.save_model:
            fscores_ = np.array(valid_fscores1).transpose()
            state = {'model': model.state_dict(), 'epoch': e, 'seed': seed}
            if fscores_.ndim == 1:
                array_size = len(fscores_)
            else:
                array_size = np.size(fscores_, 1)
            test_fscores_arr = np.array(test_fscores1).transpose()
            if np.argmax(test_fscores_arr) == (array_size - 1):
                save_path = './save_models/' + str(saved_model_number) + '_best_model.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(state, save_path)
 
    
    save_dir = './images/'+ str(saved_model_number)
    os.makedirs(save_dir, exist_ok=True)  # create directory if it does not exist
    # plot loss curve
    plt.figure(figsize=(10,5))
    plt.plot(train_losses1, label="Train Loss")
    plt.plot(valid_losses1, label="Valid Loss")
    plt.plot(test_losses1, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    # plot F1 score curve
    plt.figure(figsize=(10,5))
    plt.plot(train_fscores1, label="Train F1")
    plt.plot(valid_fscores1, label="Valid F1")
    plt.plot(test_fscores1, label="Test F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "f1_score_curve.png"))
    #plt.show()

    # plot loss1 curves for train/valid/test on the same figure
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss1_list, label="Train Loss1")
    plt.plot(valid_loss1_list, label="Valid Loss1")
    plt.plot(test_loss1_list, label="Test Loss1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss1")
    plt.title("Loss1 Curve ")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss1_curve.png"))

    # plot contrastive loss curves
    plt.figure(figsize=(10,5))
    plt.plot(train_contrastive_losses, label="Train Contrastive Loss")
    plt.plot(valid_contrastive_losses, label="Valid Contrastive Loss")
    plt.plot(test_contrastive_losses, label="Test Contrastive Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.title(f"Contrastive Loss ({args.contrastive})")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"contrastive_loss_{args.contrastive}.png"))

    print('saved_model_number is: ' + str(saved_model_number))
    #print('seed: ', seed) 
     ## main task results
    result_helper(valid_fscores1, test_fscores1, valid_losses1, rf_main, lf_main, best_label1, best_pred1, best_mask1, 'main')
     # close files
     rf_main.close()
     lf_main.close()


