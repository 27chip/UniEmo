import os
import time
import argparse
import torch
import numpy as np
from dataloader import DialogLoader
from model import JointModel, MaskedNLLLoss


def load_model(checkpoint_path, device, D_h=512, classification_model='EBERC', transformer_model='bert', transformer_mode='1', n_classes=6, n_subclasses=3, context_encoder_layer=6, no_lstm=False):
    model = JointModel(D_h, classification_model, transformer_model, transformer_mode, n_classes, n_subclasses, context_encoder_layer, False, False, no_lstm)
    model.to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    # try common nested keys used by different saving conventions
    state = None
    for k in ('state_dict', 'model', 'model_state_dict', 'state'):
        if k in ckpt:
            state = ckpt[k]
            break
    if state is None:
        state = ckpt

    # if state itself is a wrapper (contains non-tensor values), try to find nested dict of tensors
    def is_state_dict(d):
        if not isinstance(d, dict):
            return False
        for v in d.values():
            if isinstance(v, torch.Tensor):
                return True
        return False

    if not is_state_dict(state):
        # search one level deeper
        if isinstance(state, dict):
            for v in state.values():
                if is_state_dict(v):
                    state = v
                    break

    # normalize keys (remove DataParallel/module prefix)
    normalized = {}
    if isinstance(state, dict):
        for k, v in state.items():
            nk = k.replace('module.', '')
            normalized[nk] = v
    else:
        raise RuntimeError('No valid state_dict found in checkpoint')

    # attempt to load; use strict=False to allow partial matches and different key prefixes
    try:
        model.load_state_dict(normalized, strict=False)
    except Exception as e:
        raise RuntimeError(f'Failed to load checkpoint state_dict: {e}')
    model.eval()
    return model


def configure_dataloader_for_eval(dataset, classify, batch_size):
    base = './datasets/' + dataset + '/'
    test_loader = DialogLoader(
        base + dataset + '_test_utterances.tsv',
        base + dataset + '_test_' + classify + '.tsv',
        base + dataset + '_test_subtask01_index.tsv',
        base + dataset + '_test_loss_mask.tsv',
        base + dataset + '_test_speakers.tsv',
        base + dataset + '_test_v_diff.tsv',
        base + dataset + '_test_a_diff.tsv',
        base + dataset + '_test_d_diff.tsv',
        base + dataset + '_test_valence.tsv',
        base + dataset + '_test_arousal.tsv',
        base + dataset + '_test_dominance.tsv',
        batch_size,
        shuffle=False
    )
    return test_loader


def evaluate(model, dataloader, device):
    import torch.nn.functional as F
    from sklearn.metrics import f1_score, accuracy_score

    preds_all = []
    labels_all = []
    masks_all = []

    with torch.no_grad():
        for conversations, label, subindex, loss_mask, speaker_mask, diff_valence, diff_arousal, diff_dominance, valence, arousal, dominance in dataloader:
            lengths = [len(item) for item in conversations]
            max_len = max(lengths)
            umask = torch.zeros(len(lengths), max_len).long().to(device)
            for j in range(len(lengths)):
                umask[j, :lengths[j]] = 1

            # pad subindex to shape (max_len, batch) with -1
            subindex_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in subindex],
                                                              batch_first=False, padding_value=-1).long().to(device)
            # pad speaker mask (qmask) to shape (batch, max_len)
            qmask = torch.nn.utils.rnn.pad_sequence([
                torch.tensor([int(s) for s in item]) for item in speaker_mask
            ], batch_first=True).long().to(device)

            # call model with signature: (conversations, subindex, lengths, umask, qmask)
            outputs = model(conversations, subindex_tensor, lengths, umask, qmask)
            # model expected to return log_prob1 and other predictions like in train3
            # Try unpack
            if isinstance(outputs, tuple) or isinstance(outputs, list):
                log_prob1 = outputs[0]
            else:
                log_prob1 = outputs

            lp1_ = log_prob1.transpose(0, 1).contiguous().view(-1, log_prob1.size()[2])
            labels1 = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in label], batch_first=True, padding_value=0).view(-1)
            loss_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in loss_mask], batch_first=True, padding_value=0).view(-1)

            preds = torch.argmax(lp1_, 1).cpu().numpy()
            labels_np = labels1.cpu().numpy()
            masks_np = loss_mask.cpu().numpy()

            preds_all.append(preds)
            labels_all.append(labels_np)
            masks_all.append(masks_np)

    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    masks = np.concatenate(masks_all)
    acc = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    f1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    return acc, f1, preds, labels, masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--checkpoint', required=True, help='Path to model .pth')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', default='bert')
    parser.add_argument('--mode', default='1')
    parser.add_argument('--no_lstm', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, transformer_model=args.model, transformer_mode=args.mode, no_lstm=args.no_lstm)
    test_loader = configure_dataloader_for_eval(args.dataset, 'emotion', args.batch_size)
    acc, f1, preds, labels, masks = evaluate(model, test_loader, device)
    print(f"Evaluation on {args.dataset} — Accuracy: {acc}%, F1(weighted): {f1}%")

    # print only statistics for mask==1 positions
    from sklearn.metrics import confusion_matrix, classification_report
    mask_idx = (masks != 0)
    y_true = labels[mask_idx]
    y_pred = preds[mask_idx]
    cm = confusion_matrix(y_true, y_pred)
    print('\nConfusion matrix:')
    print(cm)
    #  iemocap
    if args.dataset == 'iemocap':
        target_names = ['ang', 'neu', 'hap', 'sad', 'fru', 'exc']
    else:
        # 默认用数字标签
        max_label = max(int(y_true.max()), int(y_pred.max()))
        target_names = [str(i) for i in range(max_label+1)]
    print('\nClassification report:')
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))


if __name__ == '__main__':
    main()
