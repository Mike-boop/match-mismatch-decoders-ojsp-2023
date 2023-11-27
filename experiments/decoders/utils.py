import os
import torch
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset


class SessionDataset(Dataset):

    def __init__(
            self,
            eeg_session_file,
            feature_name,
            fs,
            dtype=np.float32,
            hop_length=None,
            spacing=None,
            speaker = None,
            window_length=None,
            attention_task=False,
            distractor=False):

        super().__init__()

        self.eeg_session_file = eeg_session_file
        self.attention_task = attention_task
        self.dtype = dtype
        
        split, sub, trial, condition, stimulus = os.path.basename(eeg_session_file).split('_-_')[:5]

        if speaker is not None:
            self.story = np.load(
                os.path.join(os.path.dirname(eeg_session_file), f'{split}_-_{stimulus}_{speaker}_-_{feature_name}.npy')
            )
        else:
            self.story = np.load(
                os.path.join(os.path.dirname(eeg_session_file), f'{split}_-_{stimulus}_-_{feature_name}.npy')
            )


        if attention_task:
            self.distractor = np.load(
                os.path.join(os.path.dirname(eeg_session_file), f'{split}_-_{stimulus}_distractor_-_{feature_name}.npy')
            )

        self.eeg = np.load(eeg_session_file)

        if window_length is None:
            self.window_size = fs*3
        else:
            self.window_size = window_length

        if spacing is None:
            self.spacing = self.window_size+fs
        else:
            self.spacing = spacing

        if hop_length is None:
            self.hop_length = fs
        else:
            self.hop_length = hop_length


    def __len__(self):

        if self.attention_task:
            min_n_samples = min(len(self.eeg), len(self.story), len(self.distractor))
        else:
            min_n_samples = min(len(self.eeg), len(self.story))
        return (min_n_samples - self.window_size - self.spacing)//self.hop_length
    
    
    def __getitem__(self, index):
        matched_segment_onset = index*self.hop_length
        mismatched_segment_onset = matched_segment_onset+self.spacing

        eeg_segment = self.eeg[matched_segment_onset:matched_segment_onset+self.window_size]
        matched_segment = self.story[matched_segment_onset:matched_segment_onset+self.window_size]

        if self.attention_task:
            mismatched_segment = self.distractor[matched_segment_onset:matched_segment_onset+self.window_size]
        else:
            mismatched_segment = self.story[mismatched_segment_onset:mismatched_segment_onset+self.window_size]

        return (eeg_segment, matched_segment, mismatched_segment)

        
def get_session_datasets_from_session_files(session_files_list, feature_name, fs=64, shuffle=True, hop_length=None, attention_task=False, distractor=False, window_length=None):

    # to feed to ChainDataset constructor
    datasets = [
        SessionDataset(file,
                       feature_name=feature_name,
                       fs=fs,
                       hop_length=hop_length,
                       attention_task=attention_task,
                       distractor=distractor,
                       window_length=window_length)

        for file in session_files_list
        ]
    
    if shuffle:
        indices = np.random.permutation(len(datasets))
        datasets = [datasets[i] for i in indices]

    return datasets


def training_loop(model, optimizer, loss_fn, train_loader, tqdm_description_preamble = "", device='cuda'):

    train_loop = tqdm(train_loader)
    losses = []
    accs = []

    model.train()

    for b, (eeg, m, mm) in enumerate(train_loop):

        eeg = torch.transpose(eeg.to(device, dtype=torch.float), 1, 2)
        m = torch.transpose(m.to(device, dtype=torch.float), 1, 2)
        mm = torch.transpose(mm.to(device, dtype=torch.float), 1, 2)

        targets = torch.hstack(
            [torch.ones((m.shape[0],), device=device), torch.zeros((m.shape[0],), device=device)]
            )

        pred = model(torch.vstack([eeg, eeg]), torch.vstack([m, mm]), torch.vstack([mm, m])).flatten()

        optimizer.zero_grad()
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()

        pred = pred>0
        tmp_acc = (pred==targets).cpu()
        accs.append(torch.sum(tmp_acc)/len(tmp_acc))

        losses.append(loss.item())

        if b%50 == 0:
            train_loop.set_description(tqdm_description_preamble+" "+f"{b}it/{len(train_loader.dataset)}")
            train_loop.set_postfix(loss=np.mean(losses), acc=np.mean(accs))
    
    return model.state_dict(), np.mean(accs), np.mean(losses)


def validation_loop(model, loss_fn, val_loader, device='cuda'):

    # validation
    val_loop = tqdm(val_loader)
    accs = []
    losses = []

    model.eval()

    with torch.no_grad():

        tmp = []

        for b, (eeg, m, mm) in enumerate(val_loop):

            eeg = torch.transpose(eeg.to(device, dtype=torch.float), 1, 2)
            m = torch.transpose(m.to(device, dtype=torch.float), 1, 2)
            mm = torch.transpose(mm.to(device, dtype=torch.float), 1, 2)

            targets = torch.hstack(
                [torch.ones((m.shape[0],), device=device), torch.zeros((m.shape[0],), device=device)]
                )

            pred = model(torch.vstack([eeg, eeg]), torch.vstack([m, mm]), torch.vstack([mm, m])).flatten()

            loss = loss_fn(pred, targets)
            losses.append(loss.item())

            pred = pred>0
            tmp_acc = (pred==targets).cpu()
            tmp.append(tmp_acc)
            accs.append(torch.sum(tmp_acc)/len(tmp_acc))

            if b%50 == 0:
                val_loop.set_description(f"Validating {b}it/{len(val_loader.dataset)}")
                val_loop.set_postfix(loss=np.mean(losses), acc=np.mean(accs))
            val_loop.set_description(f"Validating {b}it/{len(val_loader.dataset)}")
            val_loop.set_postfix(loss=np.mean(losses), acc=np.mean(accs))

        accs = torch.hstack(accs)
        acc=torch.sum(accs)/len(accs)
        tmp = np.hstack(tmp)

        return tmp.sum()/len(tmp), np.mean(losses)
    

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)