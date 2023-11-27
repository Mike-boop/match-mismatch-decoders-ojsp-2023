import torch.nn as nn
import torch

class DilatedConvNet(nn.Module):

    '''
    This is the dilated convolutional model of Accou et al which
    we used as a baseline
    '''
    
    def __init__(self,
                 n_eeg_channels=64,
                 n_stimfeat_channels=1,
                 n_layers=3,
                 kernel_size=3,
                 spatial_filters=8,
                 dilation_filters=16):
        
        super().__init__()

        self.n_layers = n_layers

        self.eeg_spatial_filter = nn.Conv1d(n_eeg_channels, spatial_filters, kernel_size=1)

        self.eeg_projections = nn.ModuleList(
            [nn.Conv1d(spatial_filters, dilation_filters, kernel_size=kernel_size)] +\
                  [nn.Conv1d(dilation_filters, dilation_filters,kernel_size=kernel_size, dilation=kernel_size**i) for i in range(1,n_layers)]
                  )
        
        self.stimfeat_projections = nn.ModuleList(
            [nn.Conv1d(n_stimfeat_channels, dilation_filters, kernel_size=kernel_size)] +\
                  [nn.Conv1d(dilation_filters, dilation_filters,kernel_size=kernel_size, dilation=kernel_size**i) for i in range(1,n_layers)]
                  )

        
        self.activation = nn.ReLU()

        self.linear = nn.Linear(dilation_filters**2*2, 1)

    
    def forward(self, eeg, stimfeat_1, stimfeat_2):

        eeg = self.eeg_spatial_filter(eeg)

        for i in range(self.n_layers):

            eeg = self.eeg_projections[i](eeg)
            stimfeat_1 = self.stimfeat_projections[i](stimfeat_1)
            stimfeat_2 = self.stimfeat_projections[i](stimfeat_2)

            eeg = self.activation(eeg)
            stimfeat_1 = self.activation(stimfeat_1)
            stimfeat_2 = self.activation(stimfeat_2)

        eeg = nn.functional.normalize(eeg, dim=-1)        
        stimfeat_1 = nn.functional.normalize(stimfeat_1, dim=-1)
        stimfeat_2 = nn.functional.normalize(stimfeat_2, dim=-1)

        inner_1 = torch.bmm(eeg, torch.transpose(stimfeat_1, 1, 2))
        inner_2 = torch.bmm(eeg, torch.transpose(stimfeat_2, 1, 2))

        features = torch.hstack(
            [inner_1.flatten(start_dim=1), inner_2.flatten(start_dim=1)]
            )
        
        out = self.linear(features)

        return out
    

class DilatedConvNetSymmetrisedOutputs(nn.Module):


    '''
    This is the architecture which we used for the envelope-based
    and FFR-based decoders.
    '''
    
    def __init__(self,
                 n_eeg_channels=64,
                 n_stimfeat_channels=1,
                 n_layers=3,
                 kernel_size=3,
                 spatial_filters=8,
                 dilation_filters=16):
        
        super().__init__()

        self.n_layers = n_layers

        self.eeg_spatial_filter = nn.Conv1d(n_eeg_channels, spatial_filters, kernel_size=1, bias=False)


        self.eeg_projections = nn.ModuleList(
            [nn.Conv1d(spatial_filters, dilation_filters, kernel_size=kernel_size, groups=spatial_filters)] +\
                  [nn.Conv1d(dilation_filters, dilation_filters,kernel_size=kernel_size, dilation=kernel_size**i) for i in range(1,n_layers)]
                  )
        
        self.stimfeat_projections = nn.ModuleList(
            [nn.Conv1d(n_stimfeat_channels, dilation_filters, kernel_size=kernel_size)] +\
                  [nn.Conv1d(dilation_filters, dilation_filters,kernel_size=kernel_size, dilation=kernel_size**i) for i in range(1,n_layers)]
        )
        
        self.activation = nn.ReLU()

        self.linear = nn.Linear(dilation_filters**2, 1, bias=False)

    
    def forward(self, eeg, stimfeat_1, stimfeat_2):

        eeg = self.eeg_spatial_filter(eeg)

        for i in range(self.n_layers):

            eeg = self.eeg_projections[i](eeg)
            stimfeat_1 = self.stimfeat_projections[i](stimfeat_1)
            stimfeat_2 = self.stimfeat_projections[i](stimfeat_2)

            eeg = self.activation(eeg)
            stimfeat_1 = self.activation(stimfeat_1)
            stimfeat_2 = self.activation(stimfeat_2)

        eeg = nn.functional.normalize(eeg, dim=-1)        
        stimfeat_1 = nn.functional.normalize(stimfeat_1, dim=-1)
        stimfeat_2 = nn.functional.normalize(stimfeat_2, dim=-1)

        inner_1 = torch.bmm(eeg, torch.transpose(stimfeat_1, 1, 2))
        inner_2 = torch.bmm(eeg, torch.transpose(stimfeat_2, 1, 2))

        
        out = self.linear(
            inner_1.flatten(start_dim=1)-inner_2.flatten(start_dim=1)
            )

        return out