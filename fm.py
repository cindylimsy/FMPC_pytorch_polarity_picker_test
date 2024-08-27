from typing import Any

import numpy as np
import torch
import torch.nn as nn

from seisbench.models.base import WaveformModel


class FM_model(WaveformModel):

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["stride"] = (_annotate_args["stride"][0], 1) # When applied to data > 4 secs in length

    def __init__(
        self,
        in_channels=1,
        classes=3,
        component="Z",
        phases="UDK",
        eps=1e-10,
        sampling_rate=100,
        pred_sample=200,
        original_compatible=True,
        filter_args=["bandpass"],
        filter_kwargs={'freqmin': 1, 'freqmax': 20, 'zerophase': False},
        **kwargs,
    ):
        
        super().__init__(
            # citation=citation,
            output_type="point",
            component_order=component,
            in_samples=400,
            pred_sample=pred_sample,
            sampling_rate=sampling_rate,
            labels=phases,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.eps = eps
        self.original_compatible = original_compatible
        self.filter_args = filter_args
        self.filter_kwargs = filter_kwargs
        self._phases = phases
        if phases is not None and len(phases) != classes:
            raise ValueError(
                f"Number of classes ({classes}) does not match number of labels ({len(phases)})."
            )

        self.conv1 = nn.Conv1d(in_channels, 32, 21, padding=10)
        self.bn1 = nn.BatchNorm1d(32, eps=1e-3) # Confirmed this is what original model uses
        self.conv2 = nn.Conv1d(32, 64, 15, padding=7)
        self.bn2 = nn.BatchNorm1d(64, eps=1e-3)
        self.conv3 = nn.Conv1d(64, 128, 11, padding=5)
        self.bn3 = nn.BatchNorm1d(128, eps=1e-3)

        self.fc1 = nn.Linear(6400, 512)
        self.bn4 = nn.BatchNorm1d(512, eps=1e-3)
        self.fc2 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512, eps=1e-3)
        self.fc3 = nn.Linear(512, classes)

        self.activation = torch.relu

        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x, logits=False):
        # Max normalization - confirmed this is what paper says
        x = x / (
            torch.max(
                torch.max(torch.abs(x), dim=-1, keepdims=True)[0], dim=-2, keepdims=True
            )[0]
            + self.eps
        )
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))

        if self.original_compatible:
            # Permutation is required to be consistent with the following fully connected layer
            x = x.permute(0, 2, 1)
        x = torch.flatten(x, 1)

        x = self.activation(self.bn4(self.fc1(x)))
        x = self.activation(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        if logits:
            return x
        else:
            if self.classes == 1:
                return torch.sigmoid(x)
            else:
                return torch.softmax(x, -1)
            
            
    @property
    def phases(self):
        if self._phases is not None:
            return self._phases
        else:
            return list(range(self.classes))
    
            