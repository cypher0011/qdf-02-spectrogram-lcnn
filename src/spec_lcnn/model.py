"""Light CNN (LCNN) with Max-Feature-Map activations on a log-power spectrogram.

LCNN is the classic ASVspoof countermeasure architecture (Lavrentyeva et al.,
Interspeech 2017/2019). Max-Feature-Map (MFM) splits the channels in two and keeps
the element-wise max. It works as a learned feature selector, which helps when the
evidence is a small spectral artefact rather than a broad pattern.

The front-end is a *linear-frequency* log spectrogram (257 bins up to 8 kHz).
Mel scaling would compress the high-frequency region where many vocoder
artefacts live, so we keep it linear (Model 1's LTAS analysis agrees).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Spectrogram(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 160, win: int = 400):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, 1, F, frames)
        X = torch.stft(wav, self.n_fft, self.hop, self.win, self.window, return_complex=True)
        S = torch.log(X.abs().pow(2) + 1e-7)
        S = (S - S.mean(dim=(1, 2), keepdim=True)) / (S.std(dim=(1, 2), keepdim=True) + 1e-5)
        return S.unsqueeze(1)


class MFM(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


def conv_mfm(cin, cout, k, s=1, p=0):
    return nn.Sequential(nn.Conv2d(cin, 2 * cout, k, s, p), MFM())


class LCNN(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.front = Spectrogram()
        self.body = nn.Sequential(
            conv_mfm(1, 32, 5, 1, 2), nn.MaxPool2d(2),
            conv_mfm(32, 32, 1), nn.BatchNorm2d(32), conv_mfm(32, 48, 3, 1, 1), nn.MaxPool2d(2), nn.BatchNorm2d(48),
            conv_mfm(48, 48, 1), nn.BatchNorm2d(48), conv_mfm(48, 64, 3, 1, 1), nn.MaxPool2d(2),
            conv_mfm(64, 64, 1), nn.BatchNorm2d(64), conv_mfm(64, 32, 3, 1, 1), nn.BatchNorm2d(32),
            conv_mfm(32, 32, 1), nn.BatchNorm2d(32), conv_mfm(32, 32, 3, 1, 1), nn.MaxPool2d(2),
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1))

    def features(self, spec: torch.Tensor) -> torch.Tensor:
        return self.body(spec)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        h = self.body(self.front(wav))                    # (B, 32, F', T')
        h = h.amax(dim=2).mean(dim=2)                     # max over freq, mean over time
        return self.fc(self.drop(h)).squeeze(1)           # logit, higher = more fake

    def forward_spec(self, spec: torch.Tensor) -> torch.Tensor:
        h = self.body(spec).amax(dim=2).mean(dim=2)
        return self.fc(h).squeeze(1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
