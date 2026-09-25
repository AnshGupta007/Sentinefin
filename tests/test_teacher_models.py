"""Unit tests for teacher-requested models: Hybrid XGBoost + BiLSTM, FinBERT, Proposed CNN-RNN."""

import numpy as np
import torch

from sentinefin.cnn_rnn import ProposedCNNRNN
from sentinefin.finbert_model import FinBERTClassifierHead
from sentinefin.hybrid_xgb_bilstm import BiLSTMFeatureExtractor, prepare_sequential_tensor


def test_bilstm_feature_extractor():
    batch_size = 8
    seq_len = 12
    input_dim = 32
    n_classes = 11

    model = BiLSTMFeatureExtractor(
        input_dim=input_dim,
        seq_len=seq_len,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        n_classes=n_classes,
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    logits, features = model(x)

    assert logits.shape == (batch_size, n_classes)
    assert features.shape == (batch_size, 4 * 64)


def test_prepare_sequential_tensor():
    vectors = np.random.randn(10, 384).astype(np.float32)
    seq = prepare_sequential_tensor(vectors, seq_len=12)
    assert seq.shape == (10, 12, 32)


def test_finbert_classifier_head():
    batch_size = 4
    n_classes = 11
    head = FinBERTClassifierHead(input_dim=768, hidden_dim=256, n_classes=n_classes)
    x = torch.randn(batch_size, 768)
    out = head(x)
    assert out.shape == (batch_size, n_classes)


def test_proposed_cnn_rnn():
    batch_size = 6
    input_length = 384
    n_classes = 11

    model = ProposedCNNRNN(
        input_length=input_length,
        conv_channels=32,
        rnn_hidden_dim=64,
        rnn_layers=1,
        n_classes=n_classes,
    )
    x = torch.randn(batch_size, input_length)
    logits = model(x)
    assert logits.shape == (batch_size, n_classes)
