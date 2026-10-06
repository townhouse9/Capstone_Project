"""
LLMFriendlyLogger: Production-ready PyTorch training logger optimized for Gemini 3.8 Flash parsing.

Provides structured, machine-readable telemetry on:
1. Training and validation loss progression.
2. Tensor layer names and parameter dimensions.
3. Gradient statistics (mean absolute gradient, vanishing/exploding alerts, NaN/Inf checks).
"""

from __future__ import annotations

import time
import math
from typing import Optional, Dict, Any
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = object
    TORCH_AVAILABLE = False


class LLMFriendlyLogger:
    """
    A structured, diagnostic PyTorch logger designed for seamless LLM context consumption.
    Formats terminal output with explicit structural anchors so agents can instantly
    detect dimension mismatches, vanishing gradients, and numerical instabilities.
    """

    def __init__(self,
                 model: nn.Module,
                 vanishing_threshold: float = 1e-7,
                 exploding_threshold: float = 1e2):
        """
        Args:
            model: The PyTorch nn.Module being monitored.
            vanishing_threshold: Threshold below which gradients are flagged as vanishing.
            exploding_threshold: Threshold above which gradients are flagged as exploding.
        """
        self.model = model
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.epoch_start_time: Optional[float] = None

    def on_epoch_start(self, epoch: int, total_epochs: Optional[int] = None) -> None:
        """Records epoch start timestamp."""
        self.epoch_start_time = time.time()

    def log_epoch(self,
                  epoch: int,
                  train_loss: float,
                  val_loss: Optional[float] = None,
                  learning_rate: Optional[float] = None,
                  total_epochs: Optional[int] = None,
                  extra_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Logs a structured epoch summary followed by tensor parameter states and gradient checks.

        Args:
            epoch: Current epoch number (1-indexed).
            train_loss: Training loss for the epoch.
            val_loss: Validation loss for the epoch (optional).
            learning_rate: Current optimizer learning rate (optional).
            total_epochs: Total number of epochs (optional).
            extra_metrics: Additional metrics to log (e.g., accuracy, perplexity).
        """
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0
        epoch_str = f"{epoch}/{total_epochs}" if total_epochs else f"{epoch}"

        print(f"\n{'='*25} EPOCH SUMMARY {'='*25}")
        print(f"Epoch: {epoch_str} | Train Loss: {train_loss:.6f}", end="")
        if val_loss is not None:
            print(f" | Val Loss: {val_loss:.6f}", end="")
        if learning_rate is not None:
            print(f" | LR: {learning_rate:.6e}", end="")
        print(f" | Elapsed: {elapsed:.2f}s")

        if extra_metrics:
            extra_str = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                   for k, v in extra_metrics.items())
            print(f"Metrics -> {extra_str}")

        # Parameter Dimension & Gradient Telemetry
        self._inspect_parameters_and_gradients()
        print(f"{'='*65}\n", flush=True)

    def _inspect_parameters_and_gradients(self) -> None:
        """Loops through active parameters, logging shapes, gradient norms, and health flags."""
        print("\nTensors -> State Check:")
        has_anomalies = False
        anomaly_reports = []

        for name, param in self.model.named_parameters():
            shape_list = list(param.shape)
            req_grad = param.requires_grad

            if not req_grad:
                print(f"  [Layer] {name:<35} | Shape: {shape_list} | Status: FROZEN")
                continue

            if param.grad is None:
                print(f"  [Layer] {name:<35} | Shape: {shape_list} | Status: NO GRADIENT DETECTED")
                continue

            grad_data = param.grad.detach()
            abs_grad_mean = grad_data.abs().mean().item()
            grad_max = grad_data.abs().max().item()

            status = "HEALTHY"
            if math.isnan(abs_grad_mean) or math.isinf(abs_grad_mean):
                status = "CRITICAL: NaN/Inf GRADIENT"
                has_anomalies = True
                anomaly_reports.append(f"CRITICAL: {name} contains NaN/Inf gradients.")
            elif abs_grad_mean < self.vanishing_threshold:
                status = f"WARNING: VANISHING GRADIENT (< {self.vanishing_threshold:.1e})"
                has_anomalies = True
                anomaly_reports.append(f"VANISHING: {name} mean |grad| = {abs_grad_mean:.2e}")
            elif abs_grad_mean > self.exploding_threshold:
                status = f"WARNING: EXPLODING GRADIENT (> {self.exploding_threshold:.1e})"
                has_anomalies = True
                anomaly_reports.append(f"EXPLODING: {name} mean |grad| = {abs_grad_mean:.2e}")

            print(f"  [Layer] {name:<35} | Shape: {str(shape_list):<18} | "
                  f"Mean |Grad|: {abs_grad_mean:.6e} | Max |Grad|: {grad_max:.6e} | [{status}]")

        print("\nGradients -> Health Check:")
        if has_anomalies:
            print("  Status: ANOMALIES DETECTED")
            for alert in anomaly_reports:
                print(f"  - {alert}")
        else:
            print("  Status: ALL PARAMETERS HEALTHY (No vanishing, exploding, or NaN/Inf gradients)")


# =====================================================================
# Standalone Verification Test
# =====================================================================
if __name__ == "__main__":
    print("Testing LLMFriendlyLogger...")

    if not TORCH_AVAILABLE:
        print("[INFO] PyTorch is not installed in this Python environment.")
        print("[INFO] LLMFriendlyLogger class is ready for use once PyTorch is available (`pip install torch`).")
        print("Verification completed successfully (graceful import check passed).")
    else:
        print("PyTorch detected. Running synthetic Transformer forward/backward pass...")
        class ToyTransformerBlock(nn.Module):
            def __init__(self, embed_dim=32, n_heads=4):
                super().__init__()
                self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
                self.mlp = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Linear(embed_dim * 2, embed_dim)
                )
                self.norm = nn.LayerNorm(embed_dim)

            def forward(self, x):
                attn_out, _ = self.attn(x, x, x)
                x = self.norm(x + attn_out)
                return x + self.mlp(x)

        model = ToyTransformerBlock()
        logger = LLMFriendlyLogger(model)

        dummy_input = torch.randn(2, 8, 32)
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()

        logger.on_epoch_start(1, total_epochs=5)
        logger.log_epoch(
            epoch=1,
            train_loss=float(loss.item()),
            val_loss=0.854213,
            learning_rate=1e-3,
            total_epochs=5,
            extra_metrics={"accuracy": 0.9250}
        )
        print("PyTorch verification completed successfully.")
