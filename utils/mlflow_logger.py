import mlflow
from pathlib import Path
import deepxde as dde
import numpy as np
import torch

class MLflowLogger(dde.callbacks.Callback):
    """
    Loguje do MLflow w trakcie treningu:
      - straty całkowite i składowe (train/test)
      - zmienne materiałowe (C1, C2, C3) jeśli istnieją
      - checkpoint artefaktów co 'artifact_every' (opcjonalnie)
    """
    def __init__(self, *, every=100, artifact_every=None, vars_to_log=None):
        super().__init__()
        self.every = int(every)
        self.artifact_every = None if artifact_every is None else int(artifact_every)
        self.vars_to_log = vars_to_log or []  # np. [C1, C2, C3]
        self._last_logged_step = -1

    def _safe_float(self, v):
        try:
            if hasattr(v, "numpy"):
                v = v.numpy()
            if hasattr(v, "item"):
                v = v.item()
            return float(v)
        except Exception:
            try:
                return float(v)
            except Exception:
                return np.nan

    def _log_losses(self, steps, loss_train, loss_test):
        """Obsługuje formaty: liczba, lista, lista list (DeepXDE bywa różne)."""
        step = int(steps[-1]) if len(steps) else 0

        # train
        lt = loss_train[-1] if len(loss_train) else None
        if lt is not None:
            if isinstance(lt, (list, tuple, np.ndarray)):
                mlflow.log_metric("loss_train_total", self._safe_float(np.sum(lt)), step=step)
                for i, comp in enumerate(lt):
                    mlflow.log_metric(f"loss_train_{i}", self._safe_float(comp), step=step)
            else:
                mlflow.log_metric("loss_train_total", self._safe_float(lt), step=step)

        # test
        ltst = loss_test[-1] if len(loss_test) else None
        if ltst is not None:
            if isinstance(ltst, (list, tuple, np.ndarray)):
                mlflow.log_metric("loss_test_total", self._safe_float(np.sum(ltst)), step=step)
                for i, comp in enumerate(ltst):
                    mlflow.log_metric(f"loss_test_{i}", self._safe_float(comp), step=step)
            else:
                mlflow.log_metric("loss_test_total", self._safe_float(ltst), step=step)

        # GPU pamięć (jeśli CUDA)
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1024**2)
            mlflow.log_metric("gpu_mem_alloc_mb", self._safe_float(mem_alloc), step=step)

        # Zmienne materiałowe (C1/C2/C3 itp.)
        for v in self.vars_to_log:
            try:
                val = getattr(v, "value", v)
                mlflow.log_metric(v.name if hasattr(v, "name") else "var", self._safe_float(val), step=step)
            except Exception:
                pass

    def on_epoch_end(self):
        # dostęp do historii przez model
        hist = getattr(self.model, "losshistory", None)
        if hist is None or not len(hist.steps):
            return

        step = int(hist.steps[-1])
        if self.every and (step - self._last_logged_step) < self.every:
            return

        self._log_losses(hist.steps, hist.loss_train, hist.loss_test)
        self._last_logged_step = step

        # Co pewien czas dorzuć artefakty (np. zmiany zmiennych)
        if self.artifact_every and (step % self.artifact_every == 0):
            for p in Path(".").glob("variable_history*"):
                if p.exists():
                    mlflow.log_artifact(str(p))

    def on_train_end(self):
        # Loguj finalne artefakty na koniec
        for p in Path(".").glob("variable_history*"):
            if p.exists():
                mlflow.log_artifact(str(p))
