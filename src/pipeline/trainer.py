from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

# Umbral de batches con NaN a partir del cual se re-ejecuta la época entera (retry).
NAN_SKIP_THRESHOLD = 5
# Máximo de reintentos de una misma época antes de avanzar sin guardar.
MAX_RETRIES_PER_EPOCH = 3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.dataset import TripletPointCloudDataset, normalize_unit_sphere, to_numpy, augment, sample_n
from src.models.triplet import triplet_loss_squared


def _get_embedding_from_model(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode"):
        return model.encode(x)
    if hasattr(model, "forward_once"):
        return model.forward_once(x)
    za, _, _ = model(x, x, x)
    return za


def compute_reference_embeddings_pointclouds(
    model: nn.Module,
    all_point_clouds,
    device: torch.device,
    n_points: int,
    samples_per_class: int = 5,
    use_augmentation: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    model.eval()
    reference_embeddings: Dict[str, np.ndarray] = {}
    reference_paths: Dict[str, List[str]] = {}

    class_to_indices = defaultdict(list)
    for idx, (folder, _, _) in enumerate(all_point_clouds):
        class_to_indices[folder].append(idx)

    with torch.no_grad():
        for cls, indices in class_to_indices.items():
            if len(indices) == 0:
                continue

            indices = np.array(indices)
            np.random.shuffle(indices)
            selected_indices = indices[: min(samples_per_class, len(indices))]

            embs = []
            paths = []

            for idx in selected_indices:
                folder, path, cloud = all_point_clouds[int(idx)]
                paths.append(path)

                pts = normalize_unit_sphere(to_numpy(cloud)).astype(np.float32)
                pts_proc = augment(pts, n_points) if use_augmentation else sample_n(pts, n_points)

                pc_tensor = torch.from_numpy(pts_proc.T).unsqueeze(0).float().to(device)

                emb = _get_embedding_from_model(model, pc_tensor)
                emb = emb.detach().cpu().numpy()

                if emb.ndim == 2 and emb.shape[0] == 1:
                    emb = emb[0]

                embs.append(emb)

            if embs:
                embs_arr = np.vstack(embs)
                reference_embeddings[cls] = embs_arr.mean(axis=0)
                reference_paths[cls] = paths

    return reference_embeddings, reference_paths


class TripletTrainingPipeline:
    """
    Pipeline de entrenamiento y logging (idéntico a Colab, pero a disco local).
    """

    def __init__(
        self,
        all_point_clouds,
        model_class,
        n_points: int,
        width: int,
        batch_size: int,
        lr: float,
        margin: float,
        epochs: int,
        clip_norm: float,
        seed: int,
        device: torch.device,
        runs_dir: str = "runs",
        val_size: float = 0.15,
        test_size: float = 0.15,
        run_name: str | None = None,
        early_stopping_patience: int | None = None,
    ):
        t0 = time.perf_counter()
        self.start_time = datetime.now()
        if run_name is not None:
            timestamp = run_name
        else:
            timestamp = self.start_time.strftime("run_%Y-%m-%d_%H-%M-%S")

        self.exp_dir = os.path.join(runs_dir, timestamp)
        os.makedirs(self.exp_dir, exist_ok=True)
        print(f"  [PROGRESO] Directorio de run: {self.exp_dir} ({time.perf_counter() - t0:.1f}s)", flush=True)

        self.config_path = os.path.join(self.exp_dir, "config.json")
        self.log_path = os.path.join(self.exp_dir, "training.log")
        self.csv_path = os.path.join(self.exp_dir, "metrics.csv")
        self.best_model_path = os.path.join(self.exp_dir, "model_best.pt")
        self.checkpoint_last_path = os.path.join(self.exp_dir, "checkpoint_last.pt")
        self.ref_emb_path = os.path.join(self.exp_dir, "reference_embeddings_train.npz")
        self.ref_paths_path = os.path.join(self.exp_dir, "reference_paths_train.json")

        self.splits_dir = os.path.join(self.exp_dir, "splits")
        os.makedirs(self.splits_dir, exist_ok=True)
        self.train_split_path = os.path.join(self.splits_dir, "train_paths.json")
        self.val_split_path = os.path.join(self.splits_dir, "val_paths.json")
        self.test_split_path = os.path.join(self.splits_dir, "test_paths.json")

        self.n_points = n_points
        self.ref_samples_per_class = 5
        self.ref_use_augmentation = False

        config = {
            "start_datetime": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_points": n_points,
            "width": width,
            "batch_size": batch_size,
            "lr": lr,
            "margin": margin,
            "epochs": epochs,
            "clip_norm": clip_norm,
            "seed": seed,
            "device": str(device),
            "total_clouds": len(all_point_clouds),
            "val_size": val_size,
            "test_size": test_size,
            "ref_samples_per_class": self.ref_samples_per_class,
            "ref_use_augmentation": self.ref_use_augmentation,
            "early_stopping_patience": early_stopping_patience,
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        self._log(f"Experiment directory: {self.exp_dir}", console=True)
        self._log(f"Start time: {config['start_datetime']}", console=True)
        self._log(f"Using n_points = {n_points}", console=True)
        self._log(f"Using width    = {width}", console=True)

        # -----------------------------
        # DATASET SPLIT 70/15/15 (train/val/test, estratificado por carpeta)
        # -----------------------------
        t0 = time.perf_counter()
        folders = sorted(list(set(f for f, _, _ in all_point_clouds)))

        train_clouds, val_clouds, test_clouds = [], [], []
        for folder in folders:
            class_clouds = [x for x in all_point_clouds if x[0] == folder]
            # Primero separar test (15%)
            train_val, test = train_test_split(class_clouds, test_size=test_size, random_state=seed)
            # Del resto (85%), separar val (15% del total = val_size/(1-test_size) del train_val)
            val_ratio = val_size / (1.0 - test_size)
            tr, va = train_test_split(train_val, test_size=val_ratio, random_state=seed)
            train_clouds.extend(tr)
            val_clouds.extend(va)
            test_clouds.extend(test)

        self.train_clouds = train_clouds
        self.val_clouds = val_clouds
        self.test_clouds = test_clouds
        print(
            f"  [PROGRESO] Split train/val/test: {len(train_clouds)} / {len(val_clouds)} / {len(test_clouds)} ({(time.perf_counter() - t0):.1f}s)",
            flush=True,
        )

        train_paths = [p for _, p, _ in self.train_clouds]
        val_paths = [p for _, p, _ in self.val_clouds]
        test_paths = [p for _, p, _ in self.test_clouds]

        # Solo guardar splits si no existen (para mantener consistencia al reanudar)
        if not os.path.exists(self.train_split_path):
            with open(self.train_split_path, "w", encoding="utf-8") as f:
                json.dump(train_paths, f, indent=4)
            self._log(f"Saved train split to {self.train_split_path}", console=True)
        else:
            self._log(f"Train split already exists, keeping existing: {self.train_split_path}", console=True)

        if not os.path.exists(self.val_split_path):
            with open(self.val_split_path, "w", encoding="utf-8") as f:
                json.dump(val_paths, f, indent=4)
            self._log(f"Saved val split   to {self.val_split_path}", console=True)
        else:
            self._log(f"Val split already exists, keeping existing: {self.val_split_path}", console=True)

        if not os.path.exists(self.test_split_path):
            with open(self.test_split_path, "w", encoding="utf-8") as f:
                json.dump(test_paths, f, indent=4)
            self._log(f"Saved test split  to {self.test_split_path}", console=True)
        else:
            self._log(f"Test split already exists, keeping existing: {self.test_split_path}", console=True)

        self._log(f"Train clouds: {len(train_clouds)}", console=True)
        self._log(f"Val clouds:   {len(val_clouds)}", console=True)
        self._log(f"Test clouds:  {len(test_clouds)}", console=True)
        self._log(f"Classes:      {len(folders)}", console=True)

        # -----------------------------
        # DATASETS Y DATALOADERS
        # -----------------------------
        t0 = time.perf_counter()
        self.train_ds = TripletPointCloudDataset(self.train_clouds, n_points=n_points, train=True)
        self.val_ds = TripletPointCloudDataset(self.val_clouds, n_points=n_points, train=False)
        print(f"  [PROGRESO] Datasets train/val creados ({time.perf_counter() - t0:.1f}s)", flush=True)

        t0 = time.perf_counter()
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )
        print(f"  [PROGRESO] DataLoaders creados ({time.perf_counter() - t0:.1f}s)", flush=True)

        # -----------------------------
        # MODEL
        # -----------------------------
        t0 = time.perf_counter()
        self.model = model_class(width=width).to(device)
        params = sum(p.numel() for p in self.model.parameters())
        print(f"  [PROGRESO] Modelo en {device} ({params} params) ({time.perf_counter() - t0:.1f}s)", flush=True)
        self._log(f"Model parameters: {params}", console=True)

        # CSV header (solo si no existe o está vacío)
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

        t0 = time.perf_counter()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        print(f"  [PROGRESO] Optimizer y scheduler listos ({time.perf_counter() - t0:.1f}s)", flush=True)

        self.device = device
        self.margin = margin
        self.epochs = epochs
        self.clip_norm = clip_norm
        self.best_val = float("inf")
        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0

        self.scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    def _log(self, text: str, console: bool = False) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        if console:
            print(text)

    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        """Guarda un checkpoint completo con todo el estado necesario para reanudar."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val": self.best_val,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(checkpoint, self.checkpoint_last_path)
        self._log(f"[CHECKPOINT] Saved checkpoint at epoch {epoch}", console=False)

    def _load_checkpoint(self) -> int | None:
        """
        Carga el checkpoint más reciente si existe.
        Retorna la época desde la cual continuar, o None si no hay checkpoint.
        """
        if not os.path.exists(self.checkpoint_last_path):
            return None

        try:
            checkpoint = torch.load(self.checkpoint_last_path, map_location=self.device)
            start_epoch = checkpoint["epoch"] + 1  # Continuar desde la siguiente época

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.best_val = checkpoint.get("best_val", float("inf"))
            self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

            if self.scaler is not None and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            self._log(
                f"[RESUME] Loaded checkpoint from epoch {checkpoint['epoch']}, "
                f"best_val={self.best_val:.6f}. Continuing from epoch {start_epoch}.",
                console=True,
            )
            return start_epoch
        except Exception as e:
            self._log(f"[WARNING] Failed to load checkpoint: {e}. Starting from scratch.", console=True)
            return None

    def _reload_for_retry(self) -> bool:
        """
        Restaura modelo, optimizer, scheduler y best_val desde checkpoint_last.pt.
        Se usa cuando re-ejecutamos una época por exceso de NaNs (retry de época).
        El checkpoint corresponde al fin de la época anterior, así que el estado queda
        listo para volver a correr la época actual desde el inicio.
        Retorna True si se cargó correctamente, False si no existía checkpoint.
        """
        if not os.path.exists(self.checkpoint_last_path):
            return False
        try:
            checkpoint = torch.load(self.checkpoint_last_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.best_val = checkpoint.get("best_val", float("inf"))
            self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
            if self.scaler is not None and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            return True
        except Exception:
            return False

    def train(self, resume: bool = True):
        """
        Entrena el modelo.

        Incluye manejo de NaNs por overflow:
        - Si la loss de un batch no es finita, se skipea el step (no se corrompen pesos).
        - Clipping incremental: al primer skip de la época se reduce clip_norm para el resto
          de la época, acotando pasos y reduciendo chance de más overflows.
        - Retry de época: si skip_count >= NAN_SKIP_THRESHOLD (5) o val_loss no finita,
          se recarga el estado desde el fin de la época anterior y se re-ejecuta la misma
          época con clip_norm aún más bajo, sin avanzar scheduler ni guardar hasta lograrlo.

        Args:
            resume: Si True, intenta cargar un checkpoint existente y continuar desde ahí.
        """
        start_epoch = 1

        if resume:
            loaded_epoch = self._load_checkpoint()
            if loaded_epoch is not None:
                start_epoch = loaded_epoch
                if start_epoch > self.epochs:
                    self._log(
                        f"Training already completed (epoch {start_epoch-1}/{self.epochs}). Skipping.",
                        console=True,
                    )
                    return

        for epoch in range(start_epoch, self.epochs + 1):
            epoch_success = False
            retry_count = 0
            early_stop_triggered = False

            while not epoch_success and retry_count <= MAX_RETRIES_PER_EPOCH:
                # --- Retry de época: recuperar pesos de la última época completada ---
                if retry_count > 0:
                    if epoch > 1 and self._reload_for_retry():
                        self.clip_norm = max(0.25, self.clip_norm * 0.5)
                        self._log(
                            f"[RETRY_EPOCH] epoch={epoch} retry={retry_count} "
                            f"new_clip_norm={self.clip_norm:.4f} (reload from end of epoch {epoch-1})",
                            console=True,
                        )
                    elif epoch == 1:
                        self._log(
                            f"[RETRY_EPOCH] epoch=1 no checkpoint to reload, advancing without saving",
                            console=True,
                        )
                        break
                    else:
                        self._log(
                            f"[RETRY_EPOCH] epoch={epoch} reload failed, advancing without saving",
                            console=True,
                        )
                        break

                skip_count = 0
                train_loss_sum = 0.0
                train_count = 0
                self.model.train()

                # ----- TRAIN -----
                for batch_idx, (pa, pp, pn) in enumerate(self.train_loader):
                    pa, pp, pn = pa.to(self.device), pp.to(self.device), pn.to(self.device)
                    self.optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast("cuda" if self.device.type == "cuda" else "cpu"):
                        za, zp, zn = self.model(pa, pp, pn)
                        loss = triplet_loss_squared(za, zp, zn, margin=self.margin)

                    if not torch.isfinite(loss).all():
                        skip_count += 1
                        # Clipping incremental: al primer NaN de la época, reducir clip_norm
                        # para el resto de la época y así acotar los pasos siguientes.
                        if skip_count == 1:
                            self.clip_norm = max(0.25, self.clip_norm * 0.5)
                            self._log(
                                f"[NAN_SKIP] epoch={epoch} batch_idx={batch_idx} skip_count=1 "
                                f"reducing_clip_norm_to={self.clip_norm:.4f}",
                                console=True,
                            )
                        elif skip_count == NAN_SKIP_THRESHOLD:
                            self._log(
                                f"[NAN_SKIP] epoch={epoch} batch_idx={batch_idx} "
                                f"skip_count={skip_count} (threshold={NAN_SKIP_THRESHOLD})",
                                console=True,
                            )
                        else:
                            self._log(
                                f"[NAN_SKIP] epoch={epoch} batch_idx={batch_idx} skip_count={skip_count}",
                                console=False,
                            )
                        continue

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                        self.optimizer.step()

                    train_loss_sum += loss.item() * pa.size(0)
                    train_count += pa.size(0)

                train_loss = train_loss_sum / max(train_count, 1)

                # ----- VAL -----
                self.model.eval()
                val_loss_sum = 0.0
                val_count = 0

                with torch.no_grad():
                    with torch.amp.autocast("cuda" if self.device.type == "cuda" else "cpu"):
                        for pa, pp, pn in self.val_loader:
                            pa, pp, pn = pa.to(self.device), pp.to(self.device), pn.to(self.device)
                            za, zp, zn = self.model(pa, pp, pn)
                            loss = triplet_loss_squared(za, zp, zn, margin=self.margin)
                            val_loss_sum += loss.item() * pa.size(0)
                            val_count += pa.size(0)

                val_loss = val_loss_sum / max(val_count, 1)

                # Decidir si la época fue aceptable o hay que hacer retry
                need_retry = skip_count >= NAN_SKIP_THRESHOLD or not math.isfinite(val_loss)
                if need_retry:
                    self._log(
                        f"[RETRY_EPOCH] epoch={epoch} skip_count={skip_count} threshold={NAN_SKIP_THRESHOLD} "
                        f"val_finite={math.isfinite(val_loss)}",
                        console=True,
                    )
                    if retry_count < MAX_RETRIES_PER_EPOCH:
                        retry_count += 1
                        continue
                    else:
                        self._log(
                            f"[WARNING] epoch={epoch} max_retries={MAX_RETRIES_PER_EPOCH} exceeded, advancing without saving",
                            console=True,
                        )
                        break

                epoch_success = True
                lr = self.scheduler.get_last_lr()[0]
                self.scheduler.step()

                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_loss, val_loss, lr])

                msg = f"[{epoch:02d}] train={train_loss:.6f}  val={val_loss:.6f}  lr={lr:.3e}"
                if skip_count > 0:
                    msg += f"  (skipped={skip_count})"
                self._log(msg, console=True)

                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self.epochs_without_improvement = 0
                    torch.save(self.model.state_dict(), self.best_model_path)
                    model_version_path = os.path.join(self.exp_dir, "model_version.json")
                    with open(model_version_path, "w", encoding="utf-8") as f:
                        json.dump({"epoch": epoch, "val_loss": float(val_loss)}, f, indent=2)
                    self._log(f"[BEST] Saved model with val_loss={val_loss:.6f}", console=True)
                else:
                    self.epochs_without_improvement += 1

                self._save_checkpoint(epoch, train_loss, val_loss, lr)

                if (
                    self.early_stopping_patience is not None
                    and self.epochs_without_improvement >= self.early_stopping_patience
                ):
                    self._log(
                        f"[EARLY_STOP] No improvement for {self.early_stopping_patience} epochs. "
                        f"Stopping at epoch {epoch}. Best val_loss = {self.best_val:.6f}",
                        console=True,
                    )
                    early_stop_triggered = True
                    break

            if early_stop_triggered:
                break

        self._log(f"Training finished. Best val_loss = {self.best_val:.6f}", console=True)

        # Marcar hasta qué época se entrenó (para carpetas de evaluación ep<N>)
        last_epoch_path = os.path.join(self.exp_dir, "last_epoch.json")
        with open(last_epoch_path, "w", encoding="utf-8") as f:
            json.dump({"epoch": epoch}, f, indent=2)

        # Reference embeddings en train
        self._log(
            f"Computing reference embeddings on train set "
            f"(samples_per_class={self.ref_samples_per_class}, "
            f"use_augmentation={self.ref_use_augmentation})",
            console=True,
        )

        ref_emb, ref_paths = compute_reference_embeddings_pointclouds(
            model=self.model,
            all_point_clouds=self.train_clouds,
            device=self.device,
            n_points=self.n_points,
            samples_per_class=self.ref_samples_per_class,
            use_augmentation=self.ref_use_augmentation,
        )

        if ref_emb:
            np.savez(self.ref_emb_path, **ref_emb)
            self._log(f"Saved reference embeddings for {len(ref_emb)} classes to {self.ref_emb_path}", console=True)
        else:
            self._log("WARNING: No reference embeddings were computed.", console=True)

        if ref_paths:
            with open(self.ref_paths_path, "w", encoding="utf-8") as f:
                json.dump(ref_paths, f, indent=4)
            self._log(f"Saved reference paths for {len(ref_paths)} classes to {self.ref_paths_path}", console=True)
        else:
            self._log("WARNING: No reference paths were computed.", console=True)

        return self.model
