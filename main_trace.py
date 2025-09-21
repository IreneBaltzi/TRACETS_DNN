import argparse
import datetime, time
import os, random, json
import math
import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, random_split

from torch.utils.data import Sampler 
from utils.datasets import read_data, labels2indices, build_datasets, build_datasets_sepsis
from utils.models import TRACE, TRACETS, TraceStyleEmbeddings
from utils.engine import train_one_epoch, evaluate, evaluate_challenge
from utils.losses import FocalLoss
from utils import misc
from utils.sampler import CustomSampler
from utils.config import setup
from fvcore.nn import FlopCountAnalysis, flop_count_table
from typing import List, Tuple
import torch
from utils.baselines import run_tabular_baselines

def sepsis_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    batch: list of (X, y)
      X: [T, F] or [1, T, F]
      y: [T] or [1, T]
    Returns:
      X_pad: [B, T_max, F]
      y_pad: [B, T_max]
      lengths: [B]
    """
    Xs, ys = [], []
    for X, y in batch:
        if X.dim() == 3 and X.size(0) == 1: X = X[0]  #  [T,F]
        if y.dim() == 2 and y.size(0) == 1: y = y[0]  #  [T]
        Xs.append(X)
        ys.append(y)

    B = len(Xs)
    T_max = max(x.size(0) for x in Xs)
    F = Xs[0].size(1)
    device = Xs[0].device

    X_pad = torch.zeros(B, T_max, F, dtype=Xs[0].dtype)
    y_pad = torch.zeros(B, T_max, dtype=ys[0].dtype)
    lengths = torch.zeros(B, dtype=torch.long)

    for i, (X, y) in enumerate(zip(Xs, ys)):
        T = X.size(0)
        X_pad[i, :T] = X
        y_pad[i, :T] = y
        lengths[i] = T

    return X_pad, y_pad, lengths

def split_pos_neg_indices_patient_level(dataset):
    pos_idx, neg_idx = [], []
    for i in range(len(dataset)):
        _, y = dataset[i]
        y_t = torch.as_tensor(y)
        is_pos = (y_t == 1).any().item()
        (pos_idx if is_pos else neg_idx).append(i)
    return pos_idx, neg_idx


class ThreeToOneEpochSampler(Sampler[int]):
    def __init__(
        self,
        pos_indices: list[int],
        neg_indices: list[int],
        neg_multiplier: int = 2,
        target_pos_count: int | None = None, 
        generator: torch.Generator | None = None,
        verbose: bool = True, 
    ):
        self.pos_indices = list(pos_indices)
        self.neg_indices = list(neg_indices)
        self.neg_multiplier = int(neg_multiplier)
        self.target_pos_count = target_pos_count or len(self.pos_indices)
        self.generator = generator
        self.verbose = verbose

        if len(self.pos_indices) == 0:
            raise ValueError("ThreeToOneEpochSampler: No positive patients found.")
        if self.neg_multiplier < 1:
            raise ValueError("ThreeToOneEpochSampler: neg_multiplier must be >= 1.")

        self.last_counts: tuple[int,int] | None = None
    def __iter__(self):
        P = min(self.target_pos_count, len(self.pos_indices))
        chosen_pos = random.sample(self.pos_indices, P)

        N_target = min(self.neg_multiplier * P, len(self.neg_indices))
        chosen_neg = random.sample(self.neg_indices, N_target)

        self.last_counts = (len(chosen_pos), len(chosen_neg))
        if self.verbose:
            P_, N_ = self.last_counts
            ratio = (N_ / P_) if P_ > 0 else float('inf')
            print(f"[sampler] epoch sample counts: positives={P_}  negatives={N_}  (~{ratio:.2f}:1)")

        epoch_indices = chosen_pos + chosen_neg
        if self.generator is not None:
            order = torch.randperm(len(epoch_indices), generator=self.generator).tolist()
        else:
            order = torch.randperm(len(epoch_indices)).tolist()
        for k in order:
            yield epoch_indices[k]

    def __len__(self):
        P = min(self.target_pos_count, len(self.pos_indices))
        N = min(self.neg_multiplier * P, len(self.neg_indices))
        return P + N

    # optional: expose counts safely
    def get_last_counts(self) -> tuple[int,int] | None:
        return self.last_counts

def get_args_parser():
    parser = argparse.ArgumentParser('Attention based melanoma/osc risk estimation on clinical data', add_help=True)
    parser.add_argument('--config_file', default='./config/tracets_sepsis.yaml', help='config file path')
    parser.add_argument('--output_dir', default='./results/experiment_name', help='path to save the output model')

    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--checkpoint', default='/path/to/model/ckpt_best.pth', help='Load model from checkpoint')
    
    return parser
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def main(args):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing {device} device\n")

    # fix the seed for reproducibility
    set_seed(args.train.seed)

    with open(args.data.metadata_dir, 'r') as f:
        feature_metadata = json.load(f)

    if args.data.dataset_name == 'sepsis':
        train_dataset, test_dataset, num_indices = build_datasets_sepsis(args, device='cpu')

        pos_idx, neg_idx = split_pos_neg_indices_patient_level(train_dataset)
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        print(f"[train] patient-level counts → positives={n_pos}  negatives={n_neg}")

        # target_pos = n_pos       
        # neg_mult  = getattr(args, "neg_multiplier", 3)

        # sampler = ThreeToOneEpochSampler(
        #     pos_indices=pos_idx,
        #     neg_indices=neg_idx,
        #     neg_multiplier=neg_mult,
        #     target_pos_count=target_pos,
        #     verbose=True,              
        # )

        dataloader_train = DataLoader(
            train_dataset,
            batch_size=args.train.train_batch_size,
            # sampler=sampler,              
            shuffle=True,
            drop_last=False,
            collate_fn=sepsis_collate_fn,
            pin_memory=(device == "cuda"),
        )

        dataloader_val = DataLoader(
            test_dataset,
            batch_size=args.train.val_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=sepsis_collate_fn,
            pin_memory=(device == "cuda"),
        )
    else:
        train_dataset, test_dataset, num_indices = build_datasets(args, device=device)
        if args.sampler == 'custom':
            y_train = train_dataset.tensors[1]
            train_positives = (y_train == 1).sum().item()
            positive_ratio = train_positives / len(train_dataset)
            positive_ratio = round(math.ceil(positive_ratio / 0.05) * 0.05, 2)
            custom_sampler_train = CustomSampler(train_dataset, args.train.train_batch_size, train=True, positive_ratio=positive_ratio)
            dataloader_train = DataLoader(train_dataset, batch_sampler=custom_sampler_train)
        else:
            dataloader_train = DataLoader(train_dataset, batch_size=args.train.train_batch_size, shuffle=True, drop_last=False)

        dataloader_val = DataLoader(test_dataset, batch_size=args.train.val_batch_size, shuffle=False, drop_last=False)

    print(f"\nTrain dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

   
    if args.baselines.enable and args.data.dataset_name == 'sepsis' and not args.eval:
        print("\n Running baselines (LogReg, XGBoost, LightGBM) ONLY...\n")
        Path(args.baselines.out_dir).mkdir(parents=True, exist_ok=True)
        run_tabular_baselines(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            output_dir=args.baselines.out_dir,
            window=args.baselines.window,
            run_lr=args.baselines.run_lr,
            run_xgb=args.baselines.run_xgb,
            run_lgbm=args.baselines.run_lgbm,
            threshold=args.baselines.threshold,
            horizon=getattr(args.baselines, "horizon", 0),
            tune_threshold=getattr(args.baselines, "tune_threshold", True),
            val_fraction=getattr(args.baselines, "val_fraction", 0.1),
            seed=getattr(args.baselines, "seed", 1),
        )
        return


    if args.data.dataset_name == 'sepsis':
        trace_embedder = TraceStyleEmbeddings(
            hidden_size=args.model.hidden_size,
            feature_metadata=feature_metadata,
            num_indices=num_indices,
            num_mode=args.model.num_mode,             
            use_num_norm=args.model.use_num_norm,
            use_cat_norm=args.model.use_cat_norm,
            checkbox_mode=args.model.checkbox_mode,
        )
        model = TRACETS(
            trace_embedder=trace_embedder,
            d_model=args.model.hidden_size,
            max_T=args.model.max_T if hasattr(args.model, "max_T") else 1024,
            n_layers=args.model.tran_layers,
            n_heads=args.model.heads,
            d_ff=args.model.mlp_ratio * args.model.hidden_size,
            dropout=args.model.dropout,
        )
    else:
        model = TRACE(
            hidden_size=args.model.hidden_size,
            feature_metadata=feature_metadata,
            num_indices=num_indices,
            num_mode=args.model.num_mode,
            num_labels=args.data.num_labels,
            dropout_p=args.model.dropout,
            cls_token=args.model.cls_token,
            tran_layers=args.model.tran_layers,
            heads=args.model.heads,
            mlp_ratio=args.model.mlp_ratio,
            use_num_norm=args.model.use_num_norm,
            use_cat_norm=args.model.use_cat_norm,
            checkbox_mode=args.model.checkbox_mode
        )

    if args.checkpoint and args.eval:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.checkpoint)
        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model, strict=False)
    
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nNumber of trainable params: {n_parameters}')

    ## Calculate GFLOPS
    # num_features = len(feature_metadata['continuous']) + len(feature_metadata['categorical'])
    # dummy_input = torch.ones(2, num_features).to('cuda').long() # change 2nd axis to the number of features
    # model.eval()
    # flops = FlopCountAnalysis(model.to('cuda'), dummy_input)
    # print(f"Total Flops: {flops.total() / 2}")
    # model.train()

    if args.optim.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.optim.lr, momentum=0.9, weight_decay=1e-3)
    elif args.optim.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim.lr)
    elif args.optim.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim.lr)
    else:
        raise ValueError(f"Unsupported optimzer '{args.optim.loss}'. Supported optimizer algorithms: 'adam', 'rmsprop', 'adamw'.")
    
    if args.optim.loss == 'bce':
        if args.optim.bce.use_pos_weight:
            pos_weight = torch.tensor([args.optim.bce.pos_weight], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none" if args.data.dataset_name == 'sepsis' else "mean")
        else:
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none' if args.data.dataset_name=='sepsis' else 'mean')
    elif args.optim.loss == 'focal':
        criterion = FocalLoss(alpha=args.optim.focal.alpha)
    elif args.optim.loss=="ce":
        criterion=torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function '{args.optim.loss}'. Supported loss functions: 'focal', 'bce'.")
    print(f"Criterion: {criterion}")

    loss_scaler = misc.NativeScaler()

    if args.optim.lr_scheduler == 'rop':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.9, patience=10)
    elif args.optim.lr_scheduler == 'ca':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.optim.epochs, eta_min=2e-6)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.eval:
        if args.metrics == 'challenge':
            
            challenge_stats = evaluate_challenge(dataloader_val, model, device, args=args)
            print(f"[challenge] {challenge_stats}")
        else:
            
            test_stats, cm = evaluate(dataloader_val, model, criterion, device, return_cm=True, features=feature_metadata, args=args)
            print(f"Accuracy: {test_stats['acc']:.1f}%  F1: {test_stats['f1']:.3f}  F1_w: {test_stats['f1_w']:.3f}")
            print(f"Confusion matrix:\n{cm}")
            if args.data.num_labels == 1:
                print(f"Balanced Acc: {test_stats['bal_acc']:.1f}%  Sens: {test_stats['rec_pos']:.3f}  Spec: {test_stats['rec_neg']:.3f}")
                print(f"Prec+: {test_stats['prec_pos']:.3f}  Prec-: {test_stats['prec_neg']:.3f}")

        if args.baselines.enable and args.data.dataset_name == 'sepsis':
            Path(args.baselines.out_dir).mkdir(parents=True, exist_ok=True)
            run_tabular_baselines(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                output_dir=args.baselines.out_dir,
                window=args.baselines.window,
                run_lr=args.baselines.run_lr,
                run_xgb=args.baselines.run_xgb,
                run_lgbm=args.baselines.run_lgbm,
                threshold=args.baselines.threshold,
                horizon=getattr(args.baselines, "horizon", 0),
                tune_threshold=getattr(args.baselines, "tune_threshold", True),
                val_fraction=getattr(args.baselines, "val_fraction", 0.1),
                seed=getattr(args.baselines, "seed", 1),
            )
        return
    
    print(f"Start training for {args.optim.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_f1 = 0.0
    best_auroc = 0.0
    best_auprc = 0.0
    if args.data.num_labels == 1:
        max_balanced_accuracy = 0.0
    best_utility = -1e9  
    for epoch in range(args.optim.epochs):
        train_stats = train_one_epoch(
            model, criterion, dataloader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None, args=args
        )

        # periodic checkpoint
        if args.output_dir and (epoch + 1) % 20 == 0:
            output_dir = Path(args.output_dir)
            checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % str(epoch))]
            for checkpoint_path in checkpoint_paths:
                misc.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict() if args.optim.lr_scheduler != 'rop' else None,
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
        if args.metrics == 'challenge':
            challenge_stats = evaluate_challenge(dataloader_val, model, device, args=args)

            
            improved_f1      = challenge_stats["f1"]      > max_f1
            improved_utility = challenge_stats["utility"] > best_utility
            improved_auroc   = challenge_stats["auroc"]   > best_auroc
            improved_auprc   = challenge_stats["auprc"]   > best_auprc

            if improved_f1:      max_f1      = challenge_stats["f1"]
            if improved_utility: best_utility = challenge_stats["utility"]
            if improved_auroc:   best_auroc   = challenge_stats["auroc"]
            if improved_auprc:   best_auprc   = challenge_stats["auprc"]

            print(
                "[best-so-far] "
                f"F1={max_f1:.3f} | Utility={best_utility:.4f} | "
                f"AUROC={best_auroc:.4f} | AUPRC={best_auprc:.4f}"
            )

            if args.optim.lr_scheduler == 'rop':
                lr_scheduler.step(challenge_stats["f1"])
            elif args.optim.lr_scheduler == 'ca':
                lr_scheduler.step()

            if max_f1 == challenge_stats["f1"]:
                output_dir = Path(args.output_dir)
                checkpoint_paths = [output_dir / ('ckpt_best.pth')]
                for checkpoint_path in checkpoint_paths:
                    misc.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict() if args.optim.lr_scheduler != 'rop' else None,
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        else:
            test_stats, cm = evaluate(dataloader_val, model, criterion, device, return_cm=True, features=feature_metadata)
            print(f"Accuracy of the network on the {len(test_dataset)} test samples: {test_stats['acc']:.1f}%")
            print(f'Confustion matrix: {cm}')
            max_accuracy = max(max_accuracy, test_stats["acc"])
            max_f1 = max(max_f1, test_stats["f1"])
            if args.data.num_labels == 1:
                max_balanced_accuracy = max(max_balanced_accuracy, test_stats["bal_acc"])
            print(f'Max accuracy: {max_accuracy:.2f}%, Max F1-Score: {max_f1:.3f}')

            if args.optim.lr_scheduler == 'rop':
                lr_scheduler.step(test_stats['f1'])  # choose your monitor
            elif args.optim.lr_scheduler == 'ca':
                lr_scheduler.step()

            if (max_f1 == test_stats["f1"]):
                output_dir = Path(args.output_dir)
                checkpoint_paths = [output_dir / ('ckpt_best.pth')]
                for checkpoint_path in checkpoint_paths:
                    misc.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict() if args.optim.lr_scheduler != 'rop' else None,
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

   
    if args.baselines.enable and args.data.dataset_name == 'sepsis':
        Path(args.baselines.out_dir).mkdir(parents=True, exist_ok=True)
        run_tabular_baselines(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        output_dir=args.baselines.out_dir,
        window=args.baselines.window,
        run_lr=args.baselines.run_lr,
        run_xgb=args.baselines.run_xgb,
        run_lgbm=args.baselines.run_lgbm,
        threshold=args.baselines.threshold,
        horizon=getattr(args.baselines, "horizon", 0),
        tune_threshold=getattr(args.baselines, "tune_threshold", True),
        val_fraction=getattr(args.baselines, "val_fraction", 0.1),
        seed=getattr(args.baselines, "seed", 1),
    )

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args = setup(args)
    main(args)
