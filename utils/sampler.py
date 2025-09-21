from torch.utils.data import Sampler
import random
import torch

class CustomSampler(Sampler[int]):
    def __init__(self, dataset, batch_size, train=True, positive_ratio=0.1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.train = train

        self.targets = dataset.tensors[1]

        self.positive_indices = torch.where(self.targets == 1)[0]
        self.negative_indices = torch.where(self.targets == 0)[0]

        if train:
            self.num_positive_per_batch = max(1, int(batch_size * self.positive_ratio))
            self.num_negative_per_batch = batch_size - self.num_positive_per_batch
            self.num_batches = len(self.positive_indices) // self.num_positive_per_batch
            self.remaining_positives = len(self.positive_indices) % self.num_positive_per_batch
        else:
            self.num_positive_samples = len(self.positive_indices)
            self.num_negative_samples = len(self.negative_indices)
            self.num_batches = (self.num_positive_samples+self.num_negative_samples) // self.batch_size
            # self.num_batches = (self.num_positive_samples*2) // self.batch_size

    def shuffle_indices(self):
        self.positive_indices = self.positive_indices[torch.randperm(len(self.positive_indices))]
        self.negative_indices = self.negative_indices[torch.randperm(len(self.negative_indices))]
        
    def __iter__(self):
        if self.train:
            self.shuffle_indices()
            selected_negative_indices = self.negative_indices[:self.num_batches * self.num_negative_per_batch]
        
            for i in range(self.num_batches):
                pos_batch_indices = self.positive_indices[i * self.num_positive_per_batch: (i + 1) * self.num_positive_per_batch]
                neg_batch_indices = selected_negative_indices[i * self.num_negative_per_batch: (i + 1) * self.num_negative_per_batch]

                # Add remaining positive samples to the last few batches
                if self.remaining_positives > 0 and i >= (self.num_batches - self.remaining_positives):
                    remaining_pos_index = self.positive_indices[-(self.remaining_positives - (i - (self.num_batches - self.remaining_positives)))]
                    pos_batch_indices = torch.cat((pos_batch_indices, remaining_pos_index.unsqueeze(0)))
                    neg_batch_indices = neg_batch_indices[:-1]

                batch_indices = torch.cat((pos_batch_indices, neg_batch_indices))
                batch_indices = batch_indices[torch.randperm(len(batch_indices))]
                yield batch_indices
        
        else:
            # self.negative_indices = self.negative_indices[torch.randperm(len(self.negative_indices))]
            # selected_negative_indices = self.negative_indices[:self.num_positive_samples]
            selected_negative_indices = self.negative_indices

            all_indices = torch.cat((self.positive_indices, selected_negative_indices))
            all_indices = all_indices[torch.randperm(len(all_indices))]

            # Yield batches
            for i in range(self.num_batches):
                batch_indices = all_indices[i * self.batch_size: (i + 1) * self.batch_size]
                yield batch_indices

            # Handle remaining samples if any
            if len(all_indices) % self.batch_size != 0:
                yield all_indices[self.num_batches * self.batch_size:]
    
    def __len__(self) -> int:
        if self.train:
            return self.num_batches
        else:
            return (self.num_positive_samples+self.num_negative_samples + self.batch_size - 1) // self.batch_size
        

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