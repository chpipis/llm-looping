import numpy as np
import torch
import lightning as L
from sklearn.model_selection import train_test_split


class Tokenizer:
    def __init__(self, maxNodes):
        self.maxNodes = maxNodes
        self.encoder = {str(i): i for i in range(maxNodes)}
        self.encoder["|"] = maxNodes
        self.encoder["="] = maxNodes + 1
        self.encoder["/"] = maxNodes + 2
        self.encoder["$"] = maxNodes + 3  # Padding token

        self.decoder = {i: str(i) for i in range(maxNodes)}
        self.decoder[maxNodes] = "|"
        self.decoder[maxNodes + 1] = "="
        self.decoder[maxNodes + 2] = "/"
        self.decoder[maxNodes + 3] = "$"
        self.decoder[maxNodes + 4] = ""

        self.numbers = set("0123456789")

        self.eos_token_id = maxNodes + 4
        self.pad_token_id = maxNodes + 3  # Points to the '$' padding token
        # The final +1 is not strictly needed here, but preserved from the original BST codebase.
        self.vocab_size = maxNodes + 5 + 1

    def encode(self, data):
        out = []
        i = 0
        while i < len(data):
            if data[i] == ",":
                i += 1
                continue
            s = ""
            while i < len(data) and data[i] in self.numbers:
                s += data[i]
                i += 1
            if s:
                out.append(self.encoder[s])
            else:
                out.append(self.encoder[data[i]])
                i += 1

        return out

    def decode(self, tokens, stop_at_eos=True):
        out = ""
        for t in tokens:
            if stop_at_eos and t == self.eos_token_id:
                break
            try:
                cur = self.decoder[t]
            except KeyError:
                cur = f"<UNK:{t}>"
            if len(out) > 0 and out[-1] in self.numbers and len(cur) > 0 and cur[0] in self.numbers:
                out += ","
            out += cur
        return out

    def tokenize(self, prefix):
        prefix_tokens = self.encode(prefix)
        prefix_tokens.append(self.eos_token_id)

        seq = np.array(prefix_tokens)

        return seq, len(seq)


class LoopStarGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].strip()
        seq, _ = self.tokenizer.tokenize(line)
        x = torch.tensor(seq, dtype=torch.long)
        return x


class LoopStarGraphDataModule:
    """
    PyTorch Lightning style DataModule for LoopStarGraph dataset
    """

    def __init__(self, fabric: L.Fabric, config):
        self.fabric = fabric
        self.batch_size = config.data.device_batch_size

        # Create tokenizer
        maxNodes = config.data.loop_stargraph_max_nodes
        self.tokenizer = Tokenizer(maxNodes)

        # Load data
        with open(config.data.loop_stargraph_data_path, "r") as f:
            data = f.readlines()

        # Split the data into training and validation sets
        train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
        self.train_dataset = LoopStarGraphDataset(train_data, self.tokenizer)
        self.val_dataset = LoopStarGraphDataset(val_data, self.tokenizer)

        graph_description_len, total_len = self._measure_index(data)
        print(
            f"[LoopStarGraph] Prefix Length: {graph_description_len}, Sequence Length: {total_len}"
        )

        # The final +1 is not strictly needed here, but preserved from the original BST codebase.
        self.vocab_size = maxNodes + 5 + 1
        self.graph_description_len = graph_description_len
        self.total_len = total_len

    def _measure_index(self, data):
        # For variable-length sequences, we need to scan the entire dataset to get accurate statistics
        prefix_lengths = []
        sequence_lengths = []
        
        print(f"[LoopStarGraph] Analyzing {len(data)} sequences for length statistics...")
        
        for i, line in enumerate(data):
            line = line.strip()
            prefix = line.split("=")[0] + "="
            _, prefix_len = self.tokenizer.tokenize(prefix)
            _, seq_len = self.tokenizer.tokenize(line)
            
            prefix_lengths.append(prefix_len)
            sequence_lengths.append(seq_len)
            
            # Show progress for large datasets
            if len(data) > 1000 and (i + 1) % (len(data) // 10) == 0:
                print(f"   Processed {i + 1}/{len(data)} sequences...")
        
        # Calculate statistics
        min_prefix_len = min(prefix_lengths)
        max_prefix_len = max(prefix_lengths)
        avg_prefix_len = sum(prefix_lengths) / len(prefix_lengths)
        
        min_seq_len = min(sequence_lengths)
        max_seq_len = max(sequence_lengths)
        avg_seq_len = sum(sequence_lengths) / len(sequence_lengths)
        
        # one for beginning special token at end of sequence, one so it appears on index of equals sign.
        graph_description_len = max_prefix_len - 2
        
        print(f"[LoopStarGraph] Prefix lengths: min={min_prefix_len}, max={max_prefix_len}, avg={avg_prefix_len:.1f}")
        print(f"[LoopStarGraph] Sequence lengths: min={min_seq_len}, max={max_seq_len}, avg={avg_seq_len:.1f}")
        
        # Calculate percentiles for better understanding
        sorted_seq_lens = sorted(sequence_lengths)
        p95_seq_len = sorted_seq_lens[int(0.95 * len(sorted_seq_lens))]
        p99_seq_len = sorted_seq_lens[int(0.99 * len(sorted_seq_lens))]
        
        print(f"[LoopStarGraph] Sequence length percentiles: 95th={p95_seq_len}, 99th={p99_seq_len}")
        
        # Warn if there's a big discrepancy
        if max_seq_len > p95_seq_len * 1.5:
            print(f"[LoopStarGraph] ⚠️  Warning: Max length ({max_seq_len}) is much larger than 95th percentile ({p95_seq_len})")
            print(f"[LoopStarGraph]     Consider using a smaller max_tokens limit during data generation")

        return graph_description_len, max_seq_len

    def update_config(self, config):
        config.model.vocab_size = self.vocab_size
        config.model.context_length = self.graph_description_len
        # expose pad_token_id to model config for masking
        config.model.pad_token_id = self.tokenizer.pad_token_id
        
        # Only update block_size if it's not already set to a reasonable value
        # or if the current value is smaller than what we need
        current_block_size = getattr(config.model, 'block_size', 0)
        
        if current_block_size < self.total_len:
            print(f"[LoopStarGraph] Updating block_size from {current_block_size} to {self.total_len}")
            config.model.block_size = self.total_len
        else:
            print(f"[LoopStarGraph] Keeping existing block_size ({current_block_size}) as it's >= required ({self.total_len})")

    def get_tokenizer(self):
        return self.tokenizer

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda batch: self.collate_fn(batch, self.tokenizer),
        )
        # This will automatically partition data between devices
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=lambda batch: self.collate_fn(batch, self.tokenizer),
        )
        # This will automatically partition data between devices
        return self.fabric.setup_dataloaders(dataloader, use_distributed_sampler=True)

    @staticmethod
    def collate_fn(batch, tokenizer):
        """Collate function that returns (padded_tokens, prompt_len).

        Notes:
        - Each element in `batch` is a 1D LongTensor containing the full sequence including a final EOS token.
        - We define the prompt as everything up to and including the first '=' token.
        - We left-pad sequences (right-align actual tokens). Because masking in the model is index-based from the
            left, we include the left padding tokens inside the masked region by increasing prompt_len accordingly.
            This is harmless because pad tokens are already ignored in the loss, and it keeps logic simple.
        - prompt_len is therefore: left_pad + (index_of_equal + 1).
        """
        # Handle variable-length sequences with left padding using the proper padding token
        max_len = max(len(item) for item in batch)

        pad_token_id = tokenizer.pad_token_id
        equal_token_id = tokenizer.encoder["="]

        batch_size = len(batch)
        padded_batch = torch.full(
            (batch_size, max_len), fill_value=pad_token_id, dtype=torch.long
        )
        prompt_lens = []

        for i, item in enumerate(batch):
            seq_len = len(item)
            left_pad = max_len - seq_len
            padded_batch[i, left_pad:] = item

            # Find first '=' token position within the unpadded sequence
            equals_positions = (item == equal_token_id).nonzero(as_tuple=True)[0]
            if len(equals_positions) == 0:
                # Data should always contain '='; raise explicit error for malformed sample
                raise ValueError(
                    "Malformed LoopStarGraph sample: '=' token not found."
                )
            first_eq = int(equals_positions[0])
            # '=' must not be the penultimate token (which would mean no response tokens before EOS)
            # seq layout: [ ... tokens ..., '=', <at least one response token>, ..., EOS]
            if first_eq >= seq_len - 2:  # positions counted before EOS at seq_len-1
                raise ValueError(
                    "Malformed LoopStarGraph sample: '=' at or beyond last response position (no response tokens)."
                )
            prompt_core = first_eq + 1  # inclusive of '='

            prompt_len = left_pad + prompt_core
            
            assert 1 <= prompt_len < max_len - 1, (
                f"Computed prompt_len={prompt_len} out of bounds for max_len={max_len} (first_eq={first_eq}, left_pad={left_pad})"
            )
            prompt_lens.append(prompt_len)

        prompt_lens = torch.tensor(prompt_lens, dtype=torch.long)
        return padded_batch, prompt_lens
