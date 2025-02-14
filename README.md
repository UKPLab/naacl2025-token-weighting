## Token Weighting for Long-Range Language Modeling

### NAACL 2025 Findings

This repository provides the code for our paper. To get started, clone the repo and run 

```
pip install -r requirements.txt
```
Additionally, make sure that `CUDA 11.7.` and `torch` are installed before installing flash attention
```
pip install flash_attn==2.7.3
```

If you want to preprocess (i.e. chunk into 32k sequences and save in tokenized form) the data, run

```
import os
from data_preprocessing import DataPreprocessor
from logger_setup import initialize_logger
from datasets import load_from_disk

initialize_logger()

tokenizer_name = "llama3"

PATH_RAW_DATA = "/local/path/to/pg19"
PATH_CHUNKED_DATA = os.path.join(PATH_RAW_DATA, f"preprocessed_{tokenizer_name}")

data_preprocessor = DataPreprocessor(PATH_RAW_DATA=PATH_RAW_DATA, PATH_CHUNKED_DATA=PATH_CHUNKED_DATA,
                                     tokenizer=tokenizer_name)
dataset = data_preprocessor.fetch_data_from_file()

data_preprocessor.save_locally_hf(dataset)
``` 
If `PATH_RAW_DATA` is empty or non-existent, it will be created and [PG19](https://huggingface.co/datasets/deepmind/pg19) will be downloaded into it.

If you want to log your training runs with [aim](https://aimstack.readthedocs.io/en/latest/overview.html), run 

```
aim init
```

Then you can start the self-scoring training (i.e. the unfrozen variant) via

```
python main.py --out_path /directory/for/saved/runs
```

For the frozen variant, the sequences first have to be scored and saved: 

```
python main.py --run_name llama3_32k_dense_precompute_weights --launcher 'accelerate launch'
```

After that, you can add the column with the frozen weights to the huggingface dataset via

```
import os
from data_preprocessing import DataPreprocessor
from logger_setup import initialize_logger
from datasets import load_from_disk
from run_settings import get_settings 

initialize_logger()

tokenizer_name = "llama3"
no_devices = n # the number of GPUs that were used to score the data  

PATH_RAW_DATA = "/local/path/to/pg19"
PATH_CHUNKED_DATA = os.path.join(PATH_RAW_DATA, f"preprocessed_{tokenizer_name}")
PATH_FROZEN_CACHE = os.path.join(PATH_CHUNKED_DATA, f"precomputed_weights_3.0_8B")

settings = get_settings(mode="llama3_32k_dense_precompute_weights")
data_preprocessor = DataPreprocessor(settings=settings)

data_preprocessor.frozen_collect_and_make_hf_dataset(precomputed_weights_path = PATH_FROZEN_CACHE,
                                           folder_non_frozen = PATH_CHUNKED_DATA,
                                           no_devices = no_devices)
```

### Loss variants
The loss variants are determined by the config file. First, `use_frozen_base` indicates
whether self-scoring (unfrozen) is used or not (frozen). The `base_length` determines the
length of the short-context model. `base_stride` is the stride used for scoring the long document
with the short-context model. The overlap between subsequences is `base_length - base_stride`. Increasing
the stride makes the method more efficient (less forward passes) but more inexact. Usually, you want to use the
smallest `base_stride` that leads to `chunk_size/base_length` additional forward passes. This value
can be calculated via
`(1-base_length/chunk_size)*base_length`, e.g. 6144 for 32768 context.

The basic `logit_comparison` in the loss is $\text{LongLoss} - \text{ShortLoss} = -\log(p^l) - (-\log(p^s)) = \log\left(\frac{p^s}{p^l}\right)$
The `transforms` are applied sequentially to it. Note that the minus transform leads to $log\left(\frac{p^l}{p^s}\right)$.
The `truncation` $\gamma$ clips the values higher than itself. The sparsification parameter $\kappa$ only considers the top-$\kappa$ percent of the tokens.
`interpolation` $\lambda$ applies a convex combination with the vanilla loss. ($\kappa=1$ or $\lambda=1$ lead to standard cross-entropy loss)
`normalization` normalizes the weights such that they average to 1.

### Loss Variants Table
Losses investigated in the paper can be realised as follows:

| Loss Variant                                                  | Transforms            | Interpolation | Normalization | Sparsification | Truncation | 
|---------------------------------------------------------------|-----------------------|---------------|---------------|----------------|------------|
| Dense $\lambda$                                               | [absolute]            | $\lambda$     | L1            | -              | -          |
| Sparse $\kappa$                                               | [absolute]            | -             | L1            | $\kappa$       | -          | 
| [LongCE](https://openreview.net/forum?id=fL4qWkSmtM) $\gamma$ | [minus, exp]          | -             | -             | -              | $\gamma$   | 
| PPMI s                                                        | [minus, shift s, max] | -             | L1            | $\kappa$       | -          | 
| NPMI s                                                        | [shift s, max]        | -             | L1            | $\kappa$       | -          | 
