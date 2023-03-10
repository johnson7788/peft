<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center"> <p>🤗 PEFT</p></h1>
<h3 align="center">
    <p>State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) methods</p>
</h3>
参数高效微调（PEFT）方法使预训练语言模型（PLMs）能够在不微调所有模型参数的情况下高效适应各种后续应用。微调大规模的PLMs通常成本过高。在这方面，PEFT 方法仅微调少量（额外的）模型参数，从而大大降低计算和存储成本。最近的最新 PEFT 技术实现的性能与全面微调相当。
与🤗 Accelerate 无缝集成，利用 DeepSpeed 和大型模型推理加速大规模模型。

Supported methods:

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
2. Prefix Tuning: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/), [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
3. P-Tuning: [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)
4. Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf) 

## Getting started

```python
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# output: trainable params: 2359296 || all params: 1231940608 || trainable%: 0.19151053100118282
```

## Use Cases

### 通过使用消费者硬件，将LLMs适应于下游任务，可以获得与完全微调相似的性能。

适应LLMs少样本数据集所需的GPU内存。这里考虑的设置包括完全微调、使用普通PyTorch的PEFT-LoRA和使用具有CPU卸载功能的DeepSpeed的PEFT-LoRA。[`ought/raft/twitter_complaints`](https://huggingface.co/datasets/ought/raft/viewer/twitter_complaints)

Hardware: Single A100 80GB GPU with CPU RAM above 64GB

|   Model         | Full Finetuning | PEFT-LoRA PyTorch  | PEFT-LoRA DeepSpeed with CPU Offloading |
| --------- | ---- | ---- | ---- |
| bigscience/T0_3B (3B params) | 47.14GB GPU / 2.96GB CPU  | 14.4GB GPU / 2.96GB CPU | 9.8GB GPU / 17.8GB CPU |
| bigscience/mt0-xxl (12B params) | OOM GPU | 56GB GPU / 3GB CPU | 22GB GPU / 52GB CPU |
| bigscience/bloomz-7b1 (7B params) | OOM GPU | 32GB GPU / 3.8GB CPU | 18.1GB GPU / 35GB CPU |

Performance of PEFT-LoRA tuned [`bigscience/T0_3B`](https://huggingface.co/bigscience/T0_3B) on [`ought/raft/twitter_complaints`](https://huggingface.co/datasets/ought/raft/viewer/twitter_complaints) leaderboard. 
需要注意的一点是，我们没有通过玩弄输入指令模板、LoRA超参数和其他与训练相关的超参数来挤压性能。Also, we didn't use the larger 13B [mt0-xxl](https://huggingface.co/bigscience/mt0-xxl) model.
因此，我们已经看到了与参数高效调节相当的最先进性能。 Also, the final checkpoint size is just `19MB` in comparison to `11GB` size of the backbone [`bigscience/T0_3B`](https://huggingface.co/bigscience/T0_3B) model.

|   Submission Name        | Accuracy |
| --------- | ---- |
| Human baseline (crowdsourced) |	0.897 |
| Flan-T5 | 0.892 |
| lora-t0-3b | 0.863 |

因此，我们可以看到，使用16GB和24GB GPU这样的消费级硬件，PEFT方法可以达到与SoTA相当的性能。

### Parameter Efficient Tuning of Diffusion Models

在训练过程中不同设置所需的GPU内存如下。最终的检查点大小为`8.8 MB`。

Hardware: Single A100 80GB GPU with CPU RAM above 64GB

|   Model         | Full Finetuning | PEFT-LoRA  | PEFT-LoRA with Gradient Checkpoitning  |
| --------- | ---- | ---- | ---- |
| CompVis/stable-diffusion-v1-4 | 27.5GB GPU / 3.97GB CPU | 15.5GB GPU / 3.84GB CPU | 8.12GB GPU / 3.77GB CPU | 


**Training**
给出了使用LoRA进行参数有效的Dreambooth培训的例子 in `~examples/lora_dreambooth/train_dreambooth.py`

```bash
export MODEL_NAME= "CompVis/stable-diffusion-v1-4" #"stabilityai/stable-diffusion-2-1"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 27 \
  --lora_text_encoder_r 16 \
  --lora_text_encoder_alpha 17 \
  --learning_rate=1e-4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=800
```

尝试使用 🤗 Gradio Space 在 T4 实例上无缝运行的功能：
[smangrul/peft-lora-sd-dreambooth](https://huggingface.co/spaces/smangrul/peft-lora-sd-dreambooth).

![peft lora dreambooth gradio space](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/peft_lora_dreambooth_gradio_space.png)

### Parameter Efficient Tuning of LLMs for RLHF components such as Ranker and Policy
- Here is an exmaple in [trl](https://github.com/lvwerra/trl) library using PEFT+INT8 for tuning policy model: [gpt2-sentiment_peft.py](https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt2-sentiment_peft.py) 
- Example using PEFT for both reward model and policy [ToDo]

### INT8 training of large models in Colab using PEFT LoRA and bits_and_bytes

- Here is now a demo on how to fine tune [OPT-6.7b](https://huggingface.co/facebook/opt-6.7b) (14GB in fp16) in a Google colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing)

- Here is now a demo on how to fine tune [whishper-large](openai/whisper-large-v2) (1.5B params) (14GB in fp16) in a Google colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DOkD_5OUjFa0r5Ik3SgywJLJtEo2qLxO?usp=sharing) and [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vhF8yueFqha3Y3CpTHN6q9EVcII9EYzs?usp=sharing)

### 即使对于中小型模型，也要节省计算和存储空间。

通过避免在每个下游任务/数据集上进行完整微调模型，节省存储空间。
使用PEFT方法，用户只需要存储仅有几MB的小检查点，同时保留可与完整微调相当的性能。
LoRA被应用于在'FUNSD'数据集上对'LayoutLMForTokenClassification'进行适应任务的示例在`~examples/token_classification/PEFT_LoRA_LayoutLMForTokenClassification_on_FUNSD.py`中给出。
我们可以观察到，只有可训练参数的0.62％，我们就可以达到类似于完整微调（F1 0.786）的性能（F1 0.777）（没有进行任何超参数调整的情况下提取更多性能），
而且这个检查点只有'2.8MB'。现在，如果有`N`个这样的数据集，只需为每个数据集使用这些PEFT模型，并节省大量存储空间，而不必担心骇人听闻或过拟合基础模型的问题。

Another example is fine-tuning [`roberta-large`](https://huggingface.co/roberta-large) on [`MRPC` GLUE](https://huggingface.co/datasets/glue/viewer/mrpc) dataset suing differenct PEFT methods. The notebooks are given in `~examples/sequence_classification`. 


## PEFT + 🤗 Accelerate

PEFT 模型可以直接与 🤗 Accelerate 搭配使用。在训练期间，可以使用 🤗 Accelerate 进行分布式训练，适用于各种硬件，例如GPU、苹果硅设备等。
在消费级硬件上进行推理时，可以使用 🤗 Accelerate 来处理资源较少的情况。

### Example of PEFT model training using 🤗 Accelerate's DeepSpeed integration

DeepSpeed version required `v0.8.0`. An example is provided in `~examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py`. 
  a. First, run `accelerate config --config_file ds_zero3_cpu.yaml` and answer the questionnaire. 
  Below are the contents of the config file.
  ```yaml
  compute_environment: LOCAL_MACHINE
  deepspeed_config:
    gradient_accumulation_steps: 1
    gradient_clipping: 1.0
    offload_optimizer_device: cpu
    offload_param_device: cpu
    zero3_init_flag: true
    zero3_save_16bit_model: true
    zero_stage: 3
  distributed_type: DEEPSPEED
  downcast_bf16: 'no'
  dynamo_backend: 'NO'
  fsdp_config: {}
  machine_rank: 0
  main_training_function: main
  megatron_lm_config: {}
  mixed_precision: 'no'
  num_machines: 1
  num_processes: 1
  rdzv_backend: static
  same_network: true
  use_cpu: false
  ```
  b. run the below command to launch the example script
  ```bash
  accelerate launch --config_file ds_zero3_cpu.yaml examples/peft_lora_seq2seq_accelerate_ds_zero3_offload.py
  ```

  c. output logs:
  ```bash
  GPU Memory before entering the train : 1916
  GPU Memory consumed at the end of the train (end-begin): 66
  GPU Peak Memory consumed during the train (max-begin): 7488
  GPU Total Peak Memory consumed during the train (max): 9404
  CPU Memory before entering the train : 19411
  CPU Memory consumed at the end of the train (end-begin): 0
  CPU Peak Memory consumed during the train (max-begin): 0
  CPU Total Peak Memory consumed during the train (max): 19411
  epoch=4: train_ppl=tensor(1.0705, device='cuda:0') train_epoch_loss=tensor(0.0681, device='cuda:0')
  100%|████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:27<00:00,  3.92s/it]
  GPU Memory before entering the eval : 1982
  GPU Memory consumed at the end of the eval (end-begin): -66
  GPU Peak Memory consumed during the eval (max-begin): 672
  GPU Total Peak Memory consumed during the eval (max): 2654
  CPU Memory before entering the eval : 19411
  CPU Memory consumed at the end of the eval (end-begin): 0
  CPU Peak Memory consumed during the eval (max-begin): 0
  CPU Total Peak Memory consumed during the eval (max): 19411
  accuracy=100.0
  eval_preds[:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
  dataset['train'][label_column][:10]=['no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint', 'no complaint', 'no complaint', 'complaint', 'complaint', 'no complaint']
  ```

### Example of PEFT model inference using 🤗 Accelerate's Big Model Inferencing capabilities
An example is provided in `~examples/causal_language_modeling/peft_lora_clm_accelerate_big_model_inference.ipynb`. 


## 支持的模型

### Causal Language Modeling
|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  |
| --------- | ---- | ---- | ---- | ----  |
| GPT-2          | ✅  | ✅  | ✅  | ✅  |
| Bloom          | ✅  | ✅  | ✅  | ✅  |
| OPT            | ✅  | ✅  | ✅  | ✅  |
| GPT-Neo        | ✅  | ✅  | ✅  | ✅  |
| GPT-J          | ✅  | ✅  | ✅  | ✅  |
| GPT-NeoX-20B   | ✅  | ✅  | ✅  | ✅  |

### Conditional Generation
|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  | 
| --------- | ---- | ---- | ---- | ---- |
| T5        | ✅   | ✅   | ✅   | ✅   |
| BART      | ✅   | ✅   | ✅   | ✅   |

### Sequence Classification
|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  | 
| --------- | ---- | ---- | ---- | ----  |
| BERT           | ✅  | ✅  | ✅  | ✅  |  
| RoBERTa        | ✅  | ✅  | ✅  | ✅  |
| GPT-2          | ✅  | ✅  | ✅  | ✅  | 
| Bloom          | ✅  | ✅  | ✅  | ✅  |   
| OPT            | ✅  | ✅  | ✅  | ✅  |
| GPT-Neo        | ✅  | ✅  | ✅  | ✅  |
| GPT-J          | ✅  | ✅  | ✅  | ✅  |
| Deberta        | ✅  |     | ✅  | ✅  |     
| Deberta-v2     | ✅  |     | ✅  | ✅  |    

### Token Classification
|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  | 
| --------- | ---- | ---- | ---- | ----  |
| BERT           | ✅  | ✅  |   |   |  
| RoBERTa        | ✅  | ✅  |   |   |
| GPT-2          | ✅  | ✅  |   |   | 
| Bloom          | ✅  | ✅  |   |   |   
| OPT            | ✅  | ✅  |   |   |
| GPT-Neo        | ✅  | ✅  |   |   |
| GPT-J          | ✅  | ✅  |   |   |
| Deberta        | ✅  |     |   |   | 
| Deberta-v2     | ✅  |     |   |   |

### Text-to-Image Generation

|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  | 
| --------- | ---- | ---- | ---- | ----  |
| Stable Diffusion           | ✅  |   |   |   |  


### Image Classification

|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  | 
| --------- | ---- | ---- | ---- | ----  |
| ViT           | ✅  |   |   |   | 
| Swin           | ✅  |   |   |   | 

___Note that we have tested LoRA for [ViT](https://huggingface.co/docs/transformers/model_doc/vit) and [Swin](https://huggingface.co/docs/transformers/model_doc/swin) for fine-tuning on image classification. However, it should be possible to use LoRA for any compatible model [provided](https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads&search=vit) by 🤗 Transformers. Check out the respective
examples to learn more. If you run into problems, please open an issue.___

The same principle applies to our [segmentation models](https://huggingface.co/models?pipeline_tag=image-segmentation&sort=downloads) as well. 

### Semantic Segmentation

|   Model         | LoRA | Prefix Tuning  | P-Tuning | Prompt Tuning  | 
| --------- | ---- | ---- | ---- | ----  |
| SegFormer           | ✅  |   |   |   | 


## Caveats:

1. 以下是使用PyTorch FSDP进行训练的示例。然而，它不会节省任何GPU内存。
   Please refer issue [[FSDP] FSDP with CPU offload consumes 1.65X more GPU memory when training models with most of the params frozen](https://github.com/pytorch/pytorch/issues/91165). 

  ```python
  from peft.utils.other import fsdp_auto_wrap_policy

  ...

  if os.environ.get("ACCELERATE_USE_FSDP", None) is not None:
      accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

  model = accelerator.prepare(model)
  ```

  Example of parameter efficient tuning with [`mt0-xxl`](https://huggingface.co/bigscience/mt0-xxl) base model using 🤗 Accelerate is provided in `~examples/conditional_generation/peft_lora_seq2seq_accelerate_fsdp.py`. 
  a. First, run `accelerate config --config_file fsdp_config.yaml` and answer the questionnaire. 
  Below are the contents of the config file.
  ```yaml
  command_file: null
  commands: null
  compute_environment: LOCAL_MACHINE
  deepspeed_config: {}
  distributed_type: FSDP
  downcast_bf16: 'no'
  dynamo_backend: 'NO'
  fsdp_config:
    fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
    fsdp_backward_prefetch_policy: BACKWARD_PRE
    fsdp_offload_params: true
    fsdp_sharding_strategy: 1
    fsdp_state_dict_type: FULL_STATE_DICT
    fsdp_transformer_layer_cls_to_wrap: T5Block
  gpu_ids: null
  machine_rank: 0
  main_process_ip: null
  main_process_port: null
  main_training_function: main
  megatron_lm_config: {}
  mixed_precision: 'no'
  num_machines: 1
  num_processes: 2
  rdzv_backend: static
  same_network: true
  tpu_name: null
  tpu_zone: null
  use_cpu: false
  ```
  b. run the below command to launch the example script
  ```bash
  accelerate launch --config_file fsdp_config.yaml examples/peft_lora_seq2seq_accelerate_fsdp.py
  ```

2. 当使用 `SEQ_2_SEQ` 任务的 `P_TUNING` 或 `PROMPT_TUNING` 时，请在评估期间从模型输出的左侧删除 `num_virtual_token` 个虚拟提示预测。

对于编码器-解码器模型，因为`generate`功能需要严格的`decoder_input_ids`参数，所以`P_TUNING`或`PROMPT_TUNING`不能支持transformers的`generate`功能。
但是`P_TUNING`/`PROMPT_TUNING`会将软提示嵌入附加到`input_embeds`中，以创建新的`input_embeds`，并提供给模型。
因此，`generate`目前还不支持这种情况。

## Backlog:
1. Explore and possibly integrate `(IA)^3`
2. Add tests
3. Add more use cases and examples



## Citing 🤗 PEFT

If you use 🤗 PEFT in your publication, please cite it by using the following BibTeX entry.

```bibtex
@Misc{peft,
  title =        {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author =       {Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year =         {2022}
}
```
