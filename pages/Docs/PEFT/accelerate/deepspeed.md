# DeepSpeed

[DeepSpeed](https://www.deepspeed.ai/) هي مكتبة مصممة للسرعة والتوسع في التدريب الموزع للنماذج الكبيرة التي تحتوي على مليارات من المعلمات. وفي جوهرها يوجد محسن Zero Redundancy Optimizer (ZeRO) الذي تقسم حالات المحسن (ZeRO-1) والمدرجات (ZeRO-2) والمعلمات (ZeRO-3) عبر عمليات موازية للبيانات. وهذا يقلل بشكل كبير من استخدام الذاكرة، مما يتيح لك توسيع نطاق التدريب الخاص بك إلى نماذج المعلمات مليار. ولتحقيق المزيد من كفاءة الذاكرة، تقلل ZeRO-Offload من ذاكرة الحوسبة وذاكرة GPU من خلال الاستفادة من موارد وحدة المعالجة المركزية أثناء التحسين.

يتم دعم كلا الميزتين في 🤗 Accelerate، ويمكنك استخدامهما مع 🤗 PEFT.

## التوافق مع `bitsandbytes` الكم + LoRA

فيما يلي جدول يلخص التوافق بين PEFT's LoRA، ومكتبة [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) و DeepSpeed Zero stages فيما يتعلق بالتدريب الدقيق. لن يكون لـ DeepSpeed Zero-1 و 2 أي تأثير عند الاستدلال حيث تقوم المرحلة 1 بتقسيم حالات المحسن وتقسيم المرحلة 2 لحالات المحسن والمدرجات:

| مرحلة DeepSpeed | هل هو متوافق؟ |
|---|---|
| Zero-1 |  🟢 |
| Zero-2 |  🟢 |
| Zero-3 |  🟢 |

بالنسبة لـ DeepSpeed Stage 3 + QLoRA، يرجى الرجوع إلى القسم [استخدم PEFT QLoRA و DeepSpeed مع ZeRO3 للتدريب الدقيق للنماذج الكبيرة على وحدات GPU متعددة](#use-peft-qlora-and-deepspeed-with-zero3-for-finetuning-large-models-on-multiple-gpus) أدناه.

لتأكيد هذه الملاحظات، قمنا بتشغيل SFT (التدريب الدقيق الخاضع للإشراف) [سكريبتات المثال الرسمية](https://github.com/huggingface/trl/tree/main/examples) من مكتبة Transformers Reinforcement Learning (TRL) باستخدام QLoRA + PEFT وتهيئة Accelerate المتوفرة [هنا](https://github.com/huggingface/trl/tree/main/examples/accelerate_configs). وقد أجرينا هذه التجارب على 2x NVIDIA T4 GPU.

# استخدام PEFT و DeepSpeed مع ZeRO3 للتدريب الدقيق للنماذج الكبيرة على أجهزة متعددة وعقد متعددة

سيساعدك هذا القسم من الدليل على معرفة كيفية استخدام نصنا البرمجي للتدريب DeepSpeed [training script](https://github.com/huggingface/peft/blob/main/examples/sft/train.py) لأداء SFT. ستقوم بتكوين النص البرمجي للقيام بـ SFT (التدريب الدقيق الخاضع للإشراف) لنموذج Llama-70B مع LoRA و ZeRO-3 على 8xH100 80GB GPUs على جهاز واحد. يمكنك تهيئته للتوسع إلى أجهزة متعددة عن طريق تغيير تهيئة Accelerate.

## التهيئة

ابدأ بتشغيل الأمر التالي لإنشاء ملف تهيئة DeepSpeed باستخدام 🤗 Accelerate. يسمح علم `--config_file` لك بحفظ ملف التهيئة في موقع محدد، وإلا فسيتم حفظه كملف `default_config.yaml` في ذاكرة التخزين المؤقت لـ 🤗 Accelerate.

يُستخدم ملف التكوين لتعيين الخيارات الافتراضية عند تشغيل نص التدريب.

```bash
accelerate config --config_file deepspeed_config.yaml
```

سيُطلب منك الإجابة عن بعض الأسئلة حول إعدادك، وتهيئة الحجج التالية. في هذا المثال، ستستخدم ZeRO-3 لذا تأكد من اختيار تلك الخيارات.

```bash
`zero_stage`: [0] Disabled, [1] optimizer state partitioning, [2] optimizer+gradient state partitioning and [3] optimizer+gradient+parameter partitioning
`gradient_accumulation_steps`: Number of training steps to accumulate gradients before averaging and applying them. Pass the same value as you would pass via cmd argument else you will encounter mismatch error.
`gradient_clipping`: Enable gradient clipping with value. Don't set this as you will be passing it via cmd arguments.
`offload_optimizer_device`: [none] Disable optimizer offloading, [cpu] offload optimizer to CPU, [nvme] offload optimizer to NVMe SSD. Only applicable with ZeRO >= Stage-2. Set this as `none` as don't want to enable offloading.
`offload_param_device`: [none] Disable parameter offloading, [cpu] offload parameters to CPU, [nvme] offload parameters to NVMe SSD. Only applicable with ZeRO Stage-3. Set this as `none` as don't want to enable offloading.
`zero3_init_flag`: Decides whether to enable `deepspeed.zero.Init` for constructing massive models. Only applicable with ZeRO Stage-3. Set this to `True`.
`zero3_save_16bit_model`: Decides whether to save 16-bit model weights when using ZeRO Stage-3. Set this to `True`.
`mixed_precision`: `no` for FP32 training, `fp16` for FP16 mixed-precision training and `bf16` for BF16 mixed-precision training. Set this to `True`.
```

بمجرد الانتهاء من ذلك، يجب أن يبدو التكوين المقابل كما يلي ويمكنك العثور عليه في مجلد التهيئة في [deepspeed_config.yaml](https://github.com/huggingface/peft/blob/main/examples/sft/configs/deepspeed_config.yaml):

```yml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
    deepspeed_multinode_launcher: standard
    gradient_accumulation_steps: 4
    offload_optimizer_device: none
    offload_param_device: none
    zero3_init_flag: true
    zero3_save_16bit_model: true
    zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## أمر الإطلاق

أمر الإطلاق متاح في [run_peft_deepspeed.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_peft_deepspeed.sh) وهو موضح أدناه أيضًا:

```bash
accelerate launch --config_file "configs/deepspeed_config.yaml"  train.py \
--seed 100 \
--model_name_or_path "meta-llama/Llama-2-70b-hf" \
--dataset_name "smangrul/ultrachat-10k-chatml" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_len 2048 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--push_to_hub \
--hub_private_repo True \
--hub_strategy "every_save" \
--bf16 True \
--packing True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "llama-sft-lora-deepspeed" \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing True \
--use_reentrant False \
--dataset_text_field "content" \
--use_flash_attn True \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization False
```

لاحظ أننا نستخدم LoRA مع الرتبة=8، alpha=16 واستهداف جميع الطبقات الخطية. نقوم بتمرير ملف تهيئة DeepSpeed ونقوم بالتدريب الدقيق لنموذج Llama 70B على مجموعة فرعية من مجموعة بيانات ultrachat.

## الأجزاء المهمة

دعونا نتعمق قليلاً في النص البرمجي حتى تتمكن من رؤية ما يحدث، وفهم كيفية عمله.

أول شيء يجب معرفته هو أن النص البرمجي يستخدم DeepSpeed للتدريب الموزع حيث تم تمرير تهيئة DeepSpeed. تتولى فئة `SFTTrainer` رفع جميع الأثقال لإنشاء نموذج PEFT باستخدام تهيئة PEFT التي تم تمريرها. بعد ذلك، عندما تستدعي `trainer.train()`، يستخدم `SFTTrainer` داخليًا 🤗 Accelerate لإعداد النموذج والمحسن والمدرب باستخدام تهيئة DeepSpeed لإنشاء محرك DeepSpeed الذي يتم تدريبه بعد ذلك. مقتطف الكود الرئيسي هو أدناه:

```python
# trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=data_args.packing,
    dataset_kwargs={
        "append_concat_token": data_args.append_concat_token,
        "add_special_tokens": data_args.add_special_tokens,
    },
    dataset_text_field=data_args.dataset_text_field,
    max_seq_length=data_args.max_seq_length,
)
trainer.accelerator.print(f"{trainer.model}")

# train
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
trainer.train(resume_from_checkpoint=checkpoint)

# saving final model
trainer.save_model()
```

## استخدام الذاكرة

في المثال أعلاه، تبلغ الذاكرة التي تستهلكها كل وحدة GPU 64 جيجابايت (80٪) كما هو موضح في لقطة الشاشة أدناه:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/peft_deepspeed_mem_usage.png"/>
</div>
<small>استخدام ذاكرة GPU لتشغيل التدريب</small>

## المزيد من الموارد

يمكنك أيضًا الرجوع إلى هذه التدوينة [Falcon 180B Finetuning using 🤗 PEFT and DeepSpeed](https://medium.com/@sourabmangrulkar/falcon-180b-finetuning-using-peft-and-deepspeed-b92643091d99) حول كيفية التدريب الدقيق لنموذج Falcon 180B على 16 GPU A100 على آلتين.

# استخدام PEFT QLoRA و DeepSpeed مع ZeRO3 للتدريب الدقيق للنماذج الكبيرة على وحدات GPU متعددة

في هذا القسم، سنلقي نظرة على كيفية استخدام QLoRA و DeepSpeed Stage-3 للتدريب الدقيق لنموذج Llama 70B على 2X40GB GPU.

لهذا، نحتاج أولاً إلى `bitsandbytes>=0.43.0`، `accelerate>=0.28.0`، `transformers>4.38.2`، `trl>0.7.11` و `peft>0.9.0`. نحتاج إلى تعيين `zero3_init_flag` إلى true عند استخدام تهيئة Accelerate. فيما يلي التكوين الذي يمكن العثور عليه في [deepspeed_config_z3_qlora.yaml](https://github.com/huggingface/peft/blob/main/examples/sft/configs/deepspeed_config_z3_qlora.yaml):

```yml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
    deepspeed_multinode_launcher: standard
    offload_optimizer_device: none
    offload_param_device: none
    zero3_init_flag: true
    zero3_save_16bit_model: true
    zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

أمر الإطلاق موضح أدناه وهو متاح في [run_peft_qlora_deepspeed_stage3.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_peft_deepspeed.sh):

```
accelerate launch --config_file "configs/deepspeed_config_z3_qlora.yaml"  train.py \
--seed 100 \
--model_name_or_path "meta-llama/Llama-2-70b-hf" \
--dataset_name "smangrul/ultrachat-10k-chatml" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_len 2048 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--push_to_hub \
--hub_private_repo True \
--hub_strategy "every_save" \
--bf16 True \
--packing True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "llama-sft-qlora-dsz3" \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing True \
--use_reentrant True \
--dataset_text_field "content" \
--use_flash_attn True \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization True \
--use_nested_quant True \
--bnb_4bit_compute_dtype "bfloat16" \
--bnb_4bit_quant_storage_dtype "bfloat16"
```

لاحظ الحجة الجديدة التي يتم تمريرها `bnb_4bit_quant_storage_dtype` والتي تشير إلى نوع البيانات لتعبئة المعلمات 4-بت. على سبيل المثال، عندما يتم تعيينه على `bfloat16`، يتم تعبئة **8** معلمات 4 بت معًا بعد التكميم.

من حيث تعليمات البرمجية للتدريب، تتمثل تغييرات التعليمات البرمجية المهمة في ما يلي:

```diff
....

bnb_config = BitsAndBytesConfig(
    load_in_4bit=args.use_4bit_quantization,
    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=args.use_nested_quant,
+   bnb_4bit_quant_storage=quant_storage_dtype,
)

...

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
+   torch_dtype=quant_storage_dtype or torch.float32,
)

```

Notice that `torch_dtype` for `AutoModelForCausalLM` is same as the `bnb_4bit_quant_storage` data type. That's it. Everything else is handled by Trainer and TRL.

## Memory usage

In the above example, the memory consumed per GPU is **36.6 GB**. Therefore, what took 8X80GB GPUs with DeepSpeed Stage 3+LoRA and a couple of 80GB GPUs with DDP+QLoRA now requires 2X40GB GPUs. This makes finetuning of large models more accessible.

# استخدام PEFT و DeepSpeed مع ZeRO3 و CPU Offloading لضبط دقيق للنماذج الكبيرة على وحدة معالجة رسومية واحدة

سيساعدك هذا القسم من الدليل على معرفة كيفية استخدام نصنا البرمجي للتدريب DeepSpeed. ستقوم بضبط النص البرمجي لتدريب نموذج كبير للتنبؤ الشرطي باستخدام ZeRO-3 و CPU Offload.

<Tip>

💡 لمساعدتك على البدء، تحقق من نصوصنا البرمجية التدريبية لمثال [نمذجة اللغة السببية](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py) و [التنبؤ الشرطي](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py). يمكنك تكييف هذه النصوص البرمجية مع تطبيقاتك الخاصة أو حتى استخدامها كما هي إذا كانت مهمتك مشابهة لتلك الموجودة في النصوص البرمجية.

</Tip>

## التهيئة

ابدأ بتشغيل الأمر التالي لإنشاء ملف تهيئة DeepSpeed باستخدام 🤗 Accelerate. يسمح لك علم `--config_file` بحفظ ملف التهيئة في موقع محدد، وإلا فسيتم حفظه كملف `default_config.yaml` في ذاكرة التخزين المؤقتة لـ 🤗 Accelerate.

يُستخدم ملف التكوين لتعيين الخيارات الافتراضية عند تشغيل نص التدريب.

```bash
accelerate config --config_file ds_zero3_cpu.yaml
```

سيُطلب منك الإجابة عن بعض الأسئلة حول إعدادك، وتهيئة الحجج التالية. في هذا المثال، ستستخدم ZeRO-3 إلى جانب CPU-Offload، لذا تأكد من اختيار هذه الخيارات.

```bash
`zero_stage`: [0] Disabled, [1] optimizer state partitioning, [2] optimizer+gradient state partitioning and [3] optimizer+gradient+parameter partitioning
`gradient_accumulation_steps`: Number of training steps to accumulate gradients before averaging and applying them.
`gradient_clipping`: Enable gradient clipping with value.
`offload_optimizer_device`: [none] Disable optimizer offloading, [cpu] offload optimizer to CPU, [nvme] offload optimizer to NVMe SSD. Only applicable with ZeRO >= Stage-2.
`offload_param_device`: [none] Disable parameter offloading, [cpu] offload parameters to CPU, [nvme] offload parameters to NVMe SSD. Only applicable with ZeRO Stage-3.
`zero3_init_flag`: Decides whether to enable `deepspeed.zero.Init` for constructing massive models. Only applicable with ZeRO Stage-3.
`zero3_save_16bit_model`: Decides whether to save 16-bit model weights when using ZeRO Stage-3.
`mixed_precision`: `no` for FP32 training, `fp16` for FP16 mixed-precision training and `bf16` for BF16 mixed-precision training.
```

قد يبدو ملف [التهيئة](https://github.com/huggingface/peft/blob/main/examples/conditional_generation/accelerate_ds_zero3_cpu_offload_config.yaml) على النحو التالي. أهم ما يجب ملاحظته هو أن `zero_stage` تم تعيينه إلى `3`، و `offload_optimizer_device` و `offload_param_device` تم تعيينهما إلى `cpu`.

```yml
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

## الأجزاء المهمة

دعونا نتعمق قليلاً في النص البرمجي حتى تتمكن من رؤية ما يحدث، وفهم كيفية عمله.

داخل الدالة [`main`](https://github.com/huggingface/peft/blob/2822398fbe896f25d4dac5e468624dc5fd65a51b/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py#L103)، يقوم النص البرمجي بإنشاء فئة [`~accelerate.Accelerator`] لتهيئة جميع المتطلبات اللازمة للتدريب الموزع.

<Tip>

💡 لا تتردد في تغيير النموذج ومجموعة البيانات داخل دالة `main`. إذا كان تنسيق مجموعة البيانات مختلفًا عن التنسيق الموجود في النص البرمجي، فقد تحتاج أيضًا إلى كتابة دالة المعالجة المسبقة الخاصة بك.

</Tip>

يقوم النص البرمجي أيضًا بإنشاء تهيئة لطريقة 🤗 PEFT التي تستخدمها، والتي في هذه الحالة، هي LoRA. تحدد [`LoraConfig`] نوع المهمة والمعلمات المهمة مثل بُعد المصفوفات ذات الرتبة المنخفضة، وعامل قياس المصفوفات، واحتمال إسقاط الطبقات LoRA. إذا كنت تريد استخدام طريقة 🤗 PEFT مختلفة، فتأكد من استبدال `LoraConfig` بالفصل المناسب [class](../package_reference/tuners).

```diff
 def main():
+    accelerator = Accelerator()
     model_name_or_path = "facebook/bart-large"
     dataset_name = "twitter_complaints"
+    peft_config = LoraConfig(
         task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
     )
```

في جميع أنحاء النص البرمجي، سترى وظائف [`~accelerate.Accelerator.main_process_first`] و [`~accelerate.Accelerator.wait_for_everyone`] التي تساعد في التحكم في عمليات المزامنة وتنفيذها.

تأخذ وظيفة [`get_peft_model`] نموذجًا أساسيًا و [`peft_config`] الذي قمت بإعداده سابقًا لإنشاء [`PeftModel`]:

```diff
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
+ model = get_peft_model(model, peft_config)
```

مرر جميع كائنات التدريب ذات الصلة إلى وظيفة [`~accelerate.Accelerator.prepare`] الخاصة بـ 🤗 Accelerate، والتي تتأكد من أن كل شيء جاهز للتدريب:

```py
model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler
)
```

يفحص الجزء التالي من التعليمات البرمجية ما إذا كان المكون الإضافي DeepSpeed مستخدمًا في `Accelerator`، وإذا كان المكون الإضافي موجودًا، فإننا نتحقق مما إذا كنا نستخدم ZeRO-3. يتم استخدام هذا العلم الشرطي عند استدعاء وظيفة `generate` أثناء الاستدلال لمزامنة وحدات معالجة الرسومات (GPUs) عندما تكون معلمات النموذج مجزأة:

```py
is_ds_zero_3 = False
if getattr(accelerator.state, "deepspeed_plugin", None):
    is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
```

داخل حلقة التدريب، يتم استبدال `loss.backward()` المعتادة بـ [`~accelerate.Accelerator.backward`] من 🤗 Accelerate، والتي تستخدم طريقة `backward()` الصحيحة بناءً على تهيئتك:

```diff
  for epoch in range(num_epochs):
      with TorchTracemalloc() as tracemalloc:
          model.train()
          total_loss = 0
          for step, batch in enumerate(tqdm(train_dataloader)):
              outputs = model(**batch)
              loss = outputs.loss
              total_loss += loss.detach().float()
+             accelerator.backward(loss)
              optimizer.step()
              lr_scheduler.step()
              optimizer.zero_grad()
```

هذا كل شيء! تتعامل بقية التعليمات البرمجية مع حلقة التدريب والتقييم، بل وتدفعها إلى Hub من أجلك.

## التدريب

قم بتشغيل الأمر التالي لبدء نص التدريب. في وقت سابق، قمت بحفظ ملف التكوين إلى `ds_zero3_cpu.yaml`، لذا ستحتاج إلى تمرير المسار إلى launcher باستخدام وسيط `--config_file` مثل هذا:

```bash
accelerate launch --config_file ds_zero3_cpu.yaml examples/peft_lora_seq2seq_accelerate_ds_zero3_offload.py
```

سترى بعض سجلات الإخراج التي تتتبع استخدام الذاكرة أثناء التدريب، وبمجرد اكتماله، يعيد النص البرمجي الدقة ويقارن التوقعات بالملصقات:

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

# التحذيرات

1. الدمج عند استخدام PEFT و DeepSpeed غير مدعوم حاليًا وسيؤدي إلى خطأ.
2. عند استخدام التفريغ إلى وحدة المعالجة المركزية (CPU offloading)، ستتحقق المكاسب الرئيسية من استخدام PEFT لتصغير حالات المحسن والتدرجات لتكون مثل أوزان المحول في ذاكرة الوصول العشوائي (RAM) للوحدة المركزية (CPU)، ولن تكون هناك وفورات فيما يتعلق بذاكرة وحدة معالجة الرسومات (GPU).
3. يؤدي DeepSpeed Stage 3 و qlora عند استخدامهما مع تفريغ وحدة المعالجة المركزية إلى زيادة استخدام ذاكرة وحدة معالجة الرسومات (GPU) عند مقارنته بتعطيل تفريغ وحدة المعالجة المركزية.