# الاستخدام الكامل لتوازي البيانات المجزأة

تم تطوير [التوازي الكامل للبيانات المجزأة](https://pytorch.org/docs/stable/fsdp.html) (FSDP) للتدريب الموزع للنماذج المسبقة التدريب الكبيرة التي يصل حجمها إلى 1 تيرابايت من المعلمات. ويحقق FSDP ذلك من خلال تجزئة معلمات النموذج والتدرجات وحالات المحسن عبر عمليات التوازي للبيانات، ويمكنه أيضًا نقل معلمات النموذج المجزأة إلى وحدة المعالجة المركزية (CPU). تسمح كفاءة الذاكرة التي يوفرها FSDP بزيادة حجم الدُفعة أو حجم النموذج.

يتم دعم كلا الميزتين في 🤗 Accelerate، ويمكنك استخدامهما مع 🤗 PEFT.

# استخدام PEFT وFSDP

سيساعدك هذا القسم من الدليل على تعلم كيفية استخدام نصنا البرمجي للتدريب [التدريب النصي](https://github.com/huggingface/peft/blob/main/examples/sft/train.py) DeepSpeed لأداء SFT. ستقوم بتكوين النص البرمجي للقيام بـ SFT (الضبط الدقيق الخاضع للإشراف) لنموذج Llama-70B مع LoRA وFSDP على 8xH100 من وحدات معالجة الرسوميات (GPU) بسعة 80 جيجابايت على جهاز واحد. يمكنك تكوينه ليتناسب مع عدة آلات عن طريق تغيير تكوين برنامج Accelerate.

## التهيئة

ابدأ بتشغيل الأمر التالي لإنشاء ملف تهيئة FSDP باستخدام 🤗 Accelerate. يسمح علم `--config_file` بحفظ ملف التهيئة في موقع محدد، وإلا فسيتم حفظه كملف `default_config.yaml` في ذاكرة التخزين المؤقت لـ 🤗 Accelerate.

يُستخدم ملف التهيئة لضبط الخيارات الافتراضية عند تشغيل نص التدريب.

```bash
accelerate config --config_file fsdp_config.yaml
```

سيُطلب منك الإجابة عن بعض الأسئلة حول إعدادك، وتهيئة الحجج التالية. في هذا المثال، ستجيب على الاستبيان كما هو موضح في الصورة أدناه.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/fsdp-peft-config.png"/>
</div>

<small>إنشاء تهيئة Accelerate لاستخدام FSDP</small>

بمجرد الانتهاء من ذلك، يجب أن يبدو التكوين المقابل كما هو موضح أدناه، ويمكنك العثور عليه في مجلد التهيئة في [fsdp_config.yaml](https://github.com/huggingface/peft/blob/main/examples/sft/configs/fsdp_config.yaml):

```yml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
fsdp_backward_prefetch: BACKWARD_PRE
fsdp_cpu_ram_efficient_loading: true
fsdp_forward_prefetch: false
fsdp_offload_params: false
fsdp_sharding_strategy: FULL_SHARD
fsdp_state_dict_type: SHARDED_STATE_DICT
fsdp_sync_module_states: true
fsdp_use_orig_params: false
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

أمر الإطلاق متاح في [run_peft_fsdp.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_peft_fsdp.sh) وهو موضح أدناه أيضًا:

```bash
accelerate launch --config_file "configs/fsdp_config.yaml"  train.py \
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
--output_dir "llama-sft-lora-fsdp" \
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

لاحظ أننا نستخدم LoRA مع الرتبة 8، و alpha=16، ونستهدف جميع الطبقات الخطية. نمرر ملف تهيئة FSDP ونضبط دقة نموذج Llama بحجم 70 جيجابايت على جزء فرعي من [مجموعة بيانات ultrachat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k).

## الأجزاء المهمة

دعونا نتعمق قليلاً في النص البرمجي حتى تتمكن من رؤية ما يحدث، وفهم كيفية عمله.

أول شيء يجب معرفته هو أن النص البرمجي يستخدم FSDP للتدريب الموزع نظرًا لمرور تهيئة FSDP. تتولى فئة `SFTTrainer` التعامل مع جميع المهام الثقيلة لإنشاء نموذج PEFT باستخدام تهيئة PEFT التي تم تمريرها. بعد ذلك، عندما تستدعي `trainer.train()`، يستخدم برنامج التدريب داخليًا 🤗 Accelerate لإعداد النموذج والمحسن وبرنامج التدريب باستخدام تهيئة FSDP لإنشاء نموذج ملفوف بـ FSDP يتم تدريبه بعد ذلك. مقتطف الكود الرئيسي هو أدناه:

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
if model_args.use_peft_lora:
# handle PEFT+FSDP case
trainer.model.print_trainable_parameters()
if getattr(trainer.accelerator.state, "fsdp_plugin", None):
from peft.utils.other import fsdp_auto_wrap_policy

fsdp_plugin = trainer.accelerator.state.fsdp_plugin
fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

# train
checkpoint = None
if training_args.resume_from_checkpoint is not None:
checkpoint = training_args.resume_from_checkpoint
trainer.train(resume_from_checkpoint=checkpoint)

# saving final model
if trainer.is_fsdp_enabled:
trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model()
```

هناك شيء رئيسي يجب ملاحظته هنا هو أنه عند استخدام FSDP مع PEFT، يجب أن يكون `use_orig_params` `False` لتحقيق وفورات في ذاكرة GPU. وبسبب `use_orig_params=False`، يجب تغيير سياسة التغليف التلقائي لـ FSDP بحيث يتم تغليف المعلمات القابلة للتدريب وغير القابلة للتدريب بشكل منفصل. يتم ذلك بواسطة مقتطف الكود أدناه والذي يستخدم دالة المساعدة `fsdp_auto_wrap_policy` من PEFT:

```
if getattr(trainer.accelerator.state, "fsdp_plugin", None):
from peft.utils.other import fsdp_auto_wrap_policy

fsdp_plugin = trainer.accelerator.state.fsdp_plugin
fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
```

## استخدام الذاكرة

في المثال أعلاه، تبلغ الذاكرة التي تستهلكها كل وحدة GPU 72-80 جيجابايت (90-98٪) كما هو موضح في لقطة الشاشة أدناه. تحدث الزيادة الطفيفة في ذاكرة GPU في النهاية عند حفظ النموذج باستخدام نوع `FULL_STATE_DICT` من القاموس بدلاً من `SHARDED_STATE_DICT` حتى يحتوي النموذج على أوزان محول يمكن تحميلها بشكل طبيعي باستخدام طريقة `from_pretrained` أثناء الاستدلال:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/peft_fsdp_mem_usage.png"/>
</div>

<small>استخدام ذاكرة GPU لتشغيل التدريب</small>

# استخدام PEFT QLoRA وFSDP لضبط دقة النماذج الكبيرة على وحدات معالجة الرسوميات (GPU) متعددة

في هذا القسم، سنلقي نظرة على كيفية استخدام QLoRA وFSDP لضبط دقة نموذج Llama بحجم 70 جيجابايت على وحدتي GPU بسعة 24 جيجابايت. قامت شركة [Answer.AI](https://www.answer.ai/) بالتعاون مع bitsandbytes وHugging Face 🤗 بإتاحة الكود الذي يمكّن من استخدام FSDP+QLoRA وشرحت العملية بأكملها في منشور المدونة المفيد [يمكنك الآن تدريب نموذج لغة بحجم 70 جيجابايت في المنزل](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html). تم دمج هذا الآن في نظام Hugging Face البيئي.

لهذا، نحتاج أولاً إلى `bitsandbytes>=0.43.0`، و`accelerate>=0.28.0`، و`transformers>4.38.2`، و`trl>0.7.11`، و`peft>0.9.0`. نحتاج إلى تعيين `fsdp_cpu_ram_efficient_loading=true`، و`fsdp_use_orig_params=false`، و`fsdp_offload_params=true` (التخزين المؤقت للقرص الصلب) عند استخدام تهيئة برنامج Accelerate. عند عدم استخدام برنامج الإطلاق Accelerate، يمكنك بدلاً من ذلك تعيين متغير البيئة `export FSDP_CPU_RAM_EFFICIENT_LOADING=true`. هنا، سنستخدم تهيئة برنامج Accelerate والتهيئة أدناه متوفرة في [fsdp_config_qlora.yaml](https://github.com/huggingface/peft/blob/main/examples/sft/configs/fsdp_config_qlora.yaml):

```yml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
fsdp_backward_prefetch: BACKWARD_PRE
fsdp_cpu_ram_efficient_loading: true
fsdp_forward_prefetch: false
fsdp_offload_params: true
fsdp_sharding_strategy: FULL_SHARD
fsdp_state_dict_type: SHARDED_STATE_DICT
fsdp_sync_module_states: true
fsdp_use_orig_params: false
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

أمر الإطلاق موضح أدناه وهو متاح في [run_peft_qlora_fsdp.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_peft_qlora_fsdp.sh):

```
accelerate launch --config_file "configs/fsdp_config_qlora.yaml"  train.py \
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
--output_dir "llama-sft-qlora-fsdp" \
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

لاحظ الحجة الجديدة التي يتم تمريرها، `bnb_4bit_quant_storage_dtype`، والتي تشير إلى نوع البيانات لتغليف معلمات 4 بت. على سبيل المثال، عندما يتم تعيينه على `bfloat16`، يتم تغليف **16/4 = 4** من معلمات 4 بت معًا بعد التكميم. عند استخدام التدريب بالدقة المختلطة مع `bfloat16`، يمكن أن يكون `bnb_4bit_quant_storage_dtype` إما `bfloat16` للتدريب على `bfloat16` النقي، أو `float32` للتحديد التلقائي للدقة المختلطة (يستهلك هذا المزيد من ذاكرة GPU). عند استخدام التدريب بالدقة المختلطة مع `float16`، يجب تعيين `bnb_4bit_quant_storage_dtype` على `float32` للتدريب التلقائي المختلط الدقيق والمستقر.

من حيث تغييرات كود التدريب، تتمثل التغييرات المهمة في ما يلي:

```diff
...

bnb_config = BitsAndBytesConfig(
load