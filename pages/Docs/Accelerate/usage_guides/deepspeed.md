# DeepSpeed

ينفذ DeepSpeed كل ما هو موصوف في ورقة ZeRO. بعض التحسينات الملحوظة هي:

1. تجزئة حالة المحسن (المرحلة الأولى من ZeRO)
2. تجزئة التدرج (المرحلة الثانية من ZeRO)
3. تجزئة المعلمات (المرحلة الثالثة من ZeRO)
4. التعامل مع التدريب على الدقة المختلطة المخصصة
5. مجموعة من المحسنات السريعة المستندة إلى امتداد CUDA
6. ZeRO-Offload إلى CPU والقرص/NVMe
7. التجزئة الهرمية لمعلمات النموذج (ZeRO++)

لدى ZeRO-Offload ورقته الخاصة: [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840). ويتم وصف دعم NVMe في الورقة [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857).

يستخدم DeepSpeed ZeRO-2 بشكل أساسي للتدريب فقط، حيث أن ميزاته لا تفيد الاستدلال.

يمكن استخدام DeepSpeed ZeRO-3 للاستدلال أيضًا لأنه يسمح بتحميل نماذج ضخمة على وحدات معالجة الرسومات (GPU) متعددة، وهو ما لن يكون ممكنًا على وحدة معالجة الرسومات (GPU) واحدة.

يدمج 🤗 Accelerate [DeepSpeed](https://github.com/microsoft/DeepSpeed) عبر خيارين:

1. دمج ميزات DeepSpeed عبر مواصفات `deepspeed config file` في `accelerate config`. ما عليك سوى توفير ملف التكوين المخصص الخاص بك أو استخدام القالب الخاص بنا. يركز معظم هذه الوثيقة على هذه الميزة. يدعم هذا جميع الميزات الأساسية لـ DeepSpeed ويمنح المستخدم الكثير من المرونة. قد يضطر المستخدم إلى تغيير بضع سطور من التعليمات البرمجية اعتمادًا على التكوين.
2. التكامل عبر `deepspeed_plugin`. يدعم هذا جزءًا فرعيًا من ميزات DeepSpeed ويستخدم الخيارات الافتراضية لبقية التكوينات. لا يحتاج المستخدم إلى تغيير أي رمز وهو جيد لمن يرضون عن معظم الإعدادات الافتراضية لـ DeepSpeed.

## ما هو المدمج؟

التدريب:

1. يدمج 🤗 Accelerate جميع ميزات DeepSpeed ZeRO. ويشمل ذلك جميع مراحل ZeRO 1 و2 و3 بالإضافة إلى ZeRO-Offload وZeRO-Infinity (التي يمكنها تفريغ القرص/NVMe) وZeRO++. فيما يلي وصف قصير لتوازي البيانات باستخدام ZeRO - Zero Redundancy Optimizer إلى جانب رسم بياني من هذه [التدوينة](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

![توازي بيانات ZeRO](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png)

(المصدر: [الرابط](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/))

 أ. **المرحلة 1**: شظايا حالات المحسن عبر عمال/وحدات معالجة الرسومات المتوازية للبيانات
 ب. **المرحلة 2**: شظايا حالات التدرج + المحسن عبر عمال/وحدات معالجة الرسومات المتوازية للبيانات
 ج. **المرحلة 3**: شظايا حالات التدرج + المحسن + معلمات النموذج عبر عمال/وحدات معالجة الرسومات المتوازية للبيانات
 د. **تحسين التفريغ**: يقوم بتفريغ التدرجات + حالات المحسن إلى وحدة المعالجة المركزية/القرص بناءً على ZERO Stage 2
 هـ. **تفريغ المعلمة**: يقوم بتفريغ معلمات النموذج إلى وحدة المعالجة المركزية/القرص بناءً على ZERO Stage 3
 و. **التجزئة الهرمية**: تمكن التدريب متعدد العقد بكفاءة مع التدريب المتوازي للبيانات عبر العقد وتجزئة ZeRO-3 داخل العقدة، المبنية على ZeRO Stage 3.

> ملاحظة: فيما يتعلق بتفريغ القرص، يجب أن يكون القرص NVME للحصول على سرعة جيدة ولكنه يعمل من الناحية الفنية على أي قرص.

الاستدلال:

1. يدعم DeepSpeed ZeRO Inference مرحلة ZeRO 3 مع ZeRO-Infinity. يستخدم نفس بروتوكول ZeRO مثل التدريب، ولكنه لا يستخدم محسنًا ومخططًا للتعلم ومعدل الخطأ، والمرحلة 3 فقط ذات صلة. لمزيد من التفاصيل، راجع: [deepspeed-zero-inference](#deepspeed-zero-inference).

## كيف يعمل؟

**المتطلبات الأساسية**: قم بتثبيت DeepSpeed الإصدار >=0.6.5. يرجى الرجوع إلى [تفاصيل تثبيت DeepSpeed](https://github.com/microsoft/DeepSpeed#installation) لمزيد من المعلومات.

سنلقي نظرة أولاً على تكامل سهل الاستخدام عبر `accelerate config`.

تليها أكثر مرونة وغنية بالميزات `دمج ملف تكوين DeepSpeed`.

### مكون إضافي DeepSpeed المعجل

على جهاز الكمبيوتر الخاص بك (أجهزتك) فقط قم بتشغيل:

```bash
accelerate config
```

والإجابة عن الأسئلة المطروحة. سيسألك عما إذا كنت تريد استخدام ملف تكوين لـ DeepSpeed والذي يجب أن تجيب عليه بلا. ثم أجب عن الأسئلة التالية لإنشاء تكوين DeepSpeed الأساسي.

سيؤدي هذا إلى إنشاء ملف تكوين سيتم استخدامه تلقائيًا لضبط

الخيارات الافتراضية عند القيام بذلك

```bash
accelerate launch my_script.py --args_to_my_script
```

على سبيل المثال، إليك كيفية تشغيل مثال NLP `examples/nlp_example.py` (من الجذر المستودع) باستخدام مكون DeepSpeed الإضافي:

**مثال على مرحلة ZeRO Stage-2 DeepSpeed Plugin**

```bash
compute_environment: LOCAL_MACHINE
deepspeed_config:
gradient_accumulation_steps: 1
gradient_clipping: 1.0
offload_optimizer_device: none
offload_param_device: none
zero3_init_flag: true
zero_stage: 2
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

```bash
accelerate launch examples/nlp_example.py --mixed_precision fp16
```

**مثال على مرحلة ZeRO Stage-3 مع تفريغ وحدة المعالجة المركزية DeepSpeed Plugin**

```bash
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
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

```bash
accelerate launch examples/nlp_example.py --mixed_precision fp16
```

يدعم حاليًا `Accelerate` التكوين التالي عبر واجهة سطر الأوامر:

```bash
`zero_stage`: [0] Disabled، [1] تجزئة حالة المحسن، [2] تجزئة حالة التدرج + المحسن و[3] تجزئة التدرج + المحسن + تجزئة المعلمات
`gradient_accumulation_steps`: عدد خطوات التدريب لتراكم التدرجات قبل متوسطها وتطبيقها.
`gradient_clipping`: تمكين قص التدرج بالقيمة.
`offload_optimizer_device`: [none] تعطيل تفريغ المحسن، [cpu] تفريغ المحسن إلى وحدة المعالجة المركزية، [nvme] تفريغ المحسن إلى SSD NVMe. ينطبق فقط مع ZeRO >= Stage-2.
`offload_optimizer_nvme_path`: يحدد مسار Nvme لتفريغ حالات المحسن. إذا لم يتم تحديده، فسيتم تعيينه افتراضيًا على 'none'.
`offload_param_device`: [none] تعطيل تفريغ المعلمة، [cpu] تفريغ المعلمات إلى وحدة المعالجة المركزية، [nvme] تفريغ المعلمات إلى SSD NVMe. ينطبق فقط مع مرحلة ZeRO 3.
`offload_param_nvme_path`: يحدد مسار Nvme لتفريغ المعلمات. إذا لم يتم تحديده، فسيتم تعيينه افتراضيًا على 'none'.
`zero3_init_flag`: يقرر ما إذا كان سيتم تمكين `deepspeed.zero.Init` لبناء النماذج الضخمة. ينطبق فقط مع مرحلة ZeRO 3.
`zero3_save_16bit_model`: يقرر ما إذا كان سيتم حفظ أوزان النموذج 16 بت عند استخدام مرحلة ZeRO 3.
`mixed_precision`: `no` للتدريب FP32، `fp16` للتدريب على الدقة المختلطة FP16 و`bf16` للتدريب على الدقة المختلطة BF16.
`deepspeed_moe_layer_cls_names`: قائمة مفصولة بفواصل لأسماء فئات طبقة Mixture-of-Experts (MoE) المحولة (حساسة لحالة الأحرف) للتغليف، على سبيل المثال، `MixtralSparseMoeBlock`، `Qwen2MoeSparseMoeBlock`، `JetMoEAttention,JetMoEBlock` ...
`deepspeed_hostfile`: ملف المضيف DeepSpeed لتكوين موارد الحوسبة متعددة العقد.
`deepspeed_exclusion_filter`: سلسلة عامل تصفية الاستبعاد DeepSpeed عند استخدام إعداد متعدد العقد.
`deepspeed_inclusion_filter`: سلسلة عامل تصفية الإدراج DeepSpeed عند استخدام إعداد متعدد العقد.
`deepspeed_multinode_launcher`: برنامج الإطلاق متعدد العقد DeepSpeed الذي سيتم استخدامه. إذا لم يتم تحديده، فسيتم تعيينه افتراضيًا على "pdsh".
`deepspeed_config_file`: مسار إلى ملف تكوين DeepSpeed بتنسيق "json". راجع القسم التالي لمزيد من التفاصيل حول هذا.
```

للتمكن من ضبط المزيد من الخيارات، ستحتاج إلى استخدام ملف تكوين DeepSpeed.
### ملف تكوين DeepSpeed

على جهازك (أجهزتك)، قم فقط بتشغيل ما يلي:

```bash
accelerate config
```

وأجب عن الأسئلة المطروحة. سيسألك عما إذا كنت تريد استخدام ملف تكوين لـ DeepSpeed، فتجيب بـ "نعم" وتقدم مسار ملف تكوين DeepSpeed.

سيؤدي ذلك إلى إنشاء ملف تكوين سيتم استخدامه تلقائيًا لضبط الخيارات الافتراضية بشكل صحيح عند القيام بما يلي:

```bash
accelerate launch my_script.py --args_to_my_script
```

على سبيل المثال، إليك كيفية تشغيل مثال NLP `examples/by_feature/deepspeed_with_config_support.py` (من الجذر الخاص بالمستودع) باستخدام ملف تكوين DeepSpeed:

**مثال على ملف تكوين DeepSpeed لمرحلة ZeRO 2**

```bash
compute_environment: LOCAL_MACHINE
deepspeed_config:
deepspeed_config_file: /home/ubuntu/accelerate/examples/configs/deepspeed_config_templates/zero_stage2_config.json
zero3_init_flag: true
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

مع محتويات `zero_stage2_config.json` كما يلي:

```json
{
"fp16": {
"enabled": true,
"loss_scale": 0,
"loss_scale_window": 1000,
"initial_scale_power": 16,
"hysteresis": 2,
"min_loss_scale": 1
},
"optimizer": {
"type": "AdamW",
"params": {
"lr": "auto",
"weight_decay": "auto",
"torch_adam": true,
"adam_w_mode": true
}
},
"scheduler": {
"type": "WarmupDecayLR",
"params": {
"warmup_min_lr": "auto",
"warmup_max_lr": "auto",
"warmup_num_steps": "auto",
"total_num_steps": "auto"
}
},
"zero_optimization": {
"stage": 2,
"allgather_partitions": true,
"allgather_bucket_size": 2e8,
"overlap_comm": true,
"reduce_scatter": true,
"reduce_bucket_size": "auto",
"contiguous_gradients": true
},
"gradient_accumulation_steps": 1,
"gradient_clipping": "auto",
"steps_per_print": 2000,
"train_batch_size": "auto",
"train_micro_batch_size_per_gpu": "auto",
"wall_clock_breakdown": false
}
```

```bash
accelerate launch examples/by_feature/deepspeed_with_config_support.py \
--config_name "gpt2-large" \
--tokenizer_name "gpt2-large" \
--dataset_name "wikitext" \
--dataset_config_name "wikitext-2-raw-v1" \
--block_size 128 \
--output_dir "./clm/clm_deepspeed_stage2_accelerate" \
--learning_rate 5e-4 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_Multiplier 24 \
--num_train_epochs 3 \
--with_tracking \
--report_to "wandb"\
```

**مثال على ملف تكوين DeepSpeed لمرحلة ZeRO 3 مع إزاحة التحميل إلى CPU**

```bash
compute_environment: LOCAL_MACHINE
deepspeed_config:
deepspeed_config_file: /home/ubuntu/accelerate/examples/configs/deepspeed_config_templates/zero_stage3_offload_config.json
zero3_init_flag: true
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

مع محتويات `zero_stage3_offload_config.json` كما يلي:

```json
{
"fp16": {
"enabled": true,
"loss_scale": 0,
"loss_scale_window": 1000,
"initial_scale_power": 16,
"hysteresis": 2,
"min_loss_scale": 1
},
"optimizer": {
"type": "AdamW",
"params": {
"lr": "auto",
"weight_decay": "auto"
}
},
"scheduler": {
"type": "WarmupDecayLR",
"params": {
"warmup_min_lr": "auto",
"warmup_max_lr": "auto",
"warmup_num_steps": "auto",
"total_num_steps": "auto"
}
},
"zero_optimization": {
"stage": 3,
"offload_optimizer": {
"device": "cpu",
"pin_memory": true
},
"offload_param": {
"device": "cpu",
"pin_memory": true
},
"overlap_comm": true,
"contiguous_gradients": true,
"reduce_bucket_size": "auto",
"stage3_prefetch_bucket_size": "auto",
"stage3_param_persistence_threshold": "auto",
"sub_group_size": 1e9,
"stage3_max_live_parameters": 1e9,
"stage3_max_reuse_distance": 1e9,
"stage3_gather_16bit_weights_on_model_save": "auto"
},
"gradient_accumulation_steps": 1,
"gradient_clipping": "auto",
"steps_per_print": 2000,
"train_batch_size": "auto",
"train_micro_batch_size_per_gpu": "auto",
"wall_clock_breakdown": false
}
```

```bash
accelerate launch examples/by_feature/deepspeed_with_config_support.py \
--config_name "gpt2-large" \
--tokenizer_name "gpt2-large" \
--dataset_name "wikitext" \
--dataset_config_name "wikitext-2-raw-v1" \
--block_size 128 \
--output_dir "./clm/clm_deepspeed_stage3_offload_accelerate" \
--learning_rate 5e-4 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--num_train_epochs 3 \
--with_tracking \
--report_to "wandb"\
```

**مثال على تكوين ZeRO++**

يمكنك استخدام ميزات ZeRO++ من خلال استخدام معلمات التكوين المناسبة. لاحظ أن ZeRO++ هو امتداد لـ ZeRO Stage 3. فيما يلي كيفية تعديل ملف التكوين، من [دليل DeepSpeed حول ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/):

```json
{
"zero_optimization": {
"stage": 3,
"reduce_bucket_size": "auto",

"zero_quantized_weights": true,
"zero_hpz_partition_size": 8,
"zero_quantized_gradients": true,

"contiguous_gradients": true,
"overlap_comm": true
}
}
```

بالنسبة للتجزئة الهرمية، يجب تعيين حجم التجزئة `zero_hpz_partition_size` بشكل مثالي إلى عدد وحدات معالجة الرسوميات (GPU) لكل عقدة. (على سبيل المثال، يفترض ملف التكوين أعلاه 8 وحدات GPU لكل عقدة)

**تغييرات الشفرة المهمة عند استخدام ملف تكوين DeepSpeed**

1. محسنات DeepSpeed وجداولها الزمنية: للحصول على مزيد من المعلومات حول هذه الموضوعات، يرجى الاطلاع على وثائق [محسنات DeepSpeed](https://deepspeed.readthedocs.io/en/latest/optimizers.html) و[جداول DeepSpeed الزمنية](https://deepspeed.readthedocs.io/en/latest/schedulers.html). وسنلقي نظرة على التغييرات المطلوبة في الشفرة عند استخدام هذه الميزات.

   أ) محسن DeepSpeed + جدول DeepSpeed الزمني: الحالة التي يكون فيها كل من مفتاحي `optimizer` و`scheduler` موجودين في ملف تكوين DeepSpeed. في هذه الحالة، سيتم استخدامهما ويجب على المستخدم استخدام `accelerate.utils.DummyOptim` و`accelerate.utils.DummyScheduler` ليحل محل المحسنات والجداول الزمنية من PyTorch/المخصصة في شفرته.

   فيما يلي مقتطف من `examples/by_feature/deepspeed_with_config_support.py` يوضح ذلك:

   ```python
   # إنشاء محسن Dummy إذا تم تحديد "optimizer" في ملف التكوين، وإلا قم بإنشاء محسن Adam
   optimizer_cls = (
   torch.optim.AdamW
   if accelerator.state.deepspeed_plugin is None
   or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
   else DummyOptim
   )
   optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate)

   # إنشاء جدول زمني Dummy إذا تم تحديد "scheduler" في ملف التكوين، وإلا قم بإنشاء جدول زمني من النوع "args.lr_scheduler_type"
   if (
   accelerator.state.deepspeed_plugin is None
   or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
   ):
   lr_scheduler = get_scheduler(
   name=args.lr_scheduler_type,
   optimizer=optimizer,
   num_warmup_steps=args.num_warmup_steps,
   num_training_steps=args.max_train_steps,
   )
   else:
   lr_scheduler = DummyScheduler(
   optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
   )
   ```

   ب) محسن مخصص + جدول زمني مخصص: الحالة التي يكون فيها كل من مفتاحي `optimizer` و`scheduler` غائبين في ملف تكوين DeepSpeed. في هذه الحالة، لا يلزم إجراء أي تغييرات على الشفرة من قبل المستخدم، وهذه هي الحالة عند استخدام التكامل عبر مكون إضافي لـ DeepSpeed.

   في المثال أعلاه، يمكننا أن نرى أن الشفرة تبقى دون تغيير إذا كانت مفاتيح `optimizer` و`scheduler` غائبة في ملف تكوين DeepSpeed.

   ج) محسن مخصص + جدول DeepSpeed الزمني: الحالة التي يكون فيها مفتاح `scheduler` فقط موجودًا في ملف تكوين DeepSpeed. في هذه الحالة، يجب على المستخدم استخدام `accelerate.utils.DummyScheduler` ليحل محل الجدول الزمني من PyTorch/المخصص في شفرته.

   د) محسن DeepSpeed + جدول زمني مخصص: الحالة التي يكون فيها مفتاح `optimizer` فقط موجودًا في ملف تكوين DeepSpeed. سيؤدي ذلك إلى حدوث خطأ لأنك لا يمكنك استخدام جدول DeepSpeed الزمني إلا عند استخدام محسن DeepSpeed.

2. لاحظ قيم "auto" في ملفات تكوين DeepSpeed في الأمثلة أعلاه. تتم معالجة هذه القيم تلقائيًا بواسطة طريقة `prepare` بناءً على النموذج، ودورات التدريب، والمحسنات والجداول الزمنية الوهمية المقدمة إلى طريقة `prepare`.

   تتم معالجة قيم "auto" المحددة في الأمثلة أعلاه فقط بواسطة طريقة `prepare`، ويجب على المستخدم تحديد القيم الأخرى صراحة.

   يتم حساب قيم "auto" على النحو التالي:

   - `reduce_bucket_size`: `hidden_size * hidden_size`
   - `stage3_prefetch_bucket_size`: `int(0.9 * hidden_size * hidden_size)`
   - `stage3_param_persistence_threshold`: `10 * hidden_size`

   بالنسبة لميزة "auto" للعمل لهذه الإدخالات الثلاثة من ملف التكوين - سيستخدم Accelerate `model.config.hidden_size` أو `max(model.config.hidden_sizes)` كـ `hidden_size`. إذا لم يكن أي من هذين متاحًا، فسوف يفشل الإطلاق وسيتعين عليك تعيين إدخالات ملف التكوين هذه يدويًا. تذكر أن أول إدخالين لملف التكوين هما مخازن مؤقتة للاتصالات - كلما كبرت، زادت كفاءة الاتصالات، وكلما كبرت، استهلكت المزيد من ذاكرة وحدة معالجة الرسوميات (GPU)، لذا فهي قابلة للضبط وفقًا لأداء المقايضة.

**أشياء يجب ملاحظتها عند استخدام ملف تكوين DeepSpeed**

فيما يلي نموذج من الشفرة يستخدم `deepspeed_config_file` في سيناريوهات مختلفة:

شفرة `test.py`:

```python
from accelerate import Accelerator
from accelerate.state import AcceleratorState


def main():
accelerator = Accelerator()
accelerator.print(f"{AcceleratorState()}")


if __name__ == "__main__":
main()
```

**السيناريو 1**: ملف تكوين Accelerate الذي تم التلاعب به يدويًا والذي يحتوي على `deepspeed_config_file` إلى جانب إدخالات أخرى.

1. محتويات ملف تكوين `accelerate`:

   ```yaml
   command_file: null
   commands: null
   compute_environment: LOCAL_MACHINE
   deepspeed_config:
   gradient_accumulation_steps: 1
   gradient_clipping: 1.0
   offload_optimizer_device: 'cpu'
   offload_param_device: 'cpu'
   zero3_init_flag: true
   zero3_save_16bit_model: true
   zero_stage: 3
   deepspeed_config_file: 'ds_config.json'
   distributed_type: DEEPSPEED
   downcast_bf16: 'no'
   dynamo_backend: 'NO'
   fsdp_config: {}
   gpu_ids: null
   machine_rank: 0
   main_process_ip: null
   main_process_port: null
   main_training_function: main
   megatron_lm_config: {}
   num_machines: 1
   num_processes: 2
   rdzv_backend: static
   same_network: true
   tpu_name: null
   tpu_zone: null
   use_cpu: false
   ```

2. `ds_config.json`:

   ```json
   {
   "bf16": {
   "enabled": true
   },
   "zero_optimization": {
   "stage": 3,
   "stage3_gather_16bit_weights_on_model_save": false,
   "offload_optimizer": {
   "device": "none"
   },
   "offload_param": {
   "device": "none"
   }
   },
   "gradient_clipping": 1.0,
   "train_batch_size": "auto",
   "train_micro_batch_size_per_gpu": "auto",
   "gradient_accumulation_steps": 10,
   "steps_per_print": 2000000
   }
   ```

3. إخراج `accelerate launch test.py`:

   ```bash
   ValueError: When using `deepspeed_config_file`, the following accelerate config variables will be ignored:
   ['gradient_accumulation_steps', 'gradient_clipping', 'zero_stage', 'offload_optimizer_device', 'offload_param_device',
   'zero3_save_16bit_model', 'mixed_precision'].
   Please specify them appropriately in the DeepSpeed config file.
   If you are using an accelerate config file, remove other config variables mentioned in the above specified list.
   The easiest method is to create a new config following the questionnaire via `accelerate config`.
   It will only ask for the necessary config variables when using `deepspeed_config_file`.
   ```

**السيناريو 2**: استخدم حل الخطأ لإنشاء ملف تكوين جديد وتحقق من عدم ظهور خطأ الغموض الآن.

1. قم بتشغيل `accelerate config`:

   ```bash
   $ accelerate config
   -------------------------------------------------------------------------------------------------------------------------------
   On which compute environment are you running?
   This machine
   -------------------------------------------------------------------------------------------------------------------------------
   What type of machine are you using?
   multi-GPU
   How many different machines will you use (use more than 1 for multi-node training)? [1]:
   Do you wish to optimize your script with torch dynamo?[yes/NO]:
## الحفظ والتحميل

1. لم يتغير حفظ وتحميل النماذج في ZeRO Stage-1 وStage-2.
2. في ZeRO Stage-3، يحتوي `state_dict` فقط على المؤشرات المكانية لأن أوزان النموذج مجزأة عبر عدة وحدات معالجة رسومية. يوفر ZeRO Stage-3 خيارين:

أ. حفظ أوزان النموذج الكاملة ذات 16 بت لتحميلها مباشرة في وقت لاحق باستخدام `model.load_state_dict(torch.load(pytorch_model.bin))`.

لهذا، قم بتعيين `zero_optimization.stage3_gather_16bit_weights_on_model_save` إلى True في ملف تكوين DeepSpeed أو قم بتعيين `zero3_save_16bit_model` إلى True في DeepSpeed Plugin.

**لاحظ أن هذا الخيار يتطلب توحيد الأوزان على وحدة معالجة رسومية واحدة، ويمكن أن يكون بطيئًا ومستهلكًا للذاكرة، لذا استخدم هذه الميزة عند الحاجة فقط.**

فيما يلي مقتطف من `examples/by_feature/deepspeed_with_config_support.py` يوضح ذلك:

```python
unwrapped_model = accelerator.unwrap_model(model)

# New Code #
# Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
# `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
# `zero3_save_16bit_model` is True in DeepSpeed Plugin.
# For Zero Stages 1 and 2, models are saved as usual in the output directory.
# The model name saved is `pytorch_model.bin`
unwrapped_model.save_pretrained(
args.output_dir,
is_main_process=accelerator.is_main_process,
save_function=accelerator.save,
state_dict=accelerator.get_state_dict(model),
)
```

ب. للحصول على أوزان 32 بت، قم أولاً بحفظ النموذج باستخدام `model.save_checkpoint()`.

فيما يلي مقتطف من `examples/by_feature/deepspeed_with_config_support.py` يوضح ذلك:

```python
success = model.save_checkpoint(PATH, ckpt_id, checkpoint_state_dict)
status_msg = f"checkpointing: PATH={PATH}, ckpt_id={ckpt_id}"
if success:
logging.info(f"Success {status_msg}")
else:
logging.warning(f"Failure {status_msg}")
```

سيؤدي هذا إلى إنشاء أقسام نموذج ZeRO ومقسام المحسن إلى جانب `zero_to_fp32.py` script في دليل نقطة التفتيش.

يمكنك استخدام هذا البرنامج النصي للقيام بالدمج غير المتصل. لا يتطلب ملفات تكوين أو وحدات معالجة رسومية. فيما يلي مثال على استخدامه:

```bash
$ cd /path/to/checkpoint_dir
$ ./zero_to_fp32.py . pytorch_model.bin
Processing zero checkpoint at global_step1
Detected checkpoint of type zero stage 3, world_size: 2
Saving fp32 state dict to pytorch_model.bin (total_numel=60506624)
```

للحصول على نموذج 32 بت للحفظ/الاستدلال، يمكنك القيام بما يلي:

```python
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

unwrapped_model = accelerator.unwrap_model(model)
fp32_model = load_state_dict_from_zero_checkpoint(unwrapped_model, checkpoint_dir)
```

إذا كنت مهتمًا فقط بـ `state_dict`، فيمكنك القيام بما يلي:

```python
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
```

لاحظ أن جميع هذه الوظائف تتطلب ~2x ذاكرة (ذاكرة الوصول العشوائي العامة) لحجم نقطة التفتيش النهائية.

## استدلال Zero

يدعم DeepSpeed Zero Inference مرحلة Zero 3 مع Zero-Infinity.

يستخدم نفس بروتوكول Zero كما هو الحال في التدريب، ولكنه لا يستخدم محسنًا ومخططًا لمعدل التعلم، ولا تكون المرحلة 3 ذات صلة.

مع تكامل التسريع، تحتاج فقط إلى إعداد النموذج وdataloader كما هو موضح أدناه:

```python
model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
```

## بعض التحذيرات التي يجب مراعاتها

1. لا يدعم التكامل الحالي موازاة الأنابيب في DeepSpeed.
2. لا يدعم التكامل الحالي `mpu`، مما يحد من موازاة المصفوفة المدعومة في Megatron-LM.
3. لا يدعم التكامل الحالي نماذج متعددة.

## موارد DeepSpeed

يمكن العثور على الوثائق الخاصة بالجوانب الداخلية المتعلقة بـ deepspeed [هنا](../package_reference/deepspeed).

- [مشروع GitHub](https://github.com/microsoft/deepspeed)
- [وثائق الاستخدام](https://www.deepspeed.ai/getting-started/)
- [وثائق واجهة برمجة التطبيقات](https://deepspeed.readthedocs.io/en/latest/index.html)
- [المنشورات على المدونة](https://www.microsoft.com/en-us/research/search/?q=deepspeed)

الأوراق:

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- [ZeRO++: Extremely Efficient Collective Communication for Giant Model Training](https://arxiv.org/abs/2306.10209)

أخيرًا، يرجى تذكر أن 🤗 `Accelerate` يدمج DeepSpeed فقط، لذلك إذا واجهتك أي مشكلات أو أسئلة تتعلق باستخدام DeepSpeed، فيرجى إرسال مشكلة إلى [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed/issues).

<Tip>

بالنسبة لأولئك المهتمين بأوجه التشابه والاختلاف بين FSDP وDeepSpeed، يرجى الاطلاع على [دليل المفاهيم هنا](../concept_guides/fsdp_and_deepspeed.md)!

</Tip>