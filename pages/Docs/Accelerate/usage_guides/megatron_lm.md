لم يتم ترجمة الأجزاء التي طلبت عدم ترجمتها، مثل الروابط ورموز HTML وCSS والشفرات البرمجية.

# Megatron-LM

يتيح [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) تدريب نماذج اللغة الضخمة المحولة على نطاق واسع. يوفر موازاة فعالة للنسيج وخط الأنابيب والتسلسل لنماذج ما قبل التدريب المستندة إلى المحول مثل [GPT](https://arxiv.org/abs/2005.14165) (فك التشفير فقط) و [BERT](https://arxiv.org/pdf/1810.04805.pdf) (الترميز فقط) و [T5](https://arxiv.org/abs/1910.10683) (الترميز وفك التشفير). للحصول على معلومات مفصلة وكيفية عمل الأشياء خلف الكواليس، يرجى الرجوع إلى [repo](https://github.com/NVIDIA/Megatron-LM) على GitHub.

## ما هو المدمج؟

يدمج Accelerate الميزة التالية من Megatron-LM لتمكين ما قبل التدريب/التدريب الدقيق على نطاق واسع لـ BERT (Encoder) أو GPT (Decoder) أو نماذج T5 (Encoder and Decoder):

أ. **موازاة النسيج (TP)**: تقلل من البصمة الذاكرة دون الكثير من الاتصال الإضافي على رتب العقد داخل العقدة. يتم تقسيم كل تنسيق إلى عدة قطع، مع وجود كل شريحة على وحدة GPU منفصلة. في كل خطوة، تتم معالجة نفس الدفعة الصغيرة من البيانات بشكل مستقل وبالتوازي من قبل كل شريحة، يليها المزامنة عبر جميع وحدات GPU (`عملية all-reduce`). في طبقة محول بسيطة، يؤدي هذا إلى `all-reduces` 2 في المسار الأمامي و 2 في المسار الخلفي. لمزيد من التفاصيل، يرجى الرجوع إلى الورقة البحثية [Megatron-LM: تدريب نماذج لغة معلمات متعددة المليارات باستخدام موازاة النموذج](https://arxiv.org/pdf/1909.08053.pdf) وهذا القسم من منشور المدونة [التكنولوجيا وراء تدريب BLOOM](https://huggingface.co/blog/bloom-megatron-deepspeed#tensor-parallelism).

ب. **موازاة خط الأنابيب (PP)**: تقليل البصمة الذاكرة وتمكين التدريب واسع النطاق من خلال موازاة العقدة بين العقدة. تقلل فقاعة PP الساذجة من خلال جدول PipeDream-Flush schedule/1F1B وجدول 1F1B متداخل. يتم توزيع الطبقات بالتساوي عبر مراحل PP. على سبيل المثال، إذا كان لدى النموذج "24" طبقة ولدينا "4" وحدات GPU لموازاة خط الأنابيب، فستحتوي كل وحدة GPU على "6" طبقات (24/4). لمزيد من التفاصيل حول الجداول الزمنية لتقليل وقت الخمول لـ PP، يرجى الرجوع إلى الورقة البحثية [تدريب نماذج لغة واسعة النطاق بكفاءة على مجموعات GPU باستخدام Megatron-LM](https://arxiv.org/pdf/2104.04473.pdf) وهذا القسم من منشور المدونة [التكنولوجيا وراء تدريب BLOOM](https://huggingface.co/blog/bloom-megatron-deepspeed#pipeline-parallelism).

ج. **تسلسل الموازاة (SP)**: تقليل البصمة الذاكرة دون أي اتصال إضافي. لا ينطبق إلا عند استخدام TP. إنه يقلل من ذاكرة التنشيط المطلوبة لأنه يمنع وجود نفس النسخ على رتب موازاة النسيج بعد `all-reduce` عن طريق استبدالها بعملية `reduce-scatter` وسيتم استبدال عملية `no-op` بعملية `all-gather`. نظرًا لأن `all-reduce = reduce-scatter + all-gather`، فإن هذا يوفر الكثير من ذاكرة التنشيط دون أي تكلفة إضافية للاتصال. ببساطة، يقوم بتشطير مخرجات كل طبقة محول على طول البعد التسلسلي، على سبيل المثال، إذا كان طول التسلسل هو "1024" وكان حجم TP هو "4"، فستحتوي كل وحدة GPU على "256" رمزًا (1024/4) لكل عينة. وهذا يزيد من حجم الدفعة التي يمكن دعمها للتدريب. لمزيد من التفاصيل، يرجى الرجوع إلى الورقة البحثية [تقليل إعادة حساب التنشيط في نماذج المحول الكبيرة](https://arxiv.org/pdf/2205.05198.pdf).

د. **موازاة البيانات (DP)** عبر الموزع المُنَسِّق: تقلل من البصمة الذاكرة عن طريق تجزئة حالات المُحَسِّن والتدرجات عبر رتب DP (مقابل الطريقة التقليدية لتكرار حالة المُحَسِّن عبر رتب موازاة البيانات). على سبيل المثال، عند استخدام محسن Adam مع التدريب الدقيق، يحتوي كل معلمة على 12 بايت من الذاكرة. يتم توزيع هذا بالتساوي عبر وحدات GPU، أي أن كل معلمة ستحتوي على 3 بايت (12/4) إذا كان لدينا 4 وحدات GPU. لمزيد من التفاصيل، يرجى الرجوع إلى الورقة البحثية [ZeRO: تحسين الذاكرة نحو تدريب نماذج المعلمات التريليونية](https://arxiv.org/pdf/1910.02054.pdf) والقسم التالي من 🤗 blog [التكنولوجيا وراء تدريب BLOOM](https://huggingface.co/blog/bloom-megatron-deepspeed#zero-data-parallelism).

هـ. **إعادة حساب التنشيط الانتقائي**: تقليل البصمة الذاكرة للتنشيط بشكل كبير من خلال تسجيل نقاط التنشيط الذكية. لا يقوم بتخزين التنشيطات التي تشغل ذاكرة كبيرة أثناء إعادة الحساب بسرعة، وبالتالي تحقيق توازن رائع بين الذاكرة وإعادة الحساب. على سبيل المثال، بالنسبة لـ GPT-3، يؤدي هذا إلى تقليل بنسبة 70% في الذاكرة المطلوبة للتنشيط على حساب 2.7% فقط من نفقات FLOPs لإعادة حساب التنشيطات. لمزيد من التفاصيل، يرجى الرجوع إلى الورقة البحثية [تقليل إعادة حساب التنشيط في نماذج المحول الكبيرة](https://arxiv.org/pdf/2205.05198.pdf).

و. **النوى المندمجة**: Softmax المندمج، ودقة مختلطة مندمجة طبقة التطبيع، وتجميع التدرجات المندمجة لحساب التدرج الوزني لطبقة خطية. PyTorch JIT قام بتجميع GeLU المندمج والانحياز المندمج + إسقاط + إضافة بقايا.

ز. **دعم مجموعات البيانات المفهرسة**: تنسيق ثنائي فعال لمجموعات البيانات للتدريب واسع النطاق. دعم لـ `mmap`، وملف فهرس `cached`، وتنسيق محمل `lazy`.

ح. **التحقق من شكل نقطة التفتيش وقابلية التشغيل البيني**: أداة مساعدة لإعادة تشكيل نقاط تفتيش Megatron-LM ذات أحجام متوازية متغيرة للنسيج وخط الأنابيب إلى نقاط تفتيش مجزأة 🤗 Transformers المحبوبة نظرًا لدعمها الرائع مع مجموعة أدوات وفيرة مثل 🤗 Accelerate Big Model Inference و Megatron-DeepSpeed Inference، إلخ. يتوفر الدعم أيضًا لتحويل نقاط تفتيش 🤗 Transformers المجزأة إلى نقطة تفتيش Megatron-LM ذات أحجام متوازية متغيرة للنسيج وخط الأنابيب للتدريب واسع النطاق.

## المتطلبات الأساسية

ستحتاج إلى تثبيت أحدث إصدارات PyTorch وcuda وnccl و[APEX](https://github.com/NVIDIA/apex#quick-start) من NVIDIA ومكتبة nltk. راجع [الوثائق](https://github.com/NVIDIA/Megatron-LM#setup) لمزيد من التفاصيل.

طريقة أخرى لإعداد البيئة هي سحب حاوية PyTorch من NVIDIA تأتي مع جميع التثبيتات المطلوبة من NGC.

فيما يلي طريقة خطوة بخطوة لإعداد بيئة conda:

1. قم بإنشاء بيئة افتراضية:

```
conda create --name ml
```

2. افترض أن الآلة بها CUDA 11.3 مثبت، قم بتثبيت إصدار GPU المقابل لـ PyTorch:

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

3. تثبيت Nvidia APEX:

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

4. تثبيت Megatron-LM:

```
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.5.0
pip install --no-use-pep517 -e .
```
## تسريع Megatron-LM Plugin

تُدعم الميزات المهمة مباشرة عبر أمر `accelerate config`. وفيما يلي مثال على الأسئلة المقابلة لاستخدام ميزات Megatron-LM:

```bash
:~$ accelerate config --config_file "megatron_gpt_config.yaml"
في أي بيئة حوسبة تعمل؟ ([0] هذه الآلة، [1] AWS (Amazon SageMaker)): 0
ما نوع الآلة التي تستخدمها؟ ([0] لا يوجد تدريب موزع، [1] متعدد وحدات المعالجة المركزية، [2] متعدد وحدات معالجة الرسوميات، [3] وحدة معالجة فائقة): 2
كم عدد الآلات المختلفة التي ستستخدمها (استخدم أكثر من 1 للتدريب متعدد العقد)؟ [1]:
هل تريد استخدام DeepSpeed؟ [نعم/لا]:
هل تريد استخدام FullyShardedDataParallel؟ [نعم/لا]:
هل تريد استخدام Megatron-LM؟ [نعم/لا]: نعم
ما هي درجة/حجم التوازي في المعالجة التنسورية؟ [1]:2
هل تريد تمكين التسلسل المتوازي؟ [نعم/لا]:
ما هي درجة/حجم التوازي في الأنابيب؟ [1]:2
ما هو عدد الدفعات الصغرى؟ [1]:2
هل تريد تمكين إعادة حساب التنشيط الانتقائي؟ [نعم/لا]:
هل تريد استخدام المحسن الموزع الذي يقسم حالة المحسن والتدرجات عبر رتب التوازي في البيانات؟ [نعم/لا]:
ما هي قيمة تقليم التدرج بناءً على المعيار L2 العالمي (0 لإيقاف التشغيل)؟ [1.0]:
كم عدد وحدات معالجة الرسوميات (GPU) التي يجب استخدامها للتدريب الموزع؟ [1]:4
هل تريد استخدام الدقة العائمة FP16 أو BF16 (الدقة المختلطة)؟ [لا/FP16/BF16]: BF16
```

يُظهر التكوين الناتج أدناه:

```
~$ cat megatron_gpt_config.yaml
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MEGATRON_LM
downcast_bf16: 'no'
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config:
megatron_lm_gradient_clipping: 1.0
megatron_lm_num_micro_batches: 2
megatron_lm_pp_degree: 2
megatronMultiplier_lm_recompute_activations: true
megatron_lm_sequence_parallelism: true
megatron_lm_tp_degree: 2
megatron_lm_use_distributed_optimizer: true
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
use_cpu: false
```

سنأخذ مثالًا على التدريب المسبق لـ GPT. وفيما يلي التغييرات الدنيا المطلوبة في `run_clm_no_trainer.py` الرسمي لاستخدام Megatron-LM:

1. نظرًا لأن Megatron-LM يستخدم تنفيذه الخاص من المحسن، يجب استخدام الجدولة المتوافقة معه. وبالتالي، فإن الدعم متاح فقط لجدولة Megatron-LM. يجب على المستخدم إنشاء `accelerate.utils.MegatronLMDummyScheduler`. وفيما يلي مثال على ذلك:

```python
from accelerate.utils import MegatronLMDummyScheduler

if accelerator.distributed_type == DistributedType.MEGATRON_LM:
lr_scheduler = MegatronLMDummyScheduler(
optimizer=optimizer,
total_num_steps=args.max_train_steps,
warmup_num_steps=args.num_warmup_steps,
)
else:
lr_scheduler = get_scheduler(
name=args.lr_scheduler_type,
optimizer=optimizer,
num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
)
```

2. يتطلب الحصول على تفاصيل حجم الدفعة الإجمالية الآن معرفة أحجام التوازي في المعالجة التنسورية والأنابيب. وفيما يلي مثال على كيفية الحصول على حجم الدفعة الإجمالية الفعالة:

```python
if accelerator.distributed_type == DistributedType.MEGATRON_LM:
total_batch_size = accelerator.state.megatron_lm_plugin.global_batch_size
else:
total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
```

3. عند استخدام Megatron-LM، يتم بالفعل حساب متوسط الخسائر عبر مجموعة التوازي في البيانات:

```python
if accelerator.distributed_type == DistributedType.MEGATRON_LM:
losses.append(loss)
else:
losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

if accelerator.distributed_type == DistributedType.MEGATRON_LM:
losses = torch.tensor(losses)
else:
losses = torch.cat(losses)
```

4. بالنسبة إلى Megatron-LM، يجب علينا حفظ النموذج باستخدام `accelerator.save_state`:

```python
if accelerator.distributed_type == DistributedType.MEGATRON_LM:
accelerator.save_state(args.output_dir)
else:
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
)
```

هذا كل شيء! نحن مستعدون الآن للانطلاق 🚀. يمكنك العثور على مثال للنص البرمجي في مجلد الأمثلة على المسار `accelerate/examples/by_feature/megatron_lm_gpt_pretraining.py`.

دعونا ننفذ ذلك لنموذج `gpt-large` المعماري باستخدام 4 وحدات معالجة رسومية A100-80GB.

```bash
accelerate launch --config_file megatron_gpt_config.yaml \
examples/by_feature/megatron_lm_gpt_pretraining.py \
--config_name "gpt2-large" \
--tokenizer_name "gpt2-large" \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--block_size 1024 \
--learning_rate 5e-5 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 24 \
--num_train_epochs 5 \
--with_tracking \
--report_to "wandb" \
--output_dir "awesome_model"
```

فيما يلي بعض المقتطفات المهمة من سجلات الإخراج:

```bash
Loading extension module fused_dense_cuda...
>>> done with compiling and loading fused kernels. Compilation time: 3.569 seconds
> padded vocab (size: 50257) with 175 dummy tokens (new size: 50432)
Building gpt model in the pre-training mode.
The Megatron LM model weights are initialized at random in `accelerator.prepare`. Please use `accelerator.load_checkpoint` to load a pre-trained checkpoint matching the distributed setup.
Preparing dataloader
Preparing dataloader
Preparing model
> number of parameters on (tensor, pipeline) model parallel rank (1, 0): 210753280
> number of parameters on (tensor, pipeline) model parallel rank (1, 1): 209445120
> number of parameters on (tensor, pipeline) model parallel rank (0, 0): 210753280
> number of parameters on (tensor, pipeline) model parallel rank (0, 1): 209445120
Preparing optimizer
Preparing scheduler
> learning rate decay style: linear
10/10/2022 22:57:22 - INFO - __main__ - ***** Running training *****
10/10/2022 22:57:22 - INFO - __main__ -   Num examples = 2318
10/10/2022 22:57:22 - INFO - __main__ -   Num Epochs = 5
10/10/2022 22:57:22 - INFO - __main__ -   Instantaneous batch size per device = 24
10/10/2022 22:57:22 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 48
10/10/2022 22:57:22 - INFO - __main__ -   Gradient Accumulation steps = 1
10/10/2022 22:57:22 - INFO - __main__ -   Total optimization steps = 245
20%|████████████▍                                                 | 49/245 [01:04<04:09,  1.27s/it]
10/10/2022 22:58:29 - INFO - __main__ - epoch 0: perplexity: 1222.1594275215962 eval_loss: 7.10837459564209
40%|████████████████████████▊                                     | 98/245 [02:10<03:07,  1.28s/it]
10/10/2022 22:59:35 - INFO - __main__ - epoch 1: perplexity: 894.5236583794557 eval_loss: 6.796291351318359
60%|████████████████████████████████████▌                        | 147/245 [03:16<02:05,  1.28s/it]
10/10/2022 23:00:40 - INFO - __main__ - epoch 2: perplexity: 702.8458788508042 eval_loss: 6.555137634277344
80%|████████████████████████████████████████████████▊            | 196/245 [04:22<01:02,  1.28s/it]
10/10/2022 23:01:46 - INFO - __main__ - epoch 3: perplexity: 600.3220028695281 eval_loss: 6.39746618270874
100%|█████████████████████████████████████████████████████████████| 245/245 [05:27<00:00,  1.28s/it]
```

هناك عدد كبير من الخيارات/الميزات الأخرى التي يمكن تعيينها باستخدام `accelerate.utils.MegatronLMPlugin`.

## ميزات متقدمة للاستفادة من كتابة خطوة التدريب المخصصة ومجموعات بيانات Megatron-LM المفهرسة

لاستغلال المزيد من الميزات، يرجى الاطلاع على التفاصيل أدناه.

1. فيما يلي مثال على التغييرات المطلوبة لتخصيص خطوة التدريب أثناء استخدام Megatron-LM. ستقوم بتنفيذ `accelerate.utils.AbstractTrainStep` أو وراثة أحد أطفالهما `accelerate.utils.GPTTrainStep`، `accelerate.utils.BertTrainStep` أو `accelerate.utils.T5TrainStep`.

```python
from accelerate.utils import MegatronLMDummyScheduler, GPTTrainStep, avg_losses_across_data_parallel_group

# Custom loss function for the Megatron model
class GPTTrainStepWithCustomLoss(GPTTrainStep):
def __init__(self, megatron_args, **kwargs):
super().__init__(megatron_args)
self.kwargs = kwargs

def get_loss_func(self):
def loss_func(inputs, loss_mask, output_tensor):
batch_size, seq_length = output_tensor.shape
losses = output_tensor.float()
loss_mask = loss_mask.view(-1).float()
loss = losses.view(-1) * loss_mask

# Resize and average loss per sample
loss_per_sample = loss.view(batch_size, seq_length).sum(axis=1)
loss_mask_per_sample = loss_mask.view(batch_size, seq_length).sum(axis=1)
loss_per_sample = loss_per_sample / loss_mask_per_sample

# Calculate and scale weighting
weights = torch.stack([(inputs == kt).float() for kt in self.kwargs["keytoken_ids"]]).sum(axis=[0, 2])
weights = 1.0 + self.kwargs["alpha"] * weights
# Calculate weighted average
weighted_loss = (loss_per_sample * weights).mean()

# Reduce loss across data parallel groups
averaged_loss = avg_losses_across_data_parallel_group([weighted_loss])

return weighted_loss, {"lm loss": averaged_loss[0]}

return loss_func

def get_forward_step_func(self):
def forward_step(data_iterator, model):
"""Forward step."""
# Get the batch.
tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(data_iterator)
output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

return output_tensor, partial(self.loss_func, tokens, loss_mask)

return forward_step


def main():
# Custom loss function for the Megatron model
keytoken_ids = []
keywords = ["plt", "pd", "sk", "fit", "predict", " plt", " pd", " sk", " fit", " predict"]
for keyword in keywords:
ids = tokenizer([keyword]).input_ids[0]
if len(ids) == 1:
keytoken_ids.append(ids[0])
accelerator.print(f"Keytoken ids: {keytoken_ids}")
accelerator.state.megatron_lm_plugin.custom_train_step_class = GPTTrainStepWithCustomLoss
accelerator.state.megatron_lm_plugin.custom_train_step_kwargs = {
"keytoken_ids": keytoken_ids,
"alpha": 0.25,
}
```

2. لاستخدام مجموعات بيانات Megatron-LM، هناك بعض التغييرات الإضافية المطلوبة. متاح برامج تحميل البيانات لهذه المجموعات من البيانات فقط على الرتبة 0 من كل مجموعة توازي تنسورية. وبالتالي، هناك رتب حيث لن يكون برنامج تحميل البيانات متاحًا، ويتطلب ذلك إجراء تعديلات على حلقة التدريب. إن القدرة على القيام بذلك تُظهر مدى مرونة وقابلية امتداد مكتبة 🤗 Accelerate. وفيما يلي التغييرات المطلوبة:

   - بالنسبة إلى مجموعات بيانات Megatron-LM المفهرسة، يجب علينا استخدام `MegatronLMDummyDataLoader` وتمرير وسائط مجموعة البيانات المطلوبة إليه مثل `data_path`، `seq_length`، وما إلى ذلك. انظر [هنا](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/arguments.py#L804) للحصول على قائمة بالوسائط المتاحة.

   ```python
   from accelerate.utils import MegatronLMDummyDataLoader

   megatron_dataloader_config = {
   "data_path": args.data_path,
   "splits_string": args.splits_string,
   "seq_length": args.block_size,
   "micro_batch_size": args.per_device_train_batch_size,
   }
   megatron_dataloader = MegatronLMDummyDataLoader(**megatron_dataloader_config)
   accelerator.state.megatron_lm_plugin.megatron_dataset_flag = True
   ```

   - يتم تكرار `megatron_dataloader` 3 مرات للحصول على برامج تحميل بيانات التدريب والتحقق والاختبار وفقًا لنسب `args.splits_string`

   ```python
   model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, _ = accelerator.prepare(
   model, optimizer, lr
## أداة لتحويل نقطة التفتيش والتوافق التشغيلي 
1. تتوفر النصوص البرمجية لهذه الأداة في مكتبة 🤗 Transformers تحت النماذج المقابلة.
وهي متوفرة حاليًا لنموذج GPT [checkpoint_reshaping_and_interoperability.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/megatron_gpt2/checkpoint_reshaping_and_interoperability.py) 
2. فيما يلي مثال على تحويل نقطة تفتيش من Megatron-LM إلى نقطة تفتيش مجزأة عالمية في 🤗 Transformers.
```bash
python checkpoint_reshaping_and_interoperability.py \
--convert_checkpoint_from_megatron_to_transformers \
--load_path "gpt/iter_0005000" \
--save_path "gpt/trfs_checkpoint" \
--max_shard_size "200MB" \
--tokenizer_name "gpt2" \
--print-checkpoint-structure
``` 
3. تحويل نقطة التفتيش من Transformers إلى Megatron مع `tp_size=2`، `pp_size=2` و `dp_size=2`.
```bash
python checkpoint_utils/megatgron_gpt2/checkpoint_reshaping_and_interoperability.py \
--load_path "gpt/trfs_checkpoint" \
--save_path "gpt/megatron_lm_checkpoint" \
--target_tensor_model_parallel_size 2 \
--target_pipeline_model_parallel_size 2 \
--target_data_parallel_size 2 \
--target_params_dtype "bf16" \
--make_vocab_size_divisible_by 128 \
--use_distributed_optimizer \
--print-checkpoint-structure
```

## دعم نماذج Megatron-LM GPT لإرجاع logits ودالة `megatron_generate` لتوليد النص 
1. يتطلب إرجاع logits تعيين `require_logits=True` في MegatronLMPlugin كما هو موضح أدناه.
ستكون هذه القيم متاحة في المرحلة الأخيرة من الأنابيب.
```python
megatron_lm_plugin = MegatronLMPlugin(return_logits=True)
``` 
2. طريقة `megatron_generate` لنموذج Megatron-LM GPT: ستستخدم هذه الطريقة التوازي التوتري وأنابيب التوازي لإكمال الأجيال لمجموعة من المدخلات عند استخدام الجشع مع/بدون عينات top_k/top_p ولمدخلات المطال الفردية عند استخدام فك تشفير البحث الشعاعي.
يتم دعم مجموعة فرعية فقط من ميزات generate transformers. سيساعد هذا في استخدام النماذج الكبيرة عبر التوازي التوتري وأنابيب التوازي للتوليد (يتم بالفعل التخزين المؤقت للقيمة الرئيسية واستخدام النواة المندمجة بشكل افتراضي).
يتطلب هذا أن يكون حجم البيانات الموازية 1، وأن يتم تعطيل التوازي التسلسلي ونقطة تفتيش التنشيط.
كما يتطلب أيضًا تحديد مسار ملف المفردات وملف الاندماج في المحلل اللغوي.
يوضح المثال التالي كيفية تكوين واستخدام طريقة `megatron_generate` لنموذج Megatron-LM GPT.
```python
# تحديد ملف المفردات وملف الاندماج في المحلل اللغوي
vocab_file = os.path.join(args.resume_from_checkpoint, "vocab.json")
merge_file = os.path.join(args.resume_from_checkpoint, "merges.txt")
other_megatron_args = {"vocab_file": vocab_file, "merge_file": merge_file}
megatron_lm_plugin = MegatronLMPlugin(other_megatron_args=other_megatron_args)

# الاستدلال باستخدام وظيفة `megatron_generate`
tokenizer.pad_token = tokenizer.eos_token
max_new_tokens = 64
batch_texts = [
"Are you human?",
"The purpose of life is",
"The arsenal was constructed at the request of",
"How are you doing these days?",
]
batch_encodings = tokenizer(batch_texts, return_tensors="pt", padding=True)

# عينة top-p
generated_tokens = model.megatron_generate(
batch_encodings["input_ids"],
batch_encodings["attention_mask"],
max_new_tokens=max_new_tokens,
top_p=0.8,
top_p_decay=0.5,
temperature=0.9,
)
decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy())
accelerator.print(decoded_preds)

# عينة top-k
generated_tokens = model.megatron_generate(
batch_encodings["input_ids"],
batch_encodings["attention_mask"],
max_new_tokens=max_new_tokens,
top_k=50,
temperature=0.9,
)
decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy())
accelerator.print(decoded_preds)

# إضافة رمز `bos` في البداية
generated_tokens = model.megatron_generate(
batch_encodings["input_ids"], batch_encodings["attention_mask"], max_new_tokens=max_new_tokens, add_BOS=True
)
decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy())
accelerator.print(decoded_preds)

# بحث شعاعي => يأخذ موجه واحد فقط
batch_texts = ["The purpose of life is"]
batch_encodings = tokenizer(batch_texts, return_tensors="pt", padding=True)
generated_tokens = model.megatron_generate(
batch_encodings["input_ids"],
batch_encodings["attention_mask"],
max_new_tokens=max_new_tokens,
num_beams=20,
length_penalty=1.5,
)
decoded_preds = tokenizer.batch_decode(generated_tokens.cpu().numpy())
accelerator.print(decoded_preds)
``` 
3. يتوفر مثال شامل على استخدام طريقة `megatron_generate` لنموذج Megatron-LM GPT في
[megatron_gpt2_generation.py](https://github.com/pacman100/accelerate-megatron-test/blob/main/src/inference/megatron_gpt2_generation.py) مع
ملف تكوين [megatron_lm_gpt_generate_config.yaml](https://github.com/pacman100/accelerate-megatron-test/blob/main/src/Configs/megatron_lm_gpt_generate_config.yaml).
يتوفر نص Bash مع أمر الإطلاق في [megatron_lm_gpt_generate.sh](https://github.com/pacman100/accelerate-megatron-test/blob/main/megatron_lm_gpt_generate.sh).
تتوفر سجلات إخراج النص في [megatron_lm_gpt_generate.log](https://github.com/pacman100/accelerate-megatron-test/blob/main/output_logs/megatron_lm_gpt_generate.log).

## دعم تضمين الموضع ROPE وALiBi وMulti-Query Attention 
1. بالنسبة لاهتمام ROPE/ALiBi، قم بتمرير `position_embedding_type` مع `("absolute" | "rotary" | "alibi")` إلى `MegatronLMPlugin` كما هو موضح أدناه.
```python
other_megatron_args = {"position_embedding_type": "alibi"}
megatron_lm_plugin = MegatronLMPlugin(other_megatron_args=other_megatron_args)
``` 
2. بالنسبة لاهتمام Multi-Query، قم بتمرير `attention_head_type` مع `("multihead" | "multiquery")` إلى `MegatronLMPlugin` كما هو موضح أدناه.
```python
other_megatron_args = {"attention_head_type": "multiquery"}
megatron_lm_plugin = MegatronLMPlugin(other_megatron_args=other_megatron_args)
```

## التحذيرات 
1. يدعم نماذج Transformers GPT2 وMegatron-BERT وT5.
يغطي هذا فئات النماذج فك التشفير فقط والترميز فقط والترميز وفك التشفير. 
2. يتم إرجاع الخسارة فقط من تمرير النموذج إلى الأمام
نظرًا لوجود تفاعل معقد بين الأنابيب والتوازي التوتري والتوازي للبيانات خلف الكواليس.
تُرجع مكالمة `model(**batch_data)` الخسائر التي تم حساب متوسطها عبر مراتب التوازي للبيانات.
هذا جيد لمعظم الحالات التي يتم فيها تشغيل وظائف ما قبل التدريب باستخدام ميزات Megatron-LM
يمكنك بسهولة حساب "perplexity" باستخدام الخسارة.
بالنسبة لنموذج GPT، يتم دعم إرجاع logits بالإضافة إلى الخسائر.
لا يتم جمع هذه logits عبر مراتب التوازي للبيانات. استخدم `accelerator.utils.gather_across_data_parallel_groups`
لجمع logits عبر مراتب التوازي للبيانات. يمكن استخدام هذه القيم الصحيحة مع العلامات لحساب مقاييس الأداء المختلفة. 
3. العملية الرئيسية هي المرتبة الأخيرة حيث تتوفر الخسائر/logits في المرحلة الأخيرة من الأنابيب.
`accelerator.is_main_process` و `accelerator.is_local_main_process` إرجاع `True` للمرتبة الأخيرة عند استخدام
دمج Megatron-LM. 
4. في مكالمة `accelerator.prepare`، يتم إنشاء نموذج Megatron-LM المقابل لنموذج Transformers معين
مع أوزان عشوائية. يرجى استخدام `accelerator.load_state` لتحميل نقطة تفتيش Megatron-LM مع أقسام TP وPP وDP المطابقة. 
5. حاليًا، يتوفر دعم تحويل نقطة التفتيش والتوافق التشغيلي لنموذج GPT فقط.
سيتم توسيعه قريبًا ليشمل BERT وT5. 
6. يجب أن تكون `gradient_accumulation_steps` 1. عند استخدام Megatron-LM، فإن الدفعات الصغيرة في إعداد التوازي الأنبوبي
مرادف لتراكم التدرجات. 
7. عند استخدام Megatron-LM، استخدم `accelerator.save_state` و `accelerator.load_state` لحفظ وتحميل نقاط التفتيش. 
8. فيما يلي خريطة معماريات نموذج Megatron-LM إلى معماريات نموذج 🤗 transformers المكافئة.
يتم دعم نماذج 🤗 transformers هذه فقط. 
أ. Megatron-LM [BertModel](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/bert_model.py) :
🤗 نماذج المحولات مع `megatron-bert` في نوع نموذج التكوين، على سبيل المثال،
[MegatronBERT](https://huggingface.co/docs/transformers/model_doc/megatron-bert) 
ب. Megatron-LM [GPTModel](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py) :
🤗 نماذج المحولات مع `gpt2` في نوع نموذج التكوين، على سبيل المثال،
[OpenAI GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2) 
ج. Megatron-LM [T5Model](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/t5_model.py) :
🤗 نماذج المحولات مع `t5` في نوع التكوين، على سبيل المثال،
[T5](https://huggingface.co/docs/transformers/model_doc/t5) و
[MT5](https://huggingface.co/docs/transformers/model_doc/mt5)