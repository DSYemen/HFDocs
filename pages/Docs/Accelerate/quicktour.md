لمحة سريعة

هناك العديد من الطرق لتشغيل وتشغيل الكود الخاص بك اعتمادًا على بيئة التدريب ([torchrun](https://pytorch.org/docs/stable/elastic/run.html)، [DeepSpeed](https://www.deepspeed.ai/)، إلخ) والأجهزة المتوفرة. يوفر Accelerate واجهة موحدة لتشغيل التدريب على إعدادات موزعة مختلفة، مما يتيح لك التركيز على كود التدريب PyTorch بدلاً من تعقيدات تكييف الكود الخاص بك مع هذه الإعدادات المختلفة. يتيح لك هذا توسيع نطاق كود PyTorch الخاص بك بسهولة للتدريب والاستنتاج على إعدادات موزعة مع أجهزة مثل وحدات معالجة الرسومات (GPUs) ووحدات معالجة الرسومات (TPUs). يوفر Accelerate أيضًا Big Model Inference لجعل تحميل وتشغيل الاستدلال باستخدام النماذج الكبيرة جدًا التي عادةً ما لا تتناسب مع الذاكرة أكثر سهولة في الوصول.

تقدم هذه الجولة السريعة الميزات الرئيسية الثلاثة لـ Accelerate:

- واجهة سطر أوامر موحدة لإطلاق النصوص البرمجية الموزعة
- مكتبة تدريب لتكييف كود التدريب PyTorch لتشغيله على إعدادات موزعة مختلفة
- الاستدلال على النماذج الكبيرة

## واجهة الإطلاق الموحدة

يحدد Accelerate تلقائيًا قيم التكوين المناسبة لأي إطار تدريب موزع معين (DeepSpeed، FSDP، إلخ) من خلال ملف تكوين موحد تم إنشاؤه من الأمر [`accelerate config`](package_reference/cli#accelerate-config). يمكنك أيضًا تمرير قيم التكوين بشكل صريح إلى سطر الأوامر، وهو ما قد يكون مفيدًا في حالات معينة إذا كنت تستخدم SLURM، على سبيل المثال.

ولكن في معظم الحالات، يجب عليك دائمًا تشغيل الأمر [`accelerate config`](package_reference/cli#accelerate-config) أولاً لمساعدة Accelerate على معرفة إعداد التدريب الخاص بك.

```bash
accelerate config
```

ينشئ الأمر [`accelerate config`](package_reference/cli#accelerate-config) ملف default_config.yaml ويحفظه في مجلد ذاكرة التخزين المؤقت لـ Accelerate. يقوم هذا الملف بتخزين التكوين لبيئة التدريب الخاصة بك، مما يساعد Accelerate على إطلاق نص التدريب البرمجي الخاص بك بشكل صحيح بناءً على جهازك.

بعد تكوين بيئتك، يمكنك اختبار إعدادك باستخدام الأمر [`accelerate test`](package_reference/cli#accelerate-test)، والذي يطلق نص برمجي قصير لاختبار البيئة الموزعة.

```bash
accelerate test
```

> [!TIP]
> أضف `--config_file` إلى الأمر `accelerate test` أو `accelerate launch` لتحديد موقع ملف التكوين إذا تم حفظه في موقع غير افتراضي مثل ذاكرة التخزين المؤقت.

بمجرد إعداد بيئتك، قم بتشغيل نص التدريب البرمجي الخاص بك باستخدام الأمر [`accelerate launch`](package_reference/cli#accelerate-launch)!

```bash
accelerate launch path_to_script.py --args_for_the_script
```

لمعرفة المزيد، راجع [إطلاق التعليمات البرمجية الموزعة](basic_tutorials/launch) للاطلاع على مزيد من المعلومات حول إطلاق النصوص البرمجية الخاصة بك.

## تكييف كود التدريب

الميزة الرئيسية التالية لـ Accelerate هي فئة [`Accelerator`] التي تقوم بتكييف كود PyTorch الخاص بك لتشغيله على إعدادات موزعة مختلفة.

كل ما تحتاج إلى فعله هو إضافة بضع سطور من الكود إلى نص التدريب البرمجي الخاص بك لتمكينه من التشغيل على وحدات معالجة الرسومات (GPU) أو وحدات معالجة الرسومات (TPU) متعددة.

```diff
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ device = accelerator.device
+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

for batch in training_dataloader:
optimizer.zero_grad()
inputs, targets = batch
-     inputs = inputs.to(device)
-     targets = targets.target.to(device)
outputs = model(inputs)
loss = loss_function(outputs, targets)
+     accelerator.backward(loss)
optimizer.step()
scheduler.step()
```

1. استورد وقم بتنفيذ فئة [`Accelerator`] في بداية نص التدريب البرمجي الخاص بك. تقوم فئة [`Accelerator`] بإعداد كل ما هو ضروري للتدريب الموزع، وتكشف تلقائيًا عن بيئة التدريب الخاصة بك (جهاز واحد مع وحدة معالجة الرسومات، أو جهاز به عدة وحدات معالجة الرسومات، أو أجهزة متعددة مع وحدات معالجة الرسومات المتعددة أو وحدة معالجة الرسومات، إلخ) بناءً على كيفية إطلاق الكود.

```python
from accelerate import Accelerator

accelerator = Accelerator()
```

2. قم بإزالة الاستدعاءات مثل `.cuda()` على نموذجك وبيانات الإدخال الخاصة بك. تقوم فئة [`Accelerator`] بوضع هذه الكائنات تلقائيًا على الجهاز المناسب لك.

> [!WARNING]
> هذه الخطوة *اختيارية* ولكن يُنصح بها باعتبارها أفضل ممارسة للسماح لـ Accelerate بالتعامل مع وضع الجهاز. يمكنك أيضًا إلغاء تنشيط وضع الجهاز التلقائي عن طريق تمرير `device_placement=False` عند تهيئة [`Accelerator`]. إذا كنت تريد وضع الكائنات بشكل صريح على جهاز باستخدام `.to(device)`، فتأكد من استخدام `accelerator.device` بدلاً من ذلك. على سبيل المثال، إذا قمت بإنشاء محسن قبل وضع نموذج على `accelerator.device`، فقد يفشل التدريب على وحدة معالجة الرسومات.

> [!WARNING]
> لا يستخدم Accelerate عمليات النقل غير المتصلة بشكل افتراضي لوضع الجهاز التلقائي، مما قد يؤدي إلى عمليات تزامن CUDA غير مرغوب فيها. يمكنك تمكين عمليات النقل غير المتصلة عن طريق تمرير [`~utils.dataclasses.DataLoaderConfiguration`] مع `non_blocking=True` المحدد كـ `dataloader_config` عند تهيئة [`Accelerator`]. كما هو معتاد، لن تعمل عمليات النقل غير المتصلة إلا إذا كان برنامج التغذية أيضًا `pin_memory=True` المحدد. كن حذرًا من أن استخدام عمليات النقل غير المتصلة من وحدة معالجة الرسومات إلى وحدة المعالجة المركزية قد يتسبب في نتائج غير صحيحة إذا أدى ذلك إلى إجراء عمليات وحدة المعالجة المركزية على تنسيقات غير جاهزة.

```py
device = accelerator.device
```

3. قم بتمرير جميع كائنات PyTorch ذات الصلة بالتدريب (المحسن، النموذج، برنامج التغذية، مخطط التعلم) إلى طريقة [`~Accelerator.prepare`] بمجرد إنشائها. تقوم هذه الطريقة بتغليف النموذج في حاوية مُستَضيفة لإعدادك الموزع، وتستخدم إصدار Accelerate من المحسن والمخطط، وتنشئ إصدارًا مجزأً من برنامج التغذية الخاص بك للتوزيع عبر وحدات معالجة الرسومات أو وحدات معالجة الرسومات.

```python
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
model, optimizer, train_dataloader, lr_scheduler
)
```

4. استبدل `loss.backward()` بـ [`~Accelerator.backward`] لاستخدام طريقة `backward()` الصحيحة لإعداد التدريب الخاص بك.

```py
accelerator.backward(loss)
```

اقرأ دليل [الآليات الداخلية لـ Accelerate](concept_guides/internal_mechanism) للتعرف على المزيد من التفاصيل حول كيفية تكييف Accelerate لكودك.

### التقييم الموزع

لإجراء التقييم الموزع، قم بتمرير برنامج التغذية الخاص بالتحقق من الصحة إلى طريقة [`~Accelerator.prepare`]:

```python
validation_dataloader = accelerator.prepare(validation_dataloader)
```

يستقبل كل جهاز في إعدادك الموزع جزءًا فقط من بيانات التقييم، مما يعني أنه يجب عليك تجميع تنبؤاتك مع طريقة [`~Accelerator.gather_for_metrics`]. تتطلب هذه الطريقة أن تكون جميع المصفوفات بنفس الحجم في كل عملية، لذا إذا كانت مصفوفاتك بأحجام مختلفة في كل عملية (على سبيل المثال عند التبطين الديناميكي إلى الطول الأقصى في دفعة)، فيجب عليك استخدام طريقة [`~Accelerator.pad_across_processes`] لتبطين المصفوفة إلى الحجم الأكبر عبر العمليات. لاحظ أن المصفوفات بحاجة إلى أن تكون أحادية البعد وأننا نقوم بدمج المصفوفات على طول البعد الأول.

```python
for inputs, targets in validation_dataloader:
predictions = model(inputs)
# قم بتجميع جميع التنبؤات والأهداف
all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
# مثال على الاستخدام مع *Datasets.Metric*
metric.add_batch(all_predictions, all_targets)
```

بالنسبة للحالات الأكثر تعقيدًا (مثل المصفوفات ثنائية الأبعاد، لا تريد دمج المصفوفات، قاموس من المصفوفات ثلاثية الأبعاد)، يمكنك تمرير `use_gather_object=True` في `gather_for_metrics`. سيتم إرجاع قائمة الكائنات بعد التجميع. لاحظ أن استخدامها مع المصفوفات GPU غير مدعوم جيدًا وغير فعال.

> [!TIP]
> قد تكون البيانات في نهاية مجموعة البيانات مكررة بحيث يمكن تقسيم الدفعة بالتساوي بين جميع العمال. تقوم طريقة [`~Accelerator.gather_for_metrics`] بإزالة البيانات المكررة تلقائيًا لحساب مقياس أكثر دقة.

## الاستدلال على النماذج الكبيرة

يحتوي الاستدلال على النماذج الكبيرة في Accelerate على ميزتين رئيسيتين، [`~accelerate.init_empty_weights`] و [`~accelerate.load_checkpoint_and_dispatch`]، لتحميل نماذج كبيرة للاستدلال والتي عادةً ما لا تتناسب مع الذاكرة.

> [!TIP]
> الق نظرة على دليل [التعامل مع النماذج الكبيرة للاستدلال](concept_guides/big_model_inference) للحصول على فهم أفضل لكيفية عمل الاستدلال على النماذج الكبيرة تحت الغطاء.

### تهيئة الأوزان الفارغة

يقوم سياق [`~accelerate.init_empty_weights`] بإنشاء نماذج من أي حجم عن طريق إنشاء "هيكل نموذج" ونقل ووضع المعلمات في كل مرة يتم إنشاؤها إلى جهاز [**meta**](https://pytorch.org/docs/main/meta.html) في PyTorch. بهذه الطريقة، لا يتم تحميل جميع الأوزان على الفور ولا يتم تحميل سوى جزء صغير من النموذج في الذاكرة في أي وقت.

على سبيل المثال، يتطلب تحميل نموذج فارغ [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) ذاكرة أقل بكثير من تحميل النماذج والأوزان بالكامل على وحدة المعالجة المركزية.

```py
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
with init_empty_weights():
model = AutoModelForCausalLM.from_config(config)
```

### تحميل الأوزان وإرسالها

تقوم الدالة [`~accelerate.load_checkpoint_and_dispatch`] بتحميل نقاط التفتيش الكاملة أو المجزأة في النموذج الفارغ، وتوزع الأوزان تلقائيًا عبر جميع الأجهزة المتوفرة.

يحدد معلمة `device_map` مكان وضع كل طبقة من النموذج، ويحدد `"auto"` وضعها على وحدة معالجة الرسومات أولاً، ثم وحدة المعالجة المركزية، وأخيرًا محرك الأقراص الثابتة كمصفوفات مُدارة الذاكرة إذا لم يكن هناك ذاكرة كافية. استخدم معلمة `no_split_module_classes` للإشارة إلى الوحدات النمطية التي لا يجب تقسيمها عبر الأجهزة (عادةً تلك التي تحتوي على اتصال باقي).

```py
from accelerate import load_checkpoint_and_dispatch

model = load_checkpoint_and_dispatch(
model, checkpoint="mistralai/Mixtral-8x7B-Instruct-v0.1", device_map="auto", no_split_module_classes=['Block']
)
```

## الخطوات التالية

الآن بعد أن تم تقديمك إلى الميزات الرئيسية لـ Accelerate، يمكن أن تشمل خطواتك التالية ما يلي:

- راجع [البرامج التعليمية](basic_tutorials/overview) للحصول على دليل مفصل حول Accelerate. هذا مفيد بشكل خاص إذا كنت جديدًا في التدريب الموزع والمكتبة.
- تعمق في [الأدلة](usage_guides/explore) لمعرفة كيفية استخدام Accelerate لحالات استخدام محددة.
- تعميق فهمك المفاهيمي لكيفية عمل Accelerate داخليًا عن طريق قراءة [أدلة المفاهيم](concept_guides/internal_mechanism).
- ابحث عن الفئات والأوامر في [مرجع API](package_reference/accelerator) لمعرفة المعلمات والخيارات المتاحة.