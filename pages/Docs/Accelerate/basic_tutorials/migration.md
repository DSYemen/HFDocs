# إضافة Accelerate إلى كودك

تتبع كل إطار عمل للتدريب الموزع طريقته الخاصة في تنفيذ الأمور، والتي قد تتطلب كتابة الكثير من الكود المخصص لتكييفه مع كود التدريب الخاص بـ PyTorch وبيئة التدريب. توفر Accelerate طريقة سهلة للتواصل مع أطر العمل هذه للتدريب الموزع دون الحاجة إلى تعلم التفاصيل المحددة لكل منها. تتعامل Accelerate مع هذه التفاصيل نيابة عنك، حتى تتمكن من التركيز على كود التدريب وتوسيع نطاقه ليتناسب مع أي بيئة تدريب موزع.

في هذا البرنامج التعليمي، ستتعلم كيفية تكييف كود PyTorch الحالي لديك مع Accelerate، وستكون في طريقك نحو التدريب على الأنظمة الموزعة بسهولة! ستبدأ بحلقة تدريب PyTorch أساسية (يفترض أن جميع كائنات التدريب مثل `model` و`optimizer` قد تم إعدادها بالفعل) وتدمج Accelerate فيها تدريجياً.

```python
device = "cuda"
model.to(device)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

## مسرع

[`Accelerator`] هو الفئة الرئيسية لتكييف كودك للعمل مع Accelerate. فهو يعرف الإعداد الموزع الذي تستخدمه، مثل عدد العمليات المختلفة ونوع الأجهزة لديك. توفر هذه الفئة أيضًا الوصول إلى العديد من الطرق اللازمة لتمكين كود PyTorch الخاص بك من العمل في أي بيئة تدريب موزع ولإدارة وتنفيذ العمليات عبر الأجهزة.

هذا هو السبب في أنه يجب دائمًا البدء باستيراد وإنشاء مثيل [`Accelerator`] في النص البرمجي الخاص بك.

```python
from accelerate import Accelerator

accelerator = Accelerator()
```

كما يعرف [`Accelerator`] الجهاز الذي يجب نقل كائنات PyTorch إليه، لذلك يوصى بالسماح لـ Accelerate بالتعامل مع ذلك نيابة عنك.

```diff
- device = "cuda"
+ device = accelerator.device
model.to(device)
```

## إعداد كائنات PyTorch

بعد ذلك، تحتاج إلى إعداد كائنات PyTorch (النموذج، المحسن، المجدول، إلخ) للتدريب الموزع. تتعامل طريقة [`~Accelerator.prepare`] مع وضع نموذجك في الحاوية المناسبة (مثل GPU واحد أو متعدد GPUs) لإعداد التدريب الخاص بك، وتكييف المحسن والمجدول لاستخدام محسن Accelerate [`~optimizer.AcceleratedOptimizer`] و [`~scheduler.AcceleratedScheduler`]. وإنشاء برنامج تحميل بيانات جديد يمكن تقسيمه عبر العمليات.

> [!TIP]
> تقوم Accelerate بإعداد الكائنات التي ترث من فئات PyTorch الخاصة بها مثل `torch.optim.Optimizer`.
يتم إرجاع كائنات PyTorch بنفس الترتيب الذي يتم إرسالها به.

```py
model, optimizer, training_dataloader, scheduler = accelerator.prepare(model, optimizer, training_dataloader, scheduler)
```

## حلقة التدريب

أخيرًا، قم بإزالة استدعاءات `to(device)` إلى المدخلات والأهداف في حلقة التدريب لأن فئات DataLoader الخاصة بـ Accelerate تقوم بوضعها تلقائيًا على الجهاز الصحيح. يجب عليك أيضًا استبدال تمريرة `backward()` المعتادة بطريقة Accelerate [`~Accelerator.backward`] والتي تقوم بضبط مقياس التدرجات نيابة عنك واستخدام طريقة `backward()` المناسبة اعتمادًا على إعدادك الموزع (على سبيل المثال، DeepSpeed أو Megatron).

```diff
-   inputs = inputs.to(device)
-   targets = targets.to(device)
outputs = model(inputs)
loss = loss_function(outputs, targets)
-   loss.backward()
+   accelerator.backward(loss)
```

ضع كل شيء معًا، ويجب أن تبدو حلقة التدريب الجديدة الخاصة بك الآن على النحو التالي!

```python
from accelerate import Accelerator
accelerator = Accelerator()

device = accelerator.device
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
```

## ميزات التدريب

تقدم Accelerate ميزات إضافية - مثل تراكم التدرجات، وقص التدرجات، والتدريب على الدقة المختلطة، والمزيد - يمكنك إضافتها إلى نصك البرمجي لتحسين عملية التدريب. دعونا نستكشف هذه الميزات الثلاث.

### تراكم التدرجات

يمكن تراكم التدرجات من التدريب على أحجام دفعات أكبر عن طريق تراكم التدرجات عبر دفعات متعددة قبل تحديث الأوزان. يمكن أن يكون هذا مفيدًا للالتفاف حول قيود الذاكرة. لتمكين هذه الميزة في Accelerate، حدد معلمة `gradient_accumulation_steps` في فئة [`Accelerator`] وأضف مدير السياق [`~Accelerator.accumulate`] إلى نصك البرمجي.

```diff
+ accelerator = Accelerator(gradient_accumulation_steps=2)
  model, optimizer, training_dataloader = accelerator.prepare(model, optimizer, training_dataloader)

  for input, label in training_dataloader:
+     with accelerator.accumulate(model):
          predictions = model(input)
          loss = loss_function(predictions, label)
          accelerator.backward(loss)
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()
```

### قص التدرجات

قص التدرجات هي تقنية لمنع "التدرجات المتفجرة"، وتقدم Accelerate:

* [`~Accelerator.clip_grad_value_`] لقص التدرجات إلى قيمة دنيا وقصوى
* [`~Accelerator.clip_grad_norm_`] لتطبيع التدرجات إلى قيمة معينة

### الدقة المختلطة

تسرع الدقة المختلطة التدريب باستخدام نوع بيانات دقة أقل مثل fp16 (نصف الدقة) لحساب التدرجات. للحصول على أفضل أداء مع Accelerate، يجب حساب الخسارة داخل نموذجك (كما هو الحال في نماذج Transformers) لأن العمليات خارج النموذج يتم حسابها بدقة كاملة.

قم بتعيين نوع الدقة المختلطة الذي سيتم استخدامه في [`Accelerator`]، ثم استخدم مدير السياق [`~Accelerator.autocast`] للصب التلقائي للقيم إلى نوع البيانات المحدد.

> [!WARNING]
> تمكّن Accelerate الدقة المختلطة التلقائية، لذلك [`~Accelerator.autocast`] مطلوب فقط إذا كانت هناك عمليات دقة مختلطة أخرى بخلاف تلك التي يتم تنفيذها على الخسارة بواسطة [`~Accelerator.backward`] والتي تتعامل بالفعل مع الضبط.

```diff
+ accelerator = Accelerator(mixed_precision="fp16")
+ with accelerator.autocast():
      loss = complex_loss_function(outputs, target):
```

## الحفظ والتحميل

يمكن لـ Accelerate أيضًا حفظ وتحميل *نموذج* بمجرد اكتمال التدريب، أو يمكنك أيضًا حفظ *حالة* النموذج والمحسن والتي قد تكون مفيدة لاستئناف التدريب.

### النموذج

بمجرد اكتمال جميع العمليات، قم بإلغاء تغليف النموذج باستخدام طريقة [`~Accelerator.unwrap_model`] قبل حفظه لأن طريقة [`~Accelerator.prepare`] قامت بتغليف نموذجك في الواجهة المناسبة للتدريب الموزع. إذا لم تقم بإلغاء تغليف النموذج، فإن حفظ قاموس حالة النموذج يحفظ أيضًا أي طبقات إضافية محتملة من النموذج الأكبر، ولن تتمكن من تحميل الأوزان مرة أخرى إلى نموذجك الأساسي.

يجب عليك استخدام طريقة [`~Accelerator.save_model`] لإلغاء تغليف قاموس حالة النموذج وحفظه. يمكن لهذه الطريقة أيضًا حفظ نموذج إلى نقاط مرجعية مجزأة أو إلى تنسيق [safetensors](https://hf.co/docs/safetensors/index).

<hfoptions id="save">
<hfoption id="single checkpoint">

```py
accelerator.wait_for_everyone()
accelerator.save_model(model, save_directory)
```

<Tip>

بالنسبة للنماذج من مكتبة [Transformers](https://hf.co/docs/transformers/index)، احفظ النموذج باستخدام طريقة [`~transformers.PreTrainedModel.save_pretrained`] بحيث يمكن إعادة تحميله باستخدام طريقة [`~transformers.PreTrainedModel.from_pretrained`].

```py
from transformers import AutoModel

unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    "path/to/my_model_directory",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)

model = AutoModel.from_pretrained("path/to/my_model_directory")
```

</Tip>

لتحميل الأوزان، استخدم طريقة [`~Accelerator.unwrap_model`] لإلغاء تغليف النموذج أولاً قبل تحميل الأوزان. جميع معلمات النموذج هي مراجع إلى tensors، لذا فإن هذا يحمل أوزانك داخل `model`.

```py
unwrapped_model = accelerator.unwrap_model(model)
path_to_checkpoint = os.path.join(save_directory,"pytorch_model.bin")
unwrapped_model.load_state_dict(torch.load(path_to_checkpoint))
```

</hfoption>
<hfoption id="sharded checkpoint">

قم بتعيين `safe_serialization=True` لحفظ النموذج بتنسيق safetensor.

```py
accelerator.wait_for_everyone()
accelerator.save_model(model, save_directory, max_shard_size="1GB", safe_serialization=True)
```

لتحميل نقطة مرجعية مجزأة أو نقطة مرجعية بتنسيق safetensor، استخدم طريقة [`~accelerate.load_checkpoint_in_model`]. تسمح هذه الطريقة بتحميل نقطة مرجعية على جهاز محدد.

```py
load_checkpoint_in_model(unwrapped_model, save_directory, device_map={"":device})
```

</hfoption>
</hfoptions>

### الحالة

أثناء التدريب، قد ترغب في حفظ الحالة الحالية للنموذج والمحسن ومولدات الأرقام العشوائية، وربما مجدولو معدلات التعلم بحيث يمكن استعادتها في *نفس النص البرمجي*. يجب عليك إضافة طريقتي [`~Accelerator.save_state`] و [`~Accelerator.load_state`] إلى نصك البرمجي لحفظ وتحميل الحالات.

لمزيد من التخصيص في المكان والطريقة التي يتم بها حفظ الحالات من خلال [`~Accelerator.save_state`]`، استخدم فئة [`~utils.ProjectConfiguration`]. على سبيل المثال، إذا تم تمكين `automatic_checkpoint_naming`، يتم تخزين كل نقطة مرجعية محفوظة في `Accelerator.project_dir/checkpoints/checkpoint_{checkpoint_number}`.

يجب تسجيل أي عناصر حالة أخرى ليتم تخزينها باستخدام طريقة [`~Accelerator.register_for_checkpointing`] بحيث يمكن حفظها وتحميلها. يجب أن يكون لكل كائن يتم تمريره إلى هذه الطريقة لحفظه وتحميله دالة `load_state_dict` و `state_dict`.