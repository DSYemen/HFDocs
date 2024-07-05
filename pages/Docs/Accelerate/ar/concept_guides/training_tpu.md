# التدريب على وحدات معالجة tensor باستخدام HuggingFace Accelerate

قد يختلف التدريب على وحدات معالجة tensor اختلافًا طفيفًا عن التدريب على معالج الرسوميات متعدد الوحدات، حتى عند استخدام HuggingFace Accelerate. يهدف هذا الدليل إلى إرشادك إلى المواضع التي يجب أن تكون حذرًا فيها وسبب ذلك، بالإضافة إلى أفضل الممارسات العامة.

## التدريب في دفتر ملاحظات

ينشأ الجانب الأهم الذي يجب مراعاته عند التدريب على وحدات معالجة tensor من وظيفة "notebook_launcher". كما ذُكر في الدرس التعليمي لدفتر الملاحظات، يجب عليك إعادة هيكلة شفرة التدريب إلى دالة يمكن تمريرها إلى وظيفة "notebook_launcher" والحرص على عدم إعلان أي وسائط على وحدة معالجة الرسوميات.

وفي حين أن الجزء الأخير ليس مهمًا بنفس القدر عند استخدام وحدة معالجة tensor، فإن الجزء الحاسم الذي يجب فهمه هو أنه عند تشغيل الشفرة من دفتر الملاحظات، فإنك تفعل ذلك من خلال عملية تسمى "التشعب".

عند التشغيل من سطر الأوامر، فإنك تؤدي "الاستنساخ"، حيث لا تعمل عملية بايثون حاليًا وتقوم باستنساخ عملية جديدة. نظرًا لأن دفتر ملاحظات Jupyter الخاص بك يستخدم بالفعل عملية بايثون، فيجب عليك تشعب عملية جديدة منه لتشغيل الشفرة الخاصة بك.

ويصبح هذا مهمًا فيما يتعلق بإعلان نموذجك. ففي عمليات وحدة معالجة tensor المتفرعة، يوصى بإنشاء مثيل لنموذجك مرة واحدة وتمريره إلى دالة التدريب الخاصة بك. وهذا يختلف عن التدريب على وحدات معالجة الرسوميات، حيث تنشئ "n" من النماذج التي تتم مزامنة تدرجاتها وتنفيذ عملية انتشارها إلى الوراء في لحظات معينة. وبدلاً من ذلك، تتم مشاركة مثيل نموذج واحد بين جميع العقد ويتم تمريره ذهابًا وإيابًا. وهذا مهم بشكل خاص عند التدريب على وحدات معالجة tensor منخفضة الموارد مثل تلك المتوفرة في نوى Kaggle أو Google Colaboratory.

وفيما يلي مثال على دالة تدريب يتم تمريرها إلى وظيفة "notebook_launcher" إذا كنت تتدرب على وحدات المعالجة المركزية أو وحدات معالجة الرسوميات:

```python
def training_function():
    # Initialize accelerator
    accelerator = Accelerator()
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    train_dataloader, eval_dataloader = create_dataloaders(
        train_batch_size=hyperparameters["train_batch_size"], eval_batch_size=hyperparameters["eval_batch_size"]
    )

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=hyperparameters["learning_rate"])

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_epochs = hyperparameters["num_epochs"]
    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()
```

```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```

ستستخدم وظيفة "notebook_launcher" افتراضيًا 8 عمليات إذا تم تكوين HuggingFace Accelerate لوحدة معالجة tensor.

إذا استخدمت هذا المثال وأعلنت عن النموذج داخل حلقة التدريب، فستظهر رسالة خطأ على نظام منخفض الموارد مثل:

```
ProcessExitedException: process 0 terminated with signal SIGSEGV
```

وهذا الخطأ غامض للغاية، ولكن التفسير الأساسي هو أنك نفدت من ذاكرة الوصول العشوائي للنظام. يمكنك تجنب هذا تمامًا عن طريق إعادة تكوين دالة التدريب لقبول وسيط "model" واحد، وإعلانه في خلية خارجية:

```python
# In another Jupyter cell
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
```

```diff
+ def training_function(model):
      # Initialize accelerator
      accelerator = Accelerator()
-     model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
      train_dataloader, eval_dataloader = create_dataloaders(
          train_batch_size=hyperparameters["train_batch_size"], eval_batch_size=hyperparameters["eval_batch_size"]
      )
  ...
```

وأخيرًا، استدع دالة التدريب باستخدام ما يلي:

```diff
  from accelerate import notebook_launcher
- notebook_launcher(training_function)
+ notebook_launcher(training_function, (model,))
```

تنطبق طريقة العمل هذه فقط عند تشغيل مثيل وحدة معالجة tensor من دفتر ملاحظات Jupyter على خادم منخفض الموارد مثل Google Colaboratory أو Kaggle. إذا كنت تستخدم برنامج نصي أو تشغيله على خادم أقوى، فليس من الضروري إعلان النموذج مسبقًا.

## الدقة المختلطة والمتغيرات العالمية

كما ذُكر في الدرس التعليمي للدقة المختلطة، يدعم HuggingFace Accelerate كل من "fp16" و"bf16"، ويمكن استخدامهما مع وحدات معالجة tensor.

ومع ذلك، يُفضل استخدام "bf16" نظرًا لكفاءته العالية في الاستخدام.

هناك مستويان عند استخدام "bf16" وHuggingFace Accelerate مع وحدات معالجة tensor، وهما مستوى القاعدة ومستوى العملية.

فعلى مستوى القاعدة، يتم تمكين هذا الخيار عند تمرير "mixed_precision="bf16" إلى "Accelerator"، كما يلي:

```python
accelerator = Accelerator(mixed_precision="bf16")
```

وبشكل افتراضي، سيتم تحويل "torch.float" و"torch.double" إلى "bfloat16" على وحدات معالجة tensor. ويتمثل التكوين المحدد في تعيين متغير بيئي "XLA_USE_BF16" إلى "1".

وهناك تكوين إضافي يمكنك تنفيذه وهو تعيين متغير البيئة "XLA_DOWNCAST_BF16". فإذا تم تعيينه على "1"، فسيتم تعيين "torch.float" إلى "bfloat16" و"torch.double" إلى "float32".

ويتم تنفيذ هذا التكوين في كائن "Accelerator" عند تمرير "downcast_bf16=True":

```python
accelerator = Accelerator(mixed_precision="bf16", downcast_bf16=True)
```

ومن الأفضل استخدام التحويل إلى دقة أقل بدلاً من استخدام "bf16" في كل مكان عند محاولة حساب المقاييس أو تسجيل القيم أو غير ذلك، حيث لا يمكن استخدام وسائط "bf16" الخام.

## أوقات التدريب على وحدات معالجة tensor

عند تشغيل البرنامج النصي الخاص بك، قد تلاحظ أن التدريب بطيء للغاية في البداية. ويرجع ذلك إلى أن وحدات معالجة tensor تعمل أولاً على تشغيل بضع دفعات من البيانات لمعرفة مقدار الذاكرة التي يجب تخصيصها قبل استخدام هذا التخصيص للذاكرة المكونة بكفاءة فائقة.

إذا لاحظت أن شفرة التقييم الخاصة بحساب مقاييس نموذجك تستغرق وقتًا أطول بسبب استخدام حجم دفعة أكبر، فيوصى بإبقاء حجم الدفعة كما هو في بيانات التدريب إذا كان بطيئًا للغاية. وإلا، فستقوم الذاكرة بإعادة التخصيص لهذا الحجم الجديد للدفعة بعد الدفعات القليلة الأولى.

مجرد تخصيص الذاكرة لا يعني أنها ستُستخدم أو أن حجم الدفعة سيزداد عند العودة إلى برنامج التدريب الخاص بك.