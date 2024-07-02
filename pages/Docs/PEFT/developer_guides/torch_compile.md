# `torch.compile`

في PEFT، يعمل [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) مع بعض الميزات ولكن ليس مع جميعها. والسبب في أنه لن يعمل دائمًا هو أن PEFT ديناميكي للغاية في أماكن معينة (مثل تحميل والتبديل بين عدة وحدات تكييف، على سبيل المثال)، الأمر الذي قد يسبب مشاكل لـ `torch.compile`. في أماكن أخرى، قد يعمل `torch.compile`، ولكنه لن يكون سريعًا كما هو متوقع بسبب انقطاع الرسم البياني.

إذا لم تشاهد أي خطأ، فهذا لا يعني بالضرورة أن `torch.compile` عمل بشكل صحيح. فقد يعطيك مخرجًا، ولكن المخرج غير صحيح. يصف هذا الدليل ما يعمل مع `torch.compile` وما لا يعمل.

> [!TIP]
> ما لم يُذكر خلاف ذلك، تم استخدام إعدادات `torch.compile` الافتراضية.

## التدريب والاستنتاج مع `torch.compile`

هذه الميزات **تعمل** مع `torch.compile`. تم اختبار كل ما هو مدرج أدناه مع نموذج لغوي سببي:

- التدريب باستخدام `Trainer` من 🤗 محولات
- التدريب باستخدام حلقة PyTorch مخصصة
- الاستنتاج
- التوليد

تم اختبار وحدات التكييف التالية بنجاح:

- AdaLoRA
- BOFT
- IA³
- ضبط معيار الطبقة
- LoHa
- LoRA
- LoRA + DoRA
- OFT
- VeRA

لا تعمل وحدات التكييف التالية بشكل صحيح للتدريب أو الاستنتاج عند استخدام `torch.compile`:

- LoKr
- LoRA التي تستهدف طبقات التضمين

## ميزات PEFT المتقدمة مع `torch.compile`

فيما يلي بعض ميزات PEFT المتقدمة التي **تعمل**. تم اختبارها جميعًا مع LoRA:

- `modules_to_save` (أي `config = LoraConfig(..., modules_to_save=...)`)
- دمج وحدات التكييف (واحدة أو أكثر)
- دمج وحدات تكييف متعددة في وحدة تكييف واحدة (أي استدعاء `model.add_weighted_adapter(...)`)

بشكل عام، يمكننا أن نتوقع أنه إذا كانت الميزة تعمل بشكل صحيح مع LoRA ومدعومة أيضًا بواسطة أنواع وحدات التكييف الأخرى، فيجب أن تعمل أيضًا لنوع وحدة التكييف تلك.

ميزات PEFT المتقدمة التالية **لا تعمل** بالاقتران مع `torch.compile`. تم تشغيل الاختبارات مع LoRA:

- استخدام وحدات تكييف PEFT مع التكميم (bitsandbytes)
- الاستنتاج باستخدام وحدات تكييف متعددة
- التفريغ (أي استدعاء `model.merge_and_unload()`)
- تعطيل وحدات التكييف (أي استخدام `with model.disable_adapter()`)
- دفعات وحدات التكييف المختلطة (أي استدعاء `model(batch, adapter_names=["__base__", "default"، "other"، ...])`)

## حالات الاختبار

تم اختبار جميع حالات الاستخدام المذكورة أعلاه داخل [`peft/tests/test_torch_compile.py`](https://github.com/huggingface/peft/blob/main/tests/test_torch_compile.py). إذا كنت تريد التحقق بمزيد من التفصيل من كيفية اختبارنا لميزة معينة، يرجى الانتقال إلى هذا الملف والتحقق من الاختبار الذي يتوافق مع حالة الاستخدام الخاصة بك.

> [!TIP]
> إذا كانت لديك حالة استخدام أخرى حيث تعرف أن `torch.compile` يعمل أو لا يعمل مع PEFT، فيرجى المساهمة من خلال إعلامنا أو فتح طلب سحب لإضافة حالة الاستخدام هذه إلى حالات الاختبار المغطاة.