لم يتم ترجمة الأجزاء المحددة في النص الأصلي بناءً على طلبك.

# تأجيل التنفيذ

عند تشغيل النص البرمجي المعتاد، يتم تنفيذ التعليمات بترتيب تسلسلي. ولكن عند استخدام HuggingFace Accelerate لتشغيل النص البرمجي على عدة وحدات معالجة رسومية (GPUs) في نفس الوقت، تظهر بعض التعقيدات: ففي حين أن كل عملية تنفذ جميع التعليمات بترتيب تسلسلي، قد تكون بعض العمليات أسرع من غيرها.

قد تحتاج إلى انتظار وصول جميع العمليات إلى نقطة معينة قبل تنفيذ تعليمة معينة. على سبيل المثال، لا يجب حفظ النموذج قبل التأكد من انتهاء جميع العمليات من التدريب، ولا تريد الاستمرار في التدريب قبل تحميل جميع أوزان النموذج. للقيام بذلك، ما عليك سوى كتابة السطر التالي في الكود الخاص بك:

```
accelerator.wait_for_everyone()
```

سوف تقوم هذه التعليمة بحظر جميع العمليات التي تصل أولاً حتى تصل جميع العمليات الأخرى إلى تلك النقطة (إذا قمت بتشغيل النص البرمجي على وحدة معالجة رسومية GPU أو وحدة المعالجة المركزية CPU واحدة، فلن يتم تنفيذ أي شيء).

فيما يلي بعض الحالات التي يمكنك فيها استخدام هذه الأداة:

<Tip>
يتم استخدام بعض هذه الحالات مع مدير السياق [`~Accelerator.main_process_first`]، والذي يستخدم [`~Accelerator.wait_for_everyone`] لتشغيل مجموعة معينة من التعليمات البرمجية على العملية الرئيسية أولاً قبل تشغيل العمليات الأخرى.
</Tip>

## تنزيل مجموعة بيانات

عند تنزيل مجموعة بيانات، يجب عليك أولاً تنزيلها على العملية الرئيسية، ثم تحميل مجموعة البيانات المخزنة مؤقتًا.

<Tip>
سيؤدي استخدام الدالة `load_dataset` إلى تنفيذ قفل في الخلفية لمنع حدوث عمليات تنزيل متعددة في نفس الوقت. ولكن إذا كنت تقوم بتنزيل شيء ما لا يستخدم هذه المكتبة، فيجب استخدام هذه الطريقة.
</Tip>

```python
with accelerator.main_process_first():
datasets = load_dataset("glue", "mrpc")
```

في الخلفية، هذا يعادل استدعاء ما يلي:

```python
# أولاً، قم بتنفيذ شيء ما على العملية الرئيسية
if accelerator.is_main_process:
datasets = load_dataset("glue", "mrpc")
else:
accelerator.wait_for_everyone()

# ثم أرسلها إلى بقية العمليات
if not accelerator.is_main_process:
datasets = load_dataset("glue", "mrpc")
else:
acceleramp.wait_for_everyone()
```

## حفظ `state_dict`

عند حفظ `state_dict` للنموذج، نظرًا لأنك عادةً ما تقوم بحفظ ملف واحد على العملية الرئيسية فقط، يجب عليك تحديد ذلك:

```python
if accelerator.is_main_process:
model = accelerator.unwrap_model(model)
torch.save(model.state_dict(), "weights.pth")
```

## تحميل `state_dict`

عند تحميل `state_dict` في نموذج أو محسن أو جدول، يجب عليك الانتظار حتى تقوم جميع العمليات بتحميل الأوزان قبل المتابعة إلى التدريب:

```python
with accelerator.main_process_first():
state = torch.load("weights.pth")
model.load_state_dict(state)
```

## تنفيذ عملية CPU متعددة العمال

يجب تنفيذ عملية `map()` على عدة عمال، مثل عملية التمييز، أولاً على العملية الرئيسية، ثم نشرها على كل عامل.

```python
datasets = load_dataset("glue", "mrpc")

with accelerator.main_process_first():
tokenized_datasets = datasets.map(
tokenize_function,
batched=True,
remove_columns=["idx", "sentence1", "sentence2"],
)
```

## تطبيق فحوصات مثل التوقف المبكر

لتنفيذ فحص يعمل باستخدام علم يتم تعيينه بواسطة عملية معينة، يجب استخدام واجهة برمجة التطبيقات API `set_trigger` و`check_trigger`. ومن الأمثلة المفيدة على ذلك حالات مثل استخدام التوقف المبكر ومراقبة الخسارة (حيث تختلف الخسارة اختلافًا طفيفًا في كل عملية).

قم باستدعاء [`Accelerator.set_trigger`] عندما يتم استيفاء الشرط، و[`Accelerator.check_trigger`] عند التحقق مما إذا كان الشرط قد تم استيفاؤه في أي عملية:

```python
for (x,y) in data_loader:
logits = model(x)
loss = loss_func(logits, y)
# افترض أن `should_do_early_stopping` هي دالة مخصصة تعيد شرطًا
if should_do_early_stopping(loss):
accelerator.set_trigger()

# لاحقاً في النص البرمجي للتدريب عند الحاجة إلى التحقق من نقطة التوقف
if accelerator.check_trigger():
break
```