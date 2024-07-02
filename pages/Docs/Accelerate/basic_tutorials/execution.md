# عملية التنفيذ

عند العمل مع أنظمة التدريب الموزعة، من المهم إدارة كيفية ووقت تنفيذ العمليات عبر وحدات معالجة الرسوميات (GPUs). تكتمل بعض العمليات بشكل أسرع من غيرها، وبعض العمليات لا يجب أن تبدأ إذا لم تنتهِ عمليات أخرى بعد. يوفر Accelerate أدوات لتنسيق وقت تنفيذ العمليات لضمان بقاء كل شيء متزامنًا عبر جميع الأجهزة.

سيوضح هذا البرنامج التعليمي كيفية تنفيذ عملية على جهاز واحد فقط، وكيفية تأخير التنفيذ حتى تصل جميع العمليات إلى نقطة معينة.

## التنفيذ على عملية واحدة

تحتاج بعض الشيفرات البرمجية إلى التشغيل مرة واحدة فقط على جهاز معين، مثل طباعة بيان سجل أو عرض شريط تقدم واحد فقط على العملية الرئيسية المحلية.

<hfoptions id="local-execution">
<hfoption id="statements">

يجب استخدام `accelerator.is_local_main_process` للإشارة إلى الشيفرة التي يجب تنفيذها مرة واحدة فقط.

```py
from tqdm.auto import tqdm

progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
```

يمكنك أيضًا تضمين عبارة باستخدام `accelerator.is_local_main_process`.

> [!TIP]
> بالنسبة لتعليمات `print` المستقلة التي لا يتم تضمينها في `accelerator.is_local_main_process`، استبدل `print` بطريقة [`~Accelerator.print`] في Accelerate للطباعة مرة واحدة فقط لكل عملية.

```py
if accelerator.is_local_main_process:
print("Accelerate is the best")
```

</hfoption>
<hfoption id="function">

بالنسبة للدالة التي يجب تنفيذها مرة واحدة فقط، استخدم [`~Accelerator.on_local_main_process`].

```py
@accelerator.on_local_main_process
def do_my_thing():
"Something done once per server"
do_thing_once_per_server()
```

</hfoption>
</hfoptions>

يمكنك أيضًا توجيه Accelerate لتنفيذ الشيفرة مرة واحدة عبر *جميع العمليات* بغض النظر عن عدد الأجهزة. هذا مفيد إذا كنت تقوم بتحميل نموذج نهائي إلى Hub.

<hfoptions id="main-execution">
<hfoption id="statement">

يجب استخدام `accelerator.is_main_process` للإشارة إلى الشيفرة التي يجب تنفيذها مرة واحدة فقط عبر جميع العمليات.

```py
if accelerator.is_main_process:
repo.push_to_hub()
```

</hfoption>
<hfoption id="function">

بالنسبة للدالة التي يجب تنفيذها مرة واحدة عبر جميع العمليات، استخدم [`~Accelerator.on_main_process`].

```py
@accelerator.on_main_process
def do_my_thing():
"Something done once"
do_thing_once()
```

</hfoption>
</hfoptions>

## التنفيذ على عملية محددة

يمكن لـ Accelerate أيضًا مساعدتك في تنفيذ الدوال التي يجب تنفيذها فقط على عملية محددة أو فهرس عملية محلية.

<hfoptions id="specific-execution">
<hfoption id="specific process">

استخدم طريقة [`~Accelerator.on_process`] وحدد فهرس العملية لتنفيذ دالة عليها.

```py
@accelerator.on_process(process_index=0)
def do_my_thing():
"Something done on process index 0"
do_thing_on_index_zero()
```

</hfoption>
<hfoption id="local process">

استخدم طريقة [`~Accelerator.on_local_process`] وحدد فهرس العملية المحلية لتنفيذ دالة عليها.

```py
@accelerator.on_local_process(local_process_idx=0)
def do_my_thing():
"Something done on process index 0 on each server"
do_thing_on_index_zero_on_each_server()
```

</hfoption>
</hfoptions>

## تأخير التنفيذ

عندما تقوم بتشغيل البرنامج النصي الخاص بك على عدة وحدات معالجة الرسوميات (GPUs) في نفس الوقت، قد يتم تنفيذ بعض الشيفرات البرمجية بشكل أسرع من غيرها. قد تحتاج إلى الانتظار حتى تصل جميع العمليات إلى نقطة معينة قبل تنفيذ المجموعة التالية من التعليمات. على سبيل المثال، لا يجب حفظ النموذج قبل التأكد من انتهاء كل عملية من التدريب.

لفعل ذلك، أضف [`~Accelerator.wait_for_everyone`] في الشيفرة البرمجية الخاصة بك. هذا يمنع جميع العمليات التي انتهت أولاً من الاستمرار حتى تصل جميع العمليات المتبقية إلى نفس النقطة (هذا ليس له تأثير إذا كنت تعمل على وحدة معالجة رسوميات (GPU) واحدة أو وحدة المعالجة المركزية (CPU)).

```py
accelerator.wait_for_everyone()
```