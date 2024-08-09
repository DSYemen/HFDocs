# أنواع المهايئات المختلطة

عادة، لا يكون من الممكن خلط أنواع مهايئات مختلفة في PEFT 🤗. يمكنك إنشاء نموذج PEFT بمهايئين LoRA مختلفين (يمكن أن يكون لهما خيارات تكوين مختلفة)، ولكن لا يمكن دمج مهايئ LoRA و LoHa. ومع ذلك، يعمل [`PeftMixedModel`] طالما أن أنواع المهايئ متوافقة. والغرض الرئيسي من السماح بأنواع مهايئات مختلطة هو الجمع بين المهايئات المدربة للاستنتاج. في حين أنه من الممكن تدريب نموذج مهايئ مختلط، إلا أن هذا لم يتم اختباره ولا يوصى به.

لتحميل أنواع مهايئات مختلفة في نموذج PEFT، استخدم [`PeftMixedModel`] بدلاً من [`PeftModel`]:

```py
from peft import PeftMixedModel

base_model = ...  # تحميل نموذج الأساس، على سبيل المثال من transformers
# تحميل المهايئ الأول، والذي سيكون اسمه "default"
peft_model = PeftMixedModel.from_pretrained(base_model, <path_to_adapter1>)
peft_model.load_adapter(<path_to_adapter2>, adapter_name="other")
peft_model.set_adapter(["default", "other"])
```

طريقة [`~PeftMixedModel.set_adapter`] ضرورية لتنشيط كلا المهايئين، وإلا فلن يكون نشطًا سوى المهايئ الأول. يمكنك الاستمرار في إضافة المزيد من المهايئات عن طريق استدعاء [`~PeftModel.add_adapter`] بشكل متكرر.

لا يدعم [`PeftMixedModel`] حفظ وتحميل المهايئات المختلطة. يجب أن تكون المهايئات مدربة بالفعل، ويتطلب تحميل النموذج تشغيل برنامج نصي في كل مرة.

## نصائح:

- لا يمكن دمج جميع أنواع المهايئات. راجع [`peft.tuners.mixed.COMPATIBLE_TUNER_TYPES`](https://github.com/huggingface/peft/blob/1c1c7fdaa6e6abaa53939b865dee1eded82ad032/src/peft/tuners/mixed/model.py#L35) للحصول على قائمة بالأنواع المتوافقة. سيتم رفع خطأ إذا حاولت دمج أنواع مهايئات غير متوافقة.
- من الممكن خلط عدة مهايئات من نفس النوع، والذي يمكن أن يكون مفيدًا لدمج المهايئات ذات التكوينات المختلفة جدًا.
- إذا كنت ترغب في دمج العديد من المهايئات المختلفة، فإن أكثر الطرق فعالية للقيام بذلك هي إضافة أنواع المهايئات نفسها بشكل متتالي. على سبيل المثال، قم بإضافة LoRA1 وLoRA2 وLoHa1 وLoHa2 بهذا الترتيب، بدلاً من LoRA1 وLoHa1 وLoRA2 وLoHa2. في حين أن الترتيب يمكن أن يؤثر على الإخراج، لا يوجد ترتيب "أفضل" متأصل، لذلك من الأفضل اختيار الأسرع.