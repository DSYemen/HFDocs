# التهيئة

[`PeftConfigMixin`] هي فئة التهيئة الأساسية لتخزين تهيئة المحول لـ [`PeftModel`]، و [`PromptLearningConfig`] هي فئة التهيئة الأساسية لطرق المطالبة الناعمة (p-tuning و prefix tuning و prompt tuning). تحتوي هذه الفئات الأساسية على طرق لحفظ وتحميل تهيئات النموذج من Hub، وتحديد طريقة PEFT التي سيتم استخدامها، ونوع المهمة التي سيتم تنفيذها، وتهيئات النموذج مثل عدد الطبقات وعدد رؤوس الاهتمام.

## PeftConfigMixin

[[autodoc]] config.PeftConfigMixin
- الكل

## PeftConfig

[[autodoc]] PeftConfig
- الكل

## PromptLearningConfig

[[autodoc]] PromptLearningConfig
- الكل