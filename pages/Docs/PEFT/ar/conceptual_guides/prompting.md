# Soft prompts

يعد تدريب النماذج اللغوية الكبيرة المسبقة التدريب عملية مكثفة من حيث الوقت والحوسبة. مع استمرار نمو حجمها، هناك اهتمام متزايد بطرق التدريب الأكثر كفاءة مثل *التحفيز*. يقوم التحفيز بإعداد نموذج مسبق التدريب مجمد لمهمة تالية محددة من خلال تضمين موجه نصي يصف المهمة أو حتى يوضح مثالًا عليها. باستخدام التحفيز، يمكنك تجنب تدريب نموذج منفصل تمامًا لكل مهمة لاحقة، واستخدام نفس النموذج المسبق التدريب المجمد بدلاً من ذلك. هذا أسهل بكثير لأنه يمكنك استخدام نفس النموذج لعدة مهام مختلفة، ومن الأكثر كفاءة بكثير تدريب مجموعة أصغر من معلمات الموجه وتخزينها بدلاً من تدريب جميع معلمات النموذج.

هناك فئتان من طرق التحفيز:

- الموجهات الصعبة هي موجهات نصية مصممة يدويًا باستخدام رموز دخول منفصلة؛ الجانب السلبي هو أنها تتطلب الكثير من الجهد لإنشاء موجه جيد
- الموجهات اللينة هي موترات قابلة للتعلم يتم ضمها إلى تضمين الإدخال ويمكن تحسينها لمجموعة بيانات؛ الجانب السلبي هو أنها ليست قابلة للقراءة بواسطة البشر لأنك لا تقوم بمطابقة هذه "الرموز الافتراضية" مع تضمينات كلمة حقيقية

يوفر هذا الدليل المفاهيمي نظرة عامة موجزة على أساليب الموجهات اللينة المدرجة في 🤗 PEFT: ضبط الموجه، وضبط البادئة، وضبط P، وضبط الموجه متعدد المهام.

## ضبط الموجه

![صورة](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/prompt-tuning.png)

<small>تدريب مجموعة فرعية أصغر بكثير من معلمات المهام المحددة للموجه وتخزينها <a href="https://hf.co/papers/2104.08691">(مصدر الصورة)</a>.</small>

تم تطوير [ضبط الموجه](https://hf.co/papers/2104.08691) لمهام تصنيف النصوص على نماذج T5، ويتم صياغة جميع المهام اللاحقة كمهمة توليد نص. على سبيل المثال، عادةً ما يقوم تصنيف التسلسل بتعيين تسمية فئة واحدة لتسلسل نصي. من خلال صياغته كمهمة توليد نص، يتم *إنشاء* الرموز التي تشكل تسمية الفئة. تتم إضافة الموجهات إلى الإدخال كسلسلة من الرموز. عادة ما تكون معلمات النموذج ثابتة، مما يعني أن رموز الموجه ثابتة أيضًا بواسطة معلمات النموذج.

الفكرة الرئيسية وراء ضبط الموجه هي أن رموز الموجه لها معلماتها الخاصة التي يتم تحديثها بشكل مستقل. وهذا يعني أنه يمكنك الاحتفاظ بمعلمات النموذج المسبق التدريب مجمدة، وتحديث تدرجات تضمين رموز الموجه فقط. النتائج قابلة للمقارنة مع الطريقة التقليدية لتدريب النموذج بالكامل، ويتم ضبط أداء الموجه مع زيادة حجم النموذج.

الق نظرة على [ضبط الموجه للنمذجة اللغوية السببية](../task_guides/clm-prompt-tuning) للحصول على دليل خطوة بخطوة حول كيفية تدريب نموذج باستخدام ضبط الموجه.

## ضبط البادئة

![صورة](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/prefix-tuning.png)

<small>تحسين معلمات البادئة لكل مهمة <a href="https://hf.co/papers/2101.00190">(مصدر الصورة)</a>.</small>

تم تصميم [ضبط البادئة](https://hf.co/papers/2101.00190) لمهام توليد اللغة الطبيعية (NLG) على نماذج GPT. إنه مشابه جدًا لضبط الموجه؛ يقوم ضبط البادئة أيضًا بإلحاق تسلسل من المتجهات المحددة للمهمة بالإدخال والتي يمكن تدريبها وتحديثها مع الاحتفاظ ببقية معلمات النموذج المسبق التدريب مجمدة.

الفرق الرئيسي هو أن معلمات البادئة يتم إدخالها في **جميع** طبقات النموذج، في حين أن ضبط الموجه يضيف معلمات الموجه فقط إلى تضمينات إدخال النموذج. أيضًا، يتم تحسين معلمات البادئة بواسطة شبكة مستقيمة للأمام (FFN) منفصلة بدلاً من التدريب مباشرة على الموجهات اللينة لأنه يتسبب في عدم الاستقرار ويضر بالأداء. يتم التخلص من FFN بعد تحديث الموجهات اللينة.

ونتيجة لذلك، وجد المؤلفون أن ضبط البادئة يوضح أداءً مماثلاً لتدريب نموذج بشكل كامل، على الرغم من وجود معلمات أقل بـ 1000 مرة، ويؤدي أداءً أفضل حتى في إعدادات البيانات المنخفضة.

الق نظرة على [ضبط البادئة للتوليد الشرطي](../task_guides/seq2seq-prefix-tuning) للحصول على دليل خطوة بخطوة حول كيفية تدريب نموذج باستخدام ضبط البادئة.

## ضبط P

![صورة](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/p-tuning.png)

<small>يمكن إدراج رموز الموجه في أي مكان في تسلسل الإدخال، ويتم تحسينها بواسطة مشفر الموجه <a href="https://hf.co/papers/2103.10385">(مصدر الصورة)</a>.</small>

تم تصميم [ضبط P](https://hf.co/papers/2103.10385) لمهام فهم اللغة الطبيعية (NLU) وجميع نماذج اللغة.

إنه تباين آخر لطريقة الموجه اللين؛ يضيف ضبط P أيضًا مصفوفة تضمين قابلة للتدريب يمكن تحسينها للعثور على موجهات أفضل، ويستخدم مشفر موجه (شبكة ذاكرة طويلة وقصيرة المدى ثنائية الاتجاه أو LSTM) لتحسين معلمات الموجه. على عكس ضبط البادئة:

- يمكن إدراج رموز الموجه في أي مكان في تسلسل الإدخال، وليس مقيدًا بالبداية فقط
- يتم إضافة رموز الموجه فقط إلى الإدخال بدلاً من إضافتها إلى كل طبقة من النموذج
- يمكن أن يؤدي تقديم رموز "المرساة" إلى تحسين الأداء لأنها تشير إلى خصائص مكون في تسلسل الإدخال

تشير النتائج إلى أن ضبط P أكثر كفاءة من صياغة الموجهات يدويًا، ويمكّن النماذج الشبيهة بـ GPT من التنافس مع النماذج الشبيهة بـ BERT في مهام NLU.

الق نظرة على [ضبط P لتصنيف التسلسل](../task_guides/ptuning-seq-classification) للحصول على دليل خطوة بخطوة حول كيفية تدريب نموذج باستخدام ضبط P.

## ضبط الموجه متعدد المهام

![صورة](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/mpt.png)

<small><a href="https://hf.co/papers/2303.02861">يُمكِّن ضبط الموجه متعدد المهام من التعلم المنقول الفعال للمعلمات</a>.</small>

يتعلم [ضبط الموجه متعدد المهام (MPT)](https://hf.co/papers/2303.02861) موجهًا واحدًا من البيانات لعدة أنواع مهام يمكن مشاركتها لمهام مستهدفة مختلفة. تقوم الأساليب الموجودة الأخرى بتعلم موجه ناعم منفصل لكل مهمة يجب استردادها أو تجميعها للتكيف مع المهام المستهدفة. يتكون MPT من مرحلتين:

1. التدريب المصدر - لكل مهمة، يتم تحليل الموجه الناعم الخاص بها إلى متجهات محددة للمهمة. يتم ضرب المتجهات المحددة للمهمة معًا لتشكيل مصفوفة أخرى W، ويتم استخدام المنتج الضربي لهادامارد بين W ومصفوفة موجه مشتركة P لتوليد مصفوفة موجه محددة للمهمة. يتم تقطير الموجهات المحددة للمهمة في مصفوفة موجه واحدة مشتركة عبر جميع المهام. يتم تدريب هذا الموجه باستخدام التدريب متعدد المهام.
2. التكيف الهدف - لتكييف الموجه الفردي لمهمة مستهدفة، يتم تهيئة موجه الهدف والتعبير عنه على أنه حاصل ضرب هادامارد لمصفوفة الموجه المشتركة ومصفوفة موجه منخفضة الترتيب محددة للمهمة.

![صورة](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/mpt-decomposition.png)

<small><a href="https://hf.co/papers/2103.10385">تحليل الموجه</a>.</small>