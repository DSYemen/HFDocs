# OFT

Orthogonal Finetuning (OFT) هي طريقة تم تطويرها لتكييف نماذج النص إلى الصورة القائمة على النشر. تعمل هذه الطريقة عن طريق إعادة معلمة مصفوفات الأوزان المسبقة التدريب باستخدام مصفوفتها المتعامدة للحفاظ على المعلومات في النموذج المُدرب مسبقًا. ولتقليل عدد المعلمات، يقدم OFT هيكلًا متعامدًا قطريًا بالمصفوفة.

ملخص الورقة البحثية هو:

*تتمتع النماذج الكبيرة للنص إلى الصورة بالقدرة على توليد صور واقعية من موجهات النص. وتتمثل المشكلة المفتوحة المهمة في كيفية توجيه هذه النماذج أو التحكم فيها بفعالية لأداء مهام مختلفة. ولمواجهة هذا التحدي، نقدم طريقة ضبط دقيقة قائمة على مبادئ - الضبط الدقيق المتعامد (OFT)، لتكييف نماذج النشر من النص إلى الصورة مع مهام المصب. على عكس الطرق الحالية، يمكن لـ OFT أن يحافظ بشكل مؤكد على الطاقة الكروية الفائقة التي تميز العلاقة العصبية الزوجية على الكرة الوحيدة الخواص. ووجدنا أن هذه الخاصية مهمة للحفاظ على قدرة النماذج القائمة على النشر من النص إلى الصورة على التوليد الدلالي. ولتحسين استقرار الضبط الدقيق، نقترح أيضًا الضبط الدقيق المتعامد المقيد (COFT) الذي يفرض قيد نصف القطر الإضافي على الكرة الوحيدة الخواص. وعلى وجه التحديد، فإننا ننظر في مهمتين مهمتين للضبط الدقيق للنص إلى الصورة: التوليد القائم على الموضوع، والذي يتمثل الهدف منه في توليد صور محددة للموضوع مع إعطاء بضع صور لموضوع وما يرتبط به من نص، والتوليد القابل للتحكم، والذي يتمثل الهدف منه في تمكين النموذج من استقبال إشارات تحكم إضافية. ونظهر تجريبياً أن إطار OFT الخاص بنا يتفوق على الطرق الحالية في جودة التوليد وسرعة التقارب*.

## OFTConfig

[[autodoc]] tuners.oft.config.OFTConfig

## OFTModel

[[autodoc]] tuners.oft.model.OFTModel