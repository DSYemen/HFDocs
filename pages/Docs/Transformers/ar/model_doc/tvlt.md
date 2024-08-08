# TVLT

<Tip warning={true}>

هذا النموذج في وضع الصيانة فقط، ولا نقبل أي طلبات سحب جديدة لتغيير شفرته.
إذا واجهتك أي مشكلات أثناء تشغيل هذا النموذج، يرجى إعادة تثبيت الإصدار الأخير الذي يدعم هذا النموذج: v4.40.2.
يمكنك القيام بذلك عن طريق تشغيل الأمر التالي: `pip install -U transformers==4.40.2`.

</Tip>

## نظرة عامة

اقترح نموذج TVLT في [TVLT: Textless Vision-Language Transformer](https://arxiv.org/abs/2209.14156) بواسطة Zineng Tang و Jaemin Cho و Yixin Nie و Mohit Bansal (ساهم المؤلفون الثلاثة الأوائل بالتساوي). محول الرؤية واللغة النصي (TVLT) هو نموذج يستخدم الإدخالات البصرية والسمعية الخام لتعلم تمثيل الرؤية واللغة، دون استخدام وحدات خاصة بالنص مثل التمييز أو التعرف التلقائي على الكلام (ASR). يمكنه أداء مهام سمعية بصرية ولغوية بصرية مختلفة مثل الاسترجاع والإجابة على الأسئلة، وما إلى ذلك.

الملخص من الورقة هو كما يلي:

*في هذا العمل، نقدم محول الرؤية واللغة النصي (TVLT)، حيث تقوم كتل المحول المتجانسة بأخذ الإدخالات البصرية والسمعية الخام لتعلم تمثيل الرؤية واللغة بتصميم محدد للنمط، ولا تستخدم وحدات خاصة بالنص مثل التمييز أو التعرف التلقائي على الكلام (ASR). يتم تدريب TVLT عن طريق إعادة بناء رقع الأقنعة لإطارات الفيديو المستمرة والمخططات الطيفية الصوتية (الترميز التلقائي المقنع) ونمذجة التباين لمواءمة الفيديو والصوت. يحقق TVLT أداءً قابلاً للمقارنة مع نظيره القائم على النص في مهام متعددة الوسائط المختلفة، مثل الإجابة على الأسئلة المرئية، واسترجاع الصور، واسترجاع الفيديو، وتحليل المشاعر متعددة الوسائط، بسرعة استدلال أسرع 28 مرة وبثلث المعلمات فقط. تشير نتائجنا إلى إمكانية تعلم تمثيلات بصرية لغوية مدمجة وفعالة من الإشارات البصرية والسمعية من المستوى المنخفض دون افتراض وجود نص مسبق.*

<p align="center">

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/tvlt_architecture.png"

alt="drawing" width="600"/>

</p>

<small> تصميم TVLT. مأخوذة من <a href="[https://arxiv.org/abs/2102.03334](https://arxiv.org/abs/2209.14156)">الورقة الأصلية</a>. </small>

يمكن العثور على الشفرة الأصلية [هنا](https://github.com/zinengtang/TVLT). تم المساهمة بهذا النموذج بواسطة [Zineng Tang](https://huggingface.co/ZinengTang).

## نصائح الاستخدام

- TVLT هو نموذج يأخذ كلاً من `pixel_values` و`audio_values` كإدخال. يمكن للمرء استخدام [`TvltProcessor`] لتحضير البيانات للنموذج.
يغلف هذا المعالج معالج صورة (لوضع الصورة/الفيديو) ومستخرج ميزات صوتية (لوضع الصوت) في واحد.

- تم تدريب TVLT باستخدام صور/مقاطع فيديو ومقاطع صوتية بأحجام مختلفة: يقوم المؤلفون بإعادة حجم المحاصيل للصور/مقاطع الفيديو المدخلة إلى 224 وتحديد طول المخطط الطيفي الصوتي بـ 2048. لجعل تجميع مقاطع الفيديو والمقاطع الصوتية ممكنًا، يستخدم المؤلفون `pixel_mask` الذي يشير إلى البكسلات الحقيقية/الحشو و`audio_mask` الذي يشير إلى القيم الصوتية الحقيقية/الحشو.

- تصميم TVLT مشابه جدًا لتصميم محول الرؤية القياسي (ViT) والترميز التلقائي المقنع (MAE) كما هو موضح في [ViTMAE]. الفرق هو أن النموذج يتضمن طبقات تضمين للنمط الصوتي.

- إصدار PyTorch من هذا النموذج متاح فقط في الإصدار 1.10 من PyTorch والإصدارات الأحدث.

## TvltConfig

[[autodoc]] TvltConfig

## TvltProcessor

[[autodoc]] TvltProcessor

- __call__

## TvltImageProcessor

[[autodoc]] TvltImageProcessor

- preprocess

## TvltFeatureExtractor

[[autodoc]] TvltFeatureExtractor

- __call__

## TvltModel

[[autodoc]] TvltModel

- forward

## TvltForPreTraining

[[autodoc]] TvltForPreTraining

- forward

## TvltForAudioVisualClassification

[[autodoc]] TvltForAudioVisualClassification

- forward