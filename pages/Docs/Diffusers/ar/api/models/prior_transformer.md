# PriorTransformer

تم تقديم Prior Transformer لأول مرة في الورقة البحثية [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://huggingface.co/papers/2204.06125) بواسطة Ramesh et al. ويتم استخدامه للتنبؤ بتشغيلات صور CLIP من تشغيلات نصوص CLIP؛ حيث يتم التنبؤ بتشغيلات الصور من خلال عملية انتشار إزالة التشويش.

وفيما يلي الملخص المستخرج من الورقة البحثية:

*أظهرت النماذج التمييزية مثل CLIP قدرتها على تعلم تمثيلات قوية للصور التي تلتقط كلًا من الدلالات والأسلوب. وللاستفادة من هذه التمثيلات في توليد الصور، نقترح نموذجًا مكونًا من مرحلتين: مرحلة سابقة لتوليد تشفير صورة CLIP بناءً على عنوان نصي، ومرحلة فك تشفير لتوليد صورة مشروطة بتشفير الصورة. ونظهر أن التوليد الصريح لتمثيلات الصور يحسن تنوع الصور مع تقليل الحد الأدنى من الخسارة في الواقعية البصرية والتشابه في العنوان. ويمكن لفك تشفيرنا المشروط بتمثيلات الصور أيضًا إنتاج متغيرات من صورة ما مع الحفاظ على دلالتها وأسلوبها، مع تغيير التفاصيل غير الأساسية الغائبة عن تمثيل الصورة. علاوة على ذلك، يمكّن مساحة التضمين المشتركة لـ CLIP من إجراء تعديلات على الصور الموجهة باللغة بطريقة Zero-shot. نستخدم نماذج الانتشار للمرحلة اللاحقة ونجري تجارب باستخدام كل من النماذج التلقائية والنماذج الانتشارية للمرحلة السابقة، ونجد أن هذه الأخيرة أكثر كفاءة من الناحية الحسابية وتنتج عينات ذات جودة أعلى.*

## PriorTransformer

[[autodoc]] PriorTransformer

## PriorTransformerOutput

[[autodoc]] models.transformers.prior_transformer.PriorTransformerOutput