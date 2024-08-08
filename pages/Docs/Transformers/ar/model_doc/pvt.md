# Pyramid Vision Transformer (PVT)

## نظرة عامة
تم اقتراح نموذج PVT في
[Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122)
بواسطة Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao. PVT هو نوع من محولات الرؤية التي تستخدم هيكل هرمي لجعله عمود فقري فعال لمهام التنبؤ الكثيف. على وجه التحديد، فإنه يسمح بإدخالات أكثر تفصيلاً (4 × 4 بكسل لكل رقعة) ليتم استخدامها، مع تقليل طول تسلسل المحول في نفس الوقت أثناء تعميقه - مما يقلل من التكلفة الحسابية. بالإضافة إلى ذلك، يتم استخدام طبقة اهتمام التخفيض المكاني (SRA) لزيادة تقليل استهلاك الموارد عند تعلم ميزات عالية الدقة.

الملخص من الورقة هو كما يلي:

*على الرغم من أن شبكات CNN قد حققت نجاحًا كبيرًا في رؤية الكمبيوتر، إلا أن هذا العمل يبحث في شبكة عمود فقري أبسط وخالية من الضربات مفيدة للعديد من مهام التنبؤ الكثيف. على عكس محول الرؤية (ViT) الذي تم اقتراحه مؤخرًا والذي تم تصميمه خصيصًا لتصنيف الصور، نقدم محول الرؤية الهرمي (PVT)، والذي يتغلب على صعوبات نقل المحول إلى مختلف مهام التنبؤ الكثيفة. يتمتع PVT بعدة مزايا مقارنة بأحدث التقنيات. على عكس ViT الذي عادة ما ينتج عنه مخرجات منخفضة الدقة ويتكبد تكاليف حسابية وذاكرية عالية، يمكن تدريب PVT على أقسام كثيفة من الصورة لتحقيق دقة إخراج عالية، وهو أمر مهم للتنبؤ الكثيف، ولكنه يستخدم أيضًا هرمًا متدرجًا للتقليل من حسابات خرائط الميزات الكبيرة. يرث PVT مزايا كل من CNN و Transformer، مما يجعله عمودًا فقريًا موحدًا لمختلف مهام الرؤية بدون ضربات، حيث يمكن استخدامه كبديل مباشر لعمود فقري CNN. نحن نتحقق من صحة PVT من خلال تجارب واسعة النطاق، مما يدل على أنه يعزز أداء العديد من المهام اللاحقة، بما في ذلك اكتشاف الأجسام، وتجزئة المثيل والتجزئة الدلالية. على سبيل المثال، مع عدد مماثل من المعلمات، يحقق PVT+RetinaNet 40.4 AP على مجموعة بيانات COCO، متجاوزًا ResNet50+RetinNet (36.3 AP) بمقدار 4.1 AP مطلق (انظر الشكل 2). نأمل أن يكون PVT بمثابة عمود فقري بديل ومفيد للتنبؤات على مستوى البكسل وتسهيل الأبحاث المستقبلية.*

تمت المساهمة بهذا النموذج بواسطة [Xrenya](https://huggingface.co/Xrenya). يمكن العثور على الكود الأصلي [هنا](https://github.com/whai362/PVT).

- PVTv1 على ImageNet-1K

| **Model variant**|**Size**|**Acc@1**|**Params (M)**|
|--------------------|:-------:|:-------:|:------------:|
| PVT-Tiny|224|75.1|13.2|
| PVT-Small|224|79.8|24.5|
| PVT-Medium|224|81.2|44.2|
| PVT-Large|224|81.7|61.4|

## PvtConfig

[[autodoc]] PvtConfig

## PvtImageProcessor

[[autodoc]] PvtImageProcessor

- preprocess

## PvtForImageClassification

[[autodoc]] PvtForImageClassification

- forward

## PvtModel

[[autodoc]] PvtModel

- forward