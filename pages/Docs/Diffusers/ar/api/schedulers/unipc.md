# UniPCMultistepScheduler

`UniPCMultistepScheduler` هو إطار عمل لا يتطلب تدريبًا مصممًا للنمذجة السريعة لنماذج الانتشار. تم تقديمه في [UniPC: إطار عمل موحد للتنبؤ والتصحيح للنمذجة السريعة لنماذج الانتشار](https://huggingface.co/papers/2302.04867) بواسطة Wenliang Zhao وLujia Bai وYongming Rao وJie Zhou وJiwen Lu.

يتكون من مصحح (UniC) ومُتنبئ (UniP) يتشاركان في شكل تحليلي موحد ويدعمان الأوامر التعسفية.

UniPC مصمم ليكون غير متحيز للنموذج، مما يدعم DPMs في مساحة البكسل/الفضاء الكامن في أخذ العينات غير المشروطة/المشروطة. يمكن أيضًا تطبيقه على كل من نماذج التنبؤ بالضوضاء والتنبؤ بالبيانات. يمكن أيضًا تطبيق المصحح UniC بعد أي من الحلول الجاهزة لزيادة دقة الترتيب.

المقتطف من الورقة هو:

*أظهرت نماذج الانتشار الاحتمالية (DPMs) قدرة واعدة للغاية في توليف الصور عالية الدقة. ومع ذلك، فإن أخذ العينات من نموذج DPM المدرب مسبقًا يستغرق وقتًا طويلاً بسبب التقييمات المتعددة لشبكة إزالة التشويش، مما يجعل تسريع نمذجة DPMs أمرًا مهمًا بشكل متزايد. على الرغم من التقدم الأخير في تصميم أجهزة أخذ العينات السريعة، إلا أن الطرق الحالية لا تزال لا تستطيع إنشاء صور مرضية في العديد من التطبيقات التي يفضل فيها عدد أقل من الخطوات (على سبيل المثال، <10). في هذه الورقة، نقوم بتطوير مصحح موحد (UniC) يمكن تطبيقه بعد أي جهاز أخذ عينات DPM موجود لزيادة ترتيب الدقة دون تقييمات نموذج إضافية، ونستمد مُتنبئًا موحدًا (UniP) يدعم أي ترتيب كمنتج ثانوي. من خلال الجمع بين UniP وUniC، نقترح إطار عمل موحد للتنبؤ والتصحيح يسمى UniPC للنمذجة السريعة لـ DPMs، والذي يتمتع بشكل تحليلي موحد لأي ترتيب ويمكن أن يحسن بشكل كبير جودة النمذجة مقارنة بالطرق السابقة، خاصة في عدد قليل جدًا من الخطوات. نقيم طرقنا من خلال تجارب واسعة النطاق تشمل كل من النمذجة غير المشروطة والمشروطة باستخدام DPMs في مساحة البكسل والفضاء الكامن. يمكن أن يحقق نظامنا UniPC 3.87 FID على CIFAR10 (غير مشروط) و7.51 FID على ImageNet 256x256 (مشروط) مع 10 تقييمات وظيفية فقط. الكود متاح في [هذا عنوان URL https](<https://github.com/wl-zhao/UniPC>.)*

## نصائح

يوصى بتعيين `solver_order` إلى 2 للنمذجة الإرشادية، و`solver_order=3` للنمذجة غير المشروطة.

يتم دعم العتبات الديناميكية من [Imagen](https://huggingface.co/papers/2205.11487)، وبالنسبة لنماذج الانتشار في مساحة البكسل، يمكنك تعيين كل من `predict_x0=True` و`thresholding=True` لاستخدام العتبات الديناميكية. طريقة العتبة هذه غير مناسبة لنماذج الانتشار في الفضاء الكامن مثل Stable Diffusion.

## UniPCMultistepScheduler

[[autodoc]] UniPCMultistepScheduler

## SchedulerOutput

[[autodoc]] schedulers.scheduling_utils.SchedulerOutput