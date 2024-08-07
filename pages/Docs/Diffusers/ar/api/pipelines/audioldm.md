لمزيد من المعلومات حول AudioLDM، يرجى الاطلاع على [AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://huggingface.co/papers/2301.12503) بقلم هاوهي ليو وآخرون. استوحى من [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview)، AudioLDM هو _نموذج انتشار خفي (LDM)_ للنص إلى الصوت يتعلم التمثيلات الصوتية المستمرة من [CLAP](https://huggingface.co/docs/transformers/main/model_doc/clap) latents. يأخذ AudioLDM موجه نص كمدخلات ويتوقع الصوت المقابل. يمكنه توليد المؤثرات الصوتية المشروطة بالنص، والكلام البشري والموسيقى.

الملخص من الورقة هو:

> *حظيت أنظمة النص إلى الصوت (TTA) مؤخرًا باهتمام بقدرتها على تركيب الصوت العام بناءً على أوصاف نصية. ومع ذلك، فإن الدراسات السابقة في TTA لها جودة توليد محدودة بتكاليف حسابية عالية. في هذه الدراسة، نقترح AudioLDM، وهو نظام TTA مبني على مساحة خفية لتعلم التمثيلات الصوتية المستمرة من latents النمذجة اللغوية الصوتية التباينية (CLAP). تمكننا نماذج CLAP المسبقة التدريب من تدريب LDMs مع تضمين الصوت أثناء توفير تضمين النص كشرط أثناء المعاينة. من خلال تعلم التمثيلات الخفية لإشارات الصوت وتكويناتها دون نمذجة العلاقة بين الوسائط، يتميز AudioLDM بميزة في كل من جودة التوليد والكفاءة الحسابية. تم تدريبه على AudioCaps باستخدام وحدة معالجة رسومات واحدة، ويحقق AudioLDM أداء TTA من الطراز الأول مقاسًا بكل من المقاييس الموضوعية والذاتية (مثل مسافة Frechet). علاوة على ذلك، يعد AudioLDM أول نظام TTA يمكّن العديد من التلاعبات الصوتية الموجهة بالنص (مثل نقل الأسلوب) بطريقة خالية من التصوير. تنفيذنا وبياناتنا متوفرة في [هذا عنوان URL https](<https://audioldm.github.io/>).*

يمكن العثور على قاعدة الكود الأصلية في [haoheliu/AudioLDM](https://github.com/haoheliu/AudioLDM).

## نصائح

عند بناء موجه، ضع في اعتبارك ما يلي:

- تعمل إدخالات الموجه الوصفية بشكل أفضل؛ يمكنك استخدام الصفات لوصف الصوت (على سبيل المثال، "عالي الجودة" أو "واضح") وجعل سياق الموجه محددًا (على سبيل المثال، "تدفق مائي في الغابة" بدلاً من "التدفق").
- من الأفضل استخدام مصطلحات عامة مثل "cat" أو "dog" بدلاً من أسماء محددة أو كائنات مجردة قد لا يكون النموذج معتادًا عليها.

أثناء الاستنتاج:

- يمكن التحكم في جودة عينة الصوت المتنبأ بها بواسطة وسيط `num_inference_steps`؛ حيث توفر الخطوات الأعلى جودة صوت أعلى على حساب الاستدلال البطيء.
- يمكن التحكم في طول عينة الصوت المتنبأ بها عن طريق تغيير وسيط `audio_length_in_s`.

<Tip>

تأكد من مراجعة دليل الجداول الزمنية [guide](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة الجدول الزمني والجودة، وانظر قسم [إعادة استخدام المكونات عبر الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.

</Tip>

## AudioLDMPipeline

[[autodoc]] AudioLDMPipeline

- all
- __call__

## AudioPipelineOutput

[[autodoc]] pipelines.AudioPipelineOutput