# UNetMotionModel

تم تقديم نموذج [UNet](https://huggingface.co/papers/1505.04597) في الأصل من قبل Ronneberger et al لقطاع الصور الطبية، ولكنه يستخدم أيضًا بشكل شائع في 🤗 Diffusers لأنه ينتج صورًا بنفس حجم الإدخال. وهو أحد المكونات المهمة لنظام الانتشار لأنه يسهل عملية الانتشار الفعلية. هناك عدة متغيرات من نموذج UNet في 🤗 Diffusers، اعتمادًا على عدد الأبعاد وما إذا كان نموذجًا مشروطًا أم لا. هذا هو نموذج UNet ثنائي الأبعاد.

المستخلص من الورقة هو:

*هناك اتفاق واسع على أن التدريب الناجح للشبكات العميقة يتطلب آلاف العينات التدريبية الموسومة. في هذه الورقة، نقدم شبكة واستراتيجية تدريب تعتمدان على الاستخدام المكثف لتعزيز البيانات لاستخدام العينات الموسومة المتاحة بشكل أكثر كفاءة. ويتكون التصميم المعماري من مسار تعاقدي لالتقاط السياق ومسار توسيع متماثل يمكّن الموضع الدقيق. نُظهر أن مثل هذه الشبكة يمكن تدريبها من البداية باستخدام عدد قليل جدًا من الصور وأنها تتفوق على أفضل طريقة سابقة (شبكة انزلاق النافذة التعاقدية) في تحدي ISBI لتجزئة البنى العصبية في المكدسات المجهرية الإلكترونية. باستخدام نفس الشبكة المدربة على صور المجهر الضوئي المنقولة (التباين الطوري وDIC) فزنا بتحدي تتبع الخلايا ISBI 2015 في هذه الفئات بهامش كبير. علاوة على ذلك، الشبكة سريعة. يستغرق تجزئة صورة 512x512 أقل من ثانية على وحدة معالجة الرسومات (GPU) الحديثة. التنفيذ الكامل (القائم على Caffe) والشبكات المدربة متوفرة على http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net.*

## UNetMotionModel

[[autodoc]] UNetMotionModel

## UNet3DConditionOutput

[[autodoc]] models.unets.unet_3d_condition.UNet3DConditionOutput