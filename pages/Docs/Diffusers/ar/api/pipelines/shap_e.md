# Shap-E

لمزيد من المعلومات حول النموذج، يرجى الاطلاع على الورقة البحثية [Shap-E: Generating Conditional 3D Implicit Functions](https://huggingface.co/papers/2305.02463) من قبل Alex Nichol و Heewoo Jun من [OpenAI] (https://github.com/openai).

الملخص من الورقة هو:

*نحن نقدم Shap-E، وهو نموذج تنموي شرطي للأصول ثلاثية الأبعاد. على عكس العمل الأخير في النماذج التنموية ثلاثية الأبعاد التي تنتج تمثيل إخراج واحد، يقوم Shap-E بتوليد معلمات الوظائف الضمنية التي يمكن عرضها كشبكات ذات نسيج وكحقول إشعاع عصبية. نقوم بتدريب Shap-E على مرحلتين: أولاً، نقوم بتدريب مشفر يقوم بخرائط محددة بشكل حتمي للأصول ثلاثية الأبعاد إلى معلمات دالة ضمنية؛ ثانيًا، نقوم بتدريب نموذج انتشار شرطي على مخرجات المشفر. عندما يتم تدريبه على مجموعة بيانات كبيرة من البيانات النصية ثلاثية الأبعاد المقترنة، تكون نماذجنا الناتجة قادرة على إنشاء أصول ثلاثية الأبعاد معقدة ومتنوعة في غضون ثوان. عند مقارنته بـ Point-E، وهو نموذج تنموي صريح فوق السحب النقطية، يتقارب Shap-E بشكل أسرع ويحقق جودة عينات مماثلة أو أفضل على الرغم من نمذجة مساحة إخراج متعددة الأبعاد ومتعددة الأبعاد.*

يمكن العثور على كود المصدر الأصلي في [openai/shap-e](https://github.com/openai/shap-e).

<Tip>

راجع قسم [إعادة استخدام المكونات عبر الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.

</Tip>

## `ShapEPipeline`

[[autodoc]] ShapEPipeline

- all

- `__call__`

## `ShapEImg2ImgPipeline`

[[autodoc]] ShapEImg2ImgPipeline

- all

- `__call__`

## `ShapEPipelineOutput`

[[autodoc]] pipelines.shap_e.pipeline_shap_e.ShapEPipelineOutput