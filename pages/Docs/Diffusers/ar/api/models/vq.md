# VQModel

تم تقديم نموذج VQ-VAE في ورقة [Neural Discrete Representation Learning](https://huggingface.co/papers/1711.00937) بواسطة Aaron van den Oord، وOriol Vinyals، وKoray Kavukcuoglu. ويُستخدم النموذج في 🤗 Diffusers لترميز التمثيلات الكامنة إلى صور. وعلى عكس [`AutoencoderKL`]، يعمل [`VQModel`] في مساحة كمية كامنة.

مقدمة الورقة هي:

*يظل تعلم التمثيلات المفيدة دون إشراف تحديًا رئيسيًا في مجال تعلم الآلة. وفي هذه الورقة، نقترح نموذجًا توليديًا بسيطًا ولكنه قوي يتعلم مثل هذه التمثيلات المنفصلة. ويختلف نموذجنا، وهو Vector Quantised-Variational AutoEncoder (VQ-VAE)، عن VAEs بطريقتين رئيسيتين: تُخرج شبكة الترميز رموزًا منفصلة بدلاً من رموز مستمرة؛ ويتم تعلم الأول بدلاً من الثبات. ولتعلم تمثيل كامن منفصل، ندمج أفكارًا من التكميم المتجهي (VQ). وباستخدام طريقة VQ، يمكن للنموذج تجنب مشكلات "انهيار الاحتمال اللاحق" - حيث يتم تجاهل الكامنات عندما يتم اقترانها بترميز تنبئي قوي - والتي يتم ملاحظتها عادةً في إطار VAE. ومن خلال اقتران هذه التمثيلات باحتمال سابق تنبئي، يمكن للنموذج توليد صور وفيديوهات وكلام عالية الجودة، بالإضافة إلى إجراء تحويل صوتي عالي الجودة وتحويل غير خاضع للإشراف للفونيمات، مما يوفر مزيدًا من الأدلة على فائدة التمثيلات المتعلمة.*

## VQModel

[[autodoc]] VQModel

## VQEncoderOutput

[[autodoc]] models.autoencoders.vq_model.VQEncoderOutput