# OLMo

## نظرة عامة
اقترح نموذج OLMo في ورقة بحثية بعنوان [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838) من قبل Dirk Groeneveld وIz Beltagy وPete Walsh وآخرين.

OLMo هو سلسلة من النماذج اللغوية المفتوحة المصممة لتمكين علم نماذج اللغة. تم تدريب نماذج OLMo على مجموعة بيانات Dolma. نقوم بإطلاق جميع التعليمات البرمجية ونقاط التفتيش والسجلات (قريباً) والتفاصيل المشاركة في تدريب هذه النماذج.

الملخص من الورقة هو كما يلي:

> "أصبحت نماذج اللغة شائعة الاستخدام في كل من أبحاث معالجة اللغة الطبيعية وفي المنتجات التجارية. ومع تزايد أهميتها التجارية، أصبحت أكثر النماذج قوة مغلقة، ومحصورة خلف واجهات مملوكة، مع عدم الكشف عن التفاصيل المهمة لبياناتها التدريبية وبنيتها وتطويرها. ونظرًا لأهمية هذه التفاصيل في دراسة هذه النماذج علميًا، بما في ذلك التحيزات والمخاطر المحتملة، نعتقد أنه من الضروري أن يحصل مجتمع البحث على نماذج لغة مفتوحة وقوية حقًا. لتحقيق هذه الغاية، يقدم هذا التقرير الفني التفاصيل الخاصة بالإصدار الأول من OLMo، وهو نموذج لغة مفتوح حقًا ومتطور وإطار عمل لبناء ودراسة علم نمذجة اللغة. على عكس معظم الجهود السابقة التي لم تطلق سوى أوزان النماذج ورمز الاستدلال، نقوم بإطلاق OLMo والإطار بالكامل، بما في ذلك بيانات التدريب ورمز التدريب والتقييم. نأمل أن يؤدي هذا الإصدار إلى تمكين وتعزيز مجتمع البحث المفتوح وإلهام موجة جديدة من الابتكار."

تمت المساهمة بهذا النموذج من قبل [shanearora](https://huggingface.co/shanearora).
يمكن العثور على الكود الأصلي [هنا](https://github.com/allenai/OLMo/tree/main/olmo).

## OlmoConfig

[[autodoc]] OlmoConfig

## OlmoModel

[[autodoc]] OlmoModel

- forward

## OlmoForCausalLM

[[autodoc]] OlmoForCausalLM

- forward