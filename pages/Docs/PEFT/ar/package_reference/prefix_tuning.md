# Prefix tuning

تضيف طريقة [Prefix tuning](https://hf.co/papers/2101.00190) سلسلة من المتجهات الخاصة بالمهمة إلى تسلسل الإدخال والتي يمكن تعلمها مع الحفاظ على تجميد النموذج مسبق التدريب. يتم إدراج معلمات البادئة في جميع طبقات النموذج.

المقتطف من الورقة هو:

> "يعد الضبط الدقيق الطريقة الفعلية للاستفادة من نماذج اللغة الكبيرة مسبقة التدريب لأداء المهام اللاحقة. ومع ذلك، فإنه يعدل جميع معلمات نموذج اللغة، وبالتالي يتطلب تخزين نسخة كاملة لكل مهمة. في هذه الورقة، نقترح الضبط المسبق، وهو بديل خفيف الوزن للضبط الدقيق لمهام توليد اللغة الطبيعية، والذي يحافظ على تجميد معلمات نموذج اللغة، ولكنه يحسن متجهًا مستمرًا صغيرًا وخاصًا بالمهمة (يطلق عليه البادئة). يستلهم الضبط المسبق الإلهام من المطالبة، مما يسمح للرموز اللاحقة بالاهتمام بهذه البادئة كما لو كانت "رموزًا افتراضية". نطبق الضبط المسبق على GPT-2 لتوليد النص من الجدول وإلى BART للتلخيص. نجد أنه من خلال تعلم 0.1٪ فقط من المعلمات، يحقق الضبط المسبق أداءً قابلاً للمقارنة في إعداد البيانات الكاملة، ويتفوق على الضبط الدقيق في إعدادات البيانات المنخفضة، ويستقرئ بشكل أفضل إلى الأمثلة ذات الموضوعات غير المرئية أثناء التدريب".

## PrefixTuningConfig

[[autodoc]] tuners.prefix_tuning.config.PrefixTuningConfig

## PrefixEncoder

[[autodoc]] tuners.prefix_tuning.model.PrefixEncoder