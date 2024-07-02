# LoKr

طريقة LoKr (Low-Rank Kronecker Product) هي طريقة مشتقة من طريقة LoRA، والتي تقارب مصفوفة الأوزان الكبيرة باستخدام مصفوفتين من الرتبة المنخفضة، ثم تجمعهما باستخدام حاصل الضرب الكرونيكري. كما توفر طريقة LoKr مصفوفة ثالثة اختيارية من الرتبة المنخفضة لتوفير تحكم أفضل أثناء الضبط الدقيق.

## LoKrConfig

[[autodoc]] tuners.lokr.config.LoKrConfig

## LoKrModel

[[autodoc]] tuners.lokr.model.LoKrModel