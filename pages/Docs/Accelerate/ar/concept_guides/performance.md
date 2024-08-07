# مقارنة الأداء بين إعدادات الأجهزة المختلفة

يمكن أن يكون تقييم الأداء ومقارنته بين الإعدادات المختلفة أمرًا معقدًا بعض الشيء إذا لم تكن تعرف ما الذي تبحث عنه. على سبيل المثال، لا يمكنك تشغيل نفس البرنامج النصي بنفس حجم الدفعة عبر TPU وmulti-GPU وsingle-GPU مع Accelerate وتتوقع أن تتوافق نتائجك.

ولكن، لماذا؟

هناك ثلاثة أسباب لذلك سوف نغطيها في هذا الدرس:

1. **ضبط البذور الصحيحة**
2. **أحجام الدفعات الملحوظة**
3. **معدلات التعلم**

## ضبط البذرة

على الرغم من أن هذه المشكلة لم تظهر كثيرًا، تأكد من استخدام [`utils.set_seed`] لضبط البذرة بشكل كامل في جميع الحالات الموزعة بحيث تكون عملية التدريب قابلة للتكرار:

```python
from accelerate.utils import set_seed

set_seed(42)
```

لماذا هذا مهم؟ خلف الكواليس، سيقوم هذا بضبط إعدادات البذور الخمسة المختلفة:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# ^^ من الآمن استدعاء دالة حتى إذا لم تكن cuda متاحة
if is_torch_xla_available():
    xm.set_rng_state(seed)
```

حالة التعشيش العشوائي، وحالة النيبي، وحالة الشعلة، وحالة كودا للشعلة، وفي حالة توفر TPUs، حالة كودا للشعلة xla.

## أحجام الدفعات الملحوظة

عند التدريب باستخدام Accelerate، يكون حجم الدفعة التي يتم تمريرها إلى برنامج تحميل البيانات هو **حجم الدفعة لكل GPU**. ما يعنيه هذا هو أن حجم الدفعة 64 على GPU واحد هو في الواقع حجم دفعة 128. ونتيجة لذلك، عند الاختبار على GPU واحد، يجب مراعاة ذلك، وكذلك بالنسبة لـ TPUs.

يمكن استخدام الجدول أدناه كمرجع سريع لتجربة أحجام دفعات مختلفة:

<Tip>
في هذا المثال، هناك GPU واحد لـ "Multi-GPU" ووحدة TPU pod مع 8 عمال
</Tip>

| حجم دفعة GPU واحدة | حجم دفعة Multi-GPU المكافئ | حجم دفعة TPU المكافئ |
|-----------------------|---------------------------------|---------------------------|
| 256                   | 128                             | 32                        |
| 128                   | 64                              | 16                        |
| 64                    | 32                              | 8                         |
| 32                    | 16                              | 4                         |

## معدلات التعلم

كما هو مذكور في مصادر متعددة [[1](https://aws.amazon.com/blogs/machine-learning/scalable-multi-node-deep-learning-training-using-gpus-in-the-aws-cloud/)][[2](https://docs.nvidia.com/clara/clara-train-sdk/pt/model.html#classification-models-multi-gpu-training)]، يجب ضبط معدل التعلم *خطيًا* بناءً على عدد الأجهزة الموجودة. يوضح المقتطف أدناه كيفية القيام بذلك باستخدام Accelerate:

<Tip>
نظرًا لأن المستخدمين يمكنهم تحديد مخططاتهم الخاصة لمعدل التعلم، فإننا نترك هذا الأمر للمستخدم لاتخاذ القرار بشأن ما إذا كان يريد ضبط معدل التعلم أم لا.
</Tip>

```python
learning_rate = 1e-3
accelerator = Accelerator()
learning_Multiplier *= accelerator.num_processes

optimizer = AdamW(params=model.parameters(), lr=learning_rate)
```

كما ستجد أن `accelerate` سيقوم بضبط معدل التعلم بناءً على عدد العمليات التي يتم التدريب عليها. ويرجع ذلك إلى حجم الدفعة الملحوظ سابقًا. لذا، في حالة وجود GPU واحد، سيتم ضبط معدل التعلم مرتين أكثر من GPU واحد لمراعاة حجم الدفعة الذي يكون ضعف الحجم (إذا لم يتم إجراء أي تغييرات على حجم الدفعة في مثيل GPU واحد).

## تراكم التدرج والدقة المختلطة

عند استخدام تراكم التدرج والدقة المختلطة، فمن المتوقع حدوث بعض التدهور في الأداء بسبب كيفية عمل متوسط التدرجات (التراكم) وفقدان الدقة (الدقة المختلطة). سيتم ملاحظة ذلك صراحةً عند مقارنة الخسارة لكل دفعة بين إعدادات الكمبيوتر المختلفة. ومع ذلك، يجب أن تكون الخسارة الإجمالية والمقاييس والأداء العام في نهاية التدريب _تقريبًا_ نفس الشيء.