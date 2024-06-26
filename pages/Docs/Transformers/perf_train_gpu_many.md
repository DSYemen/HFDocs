# التدريب الفعال على معالجات GPU متعددة

إذا كان تدريب نموذج على معالج GPU واحد بطيئًا جدًا أو إذا لم تتسع ذاكرة معالج GPU الواحد لأوزان النموذج، فقد يكون الانتقال إلى إعداد متعدد معالجات GPU خيارًا قابلًا للتطبيق. قبل إجراء هذا الانتقال، استكشف بشكل شامل جميع الاستراتيجيات المشمولة في "أساليب وأدوات التدريب الفعال على معالج GPU واحد" حيث أنها تنطبق عالميًا على تدريب النماذج على أي عدد من معالجات GPU. بمجرد استخدامك لتلك الاستراتيجيات واكتشاف أنها غير كافية لحالتك على معالج GPU واحد، فكر في الانتقال إلى معالجات GPU متعددة.

يتطلب الانتقال من معالج GPU واحد إلى معالجات GPU متعددة تقديم بعض أشكال التوازي، حيث يجب توزيع عبء العمل عبر الموارد. يمكن استخدام تقنيات متعددة لتحقيق التوازي، مثل التوازي في البيانات، والتوازي في المصفوفات، والتوازي في الأنابيب. من المهم ملاحظة أنه لا يوجد حل واحد يناسب الجميع، وأن الإعدادات المثالية تعتمد على تكوين الأجهزة المحدد الذي تستخدمه.

يقدم هذا الدليل نظرة متعمقة على أنواع فردية من التوازي، بالإضافة إلى توجيهات حول طرق الجمع بين التقنيات واختيار النهج المناسب. للحصول على دروس خطوة بخطوة حول التدريب الموزع، يرجى الرجوع إلى وثائق 🤗 Accelerate.

قبل الغوص بشكل أعمق في تفاصيل كل تقنية، دعونا نلقي نظرة على عملية اتخاذ القرار عند تدريب نماذج كبيرة على بنية تحتية كبيرة.

## استراتيجية قابلية التوسع

ابدأ بتقدير مقدار ذاكرة الوصول العشوائي الظاهري (vRAM) المطلوبة لتدريب نموذجك. بالنسبة للنماذج المستضافة على 🤗 Hub، استخدم "آلة حاسبة ذاكرة النموذج" الخاصة بنا، والتي توفر لك حسابات دقيقة ضمن هامش بنسبة 5%.

**استراتيجية الموازاة لعقدة واحدة / إعداد متعدد معالجات GPU**

عند تدريب نموذج على عقدة واحدة مع معالجات GPU متعددة، يمكن لخيار استراتيجية الموازاة أن يؤثر بشكل كبير على الأداء. فيما يلي تفصيل لخياراتك:

**الحالة 1: يناسب نموذجك معالج GPU واحد**

إذا كان نموذجك يناسب معالج GPU واحد بشكل مريح، فهناك خياران رئيسيان:

1. DDP - Distributed DataParallel
2. ZeRO - اعتمادًا على الوضع والتكوين المستخدم، قد تكون هذه الطريقة أسرع أو لا، ومع ذلك، فمن الجدير تجربتها.

**الحالة 2: لا يناسب نموذجك معالج GPU واحد**

إذا كان نموذجك كبيرًا جدًا بالنسبة لمعالج GPU واحد، فهناك عدة بدائل يجب مراعاتها:

1. PipelineParallel (PP)
2. ZeRO
3. TensorParallel (TP)

مع وجود اتصال سريع بين العقد (مثل NVLINK أو NVSwitch)، يجب أن تؤدي جميع الاستراتيجيات الثلاث (PP وZeRO وTP) إلى أداء مماثل. ومع ذلك، بدون هذه الميزات، سيكون PP أسرع من TP أو ZeRO. قد يحدث أيضًا اختلاف في درجة TP. من الأفضل تجربة إعدادك المحدد لتحديد الاستراتيجية الأنسب.

يتم استخدام TP دائمًا تقريبًا داخل عقدة واحدة. وهذا يعني أن حجم TP <= معالجات GPU لكل عقدة.

**الحالة 3: لا تناسب الطبقة الأكبر في نموذجك معالج GPU واحد**

1. إذا كنت لا تستخدم ZeRO، فيجب عليك استخدام TensorParallel (TP)، لأن PipelineParallel (PP) وحده لن يكون كافيًا لاستيعاب الطبقة الكبيرة.
2. إذا كنت تستخدم ZeRO، فاعتمد أيضًا التقنيات من "أساليب وأدوات التدريب الفعال على معالج GPU واحد".

**استراتيجية الموازاة لعقد متعددة / إعداد متعدد معالجات GPU**

* عند وجود اتصال سريع بين العقد (مثل NVLINK أو NVSwitch)، فكر في استخدام أحد الخيارين التاليين:

1. ZeRO - لأنه يتطلب إجراء تعديلات طفيفة على النموذج
2. مزيج من PipelineParallel(PP) مع TensorParallel(TP) وDataParallel(DP) - سيؤدي هذا النهج إلى تقليل الاتصالات، ولكنه يتطلب إجراء تغييرات كبيرة على النموذج

* عند وجود اتصال بطيء بين العقد ولا تزال ذاكرة معالج GPU منخفضة:

1. استخدم مزيجًا من DataParallel(DP) مع PipelineParallel(PP) وTensorParallel(TP) وZeRO.

في الأقسام التالية من هذا الدليل، سنغوص بشكل أعمق في كيفية عمل أساليب التوازي المختلفة هذه.

## التوازي في البيانات

حتى مع وجود معالجي GPU فقط، يمكنك الاستفادة بسهولة من قدرات التدريب المعجلة التي توفرها الميزات المدمجة في PyTorch، مثل `DataParallel` (DP) و`DistributedDataParallel` (DDP). لاحظ أن وثائق PyTorch توصي بتفضيل `DistributedDataParallel` (DDP) على `DataParallel` (DP) للتدريب على معالجات GPU المتعددة لأنه يعمل مع جميع النماذج.

دعونا نلقي نظرة على كيفية عمل هاتين الطريقتين وما الذي يميزهما.

### DataParallel مقابل DistributedDataParallel

لفهم الاختلافات الرئيسية في عبء الاتصال بين معالجات GPU بين الطريقتين، دعونا نراجع العمليات لكل دفعة:

[DDP]:

- في وقت البدء، تقوم العملية الرئيسية بنسخ النموذج مرة واحدة من معالج GPU 0 إلى بقية معالجات GPU.
- بعد ذلك، لكل دفعة:
1. يستهلك كل معالج GPU مباشرة مصغرة الدفعة الخاصة به من البيانات.
2. أثناء `backward`، بمجرد أن تصبح التدرجات المحلية جاهزة، يتم حساب متوسطها عبر جميع العمليات.

[DP]:

لكل دفعة:

1. يقرأ معالج GPU 0 دفعة البيانات ثم يرسل مصغرة دفعة إلى كل معالج GPU.
2. يتم نسخ النموذج المحدث من معالج GPU 0 إلى كل معالج GPU.
3. يتم تنفيذ `forward`، ويتم إرسال الإخراج من كل معالج GPU إلى معالج GPU 0 لحساب الخسارة.
4. يتم توزيع الخسارة من معالج GPU 0 إلى جميع معالجات GPU، ويتم تشغيل `backward`.
5. يتم إرسال التدرجات من كل معالج GPU إلى معالج GPU 0 ويتم حساب متوسطها.

تشمل الاختلافات الرئيسية ما يلي:

1. يقوم DDP بتنفيذ عملية اتصال واحدة فقط لكل دفعة - إرسال التدرجات، في حين أن DP يقوم بخمسة تبادلات بيانات مختلفة لكل دفعة. يقوم DDP بنسخ البيانات باستخدام `torch.distributed`، بينما يقوم DP بنسخ البيانات داخل العملية عبر خيوط Python (والتي تقدم قيودًا مرتبطة بـ GIL). نتيجة لذلك، يكون `DistributedDataParallel` (DDP) أسرع بشكل عام من `DataParallel` (DP) ما لم يكن لديك اتصال بطيء بين بطاقات معالجات GPU.

2. في DP، يقوم معالج GPU 0 بمزيد من العمل مقارنة بمعالجات GPU الأخرى، مما يؤدي إلى نقص استخدام معالج GPU.

3. يدعم DDP التدريب الموزع عبر أجهزة متعددة، في حين أن DP لا يدعم ذلك.

هذه ليست قائمة شاملة بالاختلافات بين DP وDDP، ولكن الدقائق الأخرى تخرج عن نطاق هذا الدليل. يمكنك الحصول على فهم أعمق لهذه الطرق من خلال قراءة هذه المقالة.

دعونا نوضح الاختلافات بين DP وDDP من خلال تجربة. سنقوم باختبار أداء DP وDDP مع إضافة سياق وجود NVLink:

* الأجهزة: 2x TITAN RTX 24GB لكل منها + NVlink مع 2 NVLinks (`NV2` في `nvidia-smi topo -m`).
* البرمجيات: `pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`.

لإيقاف ميزة NVLink في أحد الاختبارات، نستخدم `NCCL_P2P_DISABLE=1`.

فيما يلي كود الاختبار وإخراجاته:

**DP**

```bash
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
python examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path openai-community/gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 110.5948, 'train_samples_per_second': 1.808, 'epoch': 0.69}
```

**DDP مع NVlink**

```bash
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path openai-community/gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}
```

**DDP بدون NVlink**

```bash
rm -r /tmp/test-clm; NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path openai-community/gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

فيما يلي نتائج الاختبار نفسها مجمعة في جدول للتسهيل:

| النوع | NVlink | الوقت |
| :----- | -----  | ---: |
| 2:DP | Y | 110 ثانية |
| 2:DDP | Y | 101 ثانية |
| 2:DDP | N | 131 ثانية |

كما ترون، في هذه الحالة، كان DP أبطأ بنسبة 10% تقريبًا من DDP مع NVlink، ولكنه أسرع بنسبة 15% من DDP بدون NVlink. يعتمد الاختلاف الحقيقي على مقدار البيانات التي يحتاج كل معالج GPU إلى مزامنتها مع الآخرين - فكلما زادت البيانات التي يجب مزامنتها، كلما أدى الرابط البطيء إلى إعاقة وقت التشغيل الإجمالي.
## موازاة بيانات ZeRO

يتم توضيح موازاة البيانات المدعومة من ZeRO (ZeRO-DP) في المخطط التالي من هذه [التدوينة](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/).

في حين قد يبدو الأمر معقدًا، إلا أنه مفهوم مشابه جدًا لـ `DataParallel` (DP). الفرق هو أنه بدلاً من استنساخ معلمات النموذج الكاملة، والتدرجات، وحالات المحسن، يقوم كل معالج رسومات GPU بتخزين شريحة فقط منه. بعد ذلك، في وقت التشغيل عندما تكون معلمات الطبقة الكاملة مطلوبة فقط للطبقة المعطاة، تتم مزامنة جميع وحدات معالجة الرسوميات GPUs لمنح بعضها البعض الأجزاء التي تفتقدها.

لتوضيح هذه الفكرة، دعنا نأخذ في الاعتبار نموذجًا بسيطًا يتكون من 3 طبقات (La وLb وLc)، حيث تحتوي كل طبقة على 3 معلمات. على سبيل المثال، تحتوي الطبقة La على الأوزان a0 وa1 وa2:

```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```

إذا كان لدينا 3 وحدات معالجة رسومات GPU، فإن ZeRO-DP يقسم النموذج إلى 3 وحدات معالجة رسومات GPU كما يلي:

```
GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2
```

وبطريقة ما، هذا هو نفس التقطيع الأفقي كما هو الحال في موازاة التنسور، على عكس التقطيع الرأسي، حيث يتم وضع مجموعات الطبقات الكاملة على وحدات معالجة رسومات GPU مختلفة. الآن دعنا نرى كيف يعمل هذا:

ستحصل كل من وحدات معالجة الرسومات GPUs هذه على دفعة مصغرة المعتادة كما هو الحال في DP:

```
x0 => GPU0
x1 => GPU1
x2 => GPU2
```

يتم تمرير المدخلات دون تعديلات كما لو كانت ستتم معالجتها بواسطة النموذج الأصلي.

أولاً، تصل المدخلات إلى الطبقة "La". ماذا يحدث في هذه المرحلة؟

على GPU0: تتطلب الدفعة المصغرة x0 معلمات a0 وa1 وa2 للقيام بمسارها الأمامي عبر الطبقة، ولكن GPU0 يحتوي على a0 فقط. سوف يحصل على a1 من GPU1 وa2 من GPU2، وجمع جميع قطع النموذج معا.

بالتوازي، تحصل GPU1 على دفعة مصغرة أخرى - x1. تحتوي GPU1 على المعلمة a1، ولكنها تحتاج إلى a0 وa2، لذا فهي تحصل عليها من GPU0 وGPU2.

يحدث الشيء نفسه لـ GPU2 الذي يحصل على الدفعة المصغرة x2. يحصل على a0 وa1 من GPU0 وGPU1.

بهذه الطريقة، تحصل كل من وحدات معالجة الرسومات GPUs الثلاث على التنسورات الكاملة المعاد بناؤها وتنفذ عملية انتقال أمامية باستخدام الدفعة المصغرة الخاصة بها. بمجرد الانتهاء من الحساب، يتم إسقاط البيانات التي لم تعد هناك حاجة إليها - يتم استخدامها فقط أثناء الحساب. تتم إعادة البناء بكفاءة عبر الاسترجاع المسبق.

ثم تتكرر العملية بأكملها للطبقة Lb، ثم Lc للأمام، ثم للخلف Lc -> Lb -> La.

<Tip>
تتشابه هذه الآلية مع استراتيجية فعالة لرحلات التخييم الجماعية: يحمل الشخص A الخيمة، ويحمل الشخص B الموقد، ويحمل الشخص C الفأس. كل ليلة يتشاركون ما لديهم مع الآخرين ويحصلون من الآخرين على ما لا يملكونه، وفي الصباح يحزمون معداتهم المخصصة ويستمرون في طريقهم. هذا هو ما هو عليه ZeRO DP/Sharded DDP.

قارن هذه الاستراتيجية بالاستراتيجية البسيطة حيث يتعين على كل شخص حمل خيمته وموقده وفأسه (مشابه لـ DataParallel (DP وDDP) في PyTorch)، والتي ستكون أقل كفاءة بكثير.
</Tip>

عند قراءة الأدبيات حول هذا الموضوع، قد تصادف المرادفات التالية: Sharded وPartitioned. إذا كنت تولي اهتمامًا وثيقًا للطريقة التي تقسم بها ZeRO أوزان النموذج، فستبدو مشابهة جدًا لموازاة التنسور التي سيتم مناقشتها لاحقًا. ويرجع ذلك إلى أنها تقسم/تشظي أوزان كل طبقة، على عكس موازاة النموذج الرأسية التي سيتم مناقشتها بعد ذلك.

التطبيقات:

- [DeepSpeed](https://www.deepspeed.ai/tutorials/zero/) ZeRO-DP المراحل 1+2+3
- تكامل [Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)
- تكامل [المحولات](main_classes/trainer#trainer-integrations)
## من النموذج الساذج للتوازي إلى التوازي الأنبوبي

لشرح التوازي الأنبوبي، سنلقي نظرة أولاً على التوازي الساذج للنموذج (MP)، المعروف أيضًا باسم MP الرأسي. يتضمن هذا النهج توزيع مجموعات من طبقات النموذج عبر وحدات معالجة الرسوميات (GPU) متعددة عن طريق تعيين طبقات محددة إلى وحدات معالجة الرسوميات (GPU) محددة باستخدام `.to()`.

بينما تنتقل البيانات عبر هذه الطبقات، يتم نقلها إلى نفس وحدة معالجة الرسوميات (GPU) مثل الطبقة، في حين تظل الطبقات الأخرى دون تغيير.

نشير إلى هذا التوازي النموذجي باسم "الرأسي" بسبب طريقة تصور النماذج عادةً. على سبيل المثال، يوضح المخطط التالي نموذجًا مكونًا من 8 طبقات مقسمة رأسياً إلى شريحتين، حيث يتم وضع الطبقات من 0 إلى 3 على GPU0 والطبقات من 4 إلى 7 على GPU1:

في هذا المثال، عندما تنتقل البيانات من الطبقة 0 إلى الطبقة 3، لا يختلف الأمر عن عملية التغذية الأمامية العادية. ومع ذلك، يتطلب تمرير البيانات من الطبقة 3 إلى الطبقة 4 نقلها من GPU0 إلى GPU1، مما يؤدي إلى زيادة العبء على الاتصال. إذا كانت وحدات معالجة الرسوميات (GPU) المشاركة موجودة على نفس عقدة الحوسبة (مثل نفس الجهاز المادي)، فإن النسخ يكون سريعًا، ولكن إذا تم توزيع وحدات معالجة الرسوميات (GPU) عبر عقد حوسبة مختلفة (مثل أجهزة متعددة)، فقد يكون عبء الاتصال أكبر بكثير.

بعد ذلك، تعمل الطبقات من 4 إلى 7 كما هي في النموذج الأصلي. بعد الانتهاء من الطبقة السابعة، هناك حاجة غالبًا إلى إرسال البيانات مرة أخرى إلى الطبقة 0 حيث توجد التسميات (أو بدلاً من ذلك إرسال التسميات إلى الطبقة الأخيرة). الآن يمكن حساب الخسارة ويمكن لمُحَسِّن القيام بعمله.

يعاني التوازي النموذجي الساذج من عدة أوجه قصور:

- **جميع وحدات معالجة الرسوميات (GPU) باستثناء واحدة غير نشطة في أي وقت معين**: إذا تم استخدام 4 وحدات معالجة الرسوميات (GPU)، فهذا مشابه تقريبًا لمضاعفة حجم ذاكرة وحدة معالجة الرسوميات (GPU) الفردية بمقدار أربعة أضعاف، وتجاهل بقية الأجهزة.

- **العبء الزائد في نقل البيانات بين الأجهزة**: على سبيل المثال، يمكن لـ 4x 6GB cards استيعاب نفس الحجم مثل 1x 24GB card باستخدام MP الساذج، ولكن بطاقة 24GB واحدة ستكمل التدريب بشكل أسرع، لأنها لا تحتوي على عبء نسخ البيانات. ولكن، على سبيل المثال، إذا كان لديك بطاقات 40 جيجابايت وتحتاج إلى نموذج 45 جيجابايت، فيمكنك ذلك باستخدام 4x 40GB cards (ولكن بالكاد بسبب حالة التدرج ومحسن).

- **نسخ التعليقات التوضيحية المشتركة**: قد تحتاج التعليقات التوضيحية المشتركة إلى نسخها ذهابًا وإيابًا بين وحدات معالجة الرسوميات (GPU).

الآن بعد أن تعرفت على كيفية عمل النهج الساذج للتوازي النموذجي وأوجه القصور فيه، دعنا نلقي نظرة على التوازي الأنبوبي (PP).

PP متطابق تقريبًا مع MP الساذج، ولكنه يحل مشكلة عدم نشاط وحدة معالجة الرسوميات (GPU) عن طريق تقسيم الدفعة الواردة إلى دفعات دقيقة وإنشاء خط أنابيب بشكل مصطنع، مما يسمح لوحدات معالجة الرسوميات (GPU) المختلفة بالمشاركة المتزامنة في عملية الحساب.

يوضح الرسم التالي من ورقة GPipe النهج الساذج لـ MP في الأعلى، و PP في الأسفل:

في أسفل المخطط، يمكنك ملاحظة أن نهج التوازي الأنبوبي (PP) يقلل من عدد مناطق وحدة معالجة الرسوميات (GPU) غير النشطة، المشار إليها باسم "الفقاعات". يوضح كلا جزأي المخطط مستوى توازي من الدرجة 4، مما يعني أن هناك 4 وحدات معالجة الرسوميات (GPU) تشارك في خط الأنابيب. يمكنك أن ترى أن هناك مسارًا للأمام مكونًا من 4 مراحل أنابيب (F0، F1، F2، و F3) يليه مسار عكسي بالترتيب العكسي (B3، B2، B1، و B0).

يقدم PP مُعَامِلَة جديدة للضبط - "chunks"، والتي تحدد عدد قطع البيانات المرسلة في تسلسل عبر نفس مرحلة الأنبوب. على سبيل المثال، في المخطط السفلي، يمكنك رؤية "chunks=4". تقوم وحدة معالجة الرسوميات (GPU0) بنفس المسار الأمامي على الجزء 0 و 1 و 2 و 3 (F0،0، F0،1، F0،2، F0،3) ثم تنتظر حتى تنتهي وحدات معالجة الرسوميات (GPU) الأخرى من عملها. لا تبدأ وحدة معالجة الرسوميات (GPU0) في العمل مرة أخرى إلا عندما تبدأ وحدات معالجة الرسوميات (GPU) الأخرى في إكمال عملها، وتقوم بمسار العودة للقطع 3 و 2 و 1 و 0 (B0،3، B0،2، B0،1، B0،0).

لاحظ أن هذا هو نفس مفهوم خطوات تجميع التدرجات. يستخدم PyTorch "chunks"، بينما يشير DeepSpeed إلى نفس مُعَامِلَة الضبط على أنها خطوات تجميع التدرجات.

بسبب "chunks"، يقدم PP مفهوم الدفعات الدقيقة (MBS). يقوم DP بتقسيم حجم دفعة البيانات العالمية إلى دفعات صغيرة، لذا إذا كان لديك درجة DP تبلغ 4، يتم تقسيم حجم دفعة عالمية تبلغ 1024 إلى 4 دفعات صغيرة يبلغ حجم كل منها 256 (1024/4). وإذا كان عدد "chunks" (أو GAS) هو 32، فإننا ننتهي بحجم دفعة دقيقة يبلغ 8 (256/32). تعمل كل مرحلة من مراحل الأنابيب مع دفعة دقيقة واحدة في كل مرة. لحساب حجم الدفعة العالمية لإعداد DP + PP، استخدم الصيغة: "mbs * chunks * dp_degree" ("8 * 32 * 4 = 1024").

مع "chunks=1"، ينتهي بك الأمر بـ MP الساذج، وهو غير فعال. مع قيمة كبيرة من "chunks"، ينتهي بك الأمر بحجم دفعات دقيقة صغيرة، وهو أيضًا غير فعال. لهذا السبب، نشجعك على تجربة قيمة "chunks" للعثور على القيمة التي تؤدي إلى أكثر استخدام فعال لوحدات معالجة الرسوميات (GPU).

قد تلاحظ وجود "فقاعة" من الوقت "الميت" على المخطط الذي لا يمكن تحويله إلى توازي لأن مرحلة "التغذية الأمامية" الأخيرة يجب أن تنتظر اكتمال "العودة" لإكمال خط الأنابيب. الغرض من العثور على أفضل قيمة لـ "chunks" هو تمكين استخدام متزامن عالي لوحدة معالجة الرسوميات (GPU) عبر جميع وحدات معالجة الرسوميات (GPU) المشاركة، مما يؤدي إلى تقليل حجم "الفقاعة".

تم تنفيذ حلول واجهة برمجة التطبيقات (API) الخاصة بخط الأنابيب في:

- PyTorch
- DeepSpeed
- Megatron-LM

تأتي هذه الحلول ببعض أوجه القصور:

- يجب أن تعدل النموذج بشكل كبير، لأن خط الأنابيب يتطلب إعادة كتابة التدفق الطبيعي للوحدات النمطية إلى تسلسل "nn.Sequential" من نفس النوع، والذي قد يتطلب إجراء تغييرات على تصميم النموذج.

- حاليًا، واجهة برمجة تطبيقات (API) الخاصة بخط الأنابيب مقيدة للغاية. إذا كان لديك مجموعة من متغيرات Python التي يتم تمريرها في المرحلة الأولى من خط الأنابيب، فسوف يتعين عليك إيجاد طريقة للالتفاف عليها. حاليًا، يتطلب واجهة برمجة التطبيقات (API) الخاصة بخط الأنابيب إما Tensor واحد أو مجموعة من Tensors كإدخال وإخراج وحيدين. يجب أن يكون لهذه المصفوفات حجم دفعة كأول بُعد، نظرًا لأن خط الأنابيب سيقوم بتقسيم الدفعة المصغرة إلى دفعات دقيقة. تتم مناقشة التحسينات المحتملة هنا https://github.com/pytorch/pytorch/pull/50693

- لا يمكن إجراء تدفق التحكم الشرطي على مستوى مراحل الأنابيب - على سبيل المثال، تتطلب نماذج Encoder-Decoder مثل T5 حلولًا خاصة للتعامل مع مرحلة تشفير شرطية.

- يجب أن تقوم بترتيب كل طبقة بحيث يصبح إخراج إحدى الطبقات إدخالًا للطبقة الأخرى.

تشمل الحلول الأحدث ما يلي:

- Varuna
- Sagemaker

لم نجرب Varuna و SageMaker ولكن تشير أوراقهما إلى أنهما تغلبا على قائمة المشكلات المذكورة أعلاه وأنهما يتطلبان إجراء تغييرات أقل على نموذج المستخدم.

التطبيقات:

- [PyTorch](https://pytorch.org/docs/stable/pipeline.html) (دعم أولي في pytorch-1.8، وتحسينه تدريجيًا في 1.9 وبشكل أكبر في 1.10). بعض [الأمثلة](https://github.com/pytorch/pytorch/blob/master/benchmarks/distributed/pipeline/pipe.py)

- [DeepSpeed](https://www.deepspeed.ai/tutorials/pipeline/)

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) لديه تنفيذ داخلي - لا يوجد واجهة برمجة تطبيقات (API).

- [Varuna](https://github.com/microsoft/varuna)

- [SageMaker](https://arxiv.org/abs/2111.05972) - هذا حل مملوك لا يمكن استخدامه إلا على AWS.

- [OSLO](https://github.com/tunib-ai/oslo) - تم تنفيذه هذا بناءً على محولات Hugging Face.

حالة محولات 🤗 : اعتبارًا من وقت كتابة هذا التقرير، لا يدعم أي من النماذج PP الكامل. تمتلك نماذج GPT2 و T5 دعم MP الساذج.

العقبة الرئيسية هي عدم القدرة على تحويل النماذج إلى "nn.Sequential" وجعل جميع الإدخالات Tensors. يرجع ذلك إلى أن النماذج تحتوي حاليًا على العديد من الميزات التي تجعل التحويل معقدًا للغاية، وسيتعين إزالتها لتحقيق ذلك.

تتوفر عمليات دمج DeepSpeed و Megatron-LM في [🤗 Accelerate](https://huggingface.co/docs/accelerate/main/en/usage_guides/deepspeed)

النهج الأخرى:

يستخدم DeepSpeed و Varuna و SageMaker مفهوم [خط الأنابيب المتشابك](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features.html)

هنا يتم تقليل "الفقاعة" (الوقت غير النشط) أكثر عن طريق إعطاء الأولوية لعمليات العودة. تحاول Varuna كذلك تحسين الجدول الزمني باستخدام المحاكاة للعثور على الجدول الأكثر كفاءة.

يحتوي OSLO على تنفيذ للتوازي الأنبوبي بناءً على المحولات دون تحويل "nn.Sequential".
## توازي المصفوفات 

في توازي المصفوفات، تقوم كل وحدة معالجة رسومية بمعالجة شريحة من المصفوفة ولا تجمع المصفوفة الكاملة إلا للعمليات التي تتطلب ذلك. ولوصف هذه الطريقة، يعتمد هذا القسم من الدليل على المفاهيم والرسوم البيانية من ورقة Megatron-LM: التدريب الفعال لنموذج اللغة واسع النطاق على مجموعات وحدات معالجة الرسوميات.

الكتلة الأساسية لأي محول هي طبقة متصلة تمامًا تليها تنشيط غير خطي. ويمكن كتابة الجزء النقطي من الضرب النقطي، باتباع ترميز ورقة Megatron، على النحو التالي: Y = GeLU(XA)، حيث X هو متجه الإدخال، وY هو متجه الإخراج، وA هو مصفوفة الأوزان.

إذا نظرنا إلى الحساب في شكل مصفوفة، فيمكنك أن ترى كيف يمكن تقسيم عملية ضرب المصفوفات بين وحدات معالجة رسومية متعددة:

إذا قمنا بتقسيم مصفوفة الأوزان A على شكل أعمدة عبر N من وحدات معالجة الرسوميات وأداء عمليات ضرب المصفوفات XA_1 من خلال XA_n بالتوازي، فسنحصل على ن متجهات إخراج Y_1، Y_2، ..., Y_n والتي يمكن إدخالها في GeLU بشكل مستقل:

باستخدام هذا المبدأ، يمكننا تحديث شبكة عصبية متعددة الطبقات ذات عمق عشوائي، دون الحاجة إلى أي تزامن بين وحدات معالجة الرسوميات حتى النهاية، حيث نحتاج إلى إعادة بناء متجه الإخراج من الشظايا. ويقدم مؤلفو ورقة Megatron-LM توضيحًا مفيدًا لذلك:

إن جعل طبقات الاهتمام متعدد الرؤوس أكثر بساطة، لأنها متوازية بالفعل بطبيعتها، وذلك بسبب وجود رؤوس مستقلة متعددة!

اعتبارات خاصة: يتطلب توازي المصفوفات شبكة سريعة للغاية، لذلك لا يُنصح بالقيام بتوازي المصفوفات عبر أكثر من عقدة واحدة. ومن الناحية العملية، إذا كانت العقدة تحتوي على 4 وحدات معالجة رسومية، فإن أعلى درجة من توازي المصفوفات هي 4. إذا كنت بحاجة إلى درجة توازي مصفوفات تبلغ 8، فيجب عليك استخدام العقد التي تحتوي على 8 وحدات معالجة رسومية على الأقل.

## توازي البيانات + توازي الأنابيب 

يوضح الرسم البياني التالي من تعليمي DeepSpeed كيفية الجمع بين توازي البيانات مع توازي الأنابيب.

من المهم هنا ملاحظة كيف أن ترتيب توازي البيانات 0 لا يرى وحدة معالجة الرسوميات 2 وترتيب توازي البيانات 1 لا يرى وحدة معالجة الرسوميات 3. وبالنسبة لتوازي البيانات، هناك فقط وحدات معالجة الرسوميات 0 و1 حيث يتم إدخال البيانات كما لو كان هناك وحدتان فقط من وحدات معالجة الرسوميات. وتقوم وحدة معالجة الرسوميات 0 بتفريغ بعض حمولتها سراً إلى وحدة معالجة الرسوميات 2 باستخدام توازي الأنابيب. وتقوم وحدة معالجة الرسوميات 1 بنفس الشيء عن طريق الاستعانة بوحدة معالجة الرسوميات 3.

وبما أن كل بُعد يتطلب وحدتي معالجة رسومية على الأقل، فستحتاج هنا إلى 4 وحدات معالجة رسومية على الأقل.

## توازي البيانات + توازي الأنابيب + توازي المصفوفات 

للحصول على تدريب أكثر كفاءة، يتم استخدام توازي ثلاثي الأبعاد حيث يتم الجمع بين توازي الأنابيب مع توازي المصفوفات وتوازي البيانات. ويمكن ملاحظة ذلك في الرسم البياني التالي.

هذا الرسم البياني مأخوذ من منشور مدونة بعنوان "توازي ثلاثي الأبعاد: توسيع نطاق النماذج إلى نماذج معلمات التريليون"، وهو أيضًا قراءة جيدة.

وبما أن كل بُعد يتطلب وحدتي معالجة رسومية على الأقل، فستحتاج هنا إلى 8 وحدات معالجة رسومية على الأقل.

## توازي بيانات Zero + توازي الأنابيب + توازي المصفوفات 

من الميزات الرئيسية لبرنامج DeepSpeed هي Zero، وهي امتداد قابل للتطوير للغاية لتوازي البيانات. وقد تمت مناقشته بالفعل في قسم "توازي بيانات Zero". وعادة ما يكون ميزة مستقلة لا تتطلب توازي الأنابيب أو توازي المصفوفات. ولكنه يمكن أن يجمع مع توازي الأنابيب وتوازي المصفوفات.

عندما يتم الجمع بين Zero-DP مع PP (وبشكل اختياري TP)، فإنه يمكّن عادةً من Zero stage 1 (تجزئة المحسن) فقط.

في حين أنه من الممكن نظريًا استخدام Zero stage 2 (تجزئة التدرج) مع توازي الأنابيب، إلا أنه سيكون له تأثيرات سلبية على الأداء. سيكون هناك حاجة إلى تقليل إضافي للمجموعة لكل دفعة صغيرة لجمع التدرجات قبل التجزئة، مما يضيف تكلفة اتصال محتملة كبيرة. وبسبب طبيعة توازي الأنابيب، يتم استخدام دفعات صغيرة وتحاول موازنة كثافة الحساب (حجم الدفعة الصغيرة) مع تقليل فقاعة الأنابيب (عدد الدفعات الصغيرة). لذلك ستؤثر تكاليف الاتصال هذه على الأداء.

بالإضافة إلى ذلك، هناك بالفعل طبقات أقل من المعتاد بسبب توازي الأنابيب، لذلك لن تكون وفورات الذاكرة كبيرة. ويقلل توازي الأنابيب بالفعل من حجم التدرج بنسبة 1/PP، لذلك فإن وفورات تجزئة التدرج بالإضافة إلى ذلك أقل أهمية من توازي البيانات النقي.

كما أن Zero stage 3 ليس خيارًا جيدًا لنفس السبب - الحاجة إلى مزيد من الاتصالات بين العقد.

وبما أننا لدينا Zero، فإن الفائدة الأخرى هي Zero-Offload. وبما أن هذه هي مرحلة 1، يمكن تفريغ حالات المحسن إلى وحدة المعالجة المركزية.
## FlexFlow

يقدم FlexFlow أيضًا حلاً لمشكلة الموازاة بطريقة مختلفة قليلاً.

يقوم بتنفيذ نوع من الموازاة رباعية الأبعاد على Sample-Operator-Attribute-Parameter.

1. العينة = الموازاة على مستوى البيانات (موازاة على مستوى العينات)
2. المشغل = موازاة عملية واحدة إلى عدة عمليات فرعية
3. الصفة = الموازاة على مستوى البيانات (موازاة على مستوى الطول)
4. المعامل = موازاة النموذج (بغض النظر عن البعد - أفقي أو رأسي)

الأمثلة:

* العينة

لنأخذ 10 دفعات بطول تسلسل 512. إذا قمنا بموازاة البعد إلى جهازين، فسنحصل على 10 × 512 والتي تصبح 5 × 2 × 512.

* المشغل

إذا أجرينا تطبيع الطبقة، فإننا نحسب الانحراف المعياري أولاً ثم المتوسط، وبعد ذلك يمكننا تطبيع البيانات. تسمح موازاة المشغل بحساب الانحراف المعياري والمتوسط ​​بالتوازي. لذلك، إذا قمنا بموازاة البعد إلى جهازين (cuda:0، cuda:1)، فنحن نقوم أولاً بنسخ بيانات الإدخال إلى كلا الجهازين، ويقوم cuda:0 بحساب الانحراف المعياري، ويقوم cuda:1 بحساب المتوسط ​​في نفس الوقت.

* الصفة

لدينا 10 دفعات بطول 512. إذا قمنا بموازاة البعد إلى جهازين، فستكون 10 × 512 هي 10 × 2 × 256.

* المعامل

وهو مشابه لموازاة نموذج Tensor أو موازاة طبقة Naive.

تكمن أهمية هذا الإطار في أنه يأخذ الموارد مثل (1) GPU/TPU/CPU مقابل (2) RAM/DRAM مقابل (3) fast-intra-connect/slow-inter-connect ويقوم تلقائيًا بتحسينها جميعًا خوارزميًا، ويقرر أي موازاة يجب استخدامها وأين.

من الجوانب المهمة جدًا أن FlexFlow مصمم لتحسين موازاة DNN للنماذج ذات الأحمال العملة الثابتة والثابتة، حيث قد تفضل النماذج ذات السلوك الديناميكي استراتيجيات موازاة مختلفة عبر التكرارات.

لذلك، الوعد جذاب للغاية - فهو يقوم بتشغيل محاكاة لمدة 30 دقيقة على مجموعة من اختيارك ويخرج بأفضل استراتيجية لاستخدام هذه البيئة المحددة. إذا قمت بإضافة/إزالة/استبدال أي أجزاء، فسيقوم بتشغيلها وإعادة تحسين الخطة لذلك. وبعد ذلك يمكنك التدريب. سيكون للإعداد المختلف تحسينه المخصص.

حالة المحولات: يمكن تتبع نماذج المحولات عبر transformers.utils.fx، وهو شرط أساسي لـ FlexFlow، ولكن يلزم إجراء تغييرات على جانب FlexFlow لجعله يعمل مع نماذج المحولات.

## اختيار وحدة معالجة الرسومات (GPU)

عند التدريب على وحدات معالجة الرسومات (GPU) متعددة، يمكنك تحديد عدد وحدات معالجة الرسومات (GPU) التي تريد استخدامها والترتيب الذي تريد استخدامها به. يمكن أن يكون ذلك مفيدًا، على سبيل المثال، عندما يكون لديك وحدات معالجة رسومات (GPU) ذات قدرات حوسبة مختلفة وتريد استخدام وحدة معالجة الرسومات (GPU) الأسرع أولاً. تنطبق عملية الاختيار على كل من DistributedDataParallel و DataParallel لاستخدام مجموعة فرعية فقط من وحدات معالجة الرسومات (GPU) المتوفرة، ولا تحتاج إلى Accelerate أو تكامل DeepSpeed.

### عدد وحدات معالجة الرسومات (GPU)

على سبيل المثال، إذا كان لديك 4 وحدات معالجة رسومات (GPU) وتريد استخدام أول اثنتين فقط:

- استخدم --nproc_per_node لتحديد عدد وحدات معالجة الرسومات (GPU) التي تريد استخدامها.

```bash
torchrun --nproc_per_node=2 trainer-program.py ...
```

- استخدم --num_processes لتحديد عدد وحدات معالجة الرسومات (GPU) التي تريد استخدامها.

```bash
accelerate launch --num_processes 2 trainer-program.py ...
```

- استخدم --num_gpus لتحديد عدد وحدات معالجة الرسومات (GPU) التي تريد استخدامها.

```bash
deepspeed --num_gpus 2 trainer-program.py ...
```

### ترتيب وحدات معالجة الرسومات (GPU)

الآن، لاختيار وحدات معالجة الرسومات (GPU) التي تريد استخدامها وترتيبها، ستستخدم متغير البيئة CUDA_VISIBLE_DEVICES. من الأسهل تعيين متغير البيئة في ملف ~bashrc أو ملف تهيئة آخر. يستخدم CUDA_VISIBLE_DEVICES لتعيين وحدات معالجة الرسومات (GPU) التي يتم استخدامها. على سبيل المثال، إذا كان لديك 4 وحدات معالجة رسومات (GPU) (0، 1، 2، 3) وتريد تشغيل وحدات معالجة الرسومات (GPU) 0 و 2 فقط:

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

تكون وحدات معالجة الرسومات (GPU) المادية 0 و 2 فقط "مرئية" لـ PyTorch، ويتم تعيينها إلى cuda:0 و cuda:1 على التوالي. يمكنك أيضًا عكس ترتيب وحدات معالجة الرسومات (GPU) لاستخدام 2 أولاً. الآن، يتم تعيين الخريطة إلى cuda:1 لـ GPU 0 و cuda:0 لـ GPU 2.

```bash
CUDA_VISIBLE_DEVICES=2,0 torchrun trainer-program.py ...
```

يمكنك أيضًا تعيين متغير البيئة CUDA_VISIBLE_DEVICES إلى قيمة فارغة لإنشاء بيئة بدون وحدات معالجة رسومات (GPU).

```bash
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

نصيحة: كما هو الحال مع أي متغير بيئي، يمكن تصديرها بدلاً من إضافتها إلى سطر الأوامر. ومع ذلك، لا يوصى بذلك لأنه يمكن أن يكون مربكًا إذا نسيت كيفية إعداد متغير البيئة وقد ينتهي بك الأمر باستخدام وحدات معالجة الرسومات (GPU) الخطأ. بدلاً من ذلك، من الشائع تعيين متغير البيئة لتشغيل تدريب محدد على نفس سطر الأوامر.

CUDA_DEVICE_ORDER هو متغير بيئي بديل يمكنك استخدامه للتحكم في كيفية ترتيب وحدات معالجة الرسومات (GPU). يمكنك ترتيبها إما عن طريق:

1. معرفات حافلة PCIe التي تتطابق مع ترتيب nvidia-smi و rocm-smi لـ NVIDIA و AMD GPUs على التوالي

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

2. قدرة الحوسبة GPU

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST
```

CUDA_DEVICE_ORDER مفيد بشكل خاص إذا كانت إعدادات التدريب الخاصة بك تتكون من وحدة معالجة رسومات (GPU) أقدم وأحدث، حيث تظهر وحدة معالجة الرسومات (GPU) الأقدم أولاً، ولكن لا يمكنك التبديل فعليًا بين البطاقات لجعل وحدة معالجة الرسومات (GPU) الأحدث تظهر أولاً. في هذه الحالة، قم بتعيين CUDA_DEVICE_ORDER=FASTEST_FIRST لاستخدام وحدة معالجة الرسومات (GPU) الأحدث والأسرع أولاً (لا يزال nvidia-smi أو rocm-smi يبلغ عن وحدات معالجة الرسومات (GPU) وفقًا لترتيب PCIe الخاص بها). أو يمكنك أيضًا تعيين export CUDA_VISIBLE_DEVICES=1,0.