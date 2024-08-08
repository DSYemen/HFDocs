# المساهمة في PEFT

نحن سعداء لقبول المساهمات في PEFT. إذا كنت تخطط للمساهمة، يرجى قراءة هذا المستند لتسهيل العملية قدر الإمكان.

## التثبيت

بالنسبة للمساهمات البرمجية في PEFT، يجب عليك اختيار طريقة التثبيت ["source"](../install#source).

إذا كنت جديدًا في إنشاء طلب سحب (Pull Request)، اتبع دليل [Creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) المقدم من GitHub.

## الاختبارات وفحوصات جودة الكود

بغض النظر عن نوع المساهمة (ما لم تكن متعلقة بالوثائق فقط)، يجب عليك تشغيل الاختبارات وفحوصات جودة الكود قبل إنشاء طلب سحب (PR) للتأكد من أن مساهمتك لا تتسبب في أي أخطاء وتتوافق مع معايير المشروع.

نوفر Makefile لتنفيذ الاختبارات اللازمة. قم بتشغيل الكود التالي لاختبار الوحدة:

```sh
make test
```

قم بتشغيل أحد الأوامر التالية لفحص جودة الكود والأسلوب أو لفحصهما وإصلاحهما:

```sh
make quality  # مجرد فحص
make style  # الفحص والإصلاح
```

يمكنك أيضًا إعداد [`pre-commit`](https://pre-commit.com/) لتشغيل هذه الإصلاحات تلقائيًا كخطافات Git commit.

```bash
$ pip install pre-commit
$ pre-commit install
```

قد يستغرق تشغيل جميع الاختبارات بضع دقائق، لذلك خلال عملية التطوير، قد يكون من الأكثر كفاءة تشغيل الاختبارات الخاصة بتغييرك فقط:

```sh
pytest tests/ -k <name-of-test>
```

من المفترض أن ينتهي هذا الأمر بشكل أسرع ويسمح بحلقة تكرار أسرع. ومع ذلك، يجب عليك تشغيل جناح الاختبار بالكامل قبل إنشاء طلب سحب (PR) لأن تغييرك قد يتسبب عن غير قصد في حدوث أخطاء في الاختبارات التي تبدو غير ذات صلة للوهلة الأولى.

إذا كان تغييرك خاصًا بإعدادات الأجهزة (على سبيل المثال، يتطلب CUDA)، فراجع [tests/test_gpu_examples.py](https://github.com/huggingface/peft/blob/1c1c7fdaa6e6abaa53939b865dee1eded82ad032/tests/test_gpu_examples.py) و[tests/test_common_gpu.py](https://github.com/huggingface/peft/blob/1c1c7fdaa6e6abaa53939b865dee1eded82ad032/tests/test_common_gpu.py) لمعرفة ما إذا كان من المنطقي إضافة اختبارات هناك. إذا كان لتغييرك تأثير على حفظ النماذج وتحميلها، يرجى تشغيل الاختبارات باستخدام علم `--regression` لتشغيل اختبارات الانحدار.

قد يحدث أنه أثناء عملك على طلب السحب (PR)، يتغير الكود الأساسي بسبب دمج تغييرات أخرى. إذا حدث ذلك - خاصة عند وجود صراع في الدمج - يرجى تحديث فرعك بأحدث التغييرات. يمكن أن يكون هذا دمجًا أو إعادة قاعدة، وسنقوم بدمج طلب السحب (PR) ودمجه بمجرد أن يصبح جاهزًا.

## وصف طلب السحب (PR)

عند فتح طلب سحب (PR)، يرجى تقديم وصف جيد للتغيير الذي تقترحه. إذا كان مرتبطًا بقضايا أو طلبات سحب أخرى، يرجى الإشارة إليها. لا يساعد تقديم وصف جيد المراجعين في مراجعة الكود بشكل أفضل وأسرع فحسب، بل يمكن استخدامه لاحقًا (كأساس) لرسالة الالتزام التي تساعد في صيانة المشروع على المدى الطويل.

إذا أجرى كودك بعض التغييرات غير البديهية، فقد يكون من الجيد أيضًا إضافة تعليقات إلى الكود لشرح تلك التغييرات. على سبيل المثال، إذا اضطررت إلى تكرار تنفيذك عدة مرات لأن الطريقة الأكثر وضوحًا لم تنجح، فهذه إشارة جيدة إلى أن تعليق الكود مطلوب.

## إصلاح الأخطاء

يرجى تقديم وصف لظروف حدوث الخطأ. إذا كانت هناك قضية موجودة، يرجى ربطها (على سبيل المثال، "يحل #12345").

من الناحية المثالية، عند تقديم إصلاح خطأ، يجب أن يكون مصحوبًا باختبار للخطأ. يجب أن يفشل الاختبار مع الكود الحالي وينجح مع إصلاح الخطأ. أضف تعليقًا إلى الاختبار يشير إلى القضية أو طلب السحب (PR). بدون اختبار، سيكون من الصعب منع حدوث تراجعات في المستقبل.

## إضافة طريقة ضبط دقيق جديدة

يتم تطوير طرق الضبط الدقيق الجديدة للبارامترات باستمرار. إذا كنت ترغب في إضافة طريقة جديدة وواعدة إلى PEFT، يرجى اتباع الخطوات التالية:

1. قبل البدء في تنفيذ الطريقة الجديدة، يرجى فتح قضية على GitHub باقتراحك. بهذه الطريقة، يمكن للمسؤولين تقديم بعض الملاحظات المبكرة.

2. يرجى إضافة رابط إلى مصدر الطريقة (عادةً ما يكون ورقة بحثية). يجب تقديم بعض الأدلة على وجود اهتمام عام باستخدام الطريقة. لن نقوم بإضافة طرق جديدة تم نشرها مؤخرًا، ولكن لا يوجد دليل على الطلب عليها.

3. عند تنفيذ الطريقة، من المنطقي البحث عن التنفيذ الموجود مسبقًا كدليل. علاوة على ذلك، عند هيكلة كودك، يرجى الاستلهام من طرق PEFT الأخرى. على سبيل المثال، إذا كانت طريقتك مشابهة لـ LoRA، فمن المنطقي هيكلة كودك بشكل مشابه أو حتى إعادة استخدام بعض الوظائف أو الفئات حيثما كان ذلك مناسبًا (بعض ازدواجية الكود مقبولة، ولكن لا تبالغ فيها).

4. من الناحية المثالية، بالإضافة إلى تنفيذ الطريقة الجديدة، يجب أن تكون هناك أمثلة (دفاتر ملاحظات، نصوص)، ووثائق، ومجموعة اختبارات شاملة تثبت أن الطريقة تعمل مع مجموعة متنوعة من المهام. ومع ذلك، يمكن أن يكون هذا أكثر صعوبة، لذلك من المقبول تقديم التنفيذ ومثال واحد على الأقل يعمل. يمكن إضافة الوثائق والاختبارات في طلبات السحب (PRs) اللاحقة.

5. بمجرد أن يكون لديك شيء يعمل، لا تتردد في إنشاء طلب سحب (PR) مسودة حتى إذا لم يكن جاهزًا للدمج بعد. سيكون المسؤولون سعداء بتقديم الملاحظات والتوجيهات على طول الطريق.

## إضافة ميزات أخرى

من الأفضل أولاً فتح قضية على GitHub باقتراح لإضافة الميزة الجديدة. بهذه الطريقة، يمكنك مناقشة ما إذا كانت الميزة منطقية مع المسؤولين قبل قضاء الكثير من الوقت في تنفيذها.

يجب أن تكون الميزات الجديدة مصحوبة بشكل عام باختبارات ووثائق أو أمثلة. بدون الأخيرة، سيواجه المستخدمون صعوبة في اكتشاف ميزتك الجديدة الرائعة.

يجب تنفيذ التغييرات على الكود بطريقة متوافقة مع الإصدارات السابقة. على سبيل المثال، يجب أن يستمر الكود الموجود في العمل بنفس الطريقة بعد دمج الميزة.