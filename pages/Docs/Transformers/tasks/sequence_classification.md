# تصنيف النص

يعد تصنيف النص مهمة شائعة في معالجة اللغات الطبيعية (NLP) والتي تقوم بتعيين تسمية أو فئة للنص. وتستخدم بعض أكبر الشركات تصنيف النص في الإنتاج لمجموعة واسعة من التطبيقات العملية. أحد أكثر أشكال تصنيف النص شيوعًا هو تحليل المشاعر، والذي يقوم بتعيين تسمية مثل 🙂 إيجابية، 🙁 سلبية، أو 😐 محايدة لتسلسل نصي.

سيوضح هذا الدليل كيفية:

1. ضبط نموذج [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) الدقيق على مجموعة بيانات [IMDb](https://huggingface.co/datasets/imdb) لتحديد ما إذا كان تقييم الفيلم إيجابيًا أم سلبيًا.
2. استخدام نموذجك الدقيق للاستنتاج.

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate accelerate
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للوصول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات IMDb

ابدأ بتحميل مجموعة بيانات IMDb من مكتبة Datasets 🤗:

```py
>>> from datasets import load_dataset

>>> imdb = load_dataset("imdb")
```

ثم الق نظرة على مثال:

```py
>>> imdb["test"][0]
{
"label": 0,
"text": "أنا أحب الخيال العلمي وأنا على استعداد لتحمل الكثير. أفلام/تلفزيون الخيال العلمي عادة ما تكون ممولة وناقصة التقدير ومفهومة. حاولت أن أحب هذا، حاولت حقًا، لكنها بالنسبة لتلفزيون الخيال العلمي الجيد مثل بابل 5 إلى ستار تريك (الأصلي). أدوات تجميل سخيفة، مجموعات من الورق المقوى الرخيص، حوارات متعرجة، رسومات الكمبيوتر التي لا تتطابق مع الخلفية، وشخصيات أحادية الأبعاد مؤلمة لا يمكن التغلب عليها مع إعداد "الخيال العلمي". (أنا متأكد من أن هناك منكم هناك من يعتقد أن بابل 5 هو تلفزيون الخيال العلمي الجيد. ليس كذلك. إنه مبتذل وغير ملهم.) في حين أن المشاهدين الأمريكيين قد يحبون العاطفة وتطوير الشخصية، فإن الخيال العلمي هو نوع لا يأخذ نفسه على محمل الجد (راجع ستار تريك). قد يعالج قضايا مهمة، ولكن ليس كفلسفة جادة. من الصعب حقًا الاهتمام بالشخصيات هنا لأنها ليست سخيفة فحسب، بل تفتقر إلى شرارة الحياة. أفعالهم وردود أفعالهم خشبية ويمكن التنبؤ بها، وغالبًا ما تكون مؤلمة للمشاهدة. يعرف صانعو الأرض أنها قمامة لأن عليهم دائمًا أن يقولوا "أرض جين روددينبيري..." وإلا فلن يستمر الناس في المشاهدة. يجب أن تكون رماد روددينبيري تدور في مدارها حيث يتخبط هذا العرض الباهت الرخيص سيئ التحرير (مشاهدته دون فواصل إعلانية يجلب هذا المنزل) ترابانت متثاقل إلى الفضاء. حرق. لذلك، اقتل شخصية رئيسية. ثم أعده كممثل آخر. جيز! دالاس مرة أخرى.
```

هناك حقلان في هذه المجموعة من البيانات:

- `text`: نص مراجعة الفيلم.
- `label`: قيمة إما `0` لمراجعة سلبية أو `1` لمراجعة إيجابية.

## معالجة مسبقة

الخطوة التالية هي تحميل معالج نصوص DistilBERT لمعالجة حقل `النص`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

قم بإنشاء دالة معالجة مسبقة لتوكينز النص وتقليص التسلسلات بحيث لا تكون أطول من الطول الأقصى للإدخال في DistilBERT:

```py
>>> def preprocess_function(examples):
...     return tokenizer(examples["text"], truncation=True)
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم وظيفة [`~datasets.Dataset.map`] في مكتبة 🤗 Datasets. يمكنك تسريع `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد:

```py
tokenized_imdb = imdb.map(preprocess_function, batched=True)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DataCollatorWithPadding`]. من الأكثر كفاءة أن تقوم *بالتحديد الديناميكي* للجمل إلى الطول الأطول في دفعة أثناء التجميع، بدلاً من تحديد مجموعة البيانات بأكملها إلى الطول الأقصى.

<frameworkcontent>
<pt>
```py
>>> from transformers import DataCollatorWithPadding

>>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
</pt>
<tf>
```py
>>> from transformers import DataCollatorWithPadding

>>> data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
```
</tf>
</frameworkcontent>

## تقييم

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index). بالنسبة لهذه المهمة، قم بتحميل مقياس [الدقة](https://huggingface.co/spaces/evaluate-metric/accuracy) (راجع جولة 🤗 Evaluate [السريعة](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب مقياس):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

ثم قم بإنشاء دالة تمرر تنبؤاتك وتسمياتك إلى [`~evaluate.EvaluationModule.compute`] لحساب الدقة:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

دالتك `compute_metrics` جاهزة الآن، وستعود إليها عندما تقوم بإعداد تدريبك.
بالتأكيد، سأتبع تعليماتك وسأقوم بترجمة النص الموجود في الفقرات والعناوين فقط، مع تجاهل النصوص البرمجية والروابط وأكواد HTML وCSS.

## التدريب

قبل البدء في تدريب نموذجك، قم بإنشاء خريطة للهويات المتوقعة إلى تسمياتها التوضيحية باستخدام `id2label` و`label2id`:

بعد ذلك، أنت مستعد لبدء تدريب نموذجك! قم بتحميل DistilBERT مع [`AutoModelForSequenceClassification`] جنبًا إلى جنب مع عدد التسميات المتوقعة، وخرائط التسميات:

في هذه المرحلة، لم يتبق سوى ثلاث خطوات:

1. قم بتعريف فرط معلمات التدريب الخاصة بك في [`TrainingArguments`]. المعلمة المطلوبة الوحيدة هي `output_dir` التي تحدد مكان حفظ نموذجك. سوف تقوم بدفع هذا النموذج إلى Hub عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجلاً الدخول في Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم الدقة وحفظ نقطة تفتيش التدريب.

2. قم بتمرير الحجج التدريبية إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات ومعيّن الرموز وملف تجميع البيانات ووظيفة `compute_metrics`.

3. اتصل بـ [`~Trainer.train`] لضبط نموذجك.

<Tip>

يطبق [`Trainer`] التدبيس الديناميكي بشكل افتراضي عند تمرير `tokenizer` إليه. في هذه الحالة، لا تحتاج إلى تحديد ملف تجميع البيانات بشكل صريح.

</Tip>

بمجرد اكتمال التدريب، شارك نموذجك في Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

<Tip>

للحصول على مثال أكثر تعمقًا حول كيفية ضبط نموذج للتصنيف النصي، راجع الدفتر المناسب
[دفتر ملاحظات PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)
أو [دفتر TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb).

</Tip>

## الاستنتاج

رائع، الآن بعد أن ضبطت نموذجًا، يمكنك استخدامه للاستنتاج!

احصل على بعض النصوص التي ترغب في تشغيل الاستدلال عليها:

أبسط طريقة لتجربة نموذجك المضبوط للاستدلال هي استخدامه في [`pipeline`]. قم بتنفيذ عملية فورية لتحليل المشاعر باستخدام نموذجك، ومرر نصك إليه:

يمكنك أيضًا يدويًا تكرار نتائج `pipeline` إذا أردت:

قم بتوكينز النص وإرجاع تنسيقات PyTorch:

مرر المدخلات إلى النموذج وإرجاع `logits`:

احصل على الفئة ذات الاحتمالية الأعلى، واستخدم تعيين `id2label` للنموذج لتحويله إلى تسمية نصية:

قم بتوكينز النص وإرجاع تنسيقات TensorFlow:

مرر المدخلات إلى النموذج وإرجاع `logits`:

احصل على الفئة ذات الاحتمالية الأعلى، واستخدم تعيين `id2label` للنموذج لتحويله إلى تسمية نصية: