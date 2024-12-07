<!--حقوق النشر 2023 فريق هاجينج فيس. جميع الحقوق محفوظة.

مرخص بموجب رخصة أباتشي، الإصدار 2.0 (الرخصة)؛ لا يجوز لك استخدام هذا الملف إلا بالامتثال لـ
الرخصة. يمكنك الحصول على نسخة من الرخصة على

http://www.apache.org/licenses/LICENSE-2.0

ما لم يتطلب القانون المعمول به أو يتم الاتفاق عليه كتابيًا، يتم توزيع البرامج الموزعة بموجب الرخصة على
أساس "كما هي" بدون ضمانات أو شروط من أي نوع، سواء كانت صريحة أو ضمنية. راجع الرخصة للحصول على
اللغة المحددة التي تحكم الأذونات والقيود بموجب الرخصة.

⚠️ يرجى ملاحظة أن هذا الملف مكتوب بلغة Markdown ولكنه يحتوي على بناء جملة محددة لمبني الوثائق الخاص بنا (يشبه MDX) والذي قد لا يتم
عرضه بشكل صحيح في عارض Markdown الخاص بك.

-->

# مهام الصور مع IDEFICS

[[open-in-colab]]

في حين يمكن معالجة المهام الفردية من خلال الضبط الدقيق للنماذج المتخصصة، فقد ظهر نهج بديل
اكتسب شعبية مؤخرًا وهو استخدام النماذج الكبيرة لمجموعة متنوعة من المهام بدون ضبط دقيق.
على سبيل المثال، يمكن للنماذج اللغوية الكبيرة التعامل مع مهام NLP مثل التلخيص والترجمة والتصنيف والمزيد.
لم يعد هذا النهج مقتصرًا على نمط واحد، مثل النص، وفي هذا الدليل، سنشرح كيف يمكنك
حل مهام الصور والنصوص باستخدام نموذج متعدد الوسائط كبير يسمى IDEFICS.

[IDEFICS](../model_doc/idefics) هو نموذج مفتوح الوصول للرؤية واللغة يعتمد على [Flamingo](https://huggingface.co/papers/2204.14198)،
نموذج لغة بصرية رائد تم تطويره في البداية بواسطة DeepMind. يقبل النموذج تسلسلات تعسفية من الصور
ونصوص الإدخال وينتج نصًا متماسكًا كناتج. يمكنه الإجابة عن الأسئلة حول الصور ووصف المحتوى المرئي
وإنشاء قصص مبنية على صور متعددة، وهكذا. يأتي IDEFICS في نسختين - [80 مليار معامل](https://huggingface.co/HuggingFaceM4/idefics-80b)
و [9 مليار معامل](https://huggingface.co/HuggingFaceM4/idefics-9b)، وكلاهما متاحان على 🤗 Hub. بالنسبة لكل نسخة، يمكنك أيضًا العثور على نسخ مضبوطة بدقة
من النموذج المكيف لحالات الاستخدام المحادثية.

هذا النموذج متعدد الاستخدامات بشكل استثنائي ويمكن استخدامه لمجموعة واسعة من مهام الصور والوسائط المتعددة. ومع ذلك،
كونه نموذجًا كبيرًا يعني أنه يتطلب موارد حسابية وبنية تحتية كبيرة. الأمر متروك لك لتقرر ما إذا كان
هذا النهج يناسب حالتك الاستخدامية بشكل أفضل من الضبط الدقيق للنماذج المتخصصة لكل مهمة فردية.

في هذا الدليل، ستتعلم كيفية:
- [تحميل IDEFICS](#تحميل-النموذج) و [تحميل النسخة الكمية من النموذج](#النموذج-الكمي)
- استخدام IDEFICS لـ:
  - [وضع عنوان للصورة](#وضع-عنوان-للصورة)
  - [وضع عنوان للصورة بمحفز](#وضع-عنوان-للصورة-بمحفز)
  - [التحفيز القليل](#التحفيز-القليل)
  - [الإجابة على الأسئلة البصرية](#الإجابة-على-الأسئلة-البصرية)
  - [تصنيف الصور](#تصنيف-الصور)
  - [إنشاء نص موجه بالصور](#إنشاء-نص-موجه-بالصور)
- [تشغيل الاستنتاج في وضع الدفعات](#تشغيل-الاستنتاج-في-وضع-الدفعات)
- [تشغيل IDEFICS للتعليمات للاستخدام المحادثي](#IDEFICS-للإرشاد-للاستخدام-المحادثي)

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية.

```bash
pip install -q bitsandbytes sentencepiece accelerate transformers
```

<Tip>
لتشغيل الأمثلة التالية باستخدام نسخة غير كمية من نقطة تفتيش النموذج، ستحتاج إلى 20 جيجابايت على الأقل من ذاكرة GPU.
</Tip>

## تحميل النموذج

دعنا نبدأ بتحميل نقطة تفتيش معاملات النموذج البالغة 9 مليار:

```py
>>> checkpoint = "HuggingFaceM4/idefics-9b"
```

مثل نماذج المحولات الأخرى، تحتاج إلى تحميل معالج والنموذج نفسه من نقطة التفتيش.
يغلف معالج IDEFICS [`LlamaTokenizer`] ومعالج صور IDEFICS في معالج واحد للاهتمام بـ
إعداد إدخالات النص والصور للنموذج.

```py
>>> import torch

>>> from transformers import IdeficsForVisionText2Text, AutoProcessor

>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
```

سيحدد تعيين `device_map` إلى `"auto"` تلقائيًا كيفية تحميل وتخزين أوزان النموذج بالطريقة الأكثر تحسينًا
طريقة بالنظر إلى الأجهزة الموجودة.

### النموذج الكمي

إذا كانت مشكلة توفر ذاكرة GPU عالية، فيمكنك تحميل النسخة الكمية من النموذج. لتحميل النموذج و
المعالج بدقة 4 بت، قم بتمرير `BitsAndBytesConfig` إلى طريقة `from_pretrained` وسيتم ضغط النموذج
أثناء التحميل.

```py
>>> import torch
>>> from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig

>>> quantization_config = BitsAndBytesConfig(
...     load_in_4bit=True,
...     bnb_4bit_compute_dtype=torch.float16,
... )

>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> model = IdeficsForVisionText2Text.from_pretrained(
...     checkpoint,
...     quantization_config=quantization_config,
...     device_map="auto"
... )
```

الآن بعد أن قمت بتحميل النموذج بإحدى الطرق المقترحة، دعنا ننتقل إلى استكشاف المهام التي يمكنك استخدام IDEFICS لها.

## وضع عنوان للصورة
وضع عنوان للصورة هي مهمة التنبؤ بعنوان لصورة معينة. أحد التطبيقات الشائعة هو مساعدة الأشخاص ضعاف البصر
التنقل عبر مواقف مختلفة، على سبيل المثال، استكشاف محتوى الصور عبر الإنترنت.

لتوضيح المهمة، احصل على صورة لوضع عنوان لها، على سبيل المثال:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-im-captioning.jpg" alt="صورة جرو في سرير من الزهور"/>
</div>

صورة بواسطة [Hendo Wang](https://unsplash.com/@hendoo).

يقبل IDEFICS محفزات النص والصور. ومع ذلك، لوضع عنوان لصورة، لا يتعين عليك توفير محفز نصي للنموذج، فقط
إدخال الصورة المعالجة مسبقًا. بدون محفز نصي، سيبدأ النموذج في توليد النص من
رمز BOS (بداية التسلسل) وبالتالي إنشاء عنوان.

كإدخال صورة للنموذج، يمكنك استخدام إما كائن صورة (`PIL.Image`) أو عنوان URL يمكن استرداد الصورة منه.

```py
>>> prompt = [
...     "https://images.unsplash.com/photo-1583160247711-2191776b4b91?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3542&q=80",
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
جرو في سرير من الزهور
```
<Tip>

من الجيد تضمين `bad_words_ids` في الاستدعاء لـ `generate` لتجنب الأخطاء التي تظهر عند زيادة
`max_new_tokens`: حيث ستحاول النماذج توليد رمز `<image>` أو `<fake_token_around_image>` جديد عندما لا يكون هناك صورة يتم توليدها بواسطة النموذج.
يمكنك ضبطها أثناء التنفيذ كما في هذا الدليل، أو تخزينها في `GenerationConfig` كما هو موضح في دليل [استراتيجيات توليد النص](../generation_strategies).
</Tip>

## كتابة تعليق على الصورة

يمكنك توسيع كتابة تعليق على الصورة من خلال توفير نص موجه، والذي سيكمله النموذج بناءً على الصورة. دعنا نأخذ
صورة أخرى للتوضيح:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-prompted-im-captioning.jpg" alt="صورة لبرج إيفل في الليل"/>
</div>

صورة بواسطة [Denys Nevozhai](https://unsplash.com/@dnevozhai).

يمكن تمرير النص والصور الموجهة إلى معالج النموذج كقائمة واحدة لإنشاء مدخلات مناسبة.

```py
>>> prompt = [
...     "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...     "هذه صورة لـ ",
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
هذه صورة لبرج إيفل في باريس، فرنسا.
```

## التوجيه القليل

بينما يظهر IDEFICS نتائج ممتازة بدون توجيه، قد تتطلب مهمتك تنسيقًا معينًا للتعليق، أو تأتي
بقيود أو متطلبات أخرى تزيد من تعقيد المهمة. يمكن استخدام التوجيه القليل لتمكين التعلم في السياق.
من خلال توفير أمثلة في التوجيه، يمكنك توجيه النموذج لتوليد نتائج تحاكي تنسيق الأمثلة المعطاة.

دعنا نستخدم صورة برج إيفل السابقة كمثال للنموذج ونبني توجيهًا يوضح للنموذج
أنه بالإضافة إلى تعلم ما هو الشيء في الصورة، نود أيضًا الحصول على بعض المعلومات المثيرة للاهتمام حوله.
ثم دعنا نرى، إذا كان بإمكاننا الحصول على نفس تنسيق الاستجابة لصورة تمثال الحرية:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg" alt="صورة لتمثال الحرية"/>
</div>

صورة بواسطة [Juan Mayobre](https://unsplash.com/@jmayobres).

```py
>>> prompt = ["المستخدم:",
...            "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...            "وصف هذه الصورة.\nالمساعد: صورة لبرج إيفل في الليل. حقيقة ممتعة: برج إيفل بنفس ارتفاع مبنى مكون من 81 طابقًا.\n",
...            "المستخدم:",
...            "https://images.unsplash.com/photo-1524099163253-32b7f0256868?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3387&q=80",
...            "وصف هذه الصورة.\nالمساعد:"
...            ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=30, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
المستخدم: وصف هذه الصورة.
المساعد: صورة لبرج إيفل في الليل. حقيقة ممتعة: برج إيفل بنفس ارتفاع مبنى مكون من 81 طابقًا.
المستخدم: وصف هذه الصورة.
المساعد: صورة لتمثال الحرية. حقيقة ممتعة: تمثال الحرية يبلغ ارتفاعه 151 قدمًا.
```

لاحظ أنه من خلال مثال واحد فقط (أي، توجيه واحد)، تعلم النموذج كيفية أداء المهمة. بالنسبة للمهام الأكثر تعقيدًا،
يمكنك تجربة عدد أكبر من الأمثلة (مثل، 3 أمثلة، 5 أمثلة، إلخ).

## الإجابة على الأسئلة البصرية

الإجابة على الأسئلة البصرية (VQA) هي مهمة الإجابة على الأسئلة المفتوحة بناءً على صورة. مشابهة لكتابة تعليق على الصورة، يمكن استخدامها في تطبيقات إمكانية الوصول، ولكن أيضًا في التعليم (الاستدلال حول المواد المرئية)، وخدمة العملاء (الأسئلة حول المنتجات بناءً على الصور)، واسترجاع الصور.

دعنا نحصل على صورة جديدة لهذه المهمة:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-vqa.jpg" alt="صورة لزوجين في نزهة"/>
</div>

صورة بواسطة [Jarritos Mexican Soda](https://unsplash.com/@jarritos).

يمكنك توجيه النموذج من كتابة تعليق على الصورة إلى الإجابة على الأسئلة البصرية من خلال توجيهه بتعليمات مناسبة:

```py
>>> prompt = [
...     "التعليمات: قدم إجابة للسؤال. استخدم الصورة للإجابة.\n",
...     "https://images.unsplash.com/photo-1623944889288-cd147dbb517c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...     "السؤال: أين هؤلاء الأشخاص وما هو الطقس؟ الإجابة:"
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=20, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
التعليمات: قدم إجابة للسؤال. استخدم الصورة للإجابة.
السؤال: أين هؤلاء الأشخاص وما هو الطقس؟ الإجابة: إنهم في حديقة في مدينة نيويورك، والطقس جميل.
```
```
## تصنيف الصور

IDEFICS قادر على تصنيف الصور إلى فئات مختلفة دون تدريب صريح على بيانات تحتوي على 
أمثلة مصنفة من تلك الفئات المحددة. عند إعطاء قائمة بالفئات واستخدام قدراته على فهم الصور والنصوص، 
يمكن للنموذج استنتاج الفئة التي من المحتمل أن تنتمي إليها الصورة.

لنفترض أن لدينا هذه الصورة لطاولة خضروات:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-classification.jpg" alt="صورة لطاولة خضروات"/>
</div>

الصورة من [Peter Wendt](https://unsplash.com/@peterwendt).

يمكننا توجيه النموذج لتصنيف الصورة إلى واحدة من الفئات التي لدينا:

```py
>>> categories = ['animals','vegetables', 'city landscape', 'cars', 'office']
>>> prompt = [f"Instruction: Classify the following image into a single category from the following list: {categories}.\n",
...     "https://images.unsplash.com/photo-1471193945509-9ad0617afabf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",    
...     "Category: "
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=6, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
Instruction: Classify the following image into a single category from the following list: ['animals', 'vegetables', 'city landscape', 'cars', 'office'].
Category: Vegetables
```

في المثال أعلاه، نوجه النموذج لتصنيف الصورة إلى فئة واحدة، ولكن يمكنك أيضًا توجيه النموذج للقيام بتصنيف مرتب.

## توليد النص الموجه بالصور

للتطبيقات الأكثر إبداعًا، يمكنك استخدام توليد النص الموجه بالصور لإنشاء نص بناءً على صورة. يمكن أن يكون هذا مفيدًا لخلق أوصاف للمنتجات، والإعلانات، وأوصاف المشاهد، وما إلى ذلك.

دعنا نوجه IDEFICS لكتابة قصة بناءً على صورة بسيطة لباب أحمر:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-story-generation.jpg" alt="صورة لباب أحمر مع يقطين على الدرجات"/>
</div>

الصورة من [Craig Tidball](https://unsplash.com/@devonshiremedia).

```py
>>> prompt = ["Instruction: Use the image to write a story. \n",
...     "https://images.unsplash.com/photo-1517086822157-2b0358e7684a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2203&q=80",
...     "Story: \n"]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, num_beams=2, max_new_tokens=200, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0]) 
Instruction: Use the image to write a story. 
 Story: 
Once upon a time, there was a little girl who lived in a house with a red door.  She loved her red door.  It was the prettiest door in the whole world.

One day, the little girl was playing in her yard when she noticed a man standing on her doorstep.  He was wearing a long black coat and a top hat.

The little girl ran inside and told her mother about the man.

Her mother said, “Don’t worry, honey.  He’s just a friendly ghost.”

The little girl wasn’t sure if she believed her mother, but she went outside anyway.

When she got to the door, the man was gone.

The next day, the little girl was playing in her yard again when she noticed the man standing on her doorstep.

He was wearing a long black coat and a top hat.

The little girl ran
```

يبدو أن IDEFICS لاحظ اليقطين على عتبة الباب واختار قصة مخيفة عن شبح.

<Tip>

بالنسبة للنواتج الأطول مثل هذه، ستستفيد بشكل كبير من ضبط استراتيجية توليد النص. يمكن أن يساعدك هذا في 
تحسين جودة النص المولد بشكل كبير. اطلع على [استراتيجيات توليد النص](../generation_strategies) 
لمعرفة المزيد.
</Tip>

## تشغيل الاستدلال في وضع الدفعات

أوضحت جميع الأقسام السابقة IDEFICS لمثال واحد. بطريقة مشابهة جدًا، يمكنك تشغيل الاستدلال 
لمجموعة من الأمثلة عن طريق تمرير قائمة من المطالبات:

```py
>>> prompts = [
...     [   "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...         "This is an image of ",
...     ],
...     [   "https://images.unsplash.com/photo-1623944889288-cd147dbb517c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...         "This is an image of ",
...     ],
...     [   "https://images.unsplash.com/photo-1471193945509-9ad0617afabf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...         "This is an image of ",
...     ],
... ]

>>> inputs = processor(prompts, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> for i,t in enumerate(generated_text):
...     print(f"{i}:\n{t}\n") 
0:
This is an image of the Eiffel Tower in Paris, France.

1:
This is an image of a couple on a picnic blanket.

2:
This is an image of a vegetable stand.
```

## IDEFICS للمحادثات

بالنسبة لحالات الاستخدام المحادثية، يمكنك العثور على إصدارات مدربة من النموذج على 🤗 Hub: 
`HuggingFaceM4/idefics-80b-instruct` و `HuggingFaceM4/idefics-9b-instruct`.

هذه النقاط المرجعية هي نتيجة لتدريب النماذج الأساسية على مزيج من مجموعات البيانات المُشرفة والتدريب على التعليمات، 
والذي يعزز الأداء في الأسفل بينما يجعل النماذج أكثر قابلية للاستخدام في الإعدادات المحادثية.

الاستخدام والتوجيه للاستخدام المحادثي مشابه جدًا لاستخدام النماذج الأساسية:

```py
>>> import torch
>>> from transformers import IdeficsForVisionText2Text, AutoProcessor
>>> from accelerate.test_utils.testing import get_backend

>>> device, _, _ = get_backend() # تلقائيًا يكتشف نوع الجهاز الأساسي (CUDA، CPU، XPU، MPS، إلخ)
>>> checkpoint = "HuggingFaceM4/idefics-9b-instruct"
>>> model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> prompts = [
...     [
...         "User: What is in this image?",
...         "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
...         "<end_of_utterance>",

...         "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",

...         "\nUser:",
...         "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
...         "And who is that?<end_of_utterance>",

...         "\nAssistant:",
...     ],
... ]

>>> # --وضع الدفعات
>>> inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
>>> # --وضع العينة الواحدة
>>> # inputs = processor(prompts[0], return_tensors="pt").to(device)

>>> # معلمات التوليد
>>> exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> for i, t in enumerate(generated_text):
...     print(f"{i}:\n{t}\n")
```
```