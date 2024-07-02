# LoRA

تعد طريقة LoRA طريقة تفكيك ذات رتبة منخفضة لتقليل عدد المعلمات القابلة للتدريب، مما يسرع من عملية الضبط الدقيق للنماذج الكبيرة ويقلل من استخدام الذاكرة. في PEFT، يعد استخدام LoRA سهلاً مثل إعداد [`LoraConfig`] ولفه باستخدام [`get_peft_model`] لإنشاء [`PeftModel`] قابل للتدريب.

يستكشف هذا الدليل بمزيد من التفصيل الخيارات والميزات الأخرى لاستخدام LoRA.

## التهيئة

يتم التحكم في تهيئة أوزان LoRA بواسطة معلمة `init_lora_weights` في [`LoraConfig`]. بشكل افتراضي، يقوم PEFT بتهيئة أوزان LoRA باستخدام Kaiming-uniform للوزن A والأصفار للوزن B مما يؤدي إلى تحويل هوية (نفس تنفيذ المرجع [implementation](https://github.com/microsoft/LoRA)).

من الممكن أيضًا تمرير `init_lora_weights="gaussian"`. كما يوحي الاسم، يقوم هذا بتهيئة الوزن A بتوزيع غاوسي والأصفار للوزن B (هذا هو كيفية [Diffusers](https://huggingface.co/docs/diffusers/index) تهيئة أوزان LoRA).

```py
from peft import LoraConfig

config = LoraConfig(init_lora_weights="gaussian", ...)
```

هناك أيضًا خيار لتعيين `init_lora_weights=False` والذي يكون مفيدًا للتصحيح والاختبار. يجب أن يكون هذا هو الوقت الوحيد الذي تستخدم فيه هذا الخيار. عند اختيار هذا الخيار، يتم تهيئة أوزان LoRA بحيث لا تؤدي إلى تحويل الهوية.

```py
from peft import LoraConfig

config = LoraConfig(init_lora_weights=False, ...)
```

### PiSSA

[PiSSA](https://arxiv.org/abs/2404.02948) يقوم بتهيئة محول LoRA باستخدام القيم الفردية والمجهول الفردي الرئيسي. يسمح هذا التعديل المباشر لـ PiSSA بالتقارب بشكل أسرع من LoRA وتحقيق أداء متفوق في النهاية. علاوة على ذلك، يقلل PiSSA من خطأ التقريب مقارنة بـ QLoRA، مما يؤدي إلى مزيد من التحسينات.

قم بتهيئة طريقة التهيئة إلى "pissa"، والتي قد تستغرق عدة دقائق لتنفيذ SVD على النموذج المسبق التدريب:

```python
from peft import LoraConfig
config = LoraConfig(init_lora_weights="pissa", ...)
```

أو، قم بتنفيذ SVD السريع، والذي يستغرق بضع ثوانٍ فقط. يحدد عدد التكرارات المقايضة بين الخطأ ووقت الحساب:

```python
lora_config = LoraConfig(init_lora_weights="pissa_niter_[عدد التكرارات]", ...)
```

للحصول على تعليمات مفصلة حول استخدام PiSSA، يرجى اتباع [هذه التعليمات](https://github.com/fxmeng/peft/tree/main/examples/pissa_finetuning).

### OLoRA

[OLoRA](https://arxiv.org/abs/2406.01775) يستخدم تحليل QR لتهيئة محولات LoRA. تترجم OLoRA الأوزان الأساسية للنموذج بمعامل تحليل QR الخاص بها، أي أنها تطفر الأوزان قبل إجراء أي تدريب عليها. يحسن هذا النهج بشكل كبير من الاستقرار، ويسرع سرعة التقارب، ويحقق في النهاية أداءً متفوقًا.

كل ما عليك فعله هو تمرير خيار إضافي واحد لاستخدام OLoRA:

```python
from peft import LoraConfig
config = LoraConfig(init_lora_weights="olora", ...)
```

للحصول على استخدام أكثر تقدمًا، يرجى الرجوع إلى [وثائقنا](https://github.com/huggingface/peft/tree/main/examples/olora_finetuning).

### LoftQ

#### النهج القياسي

عند تحديد كمية النموذج الأساسي للتدريب على QLoRA، يجب مراعاة استخدام [تهيئة LoftQ](https://arxiv.org/abs/2310.08659)، والتي ثبت أنها تحسن الأداء عند تدريب النماذج الكمية. الفكرة هي أن أوزان LoRA يتم تهيئتها بحيث يتم تقليل خطأ التقريب إلى الحد الأدنى. لاستخدام LoftQ، اتبع [هذه التعليمات](https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning).

بشكل عام، حتى يعمل LoftQ بشكل أفضل، يوصى باستهداف أكبر عدد ممكن من الطبقات باستخدام LoRA، نظرًا لأن تلك التي لا تستهدفها لا يمكن تطبيق LoftQ عليها. وهذا يعني أن تمرير `LoraConfig(..., target_modules="all-linear")` من المحتمل أن يعطي أفضل النتائج. أيضًا، يجب عليك استخدام `nf4` كنوع كمي في تكوين الكمية الخاصة بك عند استخدام الكمية 4 بت، أي `BitsAndBytesConfig(load_in_4bit=True، bnb_4bit_quant_type="nf4")`.

#### طريقة أكثر ملاءمة

هناك طريقة أسهل ولكنها أكثر محدودية لتطبيق تهيئة LoftQ وهي استخدام دالة الملاءمة `replace_lora_weights_loftq`. يأخذ هذا النموذج الكمي لـ PEFT كإدخال ويستبدل أوزان LoRA في مكانها بنظيراتها المبدئية لـ LoftQ.

```python
from peft import replace_lora_weights_loftq
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
base_model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)
# note: don't pass init_lora_weights="loftq" or loftq_config!
lora_config = LoraConfig(task_type="CAUSAL_LM")
peft_model = get_peft_model(base_model, lora_config)
replace_lora_weights_loftq(peft_model)
```

يسمح `replace_lora_weights_loftq` أيضًا بتمرير حجة `callback` لمنحك مزيدًا من التحكم في الطبقات التي يجب تعديلها أو عدم تعديلها، والتي يمكن أن تحسن النتائج بشكل تجريبي كثيرًا. لمشاهدة مثال أكثر تفصيلاً على ذلك، تحقق من [دفتر الملاحظات هذا](https://github.com/huggingface/peft/blob/main/examples/loftq_finetuning/LoftQ_weight_replacement.ipynb).

ينفذ `replace_lora_weights_loftq` خطوة تكرار واحدة فقط من LoftQ. وهذا يعني أنه يتم تحديث أوزان LoRA فقط، بدلاً من تحديث أوزان LoRA وأوزان النموذج الأساسي الكمي بشكل تكراري. قد يؤدي هذا إلى انخفاض الأداء ولكنه يتمتع بميزة أنه يمكننا استخدام أوزان الكمية الأصلية المستمدة من النموذج الأساسي، بدلاً من الاضطرار إلى الاحتفاظ بنسخة إضافية من أوزان الكمية المعدلة. يتوقف هذا المزايد على حالة الاستخدام.

في الوقت الحالي، لدى `replace_lora_weights_loftq` هذه القيود الإضافية:

- يجب تخزين ملفات النموذج كملف `safetensors`.
- تدعم الكمية bitsandbytes 4 بت فقط.

<Tip>
تعرف على المزيد حول كيفية عمل PEFT مع الكمية في دليل [Quantization](quantization).
</Tip>

### LoRA المستقر للرتبة

طريقة أخرى لتهيئة [`LoraConfig`] هي باستخدام طريقة [LoRA المستقرة للرتبة (rsLoRA)](https://huggingface.co/papers/2312.03732). تقوم بنية LoRA بضبط كل محول أثناء كل تمرير للأمام بمعامل ثابت يتم تعيينه عند التهيئة ويعتمد على الرتبة `r`. المعامل هو `lora_alpha/r` في التنفيذ الأصلي، ولكن يستخدم rsLoRA `lora_alpha/math.sqrt(r)` الذي يقوم بتثبيت المحولات وزيادة إمكانات الأداء من استخدام رتبة أعلى `r`.

```py
from peft import LoraConfig

config = LoraConfig(use_rslora=True, ...)
```

### تفكيك وزن DoRA منخفض الرتبة

تقنية DoRA هذه تقوم بتفكيك تحديثات الأوزان إلى جزأين، الحجم والاتجاه. يتم التعامل مع الاتجاه بواسطة LoRA العادي، في حين يتم التعامل مع الحجم بواسطة معلمة قابلة للتعلم بشكل منفصل. يمكن أن يحسن هذا أداء LoRA، خاصة عند الرتب المنخفضة. لمزيد من المعلومات حول DoRA، راجع https://arxiv.org/abs/2402.09353.

```py
from peft import LoraConfig

config = LoraConfig(use_dora=True, ...)
```

#### التحذيرات

- يدعم DoRA فقط الطبقات الخطية وConv2d في الوقت الحالي.
- يقدم DoRA Overhead أكبر من LoRA النقي، لذا يوصى بدمج الأوزان للاستدلال، راجع [`LoraModel.merge_and_unload`].
- يجب أن يعمل DoRA مع الأوزان الكمية باستخدام bitsandbytes ("QDoRA"). ومع ذلك، تم الإبلاغ عن مشكلات عند استخدام QDoRA مع DeepSpeed Zero2.

### التدريب على طريقة QLoRA

تضيف إعدادات LoRA الافتراضية في PEFT أوزانًا قابلة للتدريب إلى طبقات الاستعلام والقيمة لكل كتلة اهتمام. ولكن [QLoRA](https://hf.co/papers/2305.14314)، الذي يضيف أوزانًا قابلة للتدريب إلى جميع الطبقات الخطية لنموذج المحول، يمكن أن يوفر أداءً مساويًا لنموذج تم ضبطه بشكل كامل. لتطبيق LoRA على جميع الطبقات الخطية، مثل QLoRA، قم بتعيين `target_modules="all-linear"` (أسهل من تحديد وحدات فردية بالاسم والتي قد تختلف اعتمادًا على البنية).

```py
config = LoraConfig(target_modules="all-linear", ...)
```

### تكرار الطبقة الموفرة للذاكرة مع LoRA

تتمثل إحدى الطرق المستخدمة لتحسين أداء النماذج في توسيع نطاق النموذج عن طريق تكرار الطبقات في النموذج لبناء نموذج أكبر من نموذج مسبق التدريب بحجم معين. على سبيل المثال، زيادة نموذج 7B إلى نموذج 10B كما هو موضح في ورقة [SOLAR](https://arxiv.org/abs/2312.15166). تدعم PEFT LoRA هذا النوع من التوسع بطريقة فعالة للذاكرة تدعم المزيد من الضبط الدقيق باستخدام محولات LoRA المرفقة بالطبقات بعد تكرار الطبقات. لا تحتاج الطبقات المكررة إلى ذاكرة إضافية لأنها تشترك في الأوزان الأساسية، لذا فإن الذاكرة الإضافية الوحيدة المطلوبة هي ذاكرة أوزان المحول. لاستخدام هذه الميزة، ستقوم بإنشاء تكوين باستخدام حجة `layer_replication`.

```py
config = LoraConfig(layer_replication=[[0,4], [2,5]], ...)
```

بافتراض أن النموذج الأصلي كان يحتوي على 5 طبقات `[0، 1، 2، 3، 4]، فسيؤدي ذلك إلى إنشاء نموذج من 7 طبقات مرتبة على النحو التالي: `[0، 1، 2، 3، 2، 3، 4]`. يتبع هذا [اتفاقية الدمج pass-through merge convention](https://github.com/arcee-ai/mergekit) حيث يتم تكديس تسلسلات الطبقات المحددة كمجموعات من البداية الشاملة والنهاية الحصرية لبناء النموذج النهائي. تحصل كل طبقة في النموذج النهائي على مجموعة متميزة خاصة بها من محولات LoRA.

[Fewshot-Metamath-OrcaVicuna-Mistral-10B](https://huggingface.co/abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B) هو مثال على نموذج تم تدريبه باستخدام هذه الطريقة على Mistral-7B الموسع إلى 10B. يوضح [adapter_config.json](https://huggingface.co/abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B/blob/main/adapter_config.json) تكوين محول عينة يطبق هذه الطريقة للضبط الدقيق.
## دمج أوزان LoRA في النموذج الأساسي

على الرغم من أن LoRA أصغر وأسرع بكثير في التدريب، فقد تواجه مشكلات في الكمون أثناء الاستدلال بسبب تحميل النموذج الأساسي ومحول LoRA بشكل منفصل. وللقضاء على الكمون، استخدم وظيفة [`~LoraModel.merge_and_unload`] لدمج أوزان المحول مع النموذج الأساسي. يسمح هذا باستخدام النموذج المندمج حديثًا كنموذج مستقل. لا تحتفظ وظيفة [`~LoraModel.merge_and_unload`] بأوزان المحول في الذاكرة.

فيما يلي رسم توضيحي يوضح الحدس وراء دمج محول LoRA:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/lora_diagram.png"/>
</div>

نُظهر في المقاطع أدناه كيفية تشغيل ذلك باستخدام PEFT.

```py
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.merge_and_unload()
```

إذا كنت بحاجة إلى الاحتفاظ بنسخة من الأوزان حتى تتمكن من إلغاء دمج المحول لاحقًا أو حذف وتحميل أوزان مختلفة، فيجب استخدام وظيفة [`~LoraModel.merge_adapter`] بدلاً من ذلك. الآن لديك خيار استخدام [`~LoraModel.unmerge_adapter`] للعودة إلى النموذج الأساسي.

```py
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.merge_adapter()

# إلغاء دمج طبقات LoRA من النموذج الأساسي
model.unmerge_adapter()
```

تعد وظيفة [`~LoraModel.add_weighted_adapter`] مفيدة لدمج عدة محولات LoRA في محول جديد بناءً على مخطط الترجيح الذي يوفره المستخدم في معلمة "الأوزان". فيما يلي مثال شامل.

أولاً، قم بتحميل النموذج الأساسي:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
"mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, device_map="auto"
)
```

بعد ذلك، نقوم بتحميل المحول الأول:

```python
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id, adapter_name="sft")
```

ثم قم بتحميل محول مختلف ودمجه مع الأول:

```python
weighted_adapter_name = "sft-dpo"
model.load_adapter("alignment-handbook/zephyr-7b-dpo-lora", adapter_name="dpo")
model.add_weighted_adapter(
adapters=["sft", "dpo"],
weights=[0.7, 0.3],
adapter_name=weighted_adapter_name,
combination_type="linear"
)
model.set_adapter(weighted_adapter_name)
```

<Tip>

هناك عدة طرق مدعومة لـ `combination_type`. راجع [التوثيق](../package_reference/lora#peft.LoraModel.add_weighted_adapter) لمزيد من التفاصيل. لاحظ أن "svd" كـ `combination_type` غير مدعوم عند استخدام `torch.float16` أو `torch.bfloat16` كنوع البيانات.

</Tip>

الآن، قم بالاستدلال:

```python
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

with torch.no_grad():
generate_ids = model.generate(**inputs, max_length=30)
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(outputs)
```

## تحميل المحولات

يمكن تحميل المحولات على نموذج مُدرب مسبقًا باستخدام [`~PeftModel.load_adapter`]]، وهو ما يفيد في تجربة محولات مختلفة لا يتم دمج أوزانها. قم بتعيين أوزان المحول النشط باستخدام وظيفة [`~LoraModel.set_adapter`].

```py
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)

# تحميل محول مختلف
model.load_adapter("alignment-handbook/zephyr-7b-dpo-lora", adapter_name="dpo")

# تعيين المحول كنشط
model.set_adapter("dpo")
```

للعودة إلى النموذج الأساسي، يمكنك استخدام [`~LoraModel.unload`] لإلغاء تحميل جميع وحدات LoRA أو [`~LoraModel.delete_adapter`] لحذف المحول بالكامل.

```py
# إلغاء تحميل المحول
model.unload()

# حذف المحول
model.delete_adapter("dpo")
```

## الاستدلال باستخدام محولات LoRA مختلفة في نفس الدفعة

عادةً، يجب أن تستخدم كل دفعة استدلال نفس المحول (المحولات) في PEFT. يمكن أن يكون هذا مزعجًا في بعض الأحيان، لأنه قد تكون لدينا دفعات تحتوي على عينات يُقصد استخدامها بمحولات LoRA مختلفة. على سبيل المثال، قد يكون لدينا نموذج أساسي يعمل بشكل جيد باللغة الإنجليزية ومحولان إضافيان، أحدهما للفرنسية والآخر للألمانية. عادةً، سيتعين علينا تقسيم دفعاتنا بحيث تحتوي كل دفعة فقط على عينات لإحدى اللغات، ولا يمكننا الجمع بين لغات مختلفة في نفس الدفعة.

لحسن الحظ، من الممكن مزج محولات LoRA المختلفة في نفس الدفعة باستخدام حجة `adapter_name`. أدناه، نُظهر مثالًا على كيفية عمل ذلك في الممارسة العملية. أولاً، دعنا نحمل النموذج الأساسي والمحولين، الإنجليزي والفرنسي والألماني، بهذه الطريقة:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_id = ...
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)
# تحميل محول LoRA للفرنسية
peft_model = PeftModel.from_pretrained(model, <path>, adapter_name="adapter_fr")
# بعد ذلك، قم بتحميل محول LoRA للألمانية
peft_model.load_adapter(<path>, adapter_name="adapter_de")
```

الآن، نريد إنشاء نص على عينة تحتوي على اللغات الثلاث: العينات الثلاث الأولى باللغة الإنجليزية، والثلاثة التالية باللغة الفرنسية، والثلاثة الأخيرة باللغة الألمانية. يمكننا استخدام حجة `adapter_names` لتحديد أي محول يجب استخدامه لكل عينة. نظرًا لأن نموذجنا الأساسي يستخدم اللغة الإنجليزية، فإننا نستخدم السلسلة الخاصة "__base__" لهذه العينات. وبالنسبة للعينات الثلاث التالية، نشير إلى اسم محول الضبط الدقيق لـ LoRA باللغة الفرنسية، في هذه الحالة "adapter_fr". وبالنسبة للعينات الثلاث الأخيرة، نشير إلى اسم محول الضبط الدقيق لـ LoRA باللغة الألمانية، في هذه الحالة "adapter_de". بهذه الطريقة، يمكننا استخدام النموذج الأساسي والمحولين في دفعة واحدة.

```python
inputs = tokenizer(
[
"Hello, my dog is cute",
"Hello, my cat is awesome",
"Hello, my fish is great",
"Salut, mon chien est mignon",
"Salut, mon chat est génial",
"Salut, mon poisson est super",
"Hallo, mein Hund ist süß",
"Hallo, meine Katze ist toll",
"Hallo, mein Fisch ist großartig",
],
return_tensors="pt",
padding=True,
)

adapter_names = [
"__base__", "__base__", "__base__",
"adapter_fr", "adapter_fr", "adapter_fr",
"adapter_de", "adapter_de", "adapter_de",
]
output = peft_model.generate(**inputs, adapter_names=adapter_names, max_new_tokens=20)
```

لاحظ أن الترتيب لا يهم هنا، أي أن العينات في الدفعة لا تحتاج إلى تجميعها حسب المحول كما هو موضح في المثال أعلاه. نحتاج فقط إلى التأكد من أن حجة `adapter_names` متوافقة بشكل صحيح مع العينات.

### التحذيرات

يشتمل استخدام هذه الميزة على بعض العيوب، وهي:

- تعمل فقط للاستدلال، وليس للتدريب.
- تعطيل المحولات باستخدام سياق `with model.disable_adapter()` له الأسبقية على `adapter_names`.
- لا يمكنك تمرير `adapter_names` عندما يتم دمج بعض أوزان المحول مع الوزن الأساسي باستخدام طريقة `merge_adapter`. يرجى إلغاء دمج جميع المحولات أولاً عن طريق استدعاء `model.unmerge_adapter()`.
- لأسباب واضحة، لا يمكن استخدام هذا بعد استدعاء `merge_and_unload()`، نظرًا لأن جميع محولات LoRA ستدمج في الأوزان الأساسية في هذه الحالة.
- لا تعمل هذه الميزة حاليًا مع DoRA، لذا قم بتعيين `use_dora=False` في تكوين LoRA الخاص بك إذا كنت تريد استخدامها.
- هناك تكلفة عامة متوقعة للاستدلال باستخدام `adapter_names`، خاصة إذا كان عدد المحولات المختلفة في الدفعة مرتفعًا. ويرجع ذلك إلى أن حجم الدفعة يقل فعليًا إلى عدد العينات لكل محول. إذا كانت أداء التشغيل هو أولويتك القصوى، فجرّب ما يلي:
- زيادة حجم الدفعة.
- حاول تجنب وجود عدد كبير من المحولات المختلفة في نفس الدفعة، ويفضل الدفعات المتجانسة. يمكن تحقيق ذلك عن طريق تخزين عينات ذات محول مماثل والقيام بالاستدلال فقط مع عدد قليل من المحولات المختلفة.
- القاء نظرة على التنفيذ البديل مثل [LoRAX](https://github.com/predibase/lorax)، [punica](https://github.com/punica-ai/punica)، أو [S-LoRA](https://github.com/S-LoRA/S-LoRA)، والتي تتخصص في العمل مع عدد كبير من المحولات المختلفة.