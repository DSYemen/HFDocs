# الضبط الكمي 

يمثل الضبط الكمي البيانات بعدد أقل من البتات، مما يجعله تقنية مفيدة لخفض استخدام الذاكرة وتسريع الاستدلال خاصة عندما يتعلق الأمر بالنماذج اللغوية الكبيرة. هناك عدة طرق لضبط نموذج ما، بما في ذلك:

- تحسين الأوزان النموذجية المضبوطة باستخدام خوارزمية AWQ
- الضبط الكمي المستقل لكل صف في مصفوفة الأوزان باستخدام خوارزمية GPTQ
- الضبط الكمي إلى دقة 8 بت و4 بت باستخدام مكتبة bitsandbytes
- الضبط الكمي إلى دقة منخفضة تصل إلى 2 بت باستخدام خوارزمية AQLM

ومع ذلك، بعد ضبط نموذج ما، لا يتم عادةً تدريبه بشكل إضافي على مهام التدفق السفلي لأن التدريب يمكن أن يكون غير مستقر بسبب انخفاض دقة الأوزان والتنشيطات. ولكن نظرًا لأن طرق PEFT تضيف فقط معلمات تدريبية إضافية، فيمكنك تدريب نموذج مضبوط مع محول PEFT في الأعلى! يمكن أن يكون الجمع بين الضبط الكمي وPEFT استراتيجية جيدة لتدريب حتى أكبر النماذج على وحدة GPU واحدة. على سبيل المثال، QLoRA هي طريقة تضبط نموذجًا إلى 4 بت ثم تدربه باستخدام LoRA. تتيح هذه الطريقة ضبط دقيق لنموذج معلمات 65B على وحدة GPU واحدة بسعة 48 جيجابايت!

في هذا الدليل، ستتعلم كيفية ضبط نموذج إلى 4 بت وتدريبه باستخدام LoRA.

## ضبط نموذج

bitsandbytes هي مكتبة ضبط كمي مع تكامل المحولات. مع هذا التكامل، يمكنك ضبط نموذج إلى 8 أو 4 بت وتمكين العديد من الخيارات الأخرى عن طريق تكوين فئة ~transformers.BitsAndBytesConfig. على سبيل المثال، يمكنك:

- تعيين load_in_4bit=True لضبط النموذج إلى 4 بت عند تحميله
- تعيين bnb_4bit_quant_type="nf4" لاستخدام نوع بيانات خاص بـ 4 بت للأوزان المبدئية من توزيع طبيعي
- تعيين bnb_4bit_use_double_quant=True لاستخدام مخطط ضبط متداخل لضبط الأوزان المضبوطة بالفعل
- تعيين bnb_4bit_compute_dtype=torch.bfloat16 لاستخدام bfloat16 للحساب الأسرع

```py
import torch
from transformers import BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

مرر config إلى طريقة ~transformers.AutoModelForCausalLM.from_pretrained.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=config)
```

بعد ذلك، يجب استدعاء وظيفة ~peft.utils.prepare_model_for_kbit_training لمعالجة النموذج المضبوط للتدريب.

```py
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
```

الآن بعد أن أصبح النموذج المضبوط جاهزًا، دعنا نقوم بإعداد تكوين.

## تكوين Lora

قم بإنشاء تكوين Lora باستخدام المعلمات التالية (أو اختر المعلمات الخاصة بك):

```py
from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

ثم استخدم وظيفة get_peft_model لإنشاء نموذج PeftModel من النموذج المضبوط والتكوين.

```py
from peft import get_peft_model

model = get_peft_model(model, config)
```

أنت مستعد الآن للتدريب باستخدام طريقة التدريب التي تفضلها!

### تهيئة LoftQ

تضبط LoftQ أوزان LoRA بحيث يتم تقليل خطأ الضبط الكمي، ويمكن أن يحسن الأداء عند تدريب النماذج المضبوطة. للبدء، اتبع هذه التعليمات.

بشكل عام، من أجل عمل LoftQ بشكل أفضل، يوصى باستهداف أكبر عدد ممكن من الطبقات باستخدام LoRA، حيث لا يمكن تطبيق LoftQ على تلك التي لا تستهدفها. وهذا يعني أن تمرير LoraConfig(..., target_modules="all-linear") من المحتمل أن يعطي أفضل النتائج. أيضًا، يجب استخدام "nf4" كنوع ضبط في تكوين الضبط الكمي عند استخدام الضبط الكمي لـ 4 بت، أي BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4").

### التدريب على طريقة QLoRA

يضيف QLoRA أوزانًا قابلة للتدريب إلى جميع الطبقات الخطية في بنية المحول. نظرًا لأن أسماء السمات لهذه الطبقات الخطية يمكن أن تختلف عبر البنى، قم بتعيين target_modules إلى "all-linear" لإضافة LoRA إلى جميع الطبقات الخطية:

```py
config = LoraConfig(target_modules="all-linear", ...)
```

## ضبط AQLM الكمي

Additive Quantization of Language Models (AQLM) هي طريقة لضغط النماذج اللغوية الكبيرة. فهو يضبط كميًا أوزان متعددة معًا ويستفيد من الترابط بينها. تمثل AQLM مجموعات من 8-16 وزنًا كمجموع رموز متجه متعددة. يتيح ذلك ضغط النماذج إلى دقة منخفضة تصل إلى 2 بت بخسائر دقة منخفضة جدًا.

نظرًا لأن عملية الضبط الكمي لـ AQLM مكلفة من الناحية الحسابية، يوصى باستخدام النماذج المضبوطة مسبقًا. يمكن العثور على قائمة جزئية بالنماذج المتاحة في مستودع AQLM الرسمي.

تدعم النماذج ضبط محول LoRA. لضبط النموذج المضبوط، ستحتاج إلى تثبيت مكتبة الاستدلال AQLM: pip install aqlm>=1.0.2. يجب حفظ محولات LoRA المضبوطة بشكل منفصل، حيث لا يمكن دمجها مع أوزان AQLM المضبوطة.

```py
quantized_model = AutoModelForCausalLM.from_pretrained(
    "BlackSamorez/Mixtral-8x7b-AQLM-2Bit-1x16-hf-test-dispatch",
    torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True,
)

peft_config = LoraConfig(...)

quantized_model = get_peft_model(quantized_model, peft_config)
```

يمكنك الرجوع إلى مثال Google Colab للحصول على نظرة عامة حول ضبط AQLM+LoRA الدقيق.

## ضبط EETQ الكمي

يمكنك أيضًا إجراء ضبط دقيق لـ LoRA على النماذج المضبوطة كميًا باستخدام EETQ. توفر حزمة EETQ طريقة بسيطة وفعالة لأداء الضبط الكمي 8 بت، والتي يُزعم أنها أسرع من خوارزمية LLM.int8(). أولاً، تأكد من أن لديك إصدار Transformers متوافق مع EETQ (عن طريق تثبيته من أحدث إصدار pypi أو من المصدر).

```py
import torch
from transformers import EetqConfig

config = EetqConfig("int8")
```

مرر config إلى طريقة ~transformers.AutoModelForCausalLM.from_pretrained.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=config)
```

وأنشئ تكوين Lora ومرره إلى get_peft_model:

```py
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
```

## ضبط HQQ الكمي

تدعم النماذج التي يتم ضبطها باستخدام Half-Quadratic Quantization of Large Machine Learning Models (HQQ) ضبط محول LoRA. لضبط النموذج المضبوط، ستحتاج إلى تثبيت مكتبة hqq باستخدام: pip install hqq.

```py
from hqq.engine.hf import HQQModelForCausalLM

quantized_model = HQQModelForCausalLM.from_quantized(save_dir_or_hfhub, device='cuda')

peft_config = LoraConfig(...)

quantized_model = get_peft_model(quantized_model, peft_config)
```

أو باستخدام إصدار Transformers المتوافق مع HQQ (عن طريق تثبيته من أحدث إصدار pypi أو من المصدر).

```python
from transformers import HqqConfig, AutoModelForCausalLM

quant_config = HqqConfig(nbits=4, group_size=64)

quantized_model = AutoModelForCausalLM.from_pretrained(save_dir_or_hfhub, device='cuda', quantization_config=quant_config)

peft_config = LoraConfig(...)

quantized_model = get_peft_model(quantized_model, peft_config)
```

## الخطوات التالية

إذا كنت مهتمًا بمعرفة المزيد عن الضبط الكمي، فقد يكون ما يلي مفيدًا:

- تعرف على المزيد من التفاصيل حول QLoRA وتحقق من بعض المعايير المرجعية لتأثيرها في منشور المدونة "جعل النماذج اللغوية كبيرة الحجم أكثر سهولة في الوصول باستخدام bitsandbytes، والضبط الكمي 4 بت، وQLoRA".
- اقرأ المزيد حول مخططات الضبط الكمي المختلفة في دليل الضبط الكمي في Transformers.