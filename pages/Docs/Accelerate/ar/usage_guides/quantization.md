# الضبط الكمي 

## تكامل `bitsandbytes` 

🤗 Accelerate يجلب الضبط الكمي `bitsandbytes` إلى نموذجك. يمكنك الآن تحميل أي نموذج PyTorch ب8 بتات أو 4 بتات في بضع سطور من التعليمات البرمجية. 

إذا كنت تريد استخدام نماذج 🤗 Transformers مع `bitsandbytes`، فيجب عليك اتباع هذه [الوثائق](https://huggingface.co/docs/transformers/main_classes/quantization). 

لمعرفة المزيد عن كيفية عمل الضبط الكمي `bitsandbytes`، اطلع على المنشورات على المدونة حول الضبط الكمي [8-بت](https://huggingface.co/blog/hf-bitsandbytes-integration) و [4-بت](https://huggingface.co/blog/4bit-transformers-bitsandbytes). 

### المتطلبات الأساسية 

ستحتاج إلى تثبيت المتطلبات التالية: 

- تثبيت مكتبة `bitsandbytes`

```bash
pip install bitsandbytes
```

- تثبيت أحدث إصدار من `accelerate` من المصدر

```bash
pip install git+https://github.com/huggingface/accelerate.git
```

- تثبيت `minGPT` و `huggingface_hub` لتشغيل الأمثلة

```bash
git clone https://github.com/karpathy/minGPT.git
pip install minGPT/
pip install huggingface_hub
```

### كيفية عمله 

أولاً، نحتاج إلى تهيئة نموذجنا. لتوفير الذاكرة، يمكننا تهيئة نموذج فارغ باستخدام مدير السياق [`init_empty_weights`]. 

لنأخذ نموذج GPT2 من مكتبة minGPT كمثال.

```py
from accelerate import init_empty_weights
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2-xl'
model_config.vocab_size = 50257
model_config.block_size = 1024

with init_empty_weights():
    empty_model = GPT(model_config)
``` 

بعد ذلك، نحتاج إلى الحصول على المسار إلى أوزان نموذجك. يمكن أن يكون المسار ملف حالة (مثل "pytorch_model.bin") أو مجلد يحتوي على نقاط تفتيش مجزأة. 

```py
from huggingface_hub import snapshot_download
weights_location = snapshot_download(repo_id="marcsun13/gpt2-xl-linear-sharded")
``` 

أخيرًا، تحتاج إلى تعيين تكوين الضبط الكمي الخاص بك باستخدام [`~utils.BnbQuantizationConfig`]. 

فيما يلي مثال على الضبط الكمي 8-بت:

```py
from accelerate.utils import BnbQuantizationConfig
bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, llm_int8_threshold = 6)
``` 

فيما يلي مثال على الضبط الكمي 4-بت:

```py
from accelerate.utils import BnbQuantizationConfig
bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
``` 

لضبط نموذجك الفارغ باستخدام التكوين المحدد، تحتاج إلى استخدام [`~utils.load_and_quantize_model`]. 

```py
from accelerate.utils import load_and_quantize_model
quantized_model = load_and_quantize_model(empty_model, weights_location=weights_location, bnb_quantization_config=bnb_quantization_config, device_map = "auto")
```

### حفظ وتحميل النموذج 8-بت 

يمكنك حفظ نموذج 8-بت الخاص بك باستخدام Accelerate باستخدام [`~Accelerator.save_model`]. 

```py
from accelerate import Accelerator
accelerate = Accelerator()
new_weights_location = "path/to/save_directory"
accelerate.save_model(quantized_model, new_weights_location)

quantized_model_from_saved = load_and_quantize_model(empty_model, weights_location=new_weights_location, bnb_quantization_config=bnb_quantization_config, device_map = "auto")
``` 

ملاحظة: لا يتم دعم تسلسل نموذج 4-بت حاليًا. 

### نقل الوحدات النمطية إلى وحدة المعالجة المركزية والقرص 

يمكنك نقل بعض الوحدات النمطية إلى وحدة المعالجة المركزية/القرص إذا لم يكن لديك مساحة كافية على وحدة معالجة الرسومات (GPU) لتخزين النموذج بالكامل على وحدات معالجة الرسومات الخاصة بك. 

يستخدم هذا النقل الوحدات النمطية إلى وحدة المعالجة المركزية والقرص في الاستدلال باستخدام النماذج الكبيرة. راجع هذه [الوثائق](https://huggingface.co/docs/accelerate/usage_guides/big_modeling) لمزيد من التفاصيل. 

بالنسبة للضبط الكمي 8-بت، سيتم تحويل الوحدات النمطية المحددة إلى دقة 8-بت. 

بالنسبة للضبط الكمي 4-بت، سيتم الاحتفاظ بالوحدات النمطية المحددة في `torch_dtype` التي مررها المستخدم في `BnbQuantizationConfig`. سنضيف دعمًا لتحويل هذه الوحدات النمطية المنقولة إلى 4-بت عندما يصبح التسلسل 4-بت ممكنًا. 

كل ما عليك فعله هو تمرير خريطة أجهزة مخصصة لنقل الوحدات النمطية إلى وحدة المعالجة المركزية/القرص. سيتم إرسال الوحدات النمطية المنقولة إلى وحدة معالجة الرسومات عند الحاجة. فيما يلي مثال: 

```py
device_map = {
    "transformer.wte": 0,
    "transformer.wpe": 0,
    "transformer.drop": 0,
    "transformer.h": "cpu",
    "transformer.ln_f": "disk",
    "lm_head": "disk",
}
```

### ضبط نموذج مضبوط بدقة 

لا يمكن إجراء تدريب 8 بتات أو 4 بتات نقي على هذه النماذج. ومع ذلك، يمكنك تدريب هذه النماذج من خلال الاستفادة من طرق الضبط الدقيق الفعالة للمعالم (PEFT) وتدريب محولات، على سبيل المثال، أعلى منها. يرجى الاطلاع على مكتبة [peft](https://github.com/huggingface/peft) لمزيد من التفاصيل. 

حاليًا، لا يمكنك إضافة محولات أعلى أي نموذج مضبوط. ومع ذلك، مع الدعم الرسمي للمحولات مع نماذج 🤗 Transformers، يمكنك ضبط النماذج المضبوطة. إذا كنت تريد ضبط نموذج 🤗 Transformers، فاتبع هذه [الوثائق](https://huggingface.co/docs/transformers/main_classes/quantization) بدلاً من ذلك. اطلع على هذا [العرض التوضيحي](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing) حول كيفية ضبط نموذج 🤗 Transformers 4-بت. 

ملاحظة: لا تحتاج إلى تمرير `device_map` عند تحميل النموذج للتدريب. سيقوم بتحميل النموذج تلقائيًا على وحدة معالجة الرسومات الخاصة بك. يرجى ملاحظة أنه يجب استخدام `device_map=auto` للاستدلال فقط. 

### مثال توضيحي - تشغيل GPT2 1.5b على Google Colab 

اطلع على العرض التوضيحي لـ Google Colab [demo](https://colab.research.google.com/drive/1T1pOgewAWVpR9gKpaEWw4orOrzPFb3yM?usp=sharing) لتشغيل النماذج المضبوطة على نموذج GTP2. نقطة تفتيش نموذج GPT2-1.5B هي FP32 التي تستخدم 6 جيجابايت من الذاكرة. بعد الضبط، يستخدم 1.6 جيجابايت مع وحدات 8 بتات و 1.2 جيجابايت مع وحدات 4 بتات.