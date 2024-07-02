# تكامل PEFT 

تمتد الفوائد العملية لـ PEFT إلى مكتبات Hugging Face الأخرى مثل [Diffusers](https://hf.co/docs/diffusers) و [Transformers](https://hf.co/docs/transformers). تتمثل إحدى المزايا الرئيسية لـ PEFT في أن ملف المحول الذي تم إنشاؤه بواسطة طريقة PEFT أصغر بكثير من النموذج الأصلي، مما يجعله سهل الإدارة والاستخدام لعدة محولات. يمكنك استخدام نموذج أساسي واحد مُدرب مسبقًا لمهام متعددة ببساطة عن طريق تحميل محول جديد مُدرب مسبقًا للمهام التي تقوم بحلها. أو يمكنك الجمع بين محولات متعددة مع نموذج انتشار النص إلى الصورة لإنشاء تأثيرات جديدة.

سيوضح هذا البرنامج التعليمي كيف يمكن لـ PEFT مساعدتك في إدارة المحولات في Diffusers و Transformers.

## Diffusers

Diffusers هي مكتبة الذكاء الاصطناعي للصور ومقاطع الفيديو النصية أو الصور باستخدام نماذج الانتشار. LoRA هي طريقة تدريب شائعة بشكل خاص لنماذج الانتشار لأنك يمكن أن تدرب بسرعة ومشاركة نماذج الانتشار لتوليد الصور بأنماط جديدة. لتسهيل استخدام ومحاولة استخدام عدة نماذج LoRA، يستخدم Diffusers مكتبة PEFT للمساعدة في إدارة محولات مختلفة للاستنتاج.

على سبيل المثال، قم بتحميل نموذج أساسي ثم قم بتحميل محول [artificialguybr/3DRedmond-V1](https://huggingface.co/artificialguybr/3DRedmond-V1) للاستدلال باستخدام طريقة [`load_lora_weights`](https://huggingface.co/docs/diffusers/v0.24.0/en/api/loaders/lora#diffusers.loaders.LoraLoaderMixin.load_lora_weights). تم تمكين وسيط `adapter_name` في طريقة التحميل بواسطة PEFT ويسمح لك بتعيين اسم للمحول بحيث يكون من السهل الإشارة إليه.

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "peft-internal-testing/artificialguybr__3DRedmond-V1", 
    weight_name="3DRedmond-3DRenderStyle-3DRenderAF.safetensors", 
    adapter_name="3d"
)
image = pipeline("sushi rolls shaped like kawaii cat faces").images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/test-lora-diffusers.png"/>
</div>

الآن دعونا نجرب نموذج LoRA رائع آخر، [ostris/super-cereal-sdxl-lora](https://huggingface.co/ostris/super-cereal-sdxl-lora). كل ما عليك فعله هو تحميل هذا المحول الجديد وتسميته باستخدام `adapter_name`، واستخدام طريقة [`set_adapters`](https://huggingface.co/docs/diffusers/api/loaders/unet#diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters) لتعيينه كمحول نشط حالي.

```py
pipeline.load_lora_weights(
    "ostris/super-cereal-sdxl-lora", 
    weight_name="cereal_box_sdxl_v1.safetensors", 
    adapter_name="cereal"
)
pipeline.set_adapters("cereal")
image = pipeline("sushi rolls shaped like kawaii cat faces").images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/test-lora-diffusers-2.png"/>
</div>

أخيرًا، يمكنك استدعاء طريقة [`disable_lora`](https://huggingface.co/docs/diffusers/api/loaders/unet#diffusers.loaders.UNet2DConditionLoadersMixin.disable_lora) لاستعادة النموذج الأساسي.

```py
pipeline.disable_lora()
```

تعرف على المزيد حول كيفية دعم PEFT لـ Diffusers في البرنامج التعليمي [الاستدلال باستخدام PEFT](https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference).

## المحولات

🤗 [المحولات](https://hf.co/docs/transformers) هي مجموعة من النماذج المُدربة مسبقًا لجميع أنواع المهام في جميع الطرائق. يمكنك تحميل هذه النماذج للتدريب أو الاستدلال. العديد من النماذج هي نماذج اللغة الكبيرة (LLMs)، لذلك من المنطقي دمج PEFT مع المحولات لإدارة وتدريب المحولات.

قم بتحميل نموذج أساسي مُدرب مسبقًا للتدريب.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
```

بعد ذلك، أضف تكوين المحول لتحديد كيفية تكييف معلمات النموذج. استدعاء طريقة [`~PeftModel.add_adapter`] لإضافة التكوين إلى النموذج الأساسي.

```py
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)
model.add_adapter(peft_config)
```

الآن يمكنك تدريب النموذج باستخدام فئة [`~transformers.Trainer`] من المحول أو أي إطار تدريب تفضله.

لاستخدام النموذج المدرب حديثًا للاستدلال، تستخدم فئة [`~transformers.AutoModel`] PEFT في الخلفية لتحميل أوزان المحول وملف التكوين في نموذج أساسي مُدرب مسبقًا.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("peft-internal-testing/opt-350m-lora")
```

بدلاً من ذلك، يمكنك استخدام أنابيب المحول [Pipelines](https://huggingface.co/docs/transformers/en/main_classes/pipelines) لتحميل النموذج لتشغيل الاستدلال بشكل مناسب:

```py
from transformers import pipeline

model = pipeline("text-generation", "peft-internal-testing/opt-350m-lora")
print(model("Hello World"))
```

إذا كنت مهتمًا بمقارنة أو استخدام أكثر من محول واحد، فيمكنك استدعاء طريقة [`~PeftModel.add_adapter`] لإضافة تكوين المحول إلى النموذج الأساسي. الشرط الوحيد هو أن نوع المحول يجب أن يكون هو نفسه (لا يمكنك خلط محول LoRA و LoHa).

```py
from transformers import AutoModelForCausalLM
from peft import LoraConfig

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
model.add_adapter(lora_config_1, adapter_name="adapter_1")
```

استدعاء [`~PeftModel.add_adapter`] مرة أخرى لربط محول جديد بالنموذج الأساسي.

```py
model.add_adapter(lora_config_2, adapter_name="adapter_2")
```

بعد ذلك، يمكنك استخدام [`~PeftModel.set_adapter`] لتعيين المحول النشط حاليًا.

```py
model.set_adapter("adapter_1")
output = model.generate(**inputs)
print(tokenizer.decode(output_disabled[0], skip_special_tokens=True))
```

لإلغاء تنشيط المحول، اتصل بطريقة [disable_adapters](https://github.com/huggingface/transformers/blob/4e3490f79b40248c53ee54365a9662611e880892/src/transformers/integrations/peft.py#L313).

```py
model.disable_adapters()
```

يمكن استخدام [enable_adapters](https://github.com/huggingface/transformers/blob/4e3490f79b40248c53ee54365a9662611e880892/src/transformers/integrations/peft.py#L336) لتمكين المحولات مرة أخرى.

إذا كنت فضوليًا، فتحقق من البرنامج التعليمي [تحميل وتدريب المحولات باستخدام PEFT](https://huggingface.co/docs/transformers/main/peft) لمعرفة المزيد.