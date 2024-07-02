# دمج النماذج

إن تدريب نموذج لكل مهمة يمكن أن يكون مكلفًا، ويستهلك مساحة تخزين، ولا تتمكن النماذج من تعلم معلومات جديدة لتحسين أدائها. ويمكن للتعلم متعدد المهام التغلب على بعض هذه القيود من خلال تدريب نموذج على تعلم عدة مهام، ولكنه مكلف في التدريب، ويصعب تصميم مجموعة بيانات له. ويوفر *دمج النماذج* حلاً لهذه التحديات من خلال دمج عدة نماذج مُدربة مسبقًا في نموذج واحد، مما يمنحه القدرات المجمعة لكل نموذج فردي دون أي تدريب إضافي.

تقدم PEFT عدة طرق لدمج النماذج مثل الجمع الخطي أو SVD. ويركز هذا الدليل على طريقتين أكثر كفاءة لدمج محولات LoRA عن طريق إزالة المعلمات الزائدة:

* [TIES](https://hf.co/papers/2306.01708) - TrIm, Elect, and Merge (TIES) هي طريقة من ثلاث خطوات لدمج النماذج. أولاً، يتم تقليم المعلمات الزائدة، ثم يتم حل التناقضات في الإشارات إلى متجه مجمع، وأخيرًا يتم حساب متوسط المعلمات التي تكون إشاراتها هي نفسها كإشارة المجمع. وتأخذ هذه الطريقة في الاعتبار أن بعض القيم (الزائدة واختلاف الإشارة) يمكن أن تضعف الأداء في النموذج المدمج.

* [DARE](https://hf.co/papers/2311.03099) - Drop And REscale هي طريقة يمكن استخدامها للإعداد لطرق دمج النماذج الأخرى مثل TIES. تعمل عن طريق إسقاط المعلمات بشكل عشوائي وفقًا لمعدل الإسقاط وإعادة ضبط المعلمات المتبقية. وهذا يساعد على تقليل عدد المعلمات الزائدة والمتداخلة المحتملة بين عدة نماذج.

يتم دمج النماذج باستخدام طريقة [`~LoraModel.add_weighted_adapter`]، ويتم تحديد طريقة دمج النماذج المحددة في معلمة `combination_type`.

## طريقة الدمج

مع TIES و DARE، يتم تمكين الدمج عن طريق تعيين `combination_type` و `density` إلى قيمة أوزان الإبقاء من النماذج الفردية. على سبيل المثال، دعنا نقوم بدمج ثلاثة نماذج تمت معايرتها بدقة [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) : [tinyllama_lora_nobots](https://huggingface.co/smangrul/tinyllama_lora_norobots)، [tinyllama_lora_sql](https://huggingface.co/smangrul/tinyllama_lora_sql)، و [tinyllama_lora_adcopy](https://huggingface.co/smangrul/tinyllama_lora_adcopy).

<Tip warninig={true}>

عند محاولة دمج النماذج المدربة بالكامل باستخدام TIES، يجب أن تكون على دراية بأي رموز خاصة قد يكون قد أضافها كل نموذج إلى طبقة التضمين والتي ليست جزءًا من مفردات نقطة التحقق الأصلية. وقد يتسبب ذلك في حدوث مشكلة لأن كل نموذج قد أضاف رمزًا خاصًا إلى نفس موضع التضمين. إذا كان الأمر كذلك، فيجب استخدام طريقة [`~transformers.PreTrainedModel.resize_token_embeddings`] لتجنب دمج الرموز الخاصة في نفس مؤشر التضمين.

<br>

لن يكون هذا مشكلة إذا كنت تقوم فقط بدمج محولات LoRA المدربة من نفس النموذج الأساسي.

</Tip>

قم بتحميل نموذج أساسي ويمكنك استخدام طريقة [`~PeftModel.load_adapter`] لتحميل وتعيين اسم لكل محول:

```py
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

config = PeftConfig.from_pretrained("smangrul/tinyllama_lora_norobots")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_4bit=True, device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("smangrul/tinyllama_lora_norobots")

model = PeftModel.from_pretrained(model, "smangrul/tinyllama_lora_norobots", adapter_name="norobots")
_ = model.load_adapter("smangrul/tinyllama_lora_sql", adapter_name="sql")
_ = model.load_adapter("smangrul/tinyllama_lora_adcopy", adapter_name="adcopy")
```

قم بتعيين المحولات والأوزان و`adapter_name` و`combination_type` و`density` باستخدام طريقة [`~LoraModel.add_weighted_adapter`].

<hfoptions id="merge-method">

<hfoption id="TIES">

عادةً ما تنتج قيم الأوزان الأكبر من `1.0` نتائج أفضل لأنها تحافظ على المقياس الصحيح. وقيمة البداية الافتراضية الجيدة للأوزان هي تعيين جميع القيم على `1.0`.

```py
adapters = ["norobots", "adcopy", "sql"]
weights = [2.0, 1.0, 1.0]
adapter_name = "merge"
density = 0.2
model.add_weighted_adapter(adapters، الأوزان، adapter_name، combination_type="ties"، density=density)
```

</hfoption>

<hfoption id="DARE">

```py
adapters = ["norobots", "adcopy", "sql"]
weights = [2.0, 0.3, 0.7]
adapter_name = "merge"
density = 0.2
model.add_weighted_adapter(adapters، الأوزان، adapter_name، combination_type="dare_ties"، density=density)
```

</hfoption>

</hfoptions>

قم بتعيين النموذج المدمج حديثًا كنموذج نشط باستخدام طريقة [`~LoraModel.set_adapter`].

```py
model.set_adapter("merge")
```

الآن يمكنك استخدام النموذج المدمج كنموذج مضبوط للتعليمات لكتابة إعلان أو استعلامات SQL!

<hfoptions id="ties">

<hfoption id="instruct">

```py
messages = [
{"role": "user", "content": "Write an essay about Generative AI."},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.2, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0]))
```

</hfoption>

<hfoption id="ad copy">

```py
messages = [
{"role": "system", "content": "Create a text ad given the following product and description."},
{"role": "user", "content": "Product: Sony PS5 PlayStation Console\nDescription: The PS5 console unleashes new gaming possibilities that you never anticipated."},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=0.95, temperature=0.2, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0]))
```

</hfoption>

<hfoption id="SQL">

```py
text = """Table: 2-11365528-2
Columns: ['Team', 'Head Coach', 'President', 'Home Ground', 'Location']
Natural Query: Who is the Head Coach of the team whose President is Mario Volarevic?
SQL Query:"""

inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1, eos_token_id=tokenizer("</s>").input_ids[-1])
print(tokenizer.decode(outputs[0]))
```

</hfoption>

</hfoptions>

## دمج نماذج (IA) ³

تسهل نماذج (IA) ³ الدمج الخطي للمحولات. لدمج المحولات في نموذج (IA) ³، استخدم طريقة `add_weighted_adapter` من فئة `IA3Model`. هذه الطريقة مماثلة لطريقة `add_weighted_adapter` المستخدمة في `LoraModel`، مع الاختلاف الرئيسي هو عدم وجود معلمة `combination_type`. على سبيل المثال، لدمج ثلاثة محولات (IA) ³ في نموذج PEFT، ستتبع ما يلي:

```py
adapters = ["adapter1", "adapter2", "adapter3"]
weights = [0.4, 0.3, 0.3]
adapter_name = "merge"
model.add_weighted_adapter(adapters، الأوزان، adapter_name)
```

من المستحسن أن يصل مجموع الأوزان إلى 1.0 للحفاظ على مقياس النموذج. ويمكن بعد ذلك تعيين النموذج المدمج كنموذج نشط باستخدام طريقة `set_adapter`:

```py
model.set_adapter("merge")
```