# أساليب الاستدعاء المستندة إلى المطالبة

يمكن أن يصف موجه مهمة أو يقدم مثالًا على مهمة تريد من النموذج تعلمها. بدلاً من إنشاء هذه المطالبات يدويًا، تضيف أساليب الاستدعاء الناعمة معلمات قابلة للتعلم إلى تضمين الإدخال والتي يمكن تحسينها لمهمة محددة مع الحفاظ على تجميد معلمات النموذج المسبق التدريب. هذا يجعل من الأسرع والأسهل ضبط نماذج اللغة الكبيرة لمهمات المصب الجديدة.

تدعم مكتبة PEFT عدة أنواع من أساليب الاستدعاء (ضبط P، ضبط البادئة، ضبط المطالبة) ويمكنك معرفة المزيد حول كيفية عمل هذه الأساليب من الناحية المفاهيمية في الدليل [Soft prompts](../conceptual_guides/prompting). إذا كنت مهتمًا بتطبيق هذه الأساليب على مهام وحالات استخدام أخرى، فألق نظرة على مجموعتنا من [دفاتر الملاحظات](https://huggingface.co/spaces/PEFT/soft-prompting) .

سيوضح هذا الدليل كيفية تدريب نموذج لغة سببية - باستخدام طريقة الاستدعاء الناعم - لـ *توليد تصنيف* لما إذا كانت التغريدة شكوى أم لا.

<Tip>

ستكون بعض المعرفة بعملية تدريب نموذج اللغة السببية مفيدة حقًا وستسمح لك بالتركيز على أساليب الاستدعاء الناعمة. إذا كنت جديدًا، فنحن نوصي بالاطلاع على دليل [نمذجة اللغة السببية](https://huggingface.co/docs/transformers/tasks/language_modeling) أولاً من وثائق المحولات. عندما تكون مستعدًا، عد وشاهد مدى سهولة إسقاط PEFT في تدريبك!

</Tip>

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية.

```bash
pip install -q peft transformers datasets
```

## مجموعة البيانات

بالنسبة لهذا الدليل، ستستخدم مجموعة فرعية `twitter_complaints` من مجموعة بيانات [RAFT](https://huggingface.co/datasets/ought/raft) . تحتوي مجموعة البيانات الفرعية `twitter_complaints` على تغريدات مصنفة على أنها `complaint` و `no complaint` ويمكنك الاطلاع على [عارض مجموعة البيانات](https://huggingface.co/datasets/ought/raft/viewer/twitter_complaints) للحصول على فكرة أفضل عن شكل البيانات.

استخدم وظيفة [`~datasets.load_dataset`] لتحميل مجموعة البيانات وإنشاء عمود `text_label` جديد حتى يكون من السهل فهم ما تعنيه قيم `Label`، `1` و `2`.

```py
from datasets import load_dataset

ds = load_dataset("ought/raft", "twitter_complaints")

classes = [k.replace("_", " ") for k in ds["train"].features["Label"].names]
ds = ds.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
ds["train"][0]
{"Tweet text": "@HMRCcustomers No this is my first job", "ID": 0, "Label": 2, "text_label": "no complaint"}
```

قم بتحميل برنامج تشفير، وحدد رمز التعبئة الذي سيتم استخدامه، وتحديد الطول الأقصى للعلامة المميزة المعلمة.

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
print(target_max_length)
```

قم بإنشاء دالة معالجة مسبقة تقوم بترميز نص التغريدة والملصقات، وقم بتبطين الإدخالات والملصقات في كل دفعة، وقم بإنشاء قناع اهتمام، وقص التسلسلات إلى `max_length`. ثم قم بتحويل `input_ids` و `attention_mask` و `labels` إلى تنسورات PyTorch.

```py
import torch

max_length = 64

def preprocess_function(examples, text_column="Tweet text", label_column="text_label"):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    classes = [k.replace("_", " ") for k in ds["train"].features["Label"].names]
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(label_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

قم بتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها باستخدام وظيفة [`~datasets.Dataset.map`] ، وإزالة الأعمدة غير المعالجة لأن النموذج لن يحتاجها.

```py
processed_ds = ds.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)
```

أخيرًا، قم بإنشاء تدريب وتقييم [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) . يمكنك تعيين `pin_memory=True` لتسريع نقل البيانات إلى وحدة معالجة الرسومات (GPU) أثناء التدريب إذا كانت العينات في مجموعة البيانات موجودة على وحدة المعالجة المركزية (CPU).

```py
from torch.utils.data import DataLoader
from transformers import default_data_collator

train_ds = processed_ds["train"]
eval_ds = processed_ds["test"]

batch_size = 16

train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
eval_dataloader = DataLoader(eval_ds, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
```

## النموذج

الآن دعنا نحمل نموذجًا مسبق التدريب لاستخدامه كنموذج أساسي لطريقة الاستدعاء الناعم. يستخدم هذا الدليل النموذج [bigscience/bloomz-560m](https://huggingface.co/bigscience/bloomz-560m) ، ولكن يمكنك استخدام أي نموذج لغة سببية تريده.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
```

### تكوين PEFT والنموذج

بالنسبة لأي طريقة PEFT، ستحتاج إلى إنشاء تكوين يحتوي على جميع المعلمات التي تحدد كيفية تطبيق طريقة PEFT. بمجرد إعداد التكوين، مرره إلى وظيفة [`~peft.get_peft_model`] جنبًا إلى جنب مع النموذج الأساسي لإنشاء [`PeftModel`] قابل للتدريب.

<Tip>

اتصل بطريقة [`~PeftModel.print_trainable_parameters`] لمقارنة عدد المعلمات القابلة للتدريب في [`PeftModel`] مقابل عدد المعلمات في النموذج الأساسي!

</Tip>

<hfoptions id="configurations">
<hfoption id="p-tuning">

يضيف [P-tuning](../conceptual_guides/prompting#p-tuning) مصفوفة تضمين قابلة للتدريب حيث يمكن إضافة رموز الاستدعاء في أي مكان في تسلسل الإدخال. قم بإنشاء [`PromptEncoderConfig`] مع نوع المهمة، وعدد الرموز الافتراضية المراد إضافتها وتعلمها، وحجم الإخفاء المشفر لتعلم معلمات الاستدعاء.

```py
from peft import PromptEncoderConfig, get_peft_model

peft_config = PromptEncoderConfig(task_type="CAUSAL_LM", num_virtual_tokens=20, encoder_hidden_size=128)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 300,288 || all params: 559,514,880 || trainable%: 0.05366935013417338"
```

</hfoption>
<hfoption id="prefix tuning">

يضيف [Prefix tuning](../conceptual_guides/prompting#prefix-tuning) معلمات خاصة بالمهمة في جميع طبقات النموذج، والتي يتم تحسينها بواسطة شبكة تغذية أمامية منفصلة. قم بإنشاء [`PrefixTuningConfig`] مع نوع المهمة وعدد الرموز الافتراضية التي سيتم إضافتها وتعلمها.

```py
from peft import PrefixTuningConfig, get_peft_model

peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 983,040 || all params: 560,197,632 || trainable%: 0.1754809274167014"
```

</hfoption>
<hfoption id="prompt tuning">

تصيغ [Prompt tuning](../conceptual_guides/prompting#prompt-tuning) جميع المهام كمهمة *توليد* وتضيف مطالبة خاصة بالمهمة إلى الإدخال الذي يتم تحديثه بشكل مستقل. يحدد معلمة `prompt_tuning_init_text` كيفية ضبط دقة النموذج (في هذه الحالة، فهو يصنف ما إذا كانت التغريدات شكاوى أم لا). للحصول على أفضل النتائج، يجب أن يكون لـ `prompt_tuning_init_text` نفس عدد الرموز التي يجب التنبؤ بها. للقيام بذلك، يمكنك تعيين `num_virtual_tokens` إلى عدد الرموز في `prompt_tuning_init_text`.

قم بإنشاء [`PromptTuningConfig`] مع نوع المهمة، ونص ضبط المطالبة الأولي لتدريب النموذج عليه، وعدد الرموز الافتراضية التي سيتم إضافتها وتعلمها، وبرنامج الترميز.

```py
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

prompt_tuning_init_text = "Classify if the tweet is a complaint or no complaint.\n"
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),
    prompt_tuning_init_text=prompt_tuning_init_text,
    tokenizer_name_or_path="bigscience/bloomz-560m",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 8,192 || all params: 559,222,784 || trainable%: 0.0014648902430985358"
```

</hfoption>
</hfoptions>

### التدريب

قم بإعداد محسن وجدول زمني لمعدل التعلم.

```py
from transformers import get_linear_schedule_with_warmup

lr = 3e-2
num_epochs = 50

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
```

قم بنقل النموذج إلى وحدة معالجة الرسومات (GPU) وإنشاء حلقة تدريب تقوم بالإبلاغ عن الخسارة والغموض لكل حقبة.

```py
from tqdm import tqdm

device = "cuda"
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
```

## شارك نموذجك

بمجرد اكتمال التدريب، يمكنك تحميل نموذجك إلى المركز باستخدام طريقة [`~transformers.PreTrainedModel.push_to_hub`] . ستحتاج إلى تسجيل الدخول إلى حساب Hugging Face الخاص بك أولاً وإدخال رمزك عند المطالبة.

```py
from huggingface_hub import notebook_login

account = <your-hf-account-name>
peft_model_id = f"{account}/bloomz-560-m-peft-method"
model.push_to_hub(peft_model_id)
```

إذا قمت بالتحقق من حجم ملف النموذج في المستودع، فستلاحظ أنه أصغر بكثير من حجم النموذج الكامل!

<div class="flex flex-col justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/PEFT-hub-screenshot.png"/>
<figcaption class="text-center">على سبيل المثال، يبلغ حجم أوزان المحول لـ opt-350m النموذج المخزن على المركز حوالي 6 ميجابايت مقارنة بحجم النموذج الكامل الذي يمكن أن يكون حوالي 700 ميجابايت.</figcaption>
</div>
## الاستنتاج

الآن، دعنا نحمل النموذج للاستنتاج ونجربه على تغريدة!

```py
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("peft_model_id").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

i = 15
inputs = tokenizer(f'{text_column} : {ds["test"][i]["نص التغريدة"]} Label : ', return_tensors="pt")
print(ds["test"][i]["نص التغريدة"])
"@NYTsupport لقد اشتكيت أكثر من عشر مرات، ومع ذلك ما زالت أوراقي تلقى بعيدًا عن بابي. لماذا من الصعب حل هذه المشكلة؟"
```

قم بالاستدعاء [`~transformers.GenerationMixin.generate`] طريقة لتوليد تسمية التصنيف المتوقعة.

```py
with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
"['Tweet text : @NYTsupport i have complained a dozen times &amp; yet my papers are still thrown FAR from my door. Why is this so hard to resolve? Label : complaint']"
```