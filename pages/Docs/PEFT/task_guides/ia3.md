# IA3

[IA3](../conceptual_guides/ia3) تضرب تنشيطات النموذج (المفاتيح والقيم في كتل الاهتمام الذاتي والتشفير وفك التشفير، والتنشيط الوسيط لشبكة التغذية الأمامية الموضعية) في ثلاثة متجهات متعلمة. وتُدخل طريقة PEFT هذه عددًا أقل من المعلمات القابلة للتدريب من طريقة LoRA التي تُدخل مصفوفات الأوزان بدلاً من المتجهات. ويتم الاحتفاظ بمعلمات النموذج الأصلي مجمدة ويتم تحديث هذه المتجهات فقط. ونتيجة لذلك، فهو أسرع وأرخص وأكثر كفاءة في الضبط الدقيق لمهمة جديدة لأسفل البث.

سيوضح لك هذا الدليل كيفية تدريب نموذج تسلسل إلى تسلسل باستخدام IA3 لـ *توليد المشاعر* نظرًا لبعض الأخبار المالية.

<Tip>

ستكون بعض المعرفة بعملية التدريب العامة لتسلسل إلى تسلسل مفيدة حقًا وستسمح لك بالتركيز على كيفية تطبيق IA3. إذا كنت جديدًا، فإننا نوصي بالاطلاع على أدلة [الترجمة](https://huggingface.co/docs/transformers/tasks/translation) و[التلخيص](https://huggingface.co/docs/transformers/tasks/summarization) أولاً من وثائق Transformers. عندما تكون مستعدًا، عد وشاهد مدى سهولة إسقاط PEFT في تدريبك!

</Tip>

## مجموعة البيانات

ستستخدم مجموعة البيانات الفرعية sentences_allagree من مجموعة بيانات [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank). تحتوي هذه المجموعة الفرعية على أخبار مالية بنسبة 100% من اتفاق المُصنِّف على تسمية المشاعر. الق نظرة على [عارض مجموعة البيانات](https://huggingface.co/datasets/financial_phrasebank/viewer/sentences_allagree) للحصول على فكرة أفضل عن البيانات والجمل التي ستعمل معها.

قم بتحميل مجموعة البيانات باستخدام دالة [`~datasets.load_dataset`]. تحتوي هذه المجموعة الفرعية من مجموعة البيانات على قسم تدريب فقط، لذا استخدم دالة [`~datasets.train_test_split`] لإنشاء قسمي تدريب وتحقق. قم بإنشاء عمود `text_label` جديد حتى يكون من السهل فهم ما تعنيه قيم `label` `0` و`1` و`2`.

```py
from datasets import load_dataset

ds = load_dataset("financial_phrasebank", "sentences_allagree")
ds = ds["train"].train_test_split(test_size=0.1)
ds["validation"] = ds["test"]
del ds["test"]

classes = ds["train"].features["label"].names
ds = ds.map(
lambda x: {"text_label": [classes[label] for label in x["label"]]},
batched=True,
num_proc=1,
)

ds["train"][0]
{'sentence': 'It will be operated by Nokia , and supported by its Nokia NetAct network and service management system .',
'label': 1,
'text_label': 'neutral'}
```

قم بتحميل برنامج Tokenizer وإنشاء دالة معالجة مسبقة تقوم بما يلي:

1. يرمز المدخلات، ويوسع التسلسل ويقطعه إلى `max_length`
2. تطبيق نفس برنامج Tokenizer على التسميات ولكن مع `max_length` أقصر الذي يتوافق مع التسمية
3. قناع رموز الحشو

```py
from transformers import AutoTokenizer

text_column = "sentence"
label_column = "text_label"
max_length = 128

tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-large")

def preprocess_function(examples):
inputs = examples[text_column]
targets = examples[label_column]
model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
labels = tokenizer(targets, max_length=3, padding="max_length", truncation=True, return_tensors="pt")
labels = labels["input_ids"]
labels[labels == tokenizer.pad_token_id] = -100
model_inputs["labels"] = labels
return model_inputs
```

استخدم دالة [`~datasets.Dataset.map`] لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها.

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

قم بإنشاء [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) للتدريب والتقييم، وقم بتعيين `pin_memory=True` لتسريع نقل البيانات إلى وحدة معالجة الرسومات (GPU) أثناء التدريب إذا كانت عينات مجموعة البيانات الخاصة بك على وحدة المعالجة المركزية (CPU).

```py
from torch.utils.data import DataLoader
from transformers import default_data_collator

train_ds = processed_ds["train"]
eval_ds = processed_ds["validation"]

batch_size = 8

train_dataloader = DataLoader(
train_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_ds, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
```

## النموذج

الآن يمكنك تحميل نموذج مُدرب مسبقًا لاستخدامه كنموذج أساسي لـ IA3. يستخدم هذا الدليل النموذج [bigscience/mt0-large](https://huggingface.co/bigscience/mt0-large)، ولكن يمكنك استخدام أي نموذج تسلسل إلى تسلسل تريده.

```py
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")
```

### تكوين PEFT والنموذج

تحتاج جميع طرق PEFT إلى تكوين يحتوي على جميع المعلمات ويحدد كيفية تطبيق طريقة PEFT. قم بإنشاء [`IA3Config`] مع نوع المهمة وتعيين وضع الاستدلال إلى `False`. يمكنك العثور على معلمات إضافية لهذا التكوين في [مرجع API](../package_reference/ia3#ia3config).

<Tip>

اتصل بطريقة [`~PeftModel.print_trainable_parameters`] لمقارنة عدد المعلمات القابلة للتدريب في [`PeftModel`] مقابل عدد المعلمات في النموذج الأساسي!

</Tip>

بمجرد إعداد التكوين، مرره إلى دالة [`get_peft_model`] جنبًا إلى جنب مع النموذج الأساسي لإنشاء [`PeftModel`] قابل للتدريب.

```py
from peft import IA3Config, get_peft_model

peft_config = IA3Config(task_type="SEQ_2_SEQ_LM")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 282,624 || all params: 1,229,863,936 || trainable%: 0.022980103060766553"
```

### التدريب

قم بإعداد مُحسن ومُخطط معدل التعلم.

```py
import torch
from transformers import get_linear_schedule_with_warmup

lr = 8e-3
num_epochs = 3

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
optimizer=optimizer,
num_warmup_steps=0,
num_training_steps=(len(train_dataloader) * num_epochs),
)
```

انقل النموذج إلى وحدة معالجة الرسومات (GPU) وأنشئ حلقة تدريب تبلغ عن الخسارة والغموض لكل حقبة.

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

بعد اكتمال التدريب، يمكنك تحميل نموذجك إلى Hub باستخدام طريقة [`~transformers.PreTrainedModel.push_to_hub`]. ستحتاج إلى تسجيل الدخول إلى حساب Hugging Face الخاص بك أولاً وإدخال رمزك عند المطالبة.

```py
from huggingface_hub import notebook_login

account = <your-hf-account-name>
peft_model_id = f"{account}/mt0-large-ia3"
model.push_to_hub(peft_model_id)
```

## الاستدلال

لتحميل النموذج للاستدلال، استخدم طريقة [`~AutoPeftModelForSeq2SeqLM.from_pretrained`]. دعونا نحمل أيضًا جملة من الأخبار المالية من مجموعة البيانات لتوليد مشاعر لها.

```py
from peft import AutoPeftModelForSeq2SeqLM

model = AutoPeftModelForSeq2SeqLM.from_pretrained("<your-hf-account-name>/mt0-large-ia3").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-large")

i = 15
inputs = tokenizer(ds["validation"][text_column][i], return_tensors="pt")
print(ds["validation"][text_column][i])
"The robust growth was the result of the inclusion of clothing chain Lindex in the Group in December 2007 ."
```

اتصل بطريقة [`~transformers.GenerationMixin.generate`] لتوليد تسمية المشاعر المتوقعة.

```py
with torch.no_grad():
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
['positive']
```