لم يتم ترجمة الأجزاء التي تحتوي على أكواد برمجية أو روابط أو رموز خاصة، كما هو مطلوب:

# أساليب LoRA

طريقة شائعة لتدريب النماذج الكبيرة بكفاءة هي إدراج مصفوفات قابلة للتدريب أصغر (عادةً في كتل الاهتمام) والتي تكون عبارة عن تفكيك للوزن التفاضلي للمصفوفة التي سيتم تعلمها أثناء الضبط الدقيق. يتم تجميد مصفوفة الأوزان الأصلية للنموذج المُدرب مسبقًا، ويتم تحديث المصفوفات الأصغر فقط أثناء التدريب. يقلل هذا من عدد المعلمات القابلة للتدريب، مما يقلل من استخدام الذاكرة ووقت التدريب الذي يمكن أن يكون مكلفًا للغاية بالنسبة للنماذج الكبيرة.

هناك عدة طرق مختلفة للتعبير عن مصفوفة الأوزان على أنها تفكيك منخفض الرتبة، ولكن [التكيف منخفض الرتبة (LoRA)](../conceptual_guides/adapter#low-rank-adaptation-lora) هو الأسلوب الأكثر شيوعًا. تدعم مكتبة PEFT العديد من المتغيرات الأخرى لـ LoRA، مثل [حاصل الضرب منخفض الرتبة لهادامارد (LoHa)](../conceptual_guides/adapter#low-rank-hadamard-product-loha)، [حاصل الضرب منخفض الرتبة لكرونكر (LoKr)](../conceptual_guides/adapter#low-rank-kronecker-product-lokr)، و [التكيف منخفض الرتبة التكيفي (AdaLoRA)](../conceptual_guides/adapter#adaptive-low-rank-adaptation-adalora). يمكنك معرفة المزيد حول كيفية عمل هذه الأساليب من الناحية النظرية في الدليل [المهايئات](../conceptual_guides/adapter). إذا كنت مهتمًا بتطبيق هذه الأساليب على مهام وحالات استخدام أخرى مثل التجزئة الدلالية، أو تصنيف الرموز، فراجع [مجموعة الدفاتر](https://huggingface.co/collections/PEFT/notebooks-6573b28b33e5a4bf5b157fc1)!

سيوضح هذا الدليل كيفية تدريب نموذج تصنيف الصور بسرعة - باستخدام طريقة التفكيك منخفض الرتبة - لتحديد فئة الطعام المعروضة في الصورة.

<Tip>

ستكون بعض المعرفة بعملية تدريب نموذج تصنيف الصور مفيدة حقًا وستسمح لك بالتركيز على أساليب التفكيك منخفض الرتبة. إذا كنت جديدًا، نوصي بالاطلاع على دليل [تصنيف الصور](https://huggingface.co/docs/transformers/tasks/image_classification) أولاً من وثائق المحولات. عندما تكون مستعدًا، عد وراجع مدى سهولة إسقاط PEFT في تدريبك!

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية.

```bash
pip install -q peft transformers datasets
```

## مجموعة البيانات

في هذا الدليل، ستستخدم مجموعة بيانات [Food-101](https://huggingface.co/datasets/food101) التي تحتوي على صور لـ 101 فئة من الطعام (الق نظرة على [عارض مجموعة البيانات](https://huggingface.co/datasets/food101/viewer/default/train) للحصول على فكرة أفضل عن شكل مجموعة البيانات).

قم بتحميل مجموعة البيانات باستخدام الدالة [`~datasets.load_dataset`].

```py
from datasets import load_dataset

ds = load_dataset("food101")
```

يتم وضع علامة على كل فئة طعام برقم صحيح، لذا لجعل الأمر أكثر سهولة لفهم ما تمثله هذه الأرقام الصحيحة، ستقوم بإنشاء قاموس `label2id` و `id2label` لتعيين الرقم الصحيح إلى تسمية الفئة الخاصة به.

```py
labels = ds["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

id2label[2]
"baklava"
```

قم بتحميل معالج الصور لتصحيح حجم بكسل صور التدريب والتقييم وتطبيع قيمها.

```py
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
```

يمكنك أيضًا استخدام معالج الصور لإعداد بعض دالات التحويل للزيادة في البيانات وتغيير مقياس البكسل.

```py
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

def preprocess_train(example_batch):
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    example_batch["pixel_values"] = [
        val_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch
```

قم بتعريف مجموعات البيانات التدريبية والتحققية، واستخدم الدالة [`~datasets.Dataset.set_transform`] لتطبيق التحويلات أثناء التنقل.

```py
train_ds = ds["train"]
val_ds = ds["validation"]

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)
```

أخيرًا، ستحتاج إلى جامع بيانات لإنشاء دفعة من بيانات التدريب والتقييم وتحويل التسميات إلى كائنات `torch.tensor`.

```py
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
```

## النموذج

الآن دعنا نقوم بتحميل نموذج مُدرب مسبقًا لاستخدامه كنموذج أساسي. يستخدم هذا الدليل النموذج [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)، ولكن يمكنك استخدام أي نموذج لتصنيف الصور تريده. قم بتمرير القاموسين `label2id` و `id2label` إلى النموذج حتى يعرف كيفية تعيين التسميات الرقمية إلى تسميات الفئات الخاصة بها، ويمكنك أيضًا تمرير المعلمة `ignore_mismatched_sizes=True` إذا كنت تقوم بضبط دقيق لنقطة تفتيش تم ضبطها الدقيق بالفعل.

```py
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
```

### تكوين نموذج PEFT والنموذج

يتطلب كل أسلوب PEFT تكوينًا يحتوي على جميع المعلمات التي تحدد كيفية تطبيق أسلوب PEFT. بمجرد إعداد التكوين، مرره إلى الدالة [`~peft.get_peft_model`] جنبًا إلى جنب مع النموذج الأساسي لإنشاء نموذج قابل للتدريب [`PeftModel`].

<Tip>

استدع الدالة [`~PeftModel.print_trainable_parameters`] لمقارنة عدد المعلمات في [`PeftModel`] مقابل عدد المعلمات في النموذج الأساسي!

</Tip>

<hfoptions id="loras">

<hfoption id="LoRA">

[LoRA](../conceptual_guides/adapter#low-rank-adaptation-lora) يقوم بتفكيك مصفوفة تحديث الوزن إلى مصفوفتين أصغر. يتحدد حجم هذه المصفوفات منخفضة الرتبة بواسطة "رتبتها" أو `r`. تشير الرتبة الأعلى إلى أن النموذج لديه المزيد من المعلمات التي سيتم تدريبها، ولكنها تعني أيضًا أن النموذج لديه قدرة أكبر على التعلم. سترغب أيضًا في تحديد `target_modules` التي تحدد المكان الذي يتم فيه إدراج المصفوفات الأصغر. بالنسبة لهذا الدليل، ستستهدف المصفوفات *query* و *value* لكتل الاهتمام. من المعلمات المهمة الأخرى التي يجب تعيينها `lora_alpha` (عامل القياس)، و `bias` (ما إذا كان `none`، أو `all` أو يجب تدريب معلمات الانحياز لـ LoRA فقط)، و `modules_to_save` (الوحدات النمطية بخلاف طبقات LoRA التي سيتم تدريبها وحفظها). يمكن العثور على جميع هذه المعلمات - وأكثر - في [`LoraConfig`].

```py
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 667,493 || all params: 86,543,818 || trainable%: 0.7712775047664294"
```

</hfoption>

<hfoption id="LoHa">

[LoHa](../conceptual_guides/adapter#low-rank-hadamard-product-loha) يقوم بتفكيك مصفوفة تحديث الأوزان إلى أربع مصفوفات أصغر، ويتم دمج كل زوج من المصفوفات الأصغر مع حاصل الضرب لهادامارد. يسمح هذا لمصفوفة تحديث الأوزان بالحفاظ على نفس عدد المعلمات القابلة للتدريب عند مقارنتها بـ LoRA، ولكن برتبة أعلى (`r^2` لـ LoHA مقارنة بـ `2*r` لـ LoRA). يتحدد حجم المصفوفات الأصغر بواسطة "رتبتها" أو `r`. سترغب أيضًا في تحديد `target_modules` التي تحدد المكان الذي يتم فيه إدراج المصفوفات الأصغر. بالنسبة لهذا الدليل، ستستهدف المصفوفات *query* و *value* لكتل الاهتمام. من المعلمات المهمة الأخرى التي يجب تعيينها `alpha` (عامل القياس)، و `modules_to_save` (الوحدات النمطية بخلاف طبقات LoHa التي سيتم تدريبها وحفظها). يمكن العثور على جميع هذه المعلمات - وأكثر - في [`LoHaConfig`].

```py
from peft import LoHaConfig, get_peft_model

config = LoHaConfig(
    r=16,
    alpha=16,
    target_modules=["query", "value"],
    module_dropout=0.1,
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 1,257,317 || all params: 87,133,642 || trainable%: 1.4429753779831676"
```

</hfoption>

<hfoption id="LoKr">

[LoKr](../conceptual_guides/adapter#low-rank-kronecker-product-lokr) يعبر عن مصفوفة تحديث الأوزان على أنها تفكيك لمنتج كرونكر، مما يخلق مصفوفة كتلة قادرة على الحفاظ على رتبة مصفوفة الأوزان الأصلية. يتحدد حجم المصفوفات الأصغر بواسطة "رتبتها" أو `r`. سترغب أيضًا في تحديد `target_modules` التي تحدد المكان الذي يتم فيه إدراج المصفوفات الأصغر. بالنسبة لهذا الدليل، ستستهدف المصفوفات *query* و *value* لكتل الاهتمام. من المعلمات المهمة الأخرى التي يجب تعيينها `alpha` (عامل القياس)، و `modules_to_save` (الوحدات النمطية بخلاف طبقات LoKr التي سيتم تدريبها وحفظها). يمكن العثور على جميع هذه المعلمات - وأكثر - في [`LoKrConfig`].

```py
from peft import LoKrConfig, get_peft_model

config = LoKrConfig(
    r=16,
    alpha=16,
    target_modules=["query", "value"],
    module_dropout=0.1,
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 116,069 || all params: 87,172,042 || trainable%: 0.13314934162033282"
```

</hfoption>

<hfoption id="AdaLoRA">

[AdaLoRA](../conceptual_guides/adapter#adaptive-low-rank-adaptation-adalora) يقوم بإدارة ميزانية معلمات LoRA بكفاءة عن طريق تعيين مصفوفات الأوزان المهمة المزيد من المعلمات وتشذيب الأقل أهمية. على النقيض من ذلك، توزع LoRA المعلمات بالتساوي عبر جميع الوحدات النمطية. يمكنك التحكم في متوسط الرتبة المرغوبة أو `r` للمصفوفات، والوحدات النمطية التي سيتم تطبيق AdaLoRA عليها باستخدام `target_modules`. من المعلمات المهمة الأخرى التي يجب تعيينها `lora_alpha` (عامل القياس)، و `modules_to_save` (الوحدات النمطية بخلاف طبقات AdaLoRA التي سيتم تدريبها وحفظها). يمكن العثور على جميع هذه المعلمات - وأكثر - في [`AdaLoraConfig`].

```py
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    r=8,
    init_r=12,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 520,325 || all params: 87,614,722 || trainable%: 0.5938785036606062"
```

</hfoption>

</hfoptions>
### التدريب

للتدريب، دعنا نستخدم فئة [`~transformers.Trainer`] من Transformers. تحتوي فئة [`Trainer`] على حلقة تدريب PyTorch، وعندما تكون جاهزًا، قم بالاستدعاء [`~transformers.Trainer.train`] لبدء التدريب. ولتخصيص عملية التدريب، قم بضبط فرط معلمات التدريب في فئة [`~transformers.TrainingArguments`]. باستخدام طرق مثل LoRA، يمكنك استخدام حجم دفعة ومعدل تعلم أعلى.

> [!WARNING]
> يمتلك AdaLoRA طريقة [`~AdaLoraModel.update_and_allocate`] يجب استدعاؤها في كل خطوة تدريب لتحديث ميزانية المعلمات والقناع، وإلا فلن يتم تنفيذ خطوة التكيف. يتطلب ذلك كتابة حلقة تدريب مخصصة أو إنشاء فئة فرعية من [`~transformers.Trainer`] لإدراج هذه الطريقة. كمثال، الق نظرة على حلقة التدريب المخصصة هذه [custom training loop](https://github.com/huggingface/peft/blob/912ad41e96e03652cabf47522cd876076f7a0c4f/examples/conditional_generation/peft_adalora_seq2seq.py#L120).

```py
from transformers import TrainingArguments, Trainer

account = "stevhliu"
peft_model_id = f"{account}/google/vit-base-patch16-224-in21k-lora"
batch_size = 128

args = TrainingArguments(
peft_model_id,
remove_unused_columns=False,
evaluation_strategy="epoch",
save_strategy="epoch",
learning_rate=5e-3,
per_device_train_batch_Multiplier size=batch_size,
gradient_accumulation_steps=4,
per_device_eval_batch_size=batch_size,
fp16=True,
num_train_epochs=5,
logging_steps=10,
load_best_model_at_end=True,
label_names=["labels"],
)
```

ابدأ التدريب مع [`~transformers.Trainer.train`].

```py
trainer = Trainer(
model,
args,
train_dataset=train_ds,
eval_dataset=val_ds,
tokenizer=image_processor,
data_collator=collate_fn,
)
trainer.train()
```

## شارك نموذجك

بمجرد اكتمال التدريب، يمكنك تحميل نموذجك إلى المحول باستخدام طريقة [`~transformers.PreTrainedModel.push_to_hub`]. ستحتاج إلى تسجيل الدخول إلى حساب Hugging Face الخاص بك أولاً وإدخال رمزك عند المطالبة.

```py
from huggingface_hub import notebook_login

notebook_login()
```

اتصل بـ [`~transformers.PreTrainedModel.push_to_hub`] لحفظ نموذجك في مستودعك.

```py
model.push_to_hub(peft_model_id)
```

## الاستنتاج

دعنا نحمل النموذج من المحول ونختبره على صورة طعام.

```py
from peft import PeftConfig, PeftModel
from transformers import AutoImageProcessor
from PIL import Image
import requests

config = PeftConfig.from_pretrained("stevhliu/vit-base-patch16-224-in21k-lora")
model = AutoModelForImageClassification.from_pretrained(
config.base_model_name_or_path,
label2id=label2id,
id2label=id2label,
ignore_mismatched_sizes=True,
)
model = PeftModel.from_pretrained(model, "stevhliu/vit-base-patch16-224-in21k-lora")

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg">
</div>

قم بتحويل الصورة إلى RGB وإرجاع التنسورات PyTorch الأساسية.

```py
encoding = image_processor(image.convert("RGB"), return_tensors="pt")
```

الآن قم بتشغيل النموذج وإرجاع الفئة المتوقعة!

```py
with torch.no_grad():
outputs = model(**encoding)
logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
"Predicted class: beignets"
```