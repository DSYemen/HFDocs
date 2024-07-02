# بدء التدريب متعدد وحدات معالجة الرسوميات (GPU) من بيئة Jupyter

يعلّمك هذا البرنامج التعليمي كيفية ضبط نموذج رؤية حاسوبية باستخدام HuggingFace Accelerate من Jupyter Notebook على نظام موزع. ستتعلم أيضًا كيفية إعداد بعض المتطلبات اللازمة لضمان تهيئة بيئتك بشكل صحيح، وإعداد بياناتك بشكل صحيح، وأخيرًا كيفية بدء التدريب.

## تكوين البيئة

قبل إجراء أي تدريب، يجب أن يكون هناك ملف تكوين HuggingFace Accelerate موجودًا في النظام. عادةً يمكن القيام بذلك عن طريق تشغيل ما يلي في المحطة الطرفية والإجابة عن المطالبات:

```bash
accelerate config
```

ومع ذلك، إذا كانت الإعدادات الافتراضية العامة مناسبة لك وكنت لا تعمل على وحدة معالجة الرسوميات (TPU)، فيمكن لـ HuggingFace Accelerate إنشاء تكوين وحدة معالجة الرسوميات (GPU) بسرعة من خلال `utils.write_basic_config`.

ستقوم الشفرة التالية بإعادة تشغيل Jupyter بعد كتابة التكوين، نظرًا لاستدعاء رمز CUDA لأداء هذه المهمة.

> **تحذير:** لا يمكن تهيئة CUDA أكثر من مرة على نظام متعدد وحدات معالجة الرسوميات (GPU). من الجيد إجراء التصحيح في الدفتر والاحتفاظ بمكالمات إلى CUDA، ولكن من أجل التدريب النهائي، يجب إجراء تنظيف وإعادة تشغيل كاملين.

```python
import os
from accelerate.utils import write_basic_config

write_basic_config()  # كتابة ملف التكوين
os._exit(00)  # إعادة تشغيل الدفتر
```

## إعداد مجموعة البيانات والنماذج

بعد ذلك، يجب عليك إعداد مجموعة البيانات. كما ذُكر سابقًا، يجب توخي الحذر عند إعداد `DataLoaders` والنموذج للتأكد من عدم وضع **أي شيء** على أي وحدة معالجة رسوميات (GPU).

إذا فعلت ذلك، فيُنصح بوضع هذا الرمز المحدد في دالة واستدعائها من داخل واجهة مشغل الدفتر، والتي سيتم عرضها لاحقًا.

تأكد من تنزيل مجموعة البيانات باتباع الإرشادات [هنا](https://github.com/huggingface/accelerate/tree/main/examples#simple-vision-example)

```python
import os, re, torch, PIL
import numpy as np

from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomResizedCrop, Resize, ToTensor

from accelerate import Accelerator
from accelerate.utils import set_seed
from timm import create_model
```

أولًا، تحتاج إلى إنشاء دالة لاستخراج اسم الفئة بناءً على اسم الملف:

```python
import os

data_dir = "../../images"
fnames = os.listdir(data_dir)
fname = fnames[0]
print(fname)
```

```python out
beagle_32.jpg
```

في هذه الحالة، يكون التصنيف هو `beagle`. يمكنك باستخدام التعبير العادي استخراج التصنيف من اسم الملف:

```python
import re


def extract_label(fname):
    stem = fname.split(os.path.sep)[-1]
    return re.search(r"^(.*)_\d+\.jpg$", stem).groups()[0]
```

```python
extract_label(fname)
```

ويمكنك رؤية الاسم الصحيح الذي تم إرجاعه لملفنا:

```python out
"beagle"
```

بعد ذلك، يجب إنشاء فئة `Dataset` للتعامل مع استرداد الصورة والتصنيف:

```python
class PetsDataset(Dataset):
    def __init__(self, file_names, image_transform=None, label_to_id=None):
        self.file_names = file_names
        self.image_transform = image_transform
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        raw_image = PIL.Image.open(fname)
        image = raw_image.convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        label = extract_label(fname)
        if self.label_to_id is not None:
            label = self.label_to_id[label]
        return {"image": image, "label": label}
```

الآن لإنشاء مجموعة البيانات. يمكنك العثور على جميع أسماء الملفات والتصنيفات وإعلانها خارج دالة التدريب واستخدامها كمراجع داخل الدالة التي تم إطلاقها:

```python
fnames = [os.path.join("../../images", fname) for fname in fnames if fname.endswith(".jpg")]
```

بعد ذلك، قم بجمع جميع التصنيفات:

```python
all_labels = [extract_label(fname) for fname in fnames]
id_to_label = list(set(all_labels))
id_to_label.sort()
label_to_id = {lbl: i for i, lbl in enumerate(id_to_label)}
```

بعد ذلك، يجب عليك إنشاء دالة `get_dataloaders` التي ستعيد برنامج التهيئة الخاص بك. كما ذُكر سابقًا، إذا تم إرسال البيانات تلقائيًا إلى وحدة معالجة الرسوميات (GPU) أو جهاز وحدة معالجة الرسوميات (TPU) عند إنشاء `DataLoaders`، فيجب إنشاؤها باستخدام هذه الطريقة.

```python
def get_dataloaders(batch_size: int = 64):
    "Build a set of dataloaders with a batch_size"
    random_perm = np.random.permutation(len(fnames))
    cut = int(0.8 * len(fnames))
    train_split = random_perm[:cut]
    eval_split = random_perm[cut:]

    # لاستخدام التدريب البسيط، سيتم استخدام RandomResizedCrop
    train_tfm = Compose([RandomResizedCrop((224, 224), scale=(0.5, 1.0)), ToTensor()])
    train_dataset = PetsDataset([fnames[i] for i in train_split], image_transform=train_tfm, label_to_id=label_to_id)

    # لتقييم حجم محدد سيتم استخدامه
    eval_tfm = Compose([Resize((224, 224)), ToTensor()])
    eval_dataset = PetsDataset([fnames[i] for i in eval_split], image_transform=eval_tfm, label_to_id=label_to_id)

    # إنشاء برنامج التهيئة
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size * 2, num_workers=4)
    return train_dataloader, eval_dataloader
```

أخيرًا، يجب عليك استيراد الجدول الزمني المراد استخدامه لاحقًا:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
```

## كتابة دالة التدريب

الآن يمكنك إنشاء حلقة التدريب. تعمل [`notebook_launcher`] من خلال تمرير دالة لاستدعائها والتي سيتم تشغيلها عبر النظام الموزع.

فيما يلي حلقة تدريب أساسية لمشكلة تصنيف الحيوانات:

> **ملاحظة:** تم تقسيم الكود للسماح بالشرح لكل قسم. ستتوفر نسخة كاملة يمكن نسخها ولصقها في الأسفل.

```python
def training_loop(mixed_precision="fp16", seed: int = 42, batch_size: int = 64):
    set_seed(seed)
    accelerator = Accelerator(mixed_precision=mixed_precision)
```

أولًا، يجب عليك تعيين البذور وإنشاء كائن [`Accelerator`] في أقرب وقت ممكن في حلقة التدريب.

> **تحذير:** إذا كنت تتدرب على وحدة معالجة الرسوميات (TPU)، فيجب أن تأخذ حلقة التدريب النموذج كمعلمة ويجب أن يتم إنشاء النموذج خارج دالة حلقة التدريب. راجع [أفضل الممارسات لوحدة معالجة الرسوميات (TPU)](../concept_guides/training_tpu) لمعرفة السبب.

بعد ذلك، يجب عليك إنشاء برنامج التهيئة الخاص بك وإنشاء نموذجك:

```python
train_dataloader, eval_dataloader = get_dataloaders(batch_size)
model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))
```

> **ملاحظة:** تقوم بإنشاء النموذج هنا حتى يتحكم البذور أيضًا في تهيئة الوزن الجديد.

نظرًا لأنك تقوم بنقل التعلم في هذا المثال، يبدأ برنامج الترميز للنموذج بالتجميد بحيث يمكن تدريب رأس النموذج فقط في البداية:

```python
for param in model.parameters():
    param.requires_grad = False
for param in model.get_classifier().parameters():
    param.requires_grad = True
```

سيؤدي تطبيع دفعات الصور إلى تسريع التدريب قليلًا:

```python
mean = torch.tensor(model.default_cfg["mean"])[None, :, None, None]
std = torch.tensor(model.default_cfg["std"])[None, :, None, None]
```

لجعل هذه الثوابت متاحة على الجهاز النشط، يجب عليك تعيينه إلى جهاز `Accelerator`:

```python
mean = mean.to(accelerator.device)
std = std.to(accelerator.device)
```

بعد ذلك، قم بتهيئة بقية فئات PyTorch المستخدمة للتدريب:

```python
optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-2 / 25)
lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=3e-2, epochs=5, steps_per_epoch=len(train_dataloader))
```

قبل تمرير كل شيء إلى [`~Accelerator.prepare`].

> **ملاحظة:** لا يوجد ترتيب محدد لتذكره، فأنت تحتاج فقط إلى فك الأشياء بنفس الترتيب الذي قدمتها به إلى طريقة الإعداد.

```python
model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)
```

الآن قم بتدريب النموذج:

```python
for epoch in range(5):
    model.train()
    for batch in train_dataloader:
        inputs = (batch["image"] - mean) / std
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, batch["label"])
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

ستبدو حلقة التقييم مختلفة قليلًا مقارنة بحلقة التدريب. سيتم إضافة عدد العناصر بالإضافة إلى إجمالي دقة كل دفعة إلى ثابتين:

```python
model.eval()
accurate = 0
num_elems = 0
```

بعد ذلك، لديك بقية حلقة PyTorch القياسية:

```python
for batch in eval_dataloader:
    inputs = (batch["image"] - mean) / std
    with torch.no_grad():
        outputs = model(inputs)
        predictions = outputs.argmax(dim=-1)
```

قبل الاختلاف الرئيسي الأخير.

عند إجراء التقييم الموزع، يجب تمرير التوقعات والتصنيفات عبر [`~Accelerator.gather`] بحيث تكون جميع البيانات متاحة على الجهاز الحالي ويمكن تحقيق المقياس المحسوب بشكل صحيح:

```python
accurate_preds = accelerator.gather(predictions) == accelerator.gather(batch["label"])
num_elems += accurate_preds.shape[0]
accurate += accurate_preds.long().sum()
```

الآن كل ما عليك هو حساب المقياس الفعلي لهذه المشكلة، ويمكنك طباعته على العملية الرئيسية باستخدام [`~Accelerator.print`]:

```python
eval_metric = accurate.item() / num_elems
accelerator.print(f"epoch {epoch}: {100 * eval_metric:.2f}")
```

تتوفر نسخة كاملة من حلقة التدريب أدناه:

```python
def training_loop(mixed_precision="fp16", seed: int = 42, batch_size: int = 64):
    set_seed(seed)
    # تهيئة المعجل
    accelerator = Accelerator(mixed_precision=mixed_precision)
    # إنشاء برنامج التهيئة
    train_dataloader, eval_dataloader = get_dataloaders(batch_size)

    # إنشاء النموذج (أنت تقوم بإنشاء النموذج هنا حتى يتحكم البذور أيضًا في تهيئة الوزن الجديدة)
    model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))

    # تجميد النموذج الأساسي
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    # يمكنك تطبيع دفعات الصور لتكون أسرع قليلًا
    mean = torch.tensor(model.default_cfg["mean"])[None, :, None, None]
    std = torch.tensor(model.default_cfg["std"])[None, :, None, None]

    # لجعل هذه الثوابت متاحة على الجهاز النشط، قم بتعيينه إلى جهاز المعجل
    mean = mean.to(accelerator.device)
    std = std.to(accelerator.device)

    # تهيئة المحسن
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-2 / 25)

    # تهيئة جدول التعلم
    lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=3e-2, epochs=5, steps_per_epoch=len(train_dataloader))

    # الإعداد
    # لا يوجد ترتيب محدد لتذكره، فأنت تحتاج فقط إلى فك الأشياء بنفس الترتيب الذي قدمتها به إلى طريقة الإعداد.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # الآن قم بتدريب النموذج
    for epoch in range(5):
        model.train()
        for batch in train_dataloader:
            inputs = (batch["image"] - mean) / std
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, batch["label"])
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    model.eval()
    accurate = 0
    num_elems = 0
    for batch in eval_dataloader:
        inputs = (batch["image"] - mean) / std
        with torch.no_grad():
            outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
        accurate_preds = accelerator.gather(predictions) == accelerator.gather(batch["label"])
        num_elems += accurate_preds.shape[0]
        accurate += accurate_preds.long().sum()

    eval_metric = accurate.item() / num_elems
    # استخدم المعجل.الطباعة للطباعة فقط على العملية الرئيسية.
    accelerator.print(f"epoch {epoch}: {100 * eval_metric:.2f}")
```
## استخدام notebook_launcher

كل ما تبقى هو استخدام [notebook_launcher].
قم بتمرير الدالة والحجج (على شكل زوج) وعدد العمليات التي سيتم التدريب عليها. (راجع [التوثيق](../package_reference/launchers) لمزيد من المعلومات)

```python
from accelerate import notebook_launcher
```

```python
args = ("fp16", 42, 64)
notebook_launcher(training_loop, args, num_processes=2)
```

في حالة التشغيل على عدة عقد، تحتاج إلى إعداد جلسة Jupyter على كل عقدة وتشغيل خلية الإطلاق في نفس الوقت.
بالنسبة للبيئة التي تحتوي على عقدتين (أجهزة كمبيوتر) مع 8 وحدات معالجة رسومية لكل منهما وعنوان IP للكمبيوتر الرئيسي هو "172.31.43.8"، سيبدو الأمر كما يلي:

```python
notebook_launcher(training_loop, args, master_addr="172.31.43.8", node_rank=0, num_nodes=2, num_processes=8)
```

وفي جلسة Jupyter الثانية على الجهاز الآخر:

<Tip>
لاحظ كيف تغيرت `node_rank`
</Tip>

```python
notebook_launcher(training_loop, args, master_addr="172.31.43.8", node_rank=1, num_nodes=2, num_processes=8)
```

في حالة التشغيل على وحدة معالجة الرسوميات المُبرمَجة (TPU)، سيبدو الأمر كما يلي:

```python
model = create_model("resnet50d", pretrained=True, num_classes=len(label_to_id))

args = (model, "fp16", 42, 64)
notebook_launcher(training_loop, args, num_processes=8)
```

لبدء عملية التدريب مع المرونة، وتمكين تحمل الأخطاء، يمكنك استخدام ميزة `elastic_launch` التي يوفرها PyTorch. يتطلب ذلك تحديد معلمات إضافية مثل `rdzv_backend` و`max_restarts`. فيما يلي مثال على كيفية استخدام `notebook_launcher` مع القدرات المرنة:

```python
notebook_launcher(
training_loop,
args,
num_processes=2,
max_restarts=3
)
```

أثناء تشغيله، سيعرض التقدم المحرز، بالإضافة إلى عدد الأجهزة التي تم تشغيلها عليها. تم تشغيل هذا البرنامج التعليمي باستخدام وحدتي معالجة رسومات:

```python out
Launching training on 2 GPUs.
epoch 0: 88.12
epoch 1: 91.73
epoch 2: 92.58
epoch 3: 93.90
epoch 4: 94.71
```

وهذا كل شيء!

يرجى ملاحظة أن [`notebook_launcher`] يتجاهل ملف تكوين 🤗 Accelerate، لبدء التشغيل بناءً على التكوين، استخدم:

```bash
accelerate launch
```

## التصحيح

من المشكلات الشائعة عند تشغيل `notebook_launcher` هي تلقي خطأ "CUDA has already been initialized". عادة ما ينشأ هذا الخطأ من استيراد أو رمز سابق في الدفتر الذي يقوم باستدعاء مكتبة PyTorch الفرعية `torch.cuda`. للمساعدة في تحديد ما حدث خطأ، يمكنك تشغيل `notebook_launcher` مع `ACCELERATE_DEBUG_MODE=yes` في بيئتك وسيتم إجراء فحص إضافي عند الإنشاء للتأكد من إمكانية إنشاء عملية عادية واستخدام CUDA دون مشكلة. (يمكنك لا تزال تشغيل رمز CUDA الخاص بك بعد ذلك).

## خاتمة

أظهر هذا الدفتر كيفية تنفيذ التدريب الموزع من داخل دفتر Jupyter. فيما يلي بعض الملاحظات الرئيسية التي يجب تذكرها:

- تأكد من حفظ أي رمز يستخدم CUDA (أو استيرادات CUDA) للدالة التي تم تمريرها إلى [`notebook_launcher`]

- قم بتعيين `num_processes` ليكون عدد الأجهزة المستخدمة للتدريب (مثل عدد وحدات معالجة الرسومات، أو وحدات المعالجة المركزية، أو وحدات معالجة الرسوميات المُبرمَجة، وما إلى ذلك)

- إذا كنت تستخدم وحدة معالجة الرسوميات المُبرمَجة، قم بإعلان نموذجك خارج دالة حلقة التدريب