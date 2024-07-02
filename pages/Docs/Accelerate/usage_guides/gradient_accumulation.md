# تنفيذ تجميع الخرج باستخدام HuggingFace Accelerate

تجميع الخرج هي تقنية تسمح بالتدريب على أحجام دفعات أكبر مما تستطيع آلتك عادةً وضعه في الذاكرة. ويتم ذلك عن طريق تجميع الخرج عبر عدة دفعات، وعدم تحديث المحسن إلا بعد تنفيذ عدد معين من الدفعات.

على الرغم من أن كود تجميع الخرج القياسي سيعمل بشكل جيد في إعداد موزع، إلا أنه ليس أكثر الطرق كفاءة للقيام بذلك، وقد تواجه بطءًا كبيرًا!

في هذا البرنامج التعليمي، ستتعلم كيفية إعداد تجميع الخرج بسرعة وتنفيذه باستخدام الأدوات المساعدة المقدمة في HuggingFace Accelerate، والتي يمكن أن تصل إلى إضافة سطر واحد فقط من الكود!

سيستخدم هذا المثال حلقة تدريب PyTorch مبسطة للغاية تقوم بتجميع الخرج كل دفعتين:

```python
device = "cuda"
model.to(device)

gradient_accumulation_steps = 2

for index, batch in enumerate(training_dataloader):
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    if (index + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

## تحويله إلى HuggingFace Accelerate

أولاً، سيتم تحويل الكود الموضح أعلاه لاستخدام HuggingFace Accelerate دون مساعد تجميع الخرج الخاص:

```diff
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

for index, batch in enumerate(training_dataloader):
    inputs, targets = batch
-     inputs = inputs.to(device)
-     targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss = loss / gradient_accumulation_steps
+     accelerator.backward(loss)
    if (index+1) % gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

<Tip warning={true}>
في حالته الحالية، لن يقوم هذا الكود بتنفيذ تجميع الخرج بكفاءة بسبب عملية تسمى مزامنة الخرج. اقرأ المزيد عن ذلك في دليل المفاهيم [Concepts tutorial](../concept_guides/gradient_synchronization)!
</Tip>

## السماح لـ HuggingFace Accelerate بالتعامل مع تجميع الخرج

كل ما تبقى الآن هو السماح لـ HuggingFace Accelerate بالتعامل مع تجميع الخرج نيابة عنا. للقيام بذلك، يجب تمرير معلمة `gradient_accumulation_steps` إلى [`Accelerator`]، والتي تحدد عدد الخطوات التي يجب تنفيذها قبل كل استدعاء لـ `step()` وكيفية ضبط الخسارة تلقائيًا أثناء استدعاء [`~Accelerator.backward`]:

```diff
from accelerate import Accelerator
- accelerator = Accelerator()
+ accelerator = Accelerator(gradient_accumulation_steps=2)
```

أو يمكنك تمرير معلمة `gradient_accumulation_plugin` إلى كائن [`Accelerator`] في `__init__`، مما سيسمح لك بتخصيص سلوك تجميع الخرج بشكل أكبر. اقرأ المزيد عن ذلك في وثائق [GradientAccumulationPlugin](../package_reference/accelerator#accelerate.utils.GradientAccumulationPlugin).

من هنا، يمكنك استخدام سياق [`~Accelerator.accumulate`] من داخل حلقة التدريب الخاصة بك لأداء تجميع الخرج تلقائيًا! ما عليك سوى لفها حول الجزء التدريبي بالكامل من الكود الخاص بنا:

```diff
- for index, batch in enumerate(training_dataloader):
+ for batch in training_dataloader:
+     with accelerator.accumulate(model):
    inputs, targets = batch
    outputs = model(inputs)
```

يمكنك إزالة جميع الفحوصات الخاصة لرقم الخطوة وضبط الخسارة:

```diff
- loss = loss / gradient_accumulation_steps
accelerator.backward(loss)
- if (index+1) % gradient_accumulation_steps == 0:
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

كما ترى، يمكن لـ [`Accelerator`] تتبع رقم الدفعة التي تعمل عليها، وسيعرف تلقائيًا ما إذا كان سيتم تنفيذ الخطوة من خلال المحسن المُعد وكيفية ضبط الخسارة.

<Tip>
عادةً مع تجميع الخرج، ستحتاج إلى ضبط عدد الخطوات لتعكس التغيير في إجمالي الدفعات التي تتدرب عليها. يقوم HuggingFace Accelerate بذلك تلقائيًا بشكل افتراضي. في الكواليس، نقوم بتهيئة [`GradientAccumulationPlugin`] مصممة للقيام بذلك.
</Tip>

<Tip warning={true}>
يتم مزامنة [`state.GradientState`] مع محمل البيانات النشط الذي يتم التنقل خلاله. وبالتالي، يفترض بشكل بسيط أنه عند الوصول إلى نهاية محمل البيانات، سيتم مزامنة كل شيء وسيتم تنفيذ خطوة. لإيقاف هذا، قم بتعيين `sync_with_dataloader` على `False` في [`GradientAccumulationPlugin`]:
```{python}
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin

plugin = GradientAccumulationPlugin(sync_with_dataloader=False)
accelerator = Accelerator(..., gradient_accumulation_plugin=plugin)
```
</Tip>

## الكود النهائي

فيما يلي التنفيذ النهائي لأداء تجميع الخرج باستخدام HuggingFace Accelerate:

```python
from accelerate import Accelerator
accelerator = Accelerator(gradient_accumulation_steps=2)
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
model, optimizer, training_dataloader, scheduler
)
for batch in training_dataloader:
with accelerator.accumulate(model):
inputs, targets = batch
outputs = model(inputs)
loss = loss_function(outputs, targets)
accelerator.backward(loss)
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

<Tip warning={true}>
من المهم أن يتم تنفيذ **تقديم/إرجاع واحد فقط** داخل مدير السياق `with accelerator.accumulate(model)`.
</Tip>

لمعرفة المزيد حول السحر الذي يلفه، اقرأ دليل مفهوم مزامنة الخرج [Gradient Synchronization concept guide](../concept_guides/gradient_synchronization)

## مثال مستقل

فيما يلي مثال مستقل يمكنك تشغيله لمشاهدة تجميع الخرج في العمل باستخدام HuggingFace Accelerate:

```python
import torch
import copy
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import TensorDataset, DataLoader

# seed
set_seed(0)

# define toy inputs and labels
x = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
y = torch.tensor([2., 4., 6., 8., 10., 12., 14., 16.])
gradient_accumulation_steps = 4
batch_size = len(x) // gradient_accumulation_steps

# define dataset and dataloader
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size)

# define model, optimizer and loss function
model = torch.zeros((1, 1), requires_grad=True)
model_clone = copy.deepcopy(model)
criterion = torch.nn.MSELoss()
model_optimizer = torch.optim.SGD([model], lr=0.02)
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
model, model_optimizer, dataloader = accelerator.prepare(model, model_optimizer, dataloader)
model_clone_optimizer = torch.optim.SGD([model_clone], lr=0.02)
print(f"initial model weight is {model.mean().item():.5f}")
print(f"initial model weight is {model_clone.mean().item():.5f}")
for i, (inputs, labels) in enumerate(dataloader):
with accelerator.accumulate(model):
inputs = inputs.view(-1, 1)
print(i, inputs.flatten())
labels = labels.view(-1, 1)
outputs = inputs @ model
loss = criterion(outputs, labels)
accelerator.backward(loss)
model_optimizer.step()
model_optimizer.zero_grad()
loss = criterion(x.view(-1, 1) @ model_clone, y.view(-1, 1))
model_clone_optimizer.zero_grad()
loss.backward()
model_clone_optimizer.step()
print(f"w/ accumulation, the final model weight is {model.mean().item():.5f}")
print(f"w/o accumulation, the final model weight is {model_clone.mean().item():.5f}")
```

```
initial model weight is 0.00000
initial model weight is 0.00000
0 tensor([1., 2.])
1 tensor([3., 4.])
2 tensor([5., 6.])
3 tensor([7., 8.])
w/ accumulation, the final model weight is 2.04000
w/o accumulation, the final model weight is 2.04000
```