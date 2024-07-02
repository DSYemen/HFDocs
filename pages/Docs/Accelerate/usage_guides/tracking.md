هذا هو النص المترجم مع اتباع التعليمات المحددة: 

<!--Copyright 2022 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# التتبع

هناك عدد كبير من واجهات برمجة التطبيقات الخاصة بتتبع التجارب، ولكن جعلها جميعًا تعمل في بيئة متعددة المعالجات يمكن أن يكون معقدًا في كثير من الأحيان. يوفر HuggingFace Accelerate واجهة برمجة تطبيقات عامة للتتبع يمكن استخدامها لتسجيل العناصر المفيدة أثناء تشغيل النص البرمجي الخاص بك من خلال [Accelerator.log]

## أجهزة التتبع المدمجة

يدعم Accelerate حاليًا سبعة أجهزة تتبع بشكل افتراضي:

- TensorBoard
- WandB
- CometML
- Aim
- MLFlow
- ClearML
- DVCLive

لاستخدام أي منها، قم بتمرير النوع المحدد (الأنواع) إلى معلمة "log_with" في [Accelerate]:

```python
from accelerate import Accelerator
from accelerate.utils import LoggerType

accelerator = Accelerator(log_with="all")  # لجميع أجهزة التتبع المتاحة في البيئة
accelerator = Accelerator(log_with="wandb")
accelerator = Accelerator(log_with=["wandb", LoggerType.TENSORBOARD])
```

في بداية تجربتك، يجب استخدام [Accelerator.init_trackers] لإعداد مشروعك، وإضافة أي معلمات للتجربة المراد تسجيلها:

```python
hps = {"num_iterations": 5, "learning_rate": 1e-2}
accelerator.init_trackers("my_project", config=hps)
```

عندما تكون مستعدًا لتسجيل أي بيانات، يجب استخدام [Accelerator.log].

يمكن أيضًا تمرير "step" لربط البيانات بخطوة معينة في حلقة التدريب.

```python
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=1)
```

بمجرد الانتهاء من التدريب، تأكد من تشغيل [Accelerator.end_training] حتى تتمكن جميع أجهزة التتبع من تشغيل وظائف الإنهاء الخاصة بها إذا كان لديها أي منها.

```python
accelerator.end_training()
```

فيما يلي مثال كامل:

```python
from accelerate import Accelerator

accelerator = Accelerator(log_with="all")
config = {
"num_iterations": 5,
"learning_rate": 1e-2,
"loss_function": str(my_loss_function),
}

accelerator.init_trackers("example_project", config=config)

my_model, my_optimizer, my_training_dataloader = accelerate.prepare(my_model, my_optimizer, my_training_dataloader)
device = accelerator.device
my_model.to(device)

for iteration in config["num_iterations"]:
for step, batch in my_training_dataloader:
my_optimizer.zero_grad()
inputs, targets = batch
inputs = inputs.to(device)
targets = targets.to(device)
outputs = my_model(inputs)
loss = my_loss_function(outputs, targets)
accelerator.backward(loss)
my_optimizer.step()
accelerator.log({"training_loss": loss}, step=step)
accelerator.end_training()
```

إذا كان أحد أجهزة التتبع يتطلب دليلًا لحفظ البيانات، مثل TensorBoard، فقم بتمرير مسار الدليل إلى project_dir. معلمة project_dir مفيدة عندما تكون هناك تكوينات أخرى يجب دمجها في فئة البيانات [~utils.ProjectConfiguration]. على سبيل المثال، يمكنك حفظ بيانات TensorBoard في project_dir وتسجيل كل شيء آخر في معلمة logging_dir من [~utils.ProjectConfiguration]:

```python
accelerator = Accelerator(log_with="tensorboard", project_dir=".")

# استخدم مع ProjectConfiguration
config = ProjectConfiguration(project_dir=".", logging_dir="another/directory")
accelerator = Accelerator(log_with="tensorboard"، project_config=config)
```

## تنفيذ أجهزة التتبع المخصصة

لتنفيذ جهاز تتبع جديد لاستخدامه في "Accelerator"، يمكن إنشاء جهاز جديد من خلال تنفيذ فئة [GeneralTracker].

يجب على كل جهاز تتبع تنفيذ ثلاث وظائف وامتلاك ثلاث خصائص:

- `__init__`:
- يجب أن تخزن "run_name" وتُهيئ جهاز تتبع واجهة برمجة التطبيقات للمكتبة المدمجة.
- إذا قام جهاز التتبع بتخزين بياناته محليًا (مثل TensorBoard)، فيمكن إضافة معلمة "logging_dir".
- `store_init_configuration`:
- يجب أن يأخذ قاموس "values" ويخزنه كتكوين تجريبي لمرة واحدة
- `log`:
- يجب أن يأخذ قاموس "values" و "step"، ويجب تسجيلها في التشغيل
- `name` (`str`):
- اسم سلسلة فريد لجهاز التتبع، مثل "wandb" لجهاز تتبع wandb.
- سيتم استخدام هذا للتفاعل مع هذا جهاز التتبع المحدد
- `requires_logging_directory` (`bool`):
- ما إذا كان "logging_dir" مطلوبًا لهذا جهاز التتبع المحدد وما إذا كان يستخدم واحدًا.
- `tracker`:
- يجب تنفيذ هذا على أنه وظيفة @property
- يجب أن يعيد آلية التتبع الداخلية التي تستخدمها المكتبة، مثل كائن "run" لـ "wandb".

يجب أن تستخدم كل طريقة أيضًا فئة [state.PartialState] إذا كان من المفترض تشغيل جهاز التتبع فقط على العملية الرئيسية على سبيل المثال.

يمكنك الاطلاع على مثال مختصر أدناه مع التكامل مع Weights and Biases، والذي يحتوي فقط على المعلومات ذات الصلة ويقوم بالتسجيل فقط على العملية الرئيسية:

```python
from accelerate.tracking import GeneralTracker, on_main_process
from typing import Optional

import wandb


class MyCustomTracker(GeneralTracker):
name = "wandb"
requires_logging_directory = False

@on_main_process
def __init__(self, run_name: str):
self.run_name = run_name
run = wandb.init(self.run_name)

@property
def tracker(self):
return self.run.run

@on_main_process
def store_init_configuration(self, values: dict):
wandb.config(values)

@on_main_process
def log(self, values: dict, step: Optional[int] = None):
wandb.log(values, step=step)
```

عندما تكون مستعدًا لبناء كائن "Accelerator"، قم بتمرير مثيل من جهاز التتبع الخاص بك إلى [Accelerator.log_with] لاستخدامه تلقائيًا مع واجهة برمجة التطبيقات:

```python
tracker = MyCustomTracker("some_run_name")
accelerator = Accelerator(log_with=tracker)
```

يمكن أيضًا مزج هذه الأجهزة مع أجهزة التتبع الموجودة، بما في ذلك مع "all":

```python
tracker = MyCustomTracker("some_run_name")
accelerator = Accelerator(log_with=[tracker, "all"])
```

## الوصول إلى جهاز التتبع الداخلي

إذا كنت تريد بعض التفاعلات المخصصة مع جهاز التتبع، فيمكنك الوصول بسرعة إلى واحد باستخدام طريقة [Accelerator.get_tracker]. ما عليك سوى تمرير السلسلة المقابلة لصفة ".name" لجهاز التتبع، وسوف يعيد ذلك جهاز التتبع على العملية الرئيسية.

يوضح هذا المثال القيام بذلك مع wandb:

```python
wandb_tracker = accelerator.get_tracker("wandb")
```

من هناك، يمكنك التفاعل مع كائن "run" الخاص بـ "wandb" كالمعتاد:

```python
wandb_run.log_artifact(some_artifact_to_log)
```

<Tip>
ستقوم أجهزة التتبع المدمجة في Accelerate بتنفيذ العملية الصحيحة تلقائيًا،
لذا إذا كان من المفترض تشغيل جهاز التتبع على العملية الرئيسية فقط، فسيتم ذلك
تلقائيًا.
</Tip>

إذا كنت تريد إزالة تغليف Accelerate تمامًا، فيمكنك
تحقيق نفس النتيجة بما يلي:

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
if accelerator.is_main_process:
wandb_tracker.log_artifact(some_artifact_to_log)
```

## عندما لا يعمل الغلاف

إذا كانت المكتبة تحتوي على واجهة برمجة تطبيقات لا تتبع `.log` الصارم مع قاموس شامل مثل Neptune.AI، فيمكن إجراء التسجيل يدويًا ضمن عبارة `if accelerator.is_main_process`:

```diff
from accelerate import Accelerator
+ import neptune.new as neptune

accelerator = Accelerator()
+ run = neptune.init(...)

my_model, my_optimizer, my_training_dataloader = accelerate.prepare(my_model, my_optimizer, my_training_dataloader)
device = accelerator.device
my_model.to(device)

for iteration in config["num_iterations"]:
for batch in my_training_dataloader:
my_optimizer.zero_grad()
inputs, targets = batch
inputs = inputs.to(device)
targets = targets.to(device)
outputs = my_model(inputs)
loss = my_loss_function(outputs, targets)
total_loss += loss
accelerator.backward(loss)
my_optimizer.step()
+         if accelerator.is_main_process:
+             run["logs/training/batch/loss"].log(loss)
```