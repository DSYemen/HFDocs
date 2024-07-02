# خطافات اتصال DDP

توفر خطافات اتصال البيانات الموزعة المتوازية (DDP) واجهة عامة للتحكم في كيفية نقل التدرجات عبر العمال عن طريق تجاوز allreduce الفانيليا في `DistributedDataParallel`. يتم توفير بعض خطافات الاتصال المدمجة، ويمكن للمستخدمين تطبيق أي من هذه الخطافات بسهولة لتحسين الاتصال.

- **خطاف ضغط FP16**: يقوم بضغط التدرجات عن طريق تحويلها إلى تنسيق النقطة العائمة ذات الدقة النصفية (`torch.float16`)، مما يقلل من النفقات العامة للاتصال.

- **خطاف ضغط BF16**: مشابه لـ FP16، ولكنه يستخدم تنسيق النقطة العائمة الدماغية (`torch.bfloat16`)، والذي يمكن أن يكون أكثر كفاءة على أجهزة معينة.

- **خطاف PowerSGD**: خوارزمية ضغط تدرج متقدمة توفر معدلات ضغط عالية ويمكن أن تسرع التدريب الموزع المحدود بالنطاق الترددي.

في هذا البرنامج التعليمي، ستتعرف على كيفية إعداد خطافات اتصال DDP بسرعة وإجراء التدريب باستخدام الأدوات المساعدة المقدمة في 🤗 Accelerate، والتي يمكن أن تكون بسيطة مثل إضافة سطر رمز واحد فقط! وهذا يوضح كيفية استخدام خطافات اتصال DDP لتحسين اتصال التدرج في التدريب الموزع باستخدام مكتبة 🤗 Accelerate.

## خطاف ضغط FP16

<hfoptions id="fp16">

<hfoption id="PyTorch">

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

model = MyModel()
model = DDP(model, device_ids=[torch.cuda.current_device()])
model.register_comm_hook(state=None, hook=default_hooks.fp16_compress_hook)

# حلقة التدريب
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

</hfoption>

<hfoption id="Accelerate">

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# إعداد خطاف اتصال DDP
ddp_kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.FP16)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# حلقة التدريب
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

</hfoption>

</hfoptions>

### خطاف ضغط BF16

<Tip warning={true}>

واجهة برمجة تطبيقات خطاف ضغط BF16 تجريبية، وهي تتطلب إصدار NCCL أحدث من 2.9.6.

</Tip>

<hfoptions id="bf16">

<hfoption id="PyTorch">

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

model = MyModel()
model = DDP(model, device_ids=[torch.cuda.current_device()])
model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)

# حلقة التدريب
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

</hfoption>

<hfoption id="Accelerate">

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class Myamodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# إعداد خطاف اتصال DDP
ddp_kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.BF16)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# حلقة التدريب
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

</hfoption>

</hfoptions>

### خطاف PowerSGD

<Tip warning={true}>

عادةً ما يتطلب PowerSGD ذاكرة إضافية بنفس حجم تدرجات النموذج لتمكين التغذية الراجعة للأخطاء، والتي يمكن أن تعوض عن الاتصال المضغوط المتحيز وتحسين الدقة.

</Tip>

<hfoptions id="powerSGD">

<hfoption id="PyTorch">

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

model = MyModel()
model = DDP(model، device_ids = [torch.cuda.current_device()])
state = powerSGD_hook.PowerSGDState(process_group=None)
model.register_comm_hook(state=state, hook=powerSGD_hook.powerSGD_hook)

# حلقة التدريب
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

</hfoption>

<hfoption id="Accelerate">

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# إعداد خطاف اتصال DDP
ddp_kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.POWER_SGD)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# حلقة التدريب
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

</hfoption>

</hfoptions>

## المرافق خطافات اتصال DDP

هناك أداتان إضافيتان لدعم الوظائف الاختيارية مع خطافات الاتصال.

### comm_wrapper

`comm_wrapper` هو خيار لتغليف خطاف اتصال بوظائف إضافية. على سبيل المثال، يمكن استخدامه لدمج ضغط FP16 مع استراتيجيات الاتصال الأخرى. برامج التغليف المدعومة حاليًا هي `no` و`fp16` و`bf16`.

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# إعداد خطاف اتصال DDP
ddp_kwargs = DistributedDataParallelKwargs(
comm_hook=DDPCommunicationHookType.POWER_SGD،
comm_wrapper=DDPCommunicationHookType.FP16
)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# حلقة التدريب
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

### comm_state_option

يسمح لك `comm_state_option` بتمرير معلومات حالة إضافية مطلوبة بواسطة خطافات اتصال معينة. هذا مفيد بشكل خاص للخطافات ذات الحالة مثل `PowerSGD`، والتي تتطلب الحفاظ على المعلمات الفائقة والحالات الداخلية عبر خطوات التدريب. فيما يلي مثال يوضح استخدام `comm_state_option` مع خطاف `PowerSGD`.

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# إعداد خطاف اتصال DDP
ddp_kwargs = DistributedDataParallelKwargs(
comm_hook=DDPCommunicationHookType.POWER_SGD،
comm_state_option={"matrix_approximation_rank": 2}
)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer، data_loader)

# حلقة التدريب
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

للحصول على استخدام أكثر تقدمًا وخطافات إضافية، راجع وثائق [خطافات اتصال PyTorch DDP](https://pytorch.org/docs/stable/ddp_comm_hooks.html).