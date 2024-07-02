# حفظ نقطة تفتيش

عند تدريب نموذج PyTorch باستخدام HuggingFace Accelerate، قد ترغب في كثير من الأحيان في حفظ حالة التدريب والاستمرار منها. يتطلب القيام بذلك حفظ وتحميل النموذج، والمُحَسِّن (optimizer)، ومولدات الأعداد العشوائية (RNG generators)، وGradScaler. وهناك دالتان ميسّرتين داخل HuggingFace Accelerate للقيام بذلك بسرعة:

- استخدم [`~Accelerator.save_state`] لحفظ كل ما سبق في مجلد.
- استخدم [`~Accelerator.load_state`] لتحميل كل ما تم حفظه سابقًا من عملية حفظ حالة سابقة.

لمزيد من التخصيص في مكان وكيفية حفظ الحالات من خلال [`~Accelerator.save_state`]، يمكن استخدام فئة [`~utils.ProjectConfiguration`]. على سبيل المثال، إذا تم تمكين `automatic_checkpoint_naming`، فسيتم وضع كل نقطة تفتيش محفوظة في `Accelerator.project_dir/checkpoints/checkpoint_{checkpoint_number}`.

يجب ملاحظة أن من المتوقع أن تأتي هذه الحالات من نفس النص البرمجي للتدريب، ولا يجب أن تكون من نصين منفصلين.

- من خلال استخدام [`~Accelerator.register_for_checkpointing`]، يمكنك تسجيل الأشياء المخصصة ليتم حفظها أو تحميلها تلقائيًا من الدالتين السابقتين، طالما أن الكائن لديه وظيفة **state_dict** و**load_state_dict**. يمكن أن يشمل ذلك أشياء مثل جدول مُعَدَّل التعلم (learning rate scheduler).

فيما يلي مثال مختصر على استخدام حفظ نقطة التفتيش لحفظ حالة وإعادة تحميلها أثناء التدريب:

```python
from accelerate import Accelerator
import torch

accelerator = Accelerator(project_dir="my/save/path")

my_scheduler = torch.optim.lr_scheduler.StepLR(my_optimizer, step_size=1, gamma=0.99)
my_model, my_optimizer, my_training_dataloader = accelerator.prepare(my_model, my_optimizer,  y_training_dataloader)

# تسجيل جدول معدل التعلم
accelerator.register_for_checkpointing(my_scheduler)

# حفظ الحالة الابتدائية
accelerator.save_state()

device = accelerator.device
my_model.to(device)

# تنفيذ التدريب
for epoch in range(num_epochs):
    for batch in my_training_dataloader:
        my_optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = my_model(inputs)
        loss = my_loss_function(outputs, targets)
        accelerator.backward(loss)
        my_optimizer.step()
        my_scheduler.step()

# استعادة الحالة السابقة
accelerator.load_state("my/save/path/checkpointing/checkpoint_0")
```

## استعادة حالة DataLoader

بعد استئناف التدريب من نقطة تفتيش، قد يكون من المستحسن أيضًا استئناف التدريب من نقطة معينة في DataLoader النشط إذا تم حفظ الحالة في منتصف حقبة التدريب (epoch). يمكنك استخدام [`~Accelerator.skip_first_batches`] للقيام بذلك.

```python
from accelerate import Accelerator

accelerator = Accelerator(project_dir="my/save/path")

train_dataloader = accelerator.prepare(train_dataloader)
accelerator.load_state("my_state")

# افترض أن نقطة التفتيش تم حفظها بعد 100 خطوة من حقبة التدريب
skipped_dataloader = accelerator.skip_first_batches(train_dataloader, 100)

# بعد التكرار الأول، عد إلى `train_dataloader`

# حقبة التدريب الأولى
for batch in skipped_dataloader:
    # قم بشيء ما
    pass

# حقبة التدريب الثانية
for batch in train_dataloader:
    # قم بشيء ما
    pass
```