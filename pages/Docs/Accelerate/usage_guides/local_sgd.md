# استخدام Local SGD مع 🤗 Accelerate

Local SGD هي تقنية للتدريب الموزع حيث لا يتم تزامن التدرجات في كل خطوة. وبالتالي، يقوم كل عملية بتحديث نسخته الخاصة من أوزان النموذج، وبعد عدد معين من الخطوات، يتم تزامن هذه الأوزان عن طريق حساب المتوسط عبر جميع العمليات. يحسن هذا من كفاءة الاتصال ويمكن أن يؤدي إلى تسريع التدريب بشكل كبير، خاصة عندما يفتقر الكمبيوتر إلى اتصال أسرع مثل NVLink.

على عكس تجميع التدرجات (حيث يتطلب تحسين كفاءة الاتصال زيادة حجم الدفعة الفعال)، لا يتطلب Local SGD تغيير حجم الدفعة أو معدل التعلم / الجدول. ومع ذلك، إذا لزم الأمر، يمكن الجمع بين Local SGD وتجميع التدرجات أيضًا.

في هذا البرنامج التعليمي، ستتعلم كيفية إعداد Local SGD 🤗 Accelerate بسرعة. مقارنة بالإعداد القياسي لـ Accelerate، يتطلب هذا سطرين فقط من التعليمات البرمجية الإضافية.

سيستخدم هذا المثال حلقة تدريب PyTorch مبسطة للغاية تقوم بتجميع التدرجات كل دفعتين:

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

## تحويله إلى 🤗 Accelerate

أولاً، سيتم تحويل الكود الموضح سابقًا لاستخدام 🤗 Accelerate بدون مساعد LocalSGD أو تجميع التدرجات:

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
```

## السماح لـ 🤗 Accelerate بالتعامل مع مزامنة النموذج

كل ما تبقى الآن هو السماح لـ 🤗 Accelerate بالتعامل مع مزامنة معلمات النموذج **وتجميع التدرجات** نيابة عنا. ولتبسيط الأمور، دعنا نفترض أننا بحاجة إلى المزامنة كل 8 خطوات. يتم تحقيق ذلك عن طريق إضافة عبارة `with LocalSGD` واستدعاء واحد `local_sgd.step()` بعد كل خطوة من خطوات المحسن:

```diff
+local_sgd_steps=8

+with LocalSGD(accelerator=accelerator, model=model, local_sgd_steps=8, enabled=True) as local_sgd:
    for batch in training_dataloader:
        with accelerator.accumulate(model):
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
+           local_sgd.step()
```

تحت الغطاء، يقوم كود Local SGD بتعطيل مزامنة التدرجات التلقائية (ولكن لا يزال التجميع يعمل كما هو متوقع!). بدلاً من ذلك، فإنه يقوم بمعدلة معلمات النموذج كل `local_sgd_steps` خطوات (وفي نهاية حلقة التدريب أيضًا).

## القيود

يعمل التنفيذ الحالي فقط مع التدريب متعدد GPU (أو CPU) الأساسي بدون، على سبيل المثال، [DeepSpeed.](https://github.com/microsoft/DeepSpeed).

## المراجع

على الرغم من أننا لا نعرف الأصول الحقيقية لهذه الطريقة البسيطة، إلا أن فكرة Local SGD قديمة جدًا وترجع على الأقل إلى:

Zhang, J., De Sa, C., Mitliagkas, I., & Ré, C. (2016). [Parallel SGD: When does averaging help?. arXiv preprint
arXiv:1606.07365.](https://arxiv.org/abs/1606.07365)

نحن ننسب مصطلح Local SGD إلى الورقة التالية (ولكن قد تكون هناك مراجع سابقة لا نعرفها).

Stich, Sebastian Urban. ["Local SGD Converges Fast and Communicates Little." ICLR 2019-International Conference on
Learning Representations. No. CONF. 2019.](https://arxiv.org/abs/1805.09767)