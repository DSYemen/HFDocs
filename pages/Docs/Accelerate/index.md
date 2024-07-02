
# Accelerate

🤗 Accelerate هي مكتبة تتيح تشغيل نفس كود PyTorch عبر أي تكوين موزع بإضافة أربعة أسطر فقط من الكود! وباختصار، فإن التدريب والاستدلال على نطاق واسع أصبح بسيطًا وفعالًا وقابلًا للتكيف.

```diff
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

for batch in training_dataloader:
optimizer.zero_grad()
inputs, targets = batch
inputs = inputs.to(device)
targets = targets.to(device)
outputs = model(inputs)
loss = loss_function(outputs, targets)
+     accelerator.backward(loss)
optimizer.step()
scheduler.step()
```

تم بناء 🤗 Accelerate على `torch_xla` و `torch.distributed`، ويتولى المهام الثقيلة، لذلك لا يتعين عليك كتابة أي كود مخصص للتكيف مع هذه المنصات.

قم بتحويل قواعد الكود الموجودة إلى استخدام [DeepSpeed](usage_guides/deepspeed)، وأداء [fully sharded data parallelism](usage_guides/fsdp)، والحصول على دعم تلقائي للتدريب عالي الدقة!

<Tip>
للحصول على فكرة أفضل عن هذه العملية، تأكد من الاطلاع على [البرامج التعليمية](basic_tutorials/overview)!
</Tip>

يمكن بعد ذلك تشغيل هذا الكود على أي نظام من خلال واجهة سطر الأوامر Accelerate:

```bash
accelerate launch {my_script.py}
```

<div class="mt-10">
<div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./basic_tutorials/overview"
><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">البرامج التعليمية</div>
<p class="text-gray-700">تعلم الأساسيات واعتد على استخدام 🤗 Accelerate. ابدأ من هنا إذا كنت تستخدم 🤗 Accelerate لأول مرة!</p>
</a>
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./usage_guides/explore"
><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">أدلة كيفية الاستخدام</div>
<p class="text-gray-700">أدلة عملية لمساعدتك على تحقيق هدف محدد. الق نظرة على هذه الأدلة لمعرفة كيفية استخدام 🤗 Accelerate لحل المشكلات الواقعية.</p>
</a>
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./concept_guides/gradient_synchronization"
><div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">الأدلة المفاهيمية</div>
<p class="text-gray-700">تفسيرات عالية المستوى لبناء فهم أفضل للمواضيع المهمة مثل تجنب الدقائق والتعقيدات الخفية في التدريب الموزع و DeepSpeed.</p>
</a>
<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./package_reference/accelerator"
><div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">المرجع</div>
<p class="text-gray-700">الأوصاف الفنية لكيفية عمل فئات 🤗 Accelerate والطرق.</p>
</a>
</div>
</div>