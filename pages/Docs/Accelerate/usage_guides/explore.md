# تعلم كيفية دمج ميزات 🤗 Accelerate بسرعة!

يرجى استخدام الأداة التفاعلية أدناه للبدء في تعلم ميزة معينة من 🤗 Accelerate وكيفية استخدامها! سيوفر لك ذلك فرقًا في التعليمات البرمجية، وتوضيحًا لما يحدث، بالإضافة إلى بعض الروابط المفيدة لاستكشاف المزيد في الوثائق!

تبدأ معظم أمثلة التعليمات البرمجية من كود Python التالي قبل دمج 🤗 Accelerate بطريقة ما:

```python
for batch in dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

<div class="block dark:hidden">
<iframe
src="https://hf-accelerate-accelerate-examples.hf.space?__theme=light"
width="850"
height="1600"
></iframe>
</div>
<div class="hidden dark:block">
<iframe
src="https://hf-accelerate-accelerate-examples.hf.space?__theme=dark"
width="850"
height="1600"
></iframe>
</div>