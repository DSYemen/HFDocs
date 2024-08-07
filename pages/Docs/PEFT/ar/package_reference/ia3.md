# IA3

طريقة (IA)^3 أو Infused Adapter by Inhibiting and Amplifying Inner Activations، هي طريقة تضيف ثلاثة متجهات متعلمة لإعادة تحجيم مفاتيح وقيم طبقات self-attention وencoder-decoder attention، والتنشيط الوسيط لشبكة التغذية الأمامية الموضعية.

الملخص من الورقة البحثية هو:

*يُمكِّن التعلم القائم على السياق القليل من الأمثلة (ICL) النماذج اللغوية المُدربة مسبقًا من أداء مهمة لم يسبق رؤيتها دون أي تدريب قائم على التدرج من خلال إدخال عدد صغير من الأمثلة التدريبية كجزء من الإدخال. يتطلب ICL تكاليف حوسبة وذاكرة وتخزين كبيرة لأنه ينطوي على معالجة جميع الأمثلة التدريبية في كل مرة يتم فيها إجراء تنبؤ. يوفر الضبط الدقيق الفعال للبارامترات (PEFT) (مثل وحدات adapter، وتناغم المطال، وطرق التحديث النادرة، وما إلى ذلك) نموذجًا بديلًا يتم فيه تدريب مجموعة صغيرة من المعلمات لتمكين النموذج من أداء المهمة الجديدة. في هذه الورقة، نقارن بشكل صارم بين ICL القليل من الأمثلة و PEFT ونثبت أن الأخير يوفر دقة أفضل وتكاليف حوسبة أقل بشكل ملحوظ. وفي الوقت نفسه، نقدم طريقة PEFT جديدة تسمى (IA)^3 التي تقوم بضبط التنشيطات عن طريق متجهات متعلمة، مما يحقق أداءً أقوى مع تقديم عدد صغير نسبيًا من المعلمات الجديدة فقط. كما نقترح وصفة بسيطة بناءً على نموذج T0 تسمى T-Few والتي يمكن تطبيقها على مهام جديدة دون ضبط أو تعديلات خاصة بالمهمة. نتحقق من فعالية T-Few على المهام غير المرئية تمامًا من خلال تطبيقها على معيار RAFT، وتحقيق أداء خارق للمرة الأولى وتفوق على أحدث التقنيات بنسبة 6% مطلقة. كل الرموز المستخدمة في تجاربنا متاحة للجمهور*.

## IA3Config

[[autodoc]] tuners.ia3.config.IA3Config

## IA3Model

[[autodoc]] tuners.ia3.model.IA3Model