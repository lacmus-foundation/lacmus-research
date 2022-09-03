<b>Multimodel Deep Learning for Person Detection in Aerial Images</b>, <i>статья от 7го сентября 2020го, авторы Vladan Papić (University of Split), Mirela Kundid Vasić
(University of Mostar)</i>. <a href="https://www.researchgate.net/publication/344269970_Multimodel_Deep_Learning_for_Person_Detection_in_Aerial_Images">Оригинал на ResearchGate</a>. 

Рассматривают two-stage подходы к детектированию людей на снимках с высоты, называя это multimodel.
Помимо основной темы можно вкратце коснуться замечаний:

- Упоминается некоторый метод сжатия, который помогает сократить количество данных, используемых при передаче изображений.
Возможно, он мог бы пригодиться в нашем сервисе по приему фоток с поисков. <br/>
Ссылаются на статью <i>Music, J.; Orovic, I.; Marasovic, T.; Papić, V.; Stankovic, S. "Gradient Compressive Sensing for Image Data Reduction in UAV Based Search and Rescue in the Wild.", Math. Probl. Eng. 2016</i>

- Один из методов, который помог им улучшить качество детекции - это использование "контекстной" информации.
В качестве контекста берётся регион вокруг патча 81х81 из датасета.
Пишут так же, что могут ещё пригодиться "image illumination (shadows, contrasts, etc.), geographical performance (GPS location, terrain type, elevation, etc.), semantic content (scene category, expected event, etc.), or the time frame (recording time, surrounding images, etc.)".
В референсах есть доки про анализ контекстуальной информации и гайд <i>Koester, R.J.  "A Search and Rescue Guide on where to Look for Land, Air, and Water"; dbS Productions: Charlottesville, VA, USA, 2008.</i>

- Для отфильтровки некоторого количества False Positive им помог расчёт стандартного отклонения (дисперсии) между пикселями региона и отбрасывание участков, где его значение ниже порогового.
Порог подбирался в ручную на конкретном датасете, так что насколько он подходит для всех случаев - непонятно. 

Основная часть статьи посвещена генерации region proposals.
Её влияние на качество итоговой модели оценивается отдельно по recall и presicion, при этом отмечено, что recall важнее, т. к. пропускать людей нужно как можно реже. 
Старые методы генерации proposals вроде edge boxes или mean shift на данной задаче показывают низкие результаты.
Наилучшие показатели получились при помощи Region Proposal Network (RPN), где-то около 95% recall.
На "простом" RPN, однако, низкий Prescision (41%). 
Попытки использовать фичи с разных масштабов при помощи Feature Pyramid Network (FPN) повысили Precision до 76%, но уменьшило Recall до 86%.
Варианты с объединением proposals от RPN и FPN, добавлением контекста или фильтрации по дисперсии дают Precision на уровне 55-68, возвращая Recall к 95%.

При этом, единственный протестированный ими one-stage detector - SSD - даёт сравнимый Recall (94.36%), но очень низкий Precision - 4.33%.
Думаю, у RetinaNet и Yolo с Precision дела всё-таки получше обстоят. 

Исходя из прочитаного могу сделать предположение, что большое количество False Positives, наблюдаемое и у нас - это особенность задачи, а не проблемы с нашим решением.
Возможно, с ними стоит бороться отдельно - например отфильтровывая найденные детекты, например, как предложено в статье, анализируя контектстную информацию/окружающий регион или фильтруя по каким-либо вычисленным статистикам. 
Если обучать/инференсить на кропах, тоже можно, мне кажется применять отфильтровывание easy background по быстро вычисленным показателям каких-либо статистики. 

Подход с two-stage detectors, однако, имеет слишком долгое время работы - в cтатье на обработку одного изображения уходило около 15 секунд (у "старых методов" - 43с).
Указания, использовалось CPU или GPU, нет, хотя характеристики приведены: Intel Xeon E5-2640v4 of 3.40 GHz, 4×16 GB DDR4 memory, and multi-GPU 4×NVIDIAGeForce GTX 1080Ti Turbo with 11 GB memory.
Изображения обрабатывались в полном масштабе, с разрезанием изначального фото 4000х3000 на блоки 500х500 px с горизонтальным перекрытием 100 и вертикальным 200.
