<b>Visual-Based Person Detection for Search-and-Rescue with UAS: Humans vs. Machine Learning Algorithm</b>, <i>статья от октября 2020, авторы Sven Gotovac
(University of Split), Danijel Zelenika (University of Mostar), Zeljko Marusic (University of Mostar), Dunja Božić-Štulić (University of Split)</i>. <a href="https://www.researchgate.net/publication/345069901_Visual-Based_Person_Detection_for_Search-and-Rescue_with_UAS_Humans_vs_Machine_Learning_Algorithm">Оригинал на ResearchGate</a>. 

Исследование посвящено вопросу, насколько помощь нейросетей помогает повысить качество последующего анализа фотографий людьми. Для анализа использовались фотографии со съёмок, максимально имитирующих настоящие поисковые операции (всего четыре иммитации в разных условиях), а так же были привлечены специалисты, занимающиеся анализом реальных аналогичных снимков. 

Результаты и модели, и людей, оценивались по Recall и Precision отдельно. Модель, которую использовали авторы, показывала хороший Recall 80-100%, но почему-то очень  низкий Precision - 7%.
В трёх из четырёх иммитациях помощь нейросетки повысила показываемый людьми Recall, в одной - ухудшила его. Precision в трёх (других) операциях ухудшился по сравнению с человеческим, в одной - улучшился. В целом удивляет, что эксперименты проводились с использованием настолько низкой по Precision модели. Даже относительно Recall в татье указано, что основная причина проседаний состоит в том, что люди концентрировались на предложениях нейросетки и меньше внимания уделяли самостоятельному изучению снимков. Если бы модель выдавала меньше False Positives, проблема, очевидно, проявлялась бы в меньшей степени.
Возможно, стоит отметить, что качество анализа снимка, проводимого людьми с большим опытом такой работы ощутимо выше, чем у случайных волонтеров, и по Recall, и по Precision, и по времени обработки. Однако описанные выше эффекты действовали схожим образом и для "экспертов" и для не-экспертов.

С технической стороны модель, которую использовали авторы, базируется на Regions of Interests (ROI), генерируемые с использованием сверточной VGG-16 аналогичным Faster-RCNN способом. 
После выделения фич из ROI авторы, однако, используют не классификацию, а высчитывают "расстояния" между парами векторов 1х1х512, чтобы выделить наиболее отличающиеся. 

<sub>(<b>Примечание</b>: подход выглядит похожим на  triplet loss и similaruty learning, концентрирующиеся не на определении класса объекта из жёсткого перечня, а на высчитываниии векторов характеристик объектов так, чтобы их потом можно было бы сравнивать с друг другом по расстоянию, и за счёт этого относить к какой-либо категории или группе близких объектов. ОДнако обычно в таких подходах всё-таки модель обучают  высчитывать векторы характеристик. В triplet loss тройки объектов "эталон-похожий-отличающийся" прогоняют через нейросеть, которая учится приписывать похожему объекту близкий к эталону вектор, а отличающемуся - далекий. Здесь же авторы практически просто берут, что получилось из region proposals и на основании этого делают выводы, что семантически близко, а что - нет. Возможно, такой низкий Precision у них как раз получился из-за незрелости данного этапа обработки.
В целом, однако, подход со сравнением сравнивнение регионов-кандидатов, кажется, имеет в себе здравое зерно. Поскольку ландшафт на одной фотке, как правило, однородный, то камень, похожий на человека, будет так же похож и на камни из других ROI того же снимка, что может дать возможность отфильтровки.)</sub> 

Скорость обработки одного изображения моделью явным образом не указана. Приведено только сокращение времени обработки человеком с её помощью - с 1495s до 1110s на 40-50 снимков. (Это 22 - 37 секунды на снимок). Включается ли в это время работа самой нейросети - неясно.
Из второстепенных моментов можно отметить наблюдение о том, что часто причиной False Positive становится тень на изображении. Возмоно, нейросети оферфитятся на тени людей, встречаемые на многих фотографиях. 
Другой сложностью, приводящей уже к False Negative становится одежда (камуфляж, белая, коричневая, зеленая, темно-синяя и т. п.). Ещё один сильно усложняющий анализ момент - сильная загороженность человека или его расположение на границе изображения. 

Авторы так же пишут, что при поисках собираются данные о потерявшемся и используется статистика и экспертиза, накопленная за время предыдущих поисков. Есть закономерности в том, где и в каком положении находят потерявшихся, причём эти закономерности зачастую зависят от категории пропавшего: ребенок, старик, ментально нездоровый, велосипедист, турист и т. п. Но для использования подобной статистики в технических решениях, очевидно, нужно ею для начала обладать :)
