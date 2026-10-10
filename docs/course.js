/* Dependency-free interactions. All added numerical models are teaching demos. */
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const chapters = [...document.querySelectorAll('.chapter')];
  const links = [...document.querySelectorAll('.course-nav-links a')];
  let current = 0;
  let presenting = false;
  function updateReading(index) {
    current = index;
    links.forEach((link, i) => i === index ? link.setAttribute('aria-current', 'true') : link.removeAttribute('aria-current'));
    byId('reading-progress').value = index + 1;
    byId('reading-status').textContent = `第 ${index + 1} / 15 章`;
    byId('focus-status').textContent = `${index + 1} / 15`;
    chapters.forEach((chapter, i) => chapter.classList.toggle('is-current', i === index));
    byId('previous-chapter').disabled = index === 0;
    byId('next-chapter').disabled = index === 14;
  }
  const observer = new IntersectionObserver(entries => {
    if (presenting) return;
    const visible = entries.filter(entry => entry.isIntersecting).sort((a,b) => a.boundingClientRect.top - b.boundingClientRect.top);
    if (visible.length) updateReading(chapters.indexOf(visible[0].target));
  }, {rootMargin: '-5% 0px -65% 0px', threshold: 0});
  chapters.forEach(chapter => observer.observe(chapter));
  function go(index) {
    updateReading(Math.max(0, Math.min(14, index)));
    if (presenting) window.scrollTo({top: 0, behavior: 'instant'});
    else chapters[current].scrollIntoView({behavior: 'smooth'});
  }
  byId('start-presentation').addEventListener('click', () => {
    presenting = true;
    document.body.classList.add('presenting');
    go(current);
    byId('exit-presentation').focus();
  });
  function exitPresentation() {
    presenting = false;
    document.body.classList.remove('presenting');
    chapters[current].scrollIntoView();
    byId('start-presentation').focus({preventScroll:true});
  }
  byId('exit-presentation').addEventListener('click', exitPresentation);
  byId('previous-chapter').addEventListener('click', () => go(current - 1));
  byId('next-chapter').addEventListener('click', () => go(current + 1));
  document.addEventListener('keydown', event => {
    if (!presenting) return;
    if (event.key === 'Escape') {exitPresentation(); return;}
    if (event.target.closest('input,select,textarea,summary,[role="tab"]')) return;
    if (event.key === 'ArrowRight') {event.preventDefault(); go(current + 1);}
    if (event.key === 'ArrowLeft') {event.preventDefault(); go(current - 1);}
  });
  byId('print-course').addEventListener('click', () => window.print());
  updateReading(0);

  // Four-stage data-driven learning explanation.
  const pipeline = [
    '收集資料：房價、病歷或顧客購買紀錄。監督式學習也需要對應的答案。',
    '學習擬合：比較預測與答案，計算誤差並調整模型參數。',
    '提煉規律：建立輸入 X 與輸出 y 的映射 y = f(X)。',
    '預測新資料：把未見過的輸入交給模型；還需評估模型能否泛化。'
  ];
  let pipelineStep = 0;
  function drawPipeline() {
    [...byId('pipeline-stages').children].forEach((stage, i) => stage.classList.toggle('active', i === pipelineStep));
    byId('pipeline-explanation').textContent = pipeline[pipelineStep];
    byId('pipeline-next').textContent = pipelineStep === 3 ? '重新開始' : '下一步 →';
  }
  byId('pipeline-next').addEventListener('click', () => {pipelineStep = (pipelineStep + 1) % 4; drawPipeline();});
  drawPipeline();

  // One logistic output trained on a single labelled dog, not image recognition.
  let logit = Math.log(.8 / .2);
  let iterations = 0;
  function drawTraining() {
    const cat = 1 / (1 + Math.exp(-logit));
    const dog = 1 - cat;
    byId('cat-probability').textContent = `${(cat * 100).toFixed(1)}%`;
    byId('dog-probability').textContent = `${(dog * 100).toFixed(1)}%`;
    byId('cat-bar').value = cat;
    byId('dog-bar').value = dog;
    byId('training-loss').textContent = (-Math.log(dog)).toFixed(4);
    byId('training-count').textContent = iterations;
    byId('training-result').textContent = `目前預測：${cat > dog ? '貓（與真實標籤不符）' : '狗（與真實標籤一致）'}`;
  }
  function train(count) {
    for (let i=0;i<count;i++) {logit -= .5 / (1 + Math.exp(-logit)); iterations++;}
    drawTraining();
  }
  byId('train-once').addEventListener('click', () => train(1));
  byId('train-ten').addEventListener('click', () => train(10));
  byId('reset-training').addEventListener('click', () => {logit = Math.log(4); iterations = 0; drawTraining();});
  drawTraining();

  // Actual ordinary least-squares fit of the six points in slide 12.
  const houses = [[18,180],[23,200],[42,400],[50,500],[66,650],[95,950]];
  const px = x => 48 + x * 4.2;
  const py = y => 284 - y * .23;
  const svgNS = 'http://www.w3.org/2000/svg';
  houses.forEach(([x,y]) => {
    const dot = document.createElementNS(svgNS, 'circle');
    dot.setAttribute('cx',px(x)); dot.setAttribute('cy',py(y)); dot.setAttribute('r',5);
    dot.setAttribute('fill','#246d58');
    const title = document.createElementNS(svgNS,'title'); title.textContent = `${x} 坪，${y} 萬元`;
    dot.append(title); byId('house-points').append(dot);
  });
  function drawRegression() {
    const slope = Number(byId('slope').value), intercept = Number(byId('intercept').value), area = Number(byId('area').value);
    byId('slope-value').textContent = slope.toFixed(2);
    byId('intercept-value').textContent = intercept.toFixed(1);
    byId('area-value').textContent = area;
    byId('regression-equation').textContent = `ŷ = ${slope.toFixed(2)} × 坪數 ${intercept < 0 ? '−' : '+'} ${Math.abs(intercept).toFixed(1)}`;
    byId('regression-mse').textContent = (houses.reduce((sum,[x,y]) => sum + (slope*x+intercept-y)**2,0) / houses.length).toFixed(2);
    byId('house-prediction').textContent = `${(slope * area + intercept).toFixed(1)} 萬元`;
    const line = byId('regression-line');
    line.setAttribute('x1',px(0)); line.setAttribute('y1',py(intercept));
    line.setAttribute('x2',px(100)); line.setAttribute('y2',py(slope*100+intercept));
    byId('prediction-point').setAttribute('cx',px(area));
    byId('prediction-point').setAttribute('cy',py(slope*area+intercept));
  }
  ['slope','intercept','area'].forEach(id => byId(id).addEventListener('input',drawRegression));
  byId('fit-regression').addEventListener('click', () => {
    const meanX = houses.reduce((s,[x])=>s+x,0)/houses.length;
    const meanY = houses.reduce((s,[,y])=>s+y,0)/houses.length;
    const slope = houses.reduce((s,[x,y])=>s+(x-meanX)*(y-meanY),0)/houses.reduce((s,[x])=>s+(x-meanX)**2,0);
    byId('slope').value = slope.toFixed(2); byId('intercept').value = (meanY-slope*meanX).toFixed(1);
    drawRegression();
    byId('fit-message').textContent = '已用六筆資料計算最小平方法；顯示的參數經四捨五入。';
  });
  byId('slide-regression').addEventListener('click', () => {
    byId('slope').value = 10.2; byId('intercept').value = -15; drawRegression();
    byId('fit-message').textContent = '已載入原簡報的示意直線 y = 10.2x − 15，與精確擬合略有差異。';
  });
  drawRegression();

  // Explicitly illustrative four-feature formula; coefficients are not in the deck.
  const inferenceForm = byId('inference-form');
  function clearInference() {
    byId('inference-result').textContent = '調整特徵後，按「執行推論」。';
    byId('inference-calculation').textContent = '';
    [...byId('inference-stages').children].forEach(stage=>stage.classList.remove('active'));
  }
  inferenceForm.addEventListener('input',clearInference);
  inferenceForm.addEventListener('submit', event => {
    event.preventDefault();
    if (!inferenceForm.reportValidity()) return;
    const area = Number(byId('infer-area').value), rooms = Number(byId('infer-rooms').value), age = Number(byId('infer-age').value);
    const location = Number(byId('infer-location').value);
    const price = 10*area + location + 5*rooms - 5*age;
    byId('inference-result').textContent = `教學模型預測：${price.toFixed(0)} 萬元`;
    byId('inference-calculation').textContent = `10 × ${area} + 地點係數 ${location} + 5 × ${rooms} − 5 × ${age} = ${price.toFixed(0)}`;
    [...byId('inference-stages').children].forEach(stage=>stage.classList.add('active'));
  });
  byId('reset-inference').addEventListener('click', () => {inferenceForm.reset();clearInference();});

  document.querySelectorAll('[data-quiz-answer]').forEach(button => button.addEventListener('click', () => {
    const quiz = button.closest('.quiz');
    quiz.querySelectorAll('button').forEach(option=>option.setAttribute('aria-pressed',String(option === button)));
    quiz.querySelector('[role="status"]').textContent = button.dataset.quizAnswer;
  }));
})();
