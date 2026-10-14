import {demo, select} from '../../shared/scene.js';
import * as M from '../../shared/math.js';
import * as G from '../../shared/geometry.js';

const C = G.colors;

export function renderLeakage(c, s) {
  const isMobile = (s.canvas?.getBoundingClientRect().width ?? 760) < 550;
  const width = isMobile ? 380 : 760;
  const height = isMobile ? 740 : 460;

  const { fit = 'train', diagnosis = 'good' } = s.params;
  const isLeakage = fit === 'all';
  const progress = s.progress; // 0 to 1

  // Raw data
  const trainData = [2, 4, 6, 8];
  const testData = [20, 24];
  const allData = [...trainData, ...testData];

  // Calculated statistics
  const muTrain = M.mean(trainData); // 5.0
  const muAll = M.mean(allData); // 10.666...
  const activeMu = isLeakage ? muAll : muTrain;

  // Transformed values
  const trainTransformed = trainData.map(v => v - activeMu);
  const testTransformed = testData.map(v => v - activeMu);

  // Diagnostics metadata
  const diagData = {
    good: {
      name: '良好擬合 (Good Generalization)',
      trainErr: 0.12,
      testErrClean: 0.15,
      testErrLeak: 0.11,
      tag: '✅ 理想泛化',
      tagColor: C.green,
      essence: '高偏差與高變異數均衡，成功學會真實規律。',
      action: '處方：表現健康穩定，可準備模型封裝與上線部署。'
    },
    under: {
      name: '欠擬合 (Underfitting)',
      trainErr: 0.74,
      testErrClean: 0.79,
      testErrLeak: 0.68,
      tag: '⚠️ 欠擬合 (高偏差)',
      tagColor: C.orange,
      essence: '模型過於簡化或特徵不足，連訓練集規律都抓不到。',
      action: '處方：增加模型複雜度、新增關鍵特徵、減少正則化約束。'
    },
    over: {
      name: '過度擬合 (Overfitting)',
      trainErr: 0.02,
      testErrClean: 0.86,
      testErrLeak: 0.16, // Data leakage tricks test error to look falsely low!
      tag: '❌ 過度擬合 (高變異)',
      tagColor: '#c93b2b',
      essence: '死記硬背訓練雜訊，泛化能力極差。',
      action: '處方：擴充訓練資料、加入正則化懲罰、提早停止 (Early Stopping)。'
    }
  }[diagnosis];

  // Animation step: 4 logical phases
  // Step 0: Data Partition & Firewall inspection (0 ~ 0.25)
  // Step 1: Fit computation (0.25 ~ 0.5)
  // Step 2: Transform applied (0.5 ~ 0.75)
  // Step 3: Diagnostics & Leakage consequence (0.75 ~ 1.0)
  const step = Math.min(3, Math.floor(progress * 4));

  // Background clear
  c.clearRect(0, 0, width, height);

  if (!isMobile) {
    renderDesktop(c, {
      fit, diagnosis, isLeakage, progress, step,
      trainData, testData, muTrain, muAll, activeMu,
      trainTransformed, testTransformed, diagData
    });
  } else {
    renderMobile(c, {
      fit, diagnosis, isLeakage, progress, step,
      trainData, testData, muTrain, muAll, activeMu,
      trainTransformed, testTransformed, diagData
    });
  }

  // Explanatory message
  let msg = '';
  if (isLeakage) {
    msg = `【資料洩漏 ❌】全資料混入計算使 μ = ${M.fmt(activeMu, 2)}（受測試集 20, 24 污染，暴增 ${Math.round((muAll - muTrain) / muTrain * 100)}%）。轉換後訓練集被往下拉扯；診斷情境：${diagData.name}。注意：洩漏會偽造虛假低誤差，欺騙評估！`;
  } else {
    msg = `【正確流程 ✅】僅用訓練集計算 μ = ${M.fmt(activeMu, 2)}。測試集受防火牆嚴格隔離，依訓練基準轉換為 [+15, +19]，評估真實客觀；診斷情境：${diagData.name}。`;
  }

  return {
    message: msg,
    metrics: {
      step,
      mean: activeMu,
      leakage: isLeakage,
      diagnosis,
      trainErr: diagData.trainErr,
      testErr: isLeakage ? diagData.testErrLeak : diagData.testErrClean
    }
  };
}

function renderDesktop(c, ctx) {
  const {
    isLeakage, progress, step,
    trainData, testData, muTrain, muAll, activeMu,
    trainTransformed, testTransformed, diagData
  } = ctx;

  // 1. Top Status Banner
  const titleColor = isLeakage ? '#c93b2b' : C.green;
  const statusTitle = isLeakage
    ? '❌ 致命資料洩漏：測試集滲入預處理 · 訓練基準遭受污染'
    : '✅ 正確標準流程：嚴格資料隔離 · 預處理只在 Train fit';
  G.text(c, statusTitle, 380, 22, 18, titleColor);

  // Step indicator chips across top
  const stepsNames = ['① 資料隔離', '② Fit 算均值 μ', '③ Transform 轉換', '④ 表現診斷與後果'];
  stepsNames.forEach((sName, i) => {
    const isCurrent = step === i;
    const isPast = step > i;
    const x = 110 + i * 180;
    const bg = isCurrent ? (isLeakage ? '#fdf2f2' : C.mint) : isPast ? '#f1f5f3' : C.white;
    const border = isCurrent ? (isLeakage ? '#c93b2b' : C.green) : isPast ? C.line : '#e2e8e4';
    const textCol = isCurrent ? (isLeakage ? '#c93b2b' : C.green) : isPast ? C.ink : C.muted;
    G.rect(c, x - 75, 38, 150, 24, bg, border, 12);
    G.text(c, sName, x, 50, 12, textCol);
  });

  // 2. Section A: Feature Preprocessing Pipeline (y: 68 to 252)
  G.rect(c, 15, 68, 730, 186, C.white, C.line, 8);
  G.text(c, '階段一：特徵預處理（標準化中心化 x - μ）', 30, 84, 13, C.muted, 'left');

  // --- Left: Original Data (Train & Test) ---
  // Train Box
  G.rect(c, 30, 98, 175, 68, C.mint, C.green, 6);
  G.text(c, 'Train 訓練集 (4 筆)', 42, 114, 13, C.green, 'left');
  trainData.forEach((v, i) => {
    const bx = 42 + i * 38;
    G.rect(c, bx, 128, 30, 26, C.white, C.green, 4);
    G.text(c, String(v), bx + 15, 141, 13, C.ink);
  });

  // Test Box
  G.rect(c, 30, 174, 175, 68, '#fef3e2', C.orange, 6);
  G.text(c, 'Test 測試集 (2 筆新樣本)', 42, 190, 13, C.orange, 'left');
  testData.forEach((v, i) => {
    const bx = 42 + i * 44;
    G.rect(c, bx, 204, 36, 26, C.white, C.orange, 4);
    G.text(c, String(v), bx + 18, 217, 13, C.ink);
  });

  // --- Middle: Firewall & Fit Engine ---
  // Firewall barrier
  const fwX = 222;
  if (!isLeakage) {
    // Solid secure firewall
    G.line(c, fwX, 168, fwX, 246, C.green, 3);
    G.rect(c, fwX - 10, 195, 20, 24, C.mint, C.green, 4);
    G.text(c, '🛡️', fwX, 207, 12);
    G.text(c, '數據防火牆', fwX + 16, 198, 11, C.green, 'left');
    G.text(c, '封鎖 Test 禁入', fwX + 16, 214, 10, C.muted, 'left');
  } else {
    // Breached red firewall
    G.line(c, fwX, 168, fwX, 246, '#c93b2b', 2, [4, 4]);
    G.rect(c, fwX - 10, 195, 20, 24, '#fdf2f2', '#c93b2b', 4);
    G.text(c, '⚠️', fwX, 207, 12);
    G.text(c, '防火牆破裂！', fwX + 16, 198, 11, '#c93b2b', 'left');
    G.text(c, 'Test 違規滲入', fwX + 16, 214, 10, '#c93b2b', 'left');
  }

  // Flow arrows into Fit box
  // Train arrow (always flows)
  G.arrow(c, 205, 132, 290, 132, C.green, 2);
  if (step === 1) {
    const tLocal = (progress - 0.25) / 0.25;
    G.travelling(c, [205, 132], [290, 132], tLocal, C.green, 6);
  }

  // Test arrow
  if (isLeakage) {
    G.arrow(c, 205, 208, 290, 165, '#c93b2b', 2);
    if (step === 1) {
      const tLocal = (progress - 0.25) / 0.25;
      G.travelling(c, [205, 208], [290, 165], tLocal, '#c93b2b', 6);
    }
  } else {
    // Blocked arrow hitting firewall
    G.line(c, 205, 208, fwX - 2, 208, C.orange, 2);
    G.line(c, fwX - 8, 202, fwX + 2, 214, '#c93b2b', 2);
    G.line(c, fwX - 8, 214, fwX + 2, 202, '#c93b2b', 2);
  }

  // Fit Box
  const fitBoxX = 295, fitBoxY = 100, fitBoxW = 165, fitBoxH = 135;
  const fitBorder = isLeakage ? '#c93b2b' : C.green;
  const fitBg = isLeakage ? '#fffbfb' : '#f9fdfa';
  G.rect(c, fitBoxX, fitBoxY, fitBoxW, fitBoxH, fitBg, fitBorder, 6);
  G.text(c, '1. Fit (計算統計量)', fitBoxX + fitBoxW / 2, fitBoxY + 18, 13, fitBorder);

  if (step === 0) {
    G.text(c, '等待資料流入...', fitBoxX + fitBoxW / 2, fitBoxY + 68, 13, C.muted);
    G.text(c, '按「下一步」開始', fitBoxX + fitBoxW / 2, fitBoxY + 92, 11, C.muted);
  } else {
    if (!isLeakage) {
      G.text(c, '只取 Train 計算：', fitBoxX + fitBoxW / 2, fitBoxY + 44, 12, C.muted);
      G.text(c, 'μ = (2+4+6+8) ÷ 4', fitBoxX + fitBoxW / 2, fitBoxY + 66, 12, C.ink);
      G.rect(c, fitBoxX + 18, fitBoxY + 84, fitBoxW - 36, 34, C.mint, C.green, 4);
      G.text(c, `平均值 μ = ${M.fmt(muTrain, 2)}`, fitBoxX + fitBoxW / 2, fitBoxY + 101, 15, C.green);
    } else {
      G.text(c, '全資料混算 (含測試)：', fitBoxX + fitBoxW / 2, fitBoxY + 42, 11, '#c93b2b');
      G.text(c, 'μ = (2+4+6+8+20+24)÷6', fitBoxX + fitBoxW / 2, fitBoxY + 62, 11, C.ink);
      G.rect(c, fitBoxX + 12, fitBoxY + 78, fitBoxW - 24, 46, '#fdf2f2', '#c93b2b', 4);
      G.text(c, `平均值 μ = ${M.fmt(muAll, 2)}`, fitBoxX + fitBoxW / 2, fitBoxY + 93, 15, '#c93b2b');
      G.text(c, `↑ 基準被拉高 +${Math.round((muAll - muTrain) / muTrain * 100)}%！`, fitBoxX + fitBoxW / 2, fitBoxY + 112, 11, '#c93b2b');
    }
  }

  // --- Right: Transform Engine & Transformed Data ---
  // Arrow from Fit to Transform
  G.arrow(c, fitBoxX + fitBoxW, 165, 485, 165, isLeakage ? '#c93b2b' : C.green, 2);
  G.text(c, '套用 μ', 472, 150, 11, C.muted);
  if (step === 2) {
    const tLocal = (progress - 0.5) / 0.25;
    G.travelling(c, [fitBoxX + fitBoxW, 165], [485, 165], tLocal, isLeakage ? '#c93b2b' : C.green, 6);
  }

  // Transform Result Area
  const transX = 490, transY = 90, transW = 240, transH = 152;
  G.rect(c, transX, transY, transW, transH, C.paper, C.line, 6);
  G.text(c, `2. Transform (特徵縮放：x' = x - μ)`, transX + transW / 2, transY + 16, 12, C.ink);

  if (step < 2) {
    G.text(c, '等待 Fit 統計量產出...', transX + transW / 2, transY + 70, 13, C.muted);
    G.text(c, '按「下一步」查看轉換結果', transX + transW / 2, transY + 95, 11, C.muted);
  } else {
    // Train transformed
    G.text(c, `Train 轉換後：`, transX + 15, transY + 38, 11, C.green, 'left');
    trainTransformed.forEach((v, i) => {
      const bx = transX + 15 + i * 53;
      G.rect(c, bx, transY + 46, 48, 22, C.white, C.green, 3);
      G.text(c, v > 0 ? `+${M.fmt(v, 1)}` : M.fmt(v, 1), bx + 24, transY + 57, 11, C.ink);
    });

    // Test transformed
    G.text(c, `Test 轉換後：`, transX + 15, transY + 84, 11, C.orange, 'left');
    testTransformed.forEach((v, i) => {
      const bx = transX + 15 + i * 58;
      G.rect(c, bx, transY + 92, 53, 22, C.white, C.orange, 3);
      G.text(c, `+${M.fmt(v, 1)}`, bx + 26, transY + 103, 11, C.ink);
    });

    // Evaluation Verdict Badge
    if (!isLeakage) {
      G.rect(c, transX + 10, transY + 120, transW - 20, 24, C.mint, C.green, 4);
      G.text(c, '✅ 基準純淨：客觀呈現測試集偏異', transX + transW / 2, transY + 132, 11, C.green);
    } else {
      G.rect(c, transX + 10, transY + 120, transW - 20, 24, '#fdf2f2', '#c93b2b', 4);
      G.text(c, '❌ 基準污染：訓練特徵提前偷看考卷', transX + transW / 2, transY + 132, 11, '#c93b2b');
    }
  }

  // 3. Section B: Performance Diagnostics Matrix (y: 262 to 445)
  G.rect(c, 15, 262, 730, 185, C.paper, C.line, 8);
  G.text(c, '階段二：模型表現診斷矩陣（誤差高低與泛化健康檢查）', 30, 278, 13, C.ink, 'left');

  // Diagnostic Sub-panels
  // Panel 1: Error Meter (x: 30 to 225)
  G.rect(c, 30, 292, 195, 145, C.white, C.line, 6);
  G.text(c, '誤差表現測量 (Error)', 127, 308, 12, C.ink);

  // Train Error Bar
  G.text(c, `訓練集誤差 (Train)`, 42, 332, 11, C.muted, 'left');
  G.rect(c, 42, 340, 130, 14, '#eef2f0', null, 3);
  const trainBarW = Math.max(6, diagData.trainErr * 130);
  G.rect(c, 42, 340, trainBarW, 14, diagData.trainErr > 0.5 ? C.orange : C.green, null, 3);
  G.text(c, `${(diagData.trainErr * 100).toFixed(0)}%`, 180, 347, 11, C.ink, 'left');

  // Test Error Bar
  G.text(c, `測試集誤差 (Test)`, 42, 372, 11, C.muted, 'left');
  G.rect(c, 42, 380, 130, 14, '#eef2f0', null, 3);
  const currentTestErr = isLeakage ? diagData.testErrLeak : diagData.testErrClean;
  const testBarW = Math.max(6, currentTestErr * 130);
  const testBarColor = currentTestErr > 0.5 ? '#c93b2b' : (isLeakage ? '#e67e22' : C.green);
  G.rect(c, 42, 380, testBarW, 14, testBarColor, null, 3);
  G.text(c, `${(currentTestErr * 100).toFixed(0)}%`, 180, 387, 11, C.ink, 'left');

  if (isLeakage) {
    G.text(c, '⚠️ 測試誤差被虛假壓低！', 42, 412, 11, '#c93b2b', 'left');
    G.text(c, `(真實未洩漏應為 ${(diagData.testErrClean * 100).toFixed(0)}%)`, 42, 426, 10, C.muted, 'left');
  } else {
    G.text(c, '🛡️ 真實反映泛化誤差能力', 42, 412, 11, C.green, 'left');
  }

  // Panel 2: Geometric Fitting Curves (x: 235 to 455)
  G.rect(c, 235, 292, 220, 145, C.white, C.line, 6);
  G.text(c, '幾何擬合狀態縮圖', 345, 308, 12, C.ink);
  drawMiniCurve(c, 245, 320, 200, 105, ctx.diagnosis);

  // Panel 3: Diagnosis Verdict & Remedy (x: 465 to 730)
  G.rect(c, 465, 292, 265, 145, C.white, C.line, 6);
  G.rect(c, 475, 302, 245, 26, diagData.tagColor === C.green ? C.mint : '#fdf2f2', diagData.tagColor, 4);
  G.text(c, diagData.tag, 597, 315, 13, diagData.tagColor);

  G.text(c, '【核心成因】', 478, 342, 11, C.ink, 'left');
  G.text(c, diagData.essence, 478, 358, 11, C.muted, 'left');

  G.text(c, '【對策處方】', 478, 384, 11, C.green, 'left');
  // Break prescription into two lines if long
  if (diagData.action.length > 20) {
    G.text(c, diagData.action.slice(0, 19), 478, 400, 11, C.ink, 'left');
    G.text(c, diagData.action.slice(19), 478, 416, 11, C.ink, 'left');
  } else {
    G.text(c, diagData.action, 478, 404, 11, C.ink, 'left');
  }
}

function renderMobile(c, ctx) {
  const {
    isLeakage, progress, step,
    trainData, testData, muTrain, muAll, activeMu,
    trainTransformed, testTransformed, diagData
  } = ctx;

  // Title
  const titleColor = isLeakage ? '#c93b2b' : C.green;
  G.text(c, isLeakage ? '❌ 資料洩漏：測試集滲入 fit' : '✅ 正確流程：僅 Train fit', 190, 20, 16, titleColor);

  // Step tracker
  const stepsNames = ['① 隔離', '② Fit', '③ 轉換', '④ 診斷'];
  stepsNames.forEach((sName, i) => {
    const isCurrent = step === i;
    const x = 50 + i * 90;
    const bg = isCurrent ? (isLeakage ? '#fdf2f2' : C.mint) : C.white;
    const border = isCurrent ? (isLeakage ? '#c93b2b' : C.green) : C.line;
    G.rect(c, x - 38, 36, 76, 22, bg, border, 11);
    G.text(c, sName, x, 47, 11, isCurrent ? (isLeakage ? '#c93b2b' : C.green) : C.muted);
  });

  // Preprocessing Box (y: 65 to 395)
  G.rect(c, 10, 65, 360, 330, C.white, C.line, 8);
  G.text(c, '階段一：特徵縮放流程 (x - μ)', 20, 80, 12, C.muted, 'left');

  // Train box
  G.rect(c, 20, 95, 160, 60, C.mint, C.green, 5);
  G.text(c, 'Train 訓練集：2, 4, 6, 8', 100, 115, 12, C.green);
  G.text(c, '內部均值 = 5.0', 100, 138, 11, C.muted);

  // Test box
  G.rect(c, 195, 95, 165, 60, '#fef3e2', C.orange, 5);
  G.text(c, 'Test 測試集：20, 24', 277, 115, 12, C.orange);
  G.text(c, '未見過之新資料', 277, 138, 11, C.muted);

  // Firewall and Fit
  const fitY = 175;
  G.rect(c, 20, fitY, 340, 95, isLeakage ? '#fffbfb' : '#f9fdfa', isLeakage ? '#c93b2b' : C.green, 6);
  G.text(c, '1. Fit 統計量計算器', 190, fitY + 16, 13, isLeakage ? '#c93b2b' : C.green);

  if (step === 0) {
    G.text(c, '按「下一步」開始計算統計量...', 190, fitY + 52, 12, C.muted);
  } else {
    if (!isLeakage) {
      G.text(c, '🛡️ 防火牆阻擋 Test，只取 Train 計算：', 190, fitY + 40, 11, C.green);
      G.text(c, `平均值 μ = ${M.fmt(muTrain, 2)} (標準純淨)`, 190, fitY + 68, 15, C.green);
    } else {
      G.text(c, '⚠️ 防火牆被破壞！Test 滲入計算：', 190, fitY + 38, 11, '#c93b2b');
      G.text(c, `平均值 μ = ${M.fmt(muAll, 2)} (暴增 +113%)`, 190, fitY + 62, 15, '#c93b2b');
      G.text(c, '測試資訊已污染訓練基準', 190, fitY + 80, 10, '#c93b2b');
    }
  }

  // Transform outputs
  const transY = 285;
  G.rect(c, 20, transY, 340, 95, C.paper, C.line, 6);
  G.text(c, "2. Transform 結果 (x' = x - μ)", 190, transY + 16, 12, C.ink);

  if (step < 2) {
    G.text(c, '等待 Fit 統計量轉換中...', 190, transY + 52, 12, C.muted);
  } else {
    const trStr = trainTransformed.map(v => v > 0 ? `+${M.fmt(v, 1)}` : M.fmt(v, 1)).join(', ');
    const teStr = testTransformed.map(v => `+${M.fmt(v, 1)}`).join(', ');
    G.text(c, `Train: [ ${trStr} ]`, 190, transY + 40, 11, C.green);
    G.text(c, `Test:  [ ${teStr} ]`, 190, transY + 60, 11, C.orange);
    G.text(c, isLeakage ? '❌ 訓練基準被往下拉扯，提前偷看考卷' : '✅ 基準公正，真實反映新資料偏異', 190, transY + 82, 11, isLeakage ? '#c93b2b' : C.green);
  }

  // Diagnostics Box (y: 405 to 725)
  G.rect(c, 10, 405, 360, 320, C.paper, C.line, 8);
  G.text(c, '階段二：模型診斷與後果', 20, 420, 12, C.ink, 'left');

  // Error bars
  G.rect(c, 20, 435, 340, 85, C.white, C.line, 5);
  G.text(c, `Train 誤差: ${(diagData.trainErr * 100).toFixed(0)}%`, 35, 455, 11, C.ink, 'left');
  G.rect(c, 130, 447, 180, 12, '#eef2f0', null, 3);
  G.rect(c, 130, 447, Math.max(6, diagData.trainErr * 180), 12, diagData.trainErr > 0.5 ? C.orange : C.green, null, 3);

  const curTestErr = isLeakage ? diagData.testErrLeak : diagData.testErrClean;
  G.text(c, `Test 誤差: ${(curTestErr * 100).toFixed(0)}%`, 35, 480, 11, isLeakage ? '#c93b2b' : C.ink, 'left');
  G.rect(c, 130, 472, 180, 12, '#eef2f0', null, 3);
  G.rect(c, 130, 472, Math.max(6, curTestErr * 180), 12, curTestErr > 0.5 ? '#c93b2b' : (isLeakage ? '#e67e22' : C.green), null, 3);

  if (isLeakage) {
    G.text(c, `⚠️ 洩漏使測試誤差虛假偏低 (真實應為 ${(diagData.testErrClean * 100).toFixed(0)}%)`, 190, 506, 10, '#c93b2b');
  }

  // Mini Curve
  G.rect(c, 20, 528, 340, 95, C.white, C.line, 5);
  drawMiniCurve(c, 30, 535, 320, 80, ctx.diagnosis);

  // Verdict
  G.rect(c, 20, 630, 340, 85, C.white, C.line, 5);
  G.rect(c, 30, 638, 320, 24, diagData.tagColor === C.green ? C.mint : '#fdf2f2', diagData.tagColor, 4);
  G.text(c, diagData.tag, 190, 650, 12, diagData.tagColor);
  G.text(c, diagData.essence, 190, 676, 10, C.muted);
  G.text(c, diagData.action, 190, 696, 10, C.ink);
}

function drawMiniCurve(c, x, y, w, h, mode) {
  // Mini axes
  G.line(c, x + 15, y + h - 15, x + w - 10, y + h - 15, '#cbd5e1', 1);
  G.line(c, x + 15, y + 10, x + 15, y + h - 15, '#cbd5e1', 1);
  G.text(c, '特徵 X', x + w - 20, y + h - 5, 9, C.muted);
  G.text(c, '目標 y', x + 15, y + 6, 9, C.muted);

  // Sample data points along a quadratic trend: y = 0.6 * (x-0.5)^2 + 0.2 + noise
  const pts = [
    [0.1, 0.65], [0.2, 0.45], [0.3, 0.32], [0.45, 0.24],
    [0.6, 0.28], [0.75, 0.42], [0.85, 0.68]
  ];

  const mapX = vx => x + 25 + vx * (w - 45);
  const mapY = vy => y + h - 20 - vy * (h - 35);

  // Draw points
  pts.forEach(([px, py], i) => {
    G.circle(c, mapX(px), mapY(py), 3.5, i % 2 === 0 ? C.green : C.orange);
  });

  // Curve according to mode
  c.beginPath();
  if (mode === 'under') {
    // Stiff straight line
    c.moveTo(mapX(0.05), mapY(0.40));
    c.lineTo(mapX(0.95), mapY(0.40));
    c.strokeStyle = C.orange;
    c.lineWidth = 2.5;
    c.stroke();
    G.text(c, '直線太僵硬 (高偏差)', x + w / 2, y + 25, 11, C.orange);
  } else if (mode === 'good') {
    // Smooth quadratic curve
    for (let i = 0; i <= 30; i++) {
      const tx = 0.05 + i * (0.9 / 30);
      const ty = 1.1 * Math.pow(tx - 0.48, 2) + 0.25;
      const sx = mapX(tx), sy = mapY(ty);
      if (i === 0) c.moveTo(sx, sy);
      else c.lineTo(sx, sy);
    }
    c.strokeStyle = C.green;
    c.lineWidth = 2.5;
    c.stroke();
    G.text(c, '自然平滑曲線 (泛化佳)', x + w / 2, y + 25, 11, C.green);
  } else {
    // Overfitting: violent polynomial oscillation
    for (let i = 0; i <= 60; i++) {
      const tx = 0.05 + i * (0.9 / 60);
      const ty = 1.1 * Math.pow(tx - 0.48, 2) + 0.25 + 0.16 * Math.sin(tx * 25);
      const sx = mapX(tx), sy = mapY(ty);
      if (i === 0) c.moveTo(sx, sy);
      else c.lineTo(sx, sy);
    }
    c.strokeStyle = '#c93b2b';
    c.lineWidth = 2;
    c.stroke();
    G.text(c, '劇烈抖動硬背 (高變異)', x + w / 2, y + 25, 11, '#c93b2b');
  }
}

export const leakage = demo(
  'leakage',
  '資料洩漏與診斷',
  [
    select('fit', '縮放器 fit 來源', [
      ['train', '只用 Train（正確標準 ✅）'],
      ['all', '全資料（資料洩漏 ❌）']
    ]),
    select('diagnosis', '誤差診斷情境', [
      ['good', '良好擬合（訓練低／驗證低）'],
      ['under', '欠擬合（訓練高／驗證高）'],
      ['over', '過度擬合（訓練低／驗證高）']
    ])
  ],
  renderLeakage,
  '按「播放」或「下一步」觀察：1. 資料隔離 ➔ 2. Fit 計算 μ ➔ 3. Transform 數值轉變 ➔ 4. 表現診斷與後果。',
  {
    steps: 4,
    viewSize: width => width < 550 ? [380, 740] : [760, 460],
    guide: [
      '① 步驟 1（資料隔離）：左側展示 Train (2,4,6,8) 與 Test (20,24)；觀察中間是否有「數據防火牆」阻止測試集流入。',
      '② 步驟 2（Fit 計算統計量）：觀察平均值 μ 的計算。只用 Train 時 μ=5.00；若全資料洩漏，Test 的 20,24 會使 μ 暴增至 10.67！',
      '③ 步驟 3（Transform 套用轉換）：看右側每筆數據扣除 μ 後的實際數值 (x - μ)。比較洩漏如何扭曲了訓練基準。',
      '④ 步驟 4（表現診斷與警示）：下方即時顯示對應的誤差長條柱與擬合曲線。注意：一旦資料洩漏，測試誤差會產生虛假的良好假象！',
      '可隨時切換「全資料（資料洩漏）」與三種「誤差診斷情境」，對照兩者變化。'
    ]
  }
);
