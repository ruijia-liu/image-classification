function results = analyse_band_importance()
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%ANALYSE_BAND_IMPORTANCE Interpretable channel importance by TIME.

cfg = config();
ensureOutputDirs(cfg);
rng(cfg.RANDOM_SEED);

warningFile = fullfile(cfg.paths.logs, 'band_importance_warnings.txt');
warningFid = fopen(warningFile, 'w');
if warningFid < 0
    error('Could not open warning log for writing: %s', warningFile);
end
cleanup = onCleanup(@() fclose(warningFid)); %#ok<NASGU>
fprintf(warningFid, 'Band importance warnings\n');
fprintf(warningFid, 'Generated: %s\n\n', datestr(now, 31));

obsTable = load_cached_data("observation");
channels = cfg.expected_channels(:).';
featureNames = "mean_band_" + string(channels);
validateFeatureColumns(obsTable, featureNames);

runSlow = getRunSlowAnalyses(cfg);
timeValues = sort(unique(obsTable.time)).';

univariateParts = {};
svmParts = {};
permutationParts = {};

for d = 1:numel(timeValues)
    timeValue = timeValues(d);
    T = obsTable(obsTable.time == timeValue, :);
    fprintf('Band importance TIME %s (%d observations)\n', string(timeValue), height(T));

    [canRun, reason] = validateTimeSubset(T);
    if ~canRun
        logWarning(warningFid, 'Skipping TIME %s: %s', string(timeValue), reason);
        continue;
    end

    univariateParts{end + 1, 1} = computeUnivariateImportance(T, featureNames, channels, warningFid); %#ok<AGROW>
    svmParts{end + 1, 1} = computeLinearSvmImportance(T, featureNames, channels, cfg, warningFid); %#ok<AGROW>

    if runSlow
        permutationParts{end + 1, 1} = computePermutationImportance(T, featureNames, channels, cfg, warningFid); %#ok<AGROW>
    end
end

univariateTable = vertcatOrEmpty(univariateParts);
svmTable = vertcatOrEmpty(svmParts);
permutationTable = vertcatOrEmpty(permutationParts);

rankedTable = makeRankedTable(univariateTable, svmTable, permutationTable);
writeOutputs(cfg, univariateTable, svmTable, permutationTable, rankedTable);
makeFigures(cfg, univariateTable, svmTable, rankedTable);

results = struct();
results.univariate_importance = univariateTable;
results.svm_importance = svmTable;
results.permutation_importance = permutationTable;
results.ranked_importance = rankedTable;
results.warning_file = warningFile;

fprintf('Band importance analysis complete.\n');
fprintf('Univariate rows: %d\n', height(univariateTable));
fprintf('SVM importance rows: %d\n', height(svmTable));
fprintf('Permutation rows: %d\n', height(permutationTable));
fprintf('Outputs saved to: %s\n', cfg.paths.tables);
fprintf('Figures saved to: %s\n', cfg.paths.figures);
fprintf('Warnings saved to: %s\n', warningFile);

end

function ensureOutputDirs(cfg)
dirs = {cfg.paths.tables, cfg.paths.logs, cfg.paths.figures, cfg.paths.cache};
for k = 1:numel(dirs)
    if ~exist(dirs{k}, 'dir')
        mkdir(dirs{k});
    end
end
end

function validateFeatureColumns(T, featureNames)
missing = setdiff(featureNames, string(T.Properties.VariableNames), 'stable');
if ~isempty(missing)
    error('Observation table is missing required feature columns: %s', strjoin(missing, ', '));
end
end

function runSlow = getRunSlowAnalyses(cfg)
runSlow = false;
if isfield(cfg, 'RUN_SLOW_ANALYSES')
    runSlow = logical(cfg.RUN_SLOW_ANALYSES);
elseif isfield(cfg, 'analysis') && isfield(cfg.analysis, 'run_optional_slow_analyses')
    runSlow = logical(cfg.analysis.run_optional_slow_analyses);
end
end

function [canRun, reason] = validateTimeSubset(T)
canRun = true;
reason = "";
if height(T) < 6
    canRun = false;
    reason = "fewer than 6 observations";
elseif numel(unique(T.analysis_label)) < 2
    canRun = false;
    reason = "only one class is present";
elseif countSubjects(T, "class1") < 2
    canRun = false;
    reason = "fewer than 2 class1 subjects";
elseif countSubjects(T, "class0") < 2
    canRun = false;
    reason = "fewer than 2 class0 subjects";
end
end

function out = computeUnivariateImportance(T, featureNames, channels, warningFid)
% One independent source contributes one value to each univariate test.
[g, ~] = findgroups(T.subject_key);
first = splitapply(@(x) x(1), (1:height(T))', g);
subjectTable = T(first,:);
for name = featureNames
    subjectTable.(name) = splitapply(@(x) mean(x,'omitnan'), T.(name), g);
end
T = subjectTable;
timeValue = T.time(1);
rows = cell(numel(featureNames), 1);
pValues = NaN(numel(featureNames), 1);

for b = 1:numel(featureNames)
    xClass1 = T.(featureNames(b))(T.analysis_label == "class1");
    xClass0 = T.(featureNames(b))(T.analysis_label == "class0");
    nInf = numel(xClass1);
    nUninf = numel(xClass0);

    class1Mean = mean(xClass1, 'omitnan');
    class0Mean = mean(xClass0, 'omitnan');
    meanDifference = class1Mean - class0Mean;
    effectSize = cohensD(xClass1, xClass0);

    pValue = NaN;
    testName = "two-sample t-test";
    if nInf >= 2 && nUninf >= 2 && all(isfinite([xClass1; xClass0]))
        if nInf < 5 || nUninf < 5
            logWarning(warningFid, 'TIME %s channel %s has small sample size for p-value: class1=%d, class0=%d.', ...
                string(timeValue), string(channels(b)), nInf, nUninf);
        end
        if nInf < 8 || nUninf < 8
            logWarning(warningFid, 'TIME %s channel %s has limited normality assessment power.', ...
                string(timeValue), string(channels(b)));
        end
        try
            [~, pValue] = ttest2(xClass1, xClass0, 'Vartype', 'unequal');
        catch
            pValue = NaN;
        end
    else
        logWarning(warningFid, 'TIME %s channel %s lacks enough finite values for p-value.', ...
            string(timeValue), string(channels(b)));
    end
    pValues(b) = pValue;

    rows{b} = table(timeValue, channels(b), featureNames(b), class1Mean, class0Mean, ...
        meanDifference, abs(meanDifference), effectSize, abs(effectSize), pValue, testName, nInf, nUninf, ...
        'VariableNames', {'time', 'channel', 'feature_name', 'class1_mean', ...
        'class0_mean', 'mean_difference', 'abs_mean_difference', ...
        'standardised_effect_size', 'abs_standardised_effect_size', 'p_value', ...
        'test_name', 'class1_observations', 'class0_observations'});
end

out = vertcat(rows{:});
out.fdr_p_value = bhFdr(pValues);
out = sortrows(out, {'time', 'abs_standardised_effect_size'}, {'ascend', 'descend'});
end

function d = cohensD(x1, x0)
x1 = x1(isfinite(x1));
x0 = x0(isfinite(x0));
n1 = numel(x1);
n0 = numel(x0);
if n1 < 2 || n0 < 2
    d = NaN;
    return;
end
s1 = var(x1, 0);
s0 = var(x0, 0);
pooled = sqrt(((n1 - 1) * s1 + (n0 - 1) * s0) / max(n1 + n0 - 2, 1));
if pooled == 0 || ~isfinite(pooled)
    d = NaN;
else
    d = (mean(x1) - mean(x0)) / pooled;
end
end

function adjusted = bhFdr(pValues)
adjusted = NaN(size(pValues));
valid = isfinite(pValues);
p = pValues(valid);
if isempty(p)
    return;
end
[sortedP, order] = sort(p);
m = numel(sortedP);
q = sortedP .* m ./ (1:m).';
for i = m-1:-1:1
    q(i) = min(q(i), q(i + 1));
end
q = min(q, 1);
validIdx = find(valid);
adjusted(validIdx(order)) = q;
end

function out = computeLinearSvmImportance(T, featureNames, channels, cfg, warningFid)
timeValue = T.time(1);
requestedFolds = getOuterFoldCount(cfg);
[foldId, ~, nFolds] = make_grouped_folds(T.subject_key, T.analysis_label, requestedFolds, cfg.RANDOM_SEED + round(timeValue));

coefAbs = NaN(nFolds, numel(featureNames));
coefSigned = NaN(nFolds, numel(featureNames));
boxValues = NaN(nFolds, 1);
for fold = 1:nFolds
    trainRows = foldId ~= fold;
    if numel(unique(T.analysis_label(trainRows))) < 2
        logWarning(warningFid, 'TIME %s fold %d has one training class; skipping SVM coefficient estimate.', string(timeValue), fold);
        continue;
    end

    trainTable = T(trainRows, :);
    bestParams = selectLinearBoxConstraint(trainTable, featureNames, cfg, warningFid);
    XRaw = trainTable{:, cellstr(featureNames)};
    [X, ~] = standardizeTrainTest(XRaw, XRaw);
    y = categorical(trainTable.analysis_label);
    try
        model = fitcsvm(X, y, 'KernelFunction', 'linear', ...
            'BoxConstraint', bestParams.BoxConstraint, 'Standardize', false, ...
            'ClassNames', categorical(["class1", "class0"]));
        beta = model.Beta(:).';
        coefSigned(fold, :) = beta;
        coefAbs(fold, :) = abs(beta);
        boxValues(fold) = bestParams.BoxConstraint;
    catch ME
        logWarning(warningFid, 'TIME %s fold %d linear SVM failed: %s', string(timeValue), fold, ME.message);
    end
end

meanAbs = mean(coefAbs, 1, 'omitnan');
stdAbs = std(coefAbs, 0, 1, 'omitnan');
meanSigned = mean(coefSigned, 1, 'omitnan');
out = table(repmat(timeValue, numel(featureNames), 1), channels(:), featureNames(:), ...
    meanAbs(:), stdAbs(:), meanSigned(:), repmat(nFolds, numel(featureNames), 1), ...
    repmat(joinOrNone(string(boxValues(isfinite(boxValues)).')), numel(featureNames), 1), ...
    'VariableNames', {'time', 'channel', 'feature_name', 'mean_abs_linear_svm_coefficient', ...
    'std_abs_linear_svm_coefficient', 'mean_signed_linear_svm_coefficient', ...
    'fold_count', 'selected_box_constraints'});
out = sortrows(out, {'time', 'mean_abs_linear_svm_coefficient'}, {'ascend', 'descend'});
end

function bestParams = selectLinearBoxConstraint(trainTable, featureNames, cfg, warningFid)
boxGrid = [0.1, 1, 10];
if isfield(cfg, 'svm') && isfield(cfg.svm, 'BoxConstraint_grid')
    boxGrid = cfg.svm.BoxConstraint_grid;
end
requestedFolds = getInnerFoldCount(cfg);

try
    [foldId, ~, nFolds] = make_grouped_folds(trainTable.subject_key, trainTable.analysis_label, ...
        requestedFolds, cfg.RANDOM_SEED + 777);
catch ME
    logWarning(warningFid, 'Inner grouped folds failed for linear SVM importance: %s. Using BoxConstraint=1.', ME.message);
    bestParams = struct('BoxConstraint', 1);
    return;
end

bestScore = -Inf;
bestBox = boxGrid(1);
for b = 1:numel(boxGrid)
    scores = NaN(nFolds, 1);
    for fold = 1:nFolds
        trainRows = foldId ~= fold;
        valRows = foldId == fold;
        if numel(unique(trainTable.analysis_label(trainRows))) < 2
            continue;
        end
        XTrainRaw = trainTable{trainRows, cellstr(featureNames)};
        XValRaw = trainTable{valRows, cellstr(featureNames)};
        [XTrain, XVal] = standardizeTrainTest(XTrainRaw, XValRaw);
        yTrain = categorical(trainTable.analysis_label(trainRows));
        try
            model = fitcsvm(XTrain, yTrain, 'KernelFunction', 'linear', ...
                'BoxConstraint', boxGrid(b), 'Standardize', false, ...
                'ClassNames', categorical(["class1", "class0"]));
            pred = predict(model, XVal);
            metrics = computeMetrics(trainTable.analysis_label(valRows), string(pred));
            scores(fold) = metrics.balanced_accuracy;
        catch ME
            logWarning(warningFid, 'Inner linear SVM failed: %s', ME.message);
        end
    end
    score = mean(scores, 'omitnan');
    if isfinite(score) && score > bestScore
        bestScore = score;
        bestBox = boxGrid(b);
    end
end
bestParams = struct('BoxConstraint', bestBox);
end

function out = computePermutationImportance(T, featureNames, channels, cfg, warningFid)
timeValue = T.time(1);
requestedFolds = getOuterFoldCount(cfg);
[foldId, ~, nFolds] = make_grouped_folds(T.subject_key, T.analysis_label, requestedFolds, cfg.RANDOM_SEED + round(timeValue) + 9000);
lossIncrease = NaN(nFolds, numel(featureNames));

for fold = 1:nFolds
    trainRows = foldId ~= fold;
    testRows = foldId == fold;
    if numel(unique(T.analysis_label(trainRows))) < 2 || numel(unique(T.analysis_label(testRows))) < 2
        continue;
    end
    trainTable = T(trainRows, :);
    bestParams = selectLinearBoxConstraint(trainTable, featureNames, cfg, warningFid);
    [XTrain, XTest] = standardizeTrainTest(trainTable{:, cellstr(featureNames)}, T{testRows, cellstr(featureNames)});
    yTrain = categorical(trainTable.analysis_label);
    yTest = string(T.analysis_label(testRows));
    try
        model = fitcsvm(XTrain, yTrain, 'KernelFunction', 'linear', ...
            'BoxConstraint', bestParams.BoxConstraint, 'Standardize', false, ...
            'ClassNames', categorical(["class1", "class0"]));
        baselinePred = string(predict(model, XTest));
        baselineMetrics = computeMetrics(yTest, baselinePred);
        baselineLoss = 1 - baselineMetrics.balanced_accuracy;
        for b = 1:numel(featureNames)
            XPerm = XTest;
            XPerm(:, b) = XPerm(randperm(size(XPerm, 1)), b);
            permPred = string(predict(model, XPerm));
            permMetrics = computeMetrics(yTest, permPred);
            lossIncrease(fold, b) = (1 - permMetrics.balanced_accuracy) - baselineLoss;
        end
    catch ME
        logWarning(warningFid, 'Permutation importance failed for TIME %s fold %d: %s', string(timeValue), fold, ME.message);
    end
end

out = table(repmat(timeValue, numel(featureNames), 1), channels(:), featureNames(:), ...
    mean(lossIncrease, 1, 'omitnan').', std(lossIncrease, 0, 1, 'omitnan').', ...
    'VariableNames', {'time', 'channel', 'feature_name', ...
    'mean_balanced_accuracy_loss_increase', 'std_balanced_accuracy_loss_increase'});
out = sortrows(out, {'time', 'mean_balanced_accuracy_loss_increase'}, {'ascend', 'descend'});
end

function metrics = computeMetrics(yTrue, yPred)
yTrue = string(yTrue(:));
yPred = string(yPred(:));
tp = sum(yTrue == "class1" & yPred == "class1");
tn = sum(yTrue == "class0" & yPred == "class0");
fp = sum(yTrue == "class0" & yPred == "class1");
fn = sum(yTrue == "class1" & yPred == "class0");
metrics = struct();
metrics.sensitivity = safeDivide(tp, tp + fn);
metrics.specificity = safeDivide(tn, tn + fp);
metrics.balanced_accuracy = mean([metrics.sensitivity, metrics.specificity], 'omitnan');
end

function value = safeDivide(a, b)
if b == 0
    value = NaN;
else
    value = a / b;
end
end

function [XTrain, XTest, mu, sigma] = standardizeTrainTest(XTrainRaw, XTestRaw)
mu = mean(XTrainRaw, 1, 'omitnan');
sigma = std(XTrainRaw, 0, 1, 'omitnan');
sigma(sigma == 0 | ~isfinite(sigma)) = 1;
XTrain = (XTrainRaw - mu) ./ sigma;
XTest = (XTestRaw - mu) ./ sigma;
end

function ranked = makeRankedTable(univariateTable, svmTable, permutationTable)
if isempty(univariateTable) || isempty(svmTable)
    ranked = table();
    return;
end
ranked = outerjoin(univariateTable, svmTable, 'Keys', {'time', 'channel', 'feature_name'}, 'MergeKeys', true);
if ~isempty(permutationTable)
    ranked = outerjoin(ranked, permutationTable, 'Keys', {'time', 'channel', 'feature_name'}, 'MergeKeys', true);
end
ranked = sortrows(ranked, {'time', 'abs_standardised_effect_size', 'mean_abs_linear_svm_coefficient'}, ...
    {'ascend', 'descend', 'descend'});
end

function writeOutputs(cfg, univariateTable, svmTable, permutationTable, rankedTable)
writetable(univariateTable, fullfile(cfg.paths.tables, 'band_importance_univariate.csv'));
writetable(svmTable, fullfile(cfg.paths.tables, 'band_importance_linear_svm.csv'));
writetable(permutationTable, fullfile(cfg.paths.tables, 'band_importance_permutation.csv'));
writetable(rankedTable, fullfile(cfg.paths.tables, 'band_importance_ranked.csv'));
save(fullfile(cfg.paths.cache, 'band_importance_results.mat'), ...
    'univariateTable', 'svmTable', 'permutationTable', 'rankedTable', '-v7.3');
end

function makeFigures(cfg, univariateTable, svmTable, rankedTable)
if ~isempty(univariateTable)
    plotHeatmap(cfg, univariateTable, 'abs_standardised_effect_size', ...
        'Band x TIME absolute effect size', 'band_importance_effect_size_heatmap.png');
end
if ~isempty(svmTable)
    plotHeatmap(cfg, svmTable, 'mean_abs_linear_svm_coefficient', ...
        'Band x TIME linear SVM importance', 'band_importance_svm_heatmap.png');
end
if ~isempty(rankedTable)
    plotTopBars(cfg, rankedTable);
end
end

function plotHeatmap(cfg, T, valueVar, titleText, filename)
timeValues = sort(unique(T.time)).';
channels = sort(unique(T.channel)).';
Z = NaN(numel(channels), numel(timeValues));
for d = 1:numel(timeValues)
    for b = 1:numel(channels)
        rows = T.time == timeValues(d) & T.channel == channels(b);
        if any(rows)
            Z(b, d) = T.(valueVar)(find(rows, 1));
        end
    end
end

fig = figure('Visible', 'off', 'Position', [100, 100, 900, 650]);
imagesc(timeValues, channels, Z);
set(gca, 'YDir', 'normal');
colorbar;
xlabel('TIME');
ylabel('Channel');
title(titleText);
saveas(fig, fullfile(cfg.paths.figures, filename));
close(fig);
end

function plotTopBars(cfg, rankedTable)
timeValues = sort(unique(rankedTable.time)).';
nShow = min(5, numel(unique(rankedTable.channel)));
fig = figure('Visible', 'off', 'Position', [100, 100, 1300, 280 * ceil(numel(timeValues) / 3)]);
tiledlayout(ceil(numel(timeValues) / 3), 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for d = 1:numel(timeValues)
    nexttile;
    rows = rankedTable.time == timeValues(d);
    T = sortrows(rankedTable(rows, :), 'abs_standardised_effect_size', 'descend');
    T = T(1:min(nShow, height(T)), :);
    bar(categorical(string(T.channel)), T.abs_standardised_effect_size);
    xlabel('Channel');
    ylabel('|effect size|');
    title("TIME " + string(timeValues(d)));
    grid on;
end
saveas(fig, fullfile(cfg.paths.figures, 'band_importance_top_effect_size_bars.png'));
close(fig);
end

function n = countSubjects(T, label)
rows = T.analysis_label == label;
if any(rows)
    n = numel(unique(T.subject_key(rows)));
else
    n = 0;
end
end

function k = getOuterFoldCount(cfg)
k = 5;
if isfield(cfg, 'speed') && isfield(cfg.speed, 'outer_grouped_folds_fast') && cfg.FAST_MODE
    k = cfg.speed.outer_grouped_folds_fast;
elseif isfield(cfg, 'analysis') && isfield(cfg.analysis, 'outer_grouped_folds')
    k = cfg.analysis.outer_grouped_folds;
end
end

function k = getInnerFoldCount(cfg)
k = 3;
if isfield(cfg, 'speed') && isfield(cfg.speed, 'inner_grouped_folds_fast') && cfg.FAST_MODE
    k = cfg.speed.inner_grouped_folds_fast;
elseif isfield(cfg, 'analysis') && isfield(cfg.analysis, 'inner_grouped_folds')
    k = cfg.analysis.inner_grouped_folds;
end
end

function out = vertcatOrEmpty(parts)
parts = parts(~cellfun(@isempty, parts));
if isempty(parts)
    out = table();
else
    out = vertcat(parts{:});
end
end

function text = joinOrNone(values)
if isempty(values)
    text = "<none>";
else
    text = strjoin(string(values(:).'), ', ');
end
end

function logWarning(fid, varargin)
fprintf(fid, 'WARNING: %s\n', sprintf(varargin{:}));
end
